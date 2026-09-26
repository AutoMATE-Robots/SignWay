#!/usr/bin/env python3
"""Live occupancy-aware trajectory refinement for SignWay (core module).

Sits between the VLA policy output (16-dim waypoint vector) and the executor:

    raw 8x(x,y) waypoints
        -> odometry-compensate into the grid's capture frame
        -> assess raw path (extended to a fixed metric lookahead)
        -> SAFE?  yes -> pass raw through
                  no  -> contest: ~100 candidates (rotation x ramped-lateral
                         x speed), scored on obstacle clearance + fidelity +
                         smoothness; best safe candidate wins
        -> if nothing is safe: emit a slowed/stopped version of raw, blocked=True
        -> return SAME 16-dim format the executor already consumes

Design rules baked in:
  * NEVER raises out of refine() — any internal error returns the raw action
    with reason="error" (the robot must behave exactly as it does today).
  * The extension beyond wp8 is SCORED, never executed.
  * Unknown cells (-1) count as free for clearance (gray cone edges must not
    squeeze the robot) but are tracked as unknown_frac.
  * Extension shrinks when the path is turning (straight extrapolation is
    wrong at junctions; do not fight the VLA's turn).
  * No RNG anywhere — same inputs, same output.

Pure numpy + cv2. No ROS imports — unit-testable off-robot. ROS wiring lives
in refinement_adapter.py / jetson_occupancy_client.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

OCC_UNKNOWN = -1
OCC_FREE = 0
OCC_OCCUPIED = 100


# --------------------------------------------------------------------------
# Config / result types
# --------------------------------------------------------------------------

@dataclass
class RefinerConfig:
    # grid geometry (must match the occupancy server; verified against
    # OccupancyGrid.info at runtime by the adapter)
    resolution: float = 0.10
    lateral_half_m: float = 5.0
    forward_m: float = 16.0
    # robot + safety
    robot_radius: float = 0.30        # Pepper footprint radius (m)
    d_safe: float = 0.50              # desired clearance beyond footprint (m)
    hard_clearance: float = 0.02      # below this = collision, candidate rejected
    # lookahead (scoring horizon — fixed METRIC length, never "N waypoints")
    ext_len: float = 2.0              # straight-ish paths (m, total incl. wp path)
    ext_len_turning: float = 0.8      # when the VLA is mid-turn
    turning_lat_thresh: float = 0.05  # |y8| above this counts as turning
    turning_heading_deg: float = 15.0 # or first->last segment heading change
    sample_step: float = 0.05         # dense sampling along scored path (m)
    # candidate set (deterministic)
    rotations_deg: Tuple[float, ...] = (-25, -18, -12, -6, 0, 6, 12, 18, 25)
    laterals_m: Tuple[float, ...] = (-0.4, -0.2, 0.0, 0.2, 0.4)
    speed_scales: Tuple[float, ...] = (1.0, 0.7)
    # score weights
    w_obs: float = 60.0
    w_fid: float = 8.0
    w_smooth: float = 200.0
    w_slow: float = 0.15              # mild penalty so it doesn't always crawl
    # staleness
    max_grid_age_s: float = 1.5       # older grid -> passthrough
    occupied_thresh: int = 50


@dataclass
class RefineResult:
    action: np.ndarray                # (16,) to execute — SAME format as raw
    raw_action: np.ndarray            # (16,) as received
    safe: bool                        # raw path met d_safe over the lookahead
    modified: bool                    # action != raw_action
    blocked: bool                     # no safe candidate existed; action is slowed raw
    reason: str                       # "safe" | "refined" | "blocked" |
                                      # "no_grid" | "stale_grid" | "error"
    candidate: str = "raw"
    min_clearance_raw: float = float("nan")
    min_clearance: float = float("nan")
    first_obstacle_m: float = float("nan")   # along-path distance to first
                                             # sub-d_safe clearance (raw path)
    unknown_frac: float = float("nan")       # of sampled raw-path cells
    grid_age_s: float = float("nan")
    ext_len_used: float = float("nan")
    compute_ms: float = 0.0

    def log_dict(self) -> dict:
        """JSON-serializable record for the per-cycle JSONL log (paper data)."""
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in self.__dict__.items()}
        return d


@dataclass
class GridSnapshot:
    """One occupancy grid plus everything precomputed for fast scoring."""
    grid: np.ndarray                  # (rows fwd, cols right) int16 -1/0/100
    stamp: float                      # capture time (s, same clock as odom)
    pose: Optional[Tuple[float, float, float]] = None  # odom (x, y, yaw) at capture
    dist_m: np.ndarray = field(default=None)  # distance-to-obstacle (m) per cell

    def __post_init__(self):
        occ = (self.grid >= 50).astype(np.uint8)
        free = ((1 - occ) * 255).astype(np.uint8)
        # unknown counts as free here on purpose (see module docstring)
        self.dist_m = cv2.distanceTransform(free, cv2.DIST_L2, 5).astype(np.float32)

    def finish(self, resolution: float) -> None:
        self.dist_m *= resolution


def make_snapshot(grid: np.ndarray, stamp: float, resolution: float,
                  pose=None) -> GridSnapshot:
    snap = GridSnapshot(grid=np.asarray(grid, dtype=np.int16), stamp=stamp,
                        pose=pose)
    snap.finish(resolution)
    return snap


# --------------------------------------------------------------------------
# Pure geometry helpers (each individually unit-tested)
# --------------------------------------------------------------------------

def waypoints_from_action(action16) -> np.ndarray:
    a = np.asarray(action16, dtype=np.float64).reshape(-1)
    if a.shape[0] != 16:
        raise ValueError(f"expected 16-dim action, got {a.shape}")
    return a.reshape(8, 2)            # [:,0]=x forward, [:,1]=y left (m)


def action_from_waypoints(wp: np.ndarray) -> np.ndarray:
    return np.asarray(wp, dtype=np.float32).reshape(16)


def meters_to_cell(pts: np.ndarray, cfg: RefinerConfig) -> np.ndarray:
    """Robot/grid frame (x fwd, y LEFT) -> native grid (row fwd, col RIGHT).
    THE y-sign flip lives here and only here."""
    rows = np.floor(pts[:, 0] / cfg.resolution).astype(int)
    cols = np.floor((cfg.lateral_half_m - pts[:, 1]) / cfg.resolution).astype(int)
    return np.stack([rows, cols], axis=1)


def compensate(pts_now: np.ndarray, pose_grid, pose_now) -> np.ndarray:
    """Express points given in the CURRENT robot frame in the GRID-CAPTURE
    robot frame, using odometry poses (x, y, yaw) from the same source.
    If either pose is missing, points pass through unchanged."""
    if pose_grid is None or pose_now is None:
        return pts_now
    x0, y0, th0 = pose_grid
    x1, y1, th1 = pose_now
    c1, s1 = math.cos(th1), math.sin(th1)
    wx = x1 + pts_now[:, 0] * c1 - pts_now[:, 1] * s1
    wy = y1 + pts_now[:, 0] * s1 + pts_now[:, 1] * c1
    c0, s0 = math.cos(th0), math.sin(th0)
    gx = (wx - x0) * c0 + (wy - y0) * s0
    gy = -(wx - x0) * s0 + (wy - y0) * c0
    return np.stack([gx, gy], axis=1)


def densify_and_extend(wp: np.ndarray, total_len: float, sample: float) -> np.ndarray:
    """Origin + waypoints densely resampled every `sample` m, then continued
    along the LAST-SEGMENT heading to `total_len` m. Scoring only."""
    path = np.vstack([[0.0, 0.0], wp])
    seg = np.diff(path, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    if seglen.sum() < 1e-6:           # stopped robot: nothing to extend
        return path[:1]
    last = seg[np.nonzero(seglen)[0][-1]]
    heading = last / np.linalg.norm(last)
    remaining = total_len - seglen.sum()
    if remaining > 0:
        n = int(remaining / sample) + 1
        ext = wp[-1] + heading[None, :] * (np.arange(1, n + 1) * sample)[:, None]
        path = np.vstack([path, ext])
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    s = np.arange(0.0, d[-1] + 1e-9, sample)
    return np.stack([np.interp(s, d, path[:, 0]), np.interp(s, d, path[:, 1])], axis=1)


def is_turning(wp: np.ndarray, cfg: RefinerConfig) -> bool:
    if abs(wp[-1, 1]) > cfg.turning_lat_thresh:
        return True
    seg = np.diff(np.vstack([[0.0, 0.0], wp]), axis=0)
    ln = np.linalg.norm(seg, axis=1)
    ok = ln > 1e-6
    if ok.sum() < 2:
        return False
    first, last = seg[ok][0] / ln[ok][0], seg[ok][-1] / ln[ok][-1]
    ang = math.degrees(math.acos(np.clip(np.dot(first, last), -1.0, 1.0)))
    return ang > cfg.turning_heading_deg


# --------------------------------------------------------------------------
# The refiner
# --------------------------------------------------------------------------

class TrajectoryRefiner:
    def __init__(self, cfg: RefinerConfig = None):
        self.cfg = cfg or RefinerConfig()

    # ---- scoring pieces ----
    def _clearance(self, dense_grid_frame: np.ndarray, snap: GridSnapshot):
        cfg = self.cfg
        cells = meters_to_cell(dense_grid_frame, cfg)
        rows = np.clip(cells[:, 0], 0, snap.grid.shape[0] - 1)
        cols = np.clip(cells[:, 1], 0, snap.grid.shape[1] - 1)
        clear = snap.dist_m[rows, cols] - cfg.robot_radius
        unknown = (snap.grid[rows, cols] == OCC_UNKNOWN)
        return clear, unknown

    def _score(self, cand_wp, raw_wp, speed_scale, snap, pose_grid, pose_now,
               ext_len):
        cfg = self.cfg
        dense = densify_and_extend(cand_wp, ext_len, cfg.sample_step)
        dense_g = compensate(dense, pose_grid, pose_now)
        clear, _ = self._clearance(dense_g, snap)
        if clear.size == 0 or (clear < cfg.hard_clearance).any():
            return None, clear
        viol = np.maximum(0.0, cfg.d_safe - clear)
        obs = float((viol ** 2).sum()) * cfg.sample_step
        fid = float(((cand_wp - raw_wp) ** 2).sum())
        dd = np.diff(cand_wp, axis=0, n=2)
        smooth = float((dd ** 2).sum())
        slow = cfg.w_slow * (1.0 - speed_scale)
        total = cfg.w_obs * obs + cfg.w_fid * fid + cfg.w_smooth * smooth + slow
        return total, clear

    def _candidates(self, raw_wp: np.ndarray):
        """Deterministic edits of the raw path. Raw itself is candidate #0 so
        the layer can never do worse than raw by its own scoring."""
        cfg = self.cfg
        ramp = (np.arange(1, len(raw_wp) + 1) / len(raw_wp))[:, None]
        yield "raw", raw_wp, 1.0
        for rot in cfg.rotations_deg:
            c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
            R = np.array([[c, -s], [s, c]])
            rotated = raw_wp @ R.T
            for lat in cfg.laterals_m:
                shifted = rotated + ramp * np.array([0.0, lat])
                for sp in cfg.speed_scales:
                    if rot == 0.0 and lat == 0.0 and sp == 1.0:
                        continue      # that's "raw"
                    yield f"rot{rot:+.0f}_lat{lat:+.1f}_sp{sp:.1f}", shifted * sp, sp

    # ---- main entry ----
    def refine(self, action16, snap: Optional[GridSnapshot],
               pose_now=None, now: Optional[float] = None) -> RefineResult:
        t0 = time.perf_counter()
        raw = np.asarray(action16, dtype=np.float32).reshape(-1)[:16].copy()
        try:
            return self._refine_inner(raw, snap, pose_now, now, t0)
        except Exception as exc:  # NEVER break the control loop
            return RefineResult(action=raw, raw_action=raw, safe=False,
                                modified=False, blocked=False,
                                reason=f"error:{type(exc).__name__}",
                                compute_ms=(time.perf_counter() - t0) * 1e3)

    def _refine_inner(self, raw, snap, pose_now, now, t0) -> RefineResult:
        cfg = self.cfg
        base = dict(raw_action=raw, modified=False, blocked=False)

        if snap is None:
            return RefineResult(action=raw, safe=False, reason="no_grid",
                                compute_ms=(time.perf_counter() - t0) * 1e3, **base)
        now = time.time() if now is None else now
        age = now - snap.stamp
        if age > cfg.max_grid_age_s:
            return RefineResult(action=raw, safe=False, reason="stale_grid",
                                grid_age_s=age,
                                compute_ms=(time.perf_counter() - t0) * 1e3, **base)

        raw_wp = waypoints_from_action(raw)
        ext_len = cfg.ext_len_turning if is_turning(raw_wp, cfg) else cfg.ext_len

        # ---- assess raw ----
        dense_raw = densify_and_extend(raw_wp, ext_len, cfg.sample_step)
        dense_raw_g = compensate(dense_raw, snap.pose, pose_now)
        clear_raw, unknown = self._clearance(dense_raw_g, snap)
        min_raw = float(clear_raw.min()) if clear_raw.size else float("nan")
        below = np.nonzero(clear_raw < cfg.d_safe)[0]
        first_obst = float(below[0] * cfg.sample_step) if below.size else float("nan")
        unk_frac = float(unknown.mean()) if unknown.size else float("nan")

        common = dict(min_clearance_raw=min_raw, first_obstacle_m=first_obst,
                      unknown_frac=unk_frac, grid_age_s=age, ext_len_used=ext_len)

        if clear_raw.size and min_raw >= cfg.d_safe:
            return RefineResult(action=raw, safe=True, reason="safe",
                                min_clearance=min_raw,
                                compute_ms=(time.perf_counter() - t0) * 1e3,
                                **base, **common)

        # ---- contest ----
        best = None
        for name, cand_wp, sp in self._candidates(raw_wp):
            score, clear = self._score(cand_wp, raw_wp, sp, snap,
                                       snap.pose, pose_now, ext_len)
            if score is None:
                continue
            if best is None or score < best[0]:
                best = (score, name, cand_wp, float(clear.min()))

        if best is not None:
            _, name, wp, min_c = best
            action = action_from_waypoints(wp)
            return RefineResult(action=action, safe=False,
                                modified=not np.allclose(action, raw),
                                blocked=False, reason="refined", candidate=name,
                                min_clearance=min_c, raw_action=raw,
                                compute_ms=(time.perf_counter() - t0) * 1e3,
                                **common)

        # ---- blocked: nothing safe — slow the raw path hard ----
        factor = float(np.clip((min_raw if np.isfinite(min_raw) else 0.0)
                               / cfg.d_safe, 0.0, 1.0)) * 0.5
        slowed = action_from_waypoints(raw_wp * max(factor, 0.05))
        return RefineResult(action=slowed, safe=False, modified=True,
                            blocked=True, reason="blocked", candidate="slowed_raw",
                            min_clearance=min_raw, raw_action=raw,
                            compute_ms=(time.perf_counter() - t0) * 1e3,
                            **common)


# --------------------------------------------------------------------------
# nav_msgs/OccupancyGrid packing (pure; ROS wrappers call these)
# Convention: map frame = robot frame at capture (REP 103: x fwd, y left),
# origin at (0, -lateral_half), identity orientation -> rviz renders it
# correctly in front of the robot.
# --------------------------------------------------------------------------

def pack_occupancy_grid(grid: np.ndarray, resolution: float,
                        lateral_half_m: float):
    """native grid (rows fwd, cols RIGHT) -> (data int8 row-major, width,
    height, origin_xy) per nav_msgs convention (data[my*width+mx])."""
    g = np.asarray(grid, dtype=np.int16)
    data = np.ascontiguousarray(g.T[::-1]).astype(np.int8).ravel()
    width, height = g.shape[0], g.shape[1]     # width along x (forward)
    return data, width, height, (0.0, -lateral_half_m)


def unpack_occupancy_grid(data, width: int, height: int) -> np.ndarray:
    """Inverse of pack_occupancy_grid -> native grid (rows fwd, cols right)."""
    arr = np.asarray(data, dtype=np.int16).reshape(height, width)
    return np.ascontiguousarray(arr[::-1].T)
