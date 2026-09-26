#!/usr/bin/env python3
"""SignWay refinement glue — runs Teja's 5-package pipeline on the MSI server.

Wraps (WITHOUT modifying) vla_path / lookahead_path / safety_corridor /
occupancy_checker / candidate_planner into a single call:

    refine_action(action16, grid, ...) -> dict (refined action + metrics)

What this glue adds around the packages:
 1. GRID MOTION COMPENSATION: the grid is in the image-capture frame; the
    action is in the robot's CURRENT frame. Using the odometry poses at both
    times, the grid is rigidly warped (cv2, nearest-neighbor, unknown fill)
    into the current frame — so all five packages run unmodified with one
    fixed GridGeometry and zero frame bugs downstream.
 2. CURVED INITIAL CHECK: the SAFE/UNSAFE gate uses the zero-offset curved
    safety band (score_candidate on the lookahead centerline), not the
    straight rectangle — so turning paths are checked along the path actually
    driven and the planner is never fought mid-turn. (This is the planned v2
    of safety_corridor; the rectangle module stays available for comparison.)
 3. CENTERLINE -> 16-DIM CONVERTER: the winning 3 m centerline is resampled
    at the RAW path's per-waypoint arc lengths, so geometry changes but the
    VLA's speed profile survives, and the executor interface is unchanged.
 4. BLOCKED FALLBACK: best_candidate is None -> raw path scaled hard toward
    stop (blocked=True). A fully blocked corridor is the AR layer's decision;
    this only buys time.
 5. FAIL-OPEN: any internal error returns the raw action with reason
    "error:<type>" — refinement can never behave worse than no refinement.

Everything here is numpy + cv2 + Teja's modules. No ROS, no torch.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from vla_path import build_vla_path
from lookahead_path import build_lookahead_path
from occupancy_checker import GridGeometry, OccupancyValues
from candidate_planner import (
    CandidatePath,
    CandidatePlannerConfig,
    plan_best_candidate,
    score_candidate,
)


@dataclass
class RefinementSettings:
    # grid geometry — MUST match the occupancy server's build parameters
    resolution_m: float = 0.10
    lateral_half_m: float = 5.0
    # lookahead (fit_segments/threshold per the noise study on straight preds)
    total_lookahead_m: float = 3.0
    sample_spacing_m: float = 0.05
    fit_segments: int = 8
    straight_threshold_per_m: float = 0.4
    # planner
    candidate_count: int = 40
    max_lateral_offset_m: float = 1.2     # raised for 0.10 m cells (see notes)
    robot_width_m: float = 0.75           # VERIFY against physical Pepper
    safety_margin_m: float = 0.20
    lateral_shift_completion_fraction: float = 0.40
    # policy
    max_grid_age_s: float = 1.5
    blocked_scale: float = 0.15           # raw path multiplier when nothing is safe
    unknown_is_unsafe: bool = False
    out_of_bounds_is_unsafe: bool = False # our 16 m x 10 m grid: OOB means "far",
                                          # not "wall"; do not punish long lookaheads

    def geometry(self) -> GridGeometry:
        return GridGeometry(
            resolution_m=self.resolution_m,
            robot_row=0,
            robot_col=int(round(self.lateral_half_m / self.resolution_m)),
            forward_direction="+row",
            left_direction="-col",
        )

    def planner_config(self) -> CandidatePlannerConfig:
        return CandidatePlannerConfig(
            candidate_count=self.candidate_count,
            max_lateral_offset_m=self.max_lateral_offset_m,
            robot_width_m=self.robot_width_m,
            safety_margin_m=self.safety_margin_m,
            sample_spacing_m=self.sample_spacing_m,
            lateral_shift_completion_fraction=self.lateral_shift_completion_fraction,
        )


VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


# ---------------------------------------------------------------------------
# 1. Grid motion compensation
# ---------------------------------------------------------------------------

def _meters_to_cell_xy(pts_m: np.ndarray, s: RefinementSettings) -> np.ndarray:
    """(x fwd, y left) meters -> (cell_x=col, cell_y=row) floats for cv2."""
    rows = pts_m[:, 0] / s.resolution_m
    cols = (s.lateral_half_m - pts_m[:, 1]) / s.resolution_m
    return np.stack([cols, rows], axis=1).astype(np.float32)


def warp_grid_to_current_frame(grid: np.ndarray,
                               pose_grid: Optional[Tuple[float, float, float]],
                               pose_now: Optional[Tuple[float, float, float]],
                               s: RefinementSettings) -> np.ndarray:
    """Rigidly re-express the grid in the robot's CURRENT frame using the two
    odometry poses. Cells that rotate/translate out of view become unknown.
    Missing poses -> grid returned unchanged (no compensation possible)."""
    if pose_grid is None or pose_now is None:
        return grid
    x0, y0, th0 = pose_grid
    x1, y1, th1 = pose_now
    if (abs(x1 - x0) + abs(y1 - y0) + abs(th1 - th0)) < 1e-6:
        return grid

    # current-frame point p_c -> grid-frame point p_g (rigid, planar)
    def to_grid_frame(p_c: np.ndarray) -> np.ndarray:
        c1, s1 = math.cos(th1), math.sin(th1)
        wx = x1 + p_c[:, 0] * c1 - p_c[:, 1] * s1
        wy = y1 + p_c[:, 0] * s1 + p_c[:, 1] * c1
        c0, s0 = math.cos(th0), math.sin(th0)
        gx = (wx - x0) * c0 + (wy - y0) * s0
        gy = -(wx - x0) * s0 + (wy - y0) * c0
        return np.stack([gx, gy], axis=1)

    # three non-collinear correspondences define the affine cell map
    anchors_c = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    dst_xy = _meters_to_cell_xy(anchors_c, s)               # cells in NEW grid
    src_xy = _meters_to_cell_xy(to_grid_frame(anchors_c), s)  # cells in OLD grid
    M = cv2.getAffineTransform(dst_xy, src_xy)              # dst -> src

    return cv2.warpAffine(
        grid.astype(np.int16), M, (grid.shape[1], grid.shape[0]),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=int(VALUES.unknown),
    )


# ---------------------------------------------------------------------------
# 2/3. Converter: winning centerline -> 16-dim action with raw speed profile
# ---------------------------------------------------------------------------

def _arc_lengths(points: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def centerline_to_action(centerline: np.ndarray, raw_wp: np.ndarray) -> np.ndarray:
    """Resample the (3 m) centerline at the raw path's per-waypoint arc
    lengths: waypoint i of the refined action sits at the same
    distance-along-path as raw waypoint i. Geometry changes; speed doesn't."""
    raw_path = np.vstack([[0.0, 0.0], raw_wp])
    s_raw = _arc_lengths(raw_path)[1:]                     # 8 targets
    if s_raw[-1] < 1e-6:                                   # stopped robot
        return raw_wp.reshape(16).astype(np.float32)
    s_c = _arc_lengths(centerline)
    x = np.interp(s_raw, s_c, centerline[:, 0])
    y = np.interp(s_raw, s_c, centerline[:, 1])
    return np.stack([x, y], axis=1).reshape(16).astype(np.float32)


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------

def refine_action(action16,
                  grid: Optional[np.ndarray],
                  grid_stamp: Optional[float] = None,
                  grid_pose: Optional[Tuple[float, float, float]] = None,
                  pose_now: Optional[Tuple[float, float, float]] = None,
                  now: Optional[float] = None,
                  settings: RefinementSettings = None) -> dict:
    """Never raises. Returns a JSON-ready dict; key "action" is always a
    16-float list safe to hand to the executor."""
    t0 = time.perf_counter()
    s = settings or RefinementSettings()
    raw = np.asarray(action16, dtype=np.float64).reshape(-1)

    def out(action, reason, *, safe=False, modified=False, blocked=False, **extra):
        return {"action": [float(v) for v in np.asarray(action).reshape(-1)[:16]],
                "raw_action": [float(v) for v in raw[:16]] if raw.size >= 16 else
                              [float(v) for v in raw],
                "reason": reason, "safe": safe, "modified": modified,
                "blocked": blocked,
                "compute_ms": (time.perf_counter() - t0) * 1e3, **extra}

    try:
        if raw.size != 16 or not np.isfinite(raw).all():
            # fail-open: hand back exactly what came in; the node's existing
            # behavior for bad actions applies, refinement adds nothing.
            return out(raw, "error:bad_action")
        if grid is None:
            return out(raw, "no_grid")
        now = time.time() if now is None else now
        if grid_stamp is not None and (now - grid_stamp) > s.max_grid_age_s:
            return out(raw, "stale_grid", grid_age_s=now - grid_stamp)

        grid_c = warp_grid_to_current_frame(grid, grid_pose, pose_now, s)
        geom, values, cfg = s.geometry(), VALUES, s.planner_config()

        vla = build_vla_path(raw)
        lookahead = build_lookahead_path(
            vla, total_lookahead_m=s.total_lookahead_m,
            sample_spacing_m=s.sample_spacing_m,
            fit_segments=s.fit_segments,
            straight_threshold_per_m=s.straight_threshold_per_m)

        # -- curved-band initial check on the zero-offset (raw) path --
        raw_cand = CandidatePath(index=-1, side="left",
                                 target_lateral_offset_m=0.0,
                                 centerline_points_m=lookahead.check_points)
        raw_scored = score_candidate(
            grid_c, raw_cand, lookahead.check_points, geom, values, cfg,
            unknown_is_unsafe=s.unknown_is_unsafe,
            out_of_bounds_is_unsafe=s.out_of_bounds_is_unsafe)

        metrics = {"raw_occupied_count": raw_scored.occupied_count,
                   "raw_unknown_count": raw_scored.unknown_count,
                   "raw_min_clearance_m": raw_scored.min_clearance_m,
                   "curvature_per_m": lookahead.estimated_curvature_per_m,
                   "grid_age_s": (now - grid_stamp) if grid_stamp else None,
                   "compensated": grid_c is not grid}

        if raw_scored.safe:
            return out(raw, "safe", safe=True, **metrics)

        # -- contest --
        plan = plan_best_candidate(
            occupancy_grid=grid_c, lookahead=lookahead, geometry=geom,
            values=values, config=cfg,
            unknown_is_unsafe=s.unknown_is_unsafe,
            out_of_bounds_is_unsafe=s.out_of_bounds_is_unsafe)
        metrics["safe_candidates"] = plan.safe_candidate_count
        metrics["rejected_candidates"] = plan.rejected_candidate_count

        if plan.best_candidate is None:
            slowed = (vla.waypoints * s.blocked_scale).reshape(16)
            return out(slowed, "blocked", modified=True, blocked=True, **metrics)

        best = plan.best_candidate
        refined = centerline_to_action(best.centerline_points_m, vla.waypoints)
        return out(refined, "refined", modified=True,
                   candidate_side=best.candidate.side,
                   candidate_offset_m=best.target_lateral_offset_m,
                   candidate_score=best.score,
                   candidate_min_clearance_m=best.min_clearance_m, **metrics)

    except Exception as exc:  # fail-open, always
        return out(raw, f"error:{type(exc).__name__}")
