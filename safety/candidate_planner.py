"""Generate and score 40 smooth alternative paths when the VLA route is unsafe.

This module plugs into the existing pipeline:

    VLA output
      -> vla_path.py
      -> lookahead_path.py
      -> normal corridor/occupancy check
      -> if UNSAFE:
           candidate_planner.py
              * generate 40 smooth paths (20 left + 20 right)
              * create a 1.15 m-wide curved safety band for each
              * hard-reject any path whose band touches occupied space
              * score remaining paths by:
                    1) obstacle clearance
                    2) staying close to the original VLA/lookahead path
                    3) smoothness
              * return the best safe candidate

No OpenCV is used. Everything is NumPy/math.

Coordinate convention:
- x = forward in meters
- y = left in meters
- negative y = right
- robot starts at (0, 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from lookahead_path import LookaheadPathInfo
from occupancy_checker import (
    GridGeometry,
    OccupancyValues,
    corridor_points_to_grid_cells,
)


Side = Literal["left", "right"]


@dataclass(frozen=True)
class CandidatePlannerConfig:
    candidate_count: int = 40
    max_lateral_offset_m: float = 1.0
    robot_width_m: float = 0.75
    robot_length_m: float = 1.10
    safety_margin_m: float = 0.20
    sample_spacing_m: float = 0.05
    # Finish the main sideways shift before the end of the 3 m horizon so
    # the robot can actually get around an obstacle that appears mid-range.
    lateral_shift_completion_fraction: float = 0.40

    # Score weights. Collision is NOT a score: collision is a hard rejection.
    clearance_weight: float = 6.0
    vla_deviation_weight: float = 2.0
    smoothness_weight: float = 1.0

    # Prefer some extra room beyond the already-inflated 1.15 m safety band.
    desired_extra_clearance_m: float = 0.25


@dataclass(frozen=True)
class CandidatePath:
    index: int
    side: Side
    target_lateral_offset_m: float
    centerline_points_m: np.ndarray


@dataclass(frozen=True)
class CandidateSafetyBand:
    filled_points_m: np.ndarray
    effective_width_m: float
    half_width_m: float


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: CandidatePath
    safe: bool
    score: float | None
    clearance_penalty: float | None
    vla_deviation_penalty: float | None
    smoothness_penalty: float | None
    min_clearance_m: float | None
    occupied_count: int
    unknown_count: int
    out_of_bounds_count: int

    @property
    def target_lateral_offset_m(self) -> float:
        return self.candidate.target_lateral_offset_m

    @property
    def centerline_points_m(self) -> np.ndarray:
        return self.candidate.centerline_points_m


@dataclass(frozen=True)
class CandidatePlanResult:
    best_candidate: ScoredCandidate | None
    candidates: tuple[ScoredCandidate, ...]
    safe_candidate_count: int
    rejected_candidate_count: int


def _validate_config(config: CandidatePlannerConfig) -> None:
    if config.candidate_count < 2 or config.candidate_count % 2 != 0:
        raise ValueError("candidate_count must be an even integer >= 2")
    if config.max_lateral_offset_m <= 0:
        raise ValueError("max_lateral_offset_m must be > 0")
    if config.robot_width_m <= 0:
        raise ValueError("robot_width_m must be > 0")
    if config.robot_length_m <= 0:
        raise ValueError("robot_length_m must be > 0")
    if config.safety_margin_m < 0:
        raise ValueError("safety_margin_m must be >= 0")
    if config.sample_spacing_m <= 0:
        raise ValueError("sample_spacing_m must be > 0")
    if not (0.0 < config.lateral_shift_completion_fraction <= 1.0):
        raise ValueError("lateral_shift_completion_fraction must be in (0,1]")
    if config.desired_extra_clearance_m < 0:
        raise ValueError("desired_extra_clearance_m must be >= 0")


def _arc_progress(points: np.ndarray) -> tuple[np.ndarray, float]:
    if len(points) < 2:
        return np.zeros((len(points),), dtype=np.float64), 0.0

    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(s[-1])
    return s, total


def _unit_normals(points: np.ndarray) -> np.ndarray:
    """Return a left-facing unit normal at every path point."""
    if len(points) == 1:
        return np.array([[0.0, 1.0]], dtype=np.float64)

    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1)

    # Fall back to the previous valid tangent if a duplicate point appears.
    for i in range(len(tangents)):
        if norms[i] <= 1e-12:
            if i > 0 and norms[i - 1] > 1e-12:
                tangents[i] = tangents[i - 1]
                norms[i] = norms[i - 1]
            else:
                tangents[i] = np.array([1.0, 0.0])
                norms[i] = 1.0

    unit_t = tangents / norms[:, None]
    # In (forward=x, left=y), rotating tangent +90 deg gives left normal.
    return np.column_stack((-unit_t[:, 1], unit_t[:, 0]))


def _quintic_smoothstep(t: np.ndarray) -> np.ndarray:
    """0->1 profile with zero slope and zero acceleration at both ends."""
    t = np.clip(t, 0.0, 1.0)
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def generate_candidate_paths(
    lookahead: LookaheadPathInfo,
    config: CandidatePlannerConfig = CandidatePlannerConfig(),
) -> list[CandidatePath]:
    """Generate 20 smooth left + 20 smooth right candidates by default.

    Each candidate starts exactly on the reference VLA/lookahead path.
    The sideways change grows smoothly and reaches its target offset near 3 m.
    """
    _validate_config(config)

    reference = np.asarray(lookahead.check_points, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2:
        raise ValueError("lookahead.check_points must have shape (N,2), N>=2")

    s, total = _arc_progress(reference)
    if total <= 1e-9:
        raise ValueError("lookahead path has near-zero length")

    t = s / total
    # Reach the chosen lateral offset before the end of the horizon, then
    # hold that offset. This starts avoidance early enough for obstacles
    # that are already inside the 3 m detection region.
    shifted_t = np.clip(t / config.lateral_shift_completion_fraction, 0.0, 1.0)
    profile = _quintic_smoothstep(shifted_t)
    normals = _unit_normals(reference)

    per_side = config.candidate_count // 2
    magnitudes = np.linspace(
        config.max_lateral_offset_m / per_side,
        config.max_lateral_offset_m,
        per_side,
        dtype=np.float64,
    )

    candidates: list[CandidatePath] = []
    idx = 0

    for side, sign in (("left", +1.0), ("right", -1.0)):
        for mag in magnitudes:
            target = float(sign * mag)
            offsets = (profile * target)[:, None] * normals
            path = reference + offsets

            candidates.append(
                CandidatePath(
                    index=idx,
                    side=side,
                    target_lateral_offset_m=target,
                    centerline_points_m=path,
                )
            )
            idx += 1

    return candidates


def _symmetric_samples(half_width: float, spacing: float) -> np.ndarray:
    count_each_side = max(1, int(np.ceil(half_width / spacing)))
    positive = np.linspace(0.0, half_width, count_each_side + 1)
    return np.concatenate((-positive[:0:-1], positive))


def build_curved_safety_band(
    centerline_points_m: np.ndarray,
    config: CandidatePlannerConfig = CandidatePlannerConfig(),
) -> CandidateSafetyBand:
    """Sweep the 1.15 m effective width along a curved candidate path."""
    _validate_config(config)

    centerline = np.asarray(centerline_points_m, dtype=np.float64)
    if centerline.ndim != 2 or centerline.shape[1] != 2 or len(centerline) < 2:
        raise ValueError("centerline_points_m must have shape (N,2), N>=2")

    effective_width = config.robot_width_m + 2.0 * config.safety_margin_m
    half_width = 0.5 * effective_width
    normals = _unit_normals(centerline)
    lateral = _symmetric_samples(half_width, config.sample_spacing_m)

    bands = []
    for c, n in zip(centerline, normals):
        bands.append(c[None, :] + lateral[:, None] * n[None, :])

    filled = np.vstack(bands)

    return CandidateSafetyBand(
        filled_points_m=filled,
        effective_width_m=float(effective_width),
        half_width_m=float(half_width),
    )


def _inspect_band(
    occupancy_grid: np.ndarray,
    band: CandidateSafetyBand,
    geometry: GridGeometry,
    values: OccupancyValues,
    *,
    unknown_is_unsafe: bool,
    out_of_bounds_is_unsafe: bool,
) -> tuple[bool, int, int, int]:
    """Check a curved band directly against the OGM."""
    grid = np.asarray(occupancy_grid)
    if grid.ndim != 2:
        raise ValueError("occupancy_grid must be 2D")

    cells, _ = corridor_points_to_grid_cells(band.filled_points_m, geometry)
    cells = np.unique(cells, axis=0)

    rows, cols = grid.shape
    in_bounds = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < rows)
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < cols)
    )

    out_count = int((~in_bounds).sum())
    valid = cells[in_bounds]

    if len(valid):
        sampled = grid[valid[:, 0], valid[:, 1]]
        occupied_count = int(np.count_nonzero(sampled == values.occupied))
        unknown_count = int(np.count_nonzero(sampled == values.unknown))
    else:
        occupied_count = 0
        unknown_count = 0

    unsafe = occupied_count > 0
    if unknown_is_unsafe and unknown_count > 0:
        unsafe = True
    if out_of_bounds_is_unsafe and out_count > 0:
        unsafe = True

    return (not unsafe), occupied_count, unknown_count, out_count


def _min_centerline_clearance_m(
    occupancy_grid: np.ndarray,
    centerline_points_m: np.ndarray,
    geometry: GridGeometry,
    values: OccupancyValues,
    half_width_m: float,
) -> float:
    """Approximate nearest obstacle distance from the outside of the safety band.

    We compute nearest centerline-to-occupied-cell distance using grid-cell
    centers, then subtract half the effective safety width.

    Returns +inf if the grid has no occupied cells.
    """
    grid = np.asarray(occupancy_grid)
    occupied_rc = np.argwhere(grid == values.occupied)
    if len(occupied_rc) == 0:
        return float("inf")

    center_rc, _ = corridor_points_to_grid_cells(centerline_points_m, geometry)

    best_cells = float("inf")

    # Chunk the path so memory stays small even on a large grid.
    chunk_size = 32
    occupied = occupied_rc.astype(np.float64)

    for i in range(0, len(center_rc), chunk_size):
        chunk = center_rc[i : i + chunk_size].astype(np.float64)
        diff = chunk[:, None, :] - occupied[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        local_best = float(np.sqrt(np.min(dist2)))
        if local_best < best_cells:
            best_cells = local_best

    center_clearance_m = best_cells * geometry.resolution_m
    return float(center_clearance_m - half_width_m)


def _vla_deviation_penalty(
    candidate_points: np.ndarray,
    reference_points: np.ndarray,
    max_offset_m: float,
) -> float:
    deviation = np.linalg.norm(candidate_points - reference_points, axis=1)
    denom = max(float(max_offset_m), 1e-9)
    return float(np.mean(deviation) / denom)


def _smoothness_penalty(points: np.ndarray) -> float:
    """Small when heading changes gradually; larger for sharp/jerky paths."""
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    valid = lengths > 1e-9
    deltas = deltas[valid]
    if len(deltas) < 2:
        return 0.0

    headings = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
    dtheta = np.diff(headings)
    # Normalize by a moderate 30-degree change so the term stays intuitive.
    return float(np.mean(np.abs(dtheta)) / np.deg2rad(30.0))


def _clearance_penalty(min_clearance_m: float, desired_extra_clearance_m: float) -> float:
    if np.isinf(min_clearance_m):
        return 0.0
    if desired_extra_clearance_m <= 1e-12:
        return 0.0

    remaining = max(0.0, desired_extra_clearance_m - max(0.0, min_clearance_m))
    ratio = remaining / desired_extra_clearance_m
    return float(ratio * ratio)


def score_candidate(
    occupancy_grid: np.ndarray,
    candidate: CandidatePath,
    reference_points_m: np.ndarray,
    geometry: GridGeometry,
    values: OccupancyValues,
    config: CandidatePlannerConfig = CandidatePlannerConfig(),
    *,
    unknown_is_unsafe: bool = False,
    out_of_bounds_is_unsafe: bool = True,
) -> ScoredCandidate:
    """Hard collision check first; weighted score only if the candidate is safe."""
    band = build_curved_safety_band(candidate.centerline_points_m, config)

    safe, occupied_count, unknown_count, out_count = _inspect_band(
        occupancy_grid,
        band,
        geometry,
        values,
        unknown_is_unsafe=unknown_is_unsafe,
        out_of_bounds_is_unsafe=out_of_bounds_is_unsafe,
    )

    if not safe:
        return ScoredCandidate(
            candidate=candidate,
            safe=False,
            score=None,
            clearance_penalty=None,
            vla_deviation_penalty=None,
            smoothness_penalty=None,
            min_clearance_m=None,
            occupied_count=occupied_count,
            unknown_count=unknown_count,
            out_of_bounds_count=out_count,
        )

    min_clearance = _min_centerline_clearance_m(
        occupancy_grid,
        candidate.centerline_points_m,
        geometry,
        values,
        band.half_width_m,
    )
    clearance_pen = _clearance_penalty(
        min_clearance,
        config.desired_extra_clearance_m,
    )
    deviation_pen = _vla_deviation_penalty(
        candidate.centerline_points_m,
        reference_points_m,
        config.max_lateral_offset_m,
    )
    smooth_pen = _smoothness_penalty(candidate.centerline_points_m)

    score = (
        config.clearance_weight * clearance_pen
        + config.vla_deviation_weight * deviation_pen
        + config.smoothness_weight * smooth_pen
    )

    return ScoredCandidate(
        candidate=candidate,
        safe=True,
        score=float(score),
        clearance_penalty=float(clearance_pen),
        vla_deviation_penalty=float(deviation_pen),
        smoothness_penalty=float(smooth_pen),
        min_clearance_m=float(min_clearance),
        occupied_count=occupied_count,
        unknown_count=unknown_count,
        out_of_bounds_count=out_count,
    )


def plan_best_candidate(
    *,
    occupancy_grid: np.ndarray,
    lookahead: LookaheadPathInfo,
    geometry: GridGeometry,
    values: OccupancyValues,
    config: CandidatePlannerConfig = CandidatePlannerConfig(),
    unknown_is_unsafe: bool = False,
    out_of_bounds_is_unsafe: bool = True,
) -> CandidatePlanResult:
    """Generate all candidates, reject collisions, score survivors, return best."""
    candidates = generate_candidate_paths(lookahead, config)
    reference = np.asarray(lookahead.check_points, dtype=np.float64)

    scored = [
        score_candidate(
            occupancy_grid,
            candidate,
            reference,
            geometry,
            values,
            config,
            unknown_is_unsafe=unknown_is_unsafe,
            out_of_bounds_is_unsafe=out_of_bounds_is_unsafe,
        )
        for candidate in candidates
    ]

    safe = [c for c in scored if c.safe and c.score is not None]
    best = min(safe, key=lambda c: c.score) if safe else None

    return CandidatePlanResult(
        best_candidate=best,
        candidates=tuple(scored),
        safe_candidate_count=len(safe),
        rejected_candidate_count=len(scored) - len(safe),
    )
