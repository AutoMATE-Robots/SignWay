"""Compare the filled safety corridor against a 2D occupancy grid.

No OpenCV and no AI are used here.

Input:
- a 2D occupancy-grid array
- the filled 3.0 m x 1.15 m safety corridor from safety_corridor.py
- grid geometry: resolution, robot cell, and axis directions
- occupancy values: free / occupied / unknown

Output:
- CLEAR / UNSAFE
- counts of occupied / unknown / out-of-bounds cells
- nearest occupied distance ahead when one is found

Coordinate convention coming from the VLA side:
- x = forward in meters
- y = left in meters
- negative y = right
- robot = (0, 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from safety_corridor import SafetyCorridorInfo


GridDirection = Literal["+row", "-row", "+col", "-col"]


@dataclass(frozen=True)
class GridGeometry:
    resolution_m: float
    robot_row: int
    robot_col: int
    forward_direction: GridDirection
    left_direction: GridDirection


@dataclass(frozen=True)
class OccupancyValues:
    free: int | float
    occupied: int | float
    unknown: int | float


@dataclass(frozen=True)
class OccupancyCheckResult:
    safe: bool
    occupied_count: int
    unknown_count: int
    out_of_bounds_count: int
    checked_cell_count: int
    first_occupied_distance_m: float | None
    occupied_cells: np.ndarray
    unknown_cells: np.ndarray


_DIRECTION_TO_RC = {
    "+row": np.array([+1.0, 0.0]),
    "-row": np.array([-1.0, 0.0]),
    "+col": np.array([0.0, +1.0]),
    "-col": np.array([0.0, -1.0]),
}


def _validate_geometry(geometry: GridGeometry) -> None:
    if geometry.resolution_m <= 0:
        raise ValueError("resolution_m must be > 0")

    if geometry.forward_direction not in _DIRECTION_TO_RC:
        raise ValueError(f"Invalid forward_direction: {geometry.forward_direction}")
    if geometry.left_direction not in _DIRECTION_TO_RC:
        raise ValueError(f"Invalid left_direction: {geometry.left_direction}")

    f = _DIRECTION_TO_RC[geometry.forward_direction]
    l = _DIRECTION_TO_RC[geometry.left_direction]

    if abs(float(np.dot(f, l))) > 1e-12:
        raise ValueError("forward_direction and left_direction must be perpendicular")


def corridor_points_to_grid_cells(
    points_m: np.ndarray,
    geometry: GridGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert robot-relative (forward, left) meter points to grid row/col.

    Returns:
        cells_rc: shape (N, 2), integer [row, col]
        forward_m: shape (N,), original forward distance for each mapped point
    """
    _validate_geometry(geometry)

    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_m must have shape (N, 2)")
    if not np.isfinite(points).all():
        raise ValueError("points_m contains NaN or infinite values")

    f_rc = _DIRECTION_TO_RC[geometry.forward_direction]
    l_rc = _DIRECTION_TO_RC[geometry.left_direction]

    forward_cells = points[:, 0] / geometry.resolution_m
    left_cells = points[:, 1] / geometry.resolution_m

    offsets = forward_cells[:, None] * f_rc + left_cells[:, None] * l_rc
    origin = np.array([geometry.robot_row, geometry.robot_col], dtype=np.float64)

    cells_rc = np.rint(origin[None, :] + offsets).astype(np.int64)
    return cells_rc, points[:, 0].copy()


def _deduplicate_cells_with_min_forward(
    cells_rc: np.ndarray,
    forward_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate repeated grid cells and keep the smallest forward distance."""
    if len(cells_rc) == 0:
        return cells_rc.copy(), forward_m.copy()

    best: dict[tuple[int, int], float] = {}
    for (row, col), x_m in zip(cells_rc, forward_m):
        key = (int(row), int(col))
        x = float(x_m)
        if key not in best or x < best[key]:
            best[key] = x

    keys = sorted(best.keys())
    unique_cells = np.asarray(keys, dtype=np.int64)
    min_forward = np.asarray([best[k] for k in keys], dtype=np.float64)
    return unique_cells, min_forward


def check_corridor_against_ogm(
    occupancy_grid: np.ndarray,
    corridor: SafetyCorridorInfo,
    geometry: GridGeometry,
    values: OccupancyValues,
    *,
    unknown_is_unsafe: bool = False,
    out_of_bounds_is_unsafe: bool = True,
) -> OccupancyCheckResult:
    """Return whether any checked corridor cell conflicts with the OGM."""
    grid = np.asarray(occupancy_grid)
    if grid.ndim != 2:
        raise ValueError("occupancy_grid must be a 2D array")

    cells_rc, forward_m = corridor_points_to_grid_cells(
        corridor.filled_points_m, geometry
    )
    cells_rc, forward_m = _deduplicate_cells_with_min_forward(cells_rc, forward_m)

    rows, cols = grid.shape
    in_bounds = (
        (cells_rc[:, 0] >= 0)
        & (cells_rc[:, 0] < rows)
        & (cells_rc[:, 1] >= 0)
        & (cells_rc[:, 1] < cols)
    )

    out_of_bounds_count = int((~in_bounds).sum())
    valid_cells = cells_rc[in_bounds]
    valid_forward = forward_m[in_bounds]

    if len(valid_cells):
        sampled = grid[valid_cells[:, 0], valid_cells[:, 1]]
    else:
        sampled = np.empty((0,), dtype=grid.dtype)

    occupied_mask = sampled == values.occupied
    unknown_mask = sampled == values.unknown

    occupied_cells = valid_cells[occupied_mask]
    unknown_cells = valid_cells[unknown_mask]

    occupied_count = int(occupied_mask.sum())
    unknown_count = int(unknown_mask.sum())

    if occupied_count:
        first_occupied_distance_m = float(np.min(valid_forward[occupied_mask]))
    else:
        first_occupied_distance_m = None

    unsafe = occupied_count > 0
    if unknown_is_unsafe and unknown_count > 0:
        unsafe = True
    if out_of_bounds_is_unsafe and out_of_bounds_count > 0:
        unsafe = True

    return OccupancyCheckResult(
        safe=not unsafe,
        occupied_count=occupied_count,
        unknown_count=unknown_count,
        out_of_bounds_count=out_of_bounds_count,
        checked_cell_count=int(len(valid_cells)),
        first_occupied_distance_m=first_occupied_distance_m,
        occupied_cells=occupied_cells.copy(),
        unknown_cells=unknown_cells.copy(),
    )
