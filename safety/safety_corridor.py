"""Build a filled forward safety corridor for occupancy-grid checking.

First simple version for normal/straight driving:
- robot physical width: 0.75 m
- safety margin: 0.20 m on the left + 0.20 m on the right
- effective checked width: 1.15 m
- forward checked length: inherited from the Lookahead Path (default 3.0 m)
- area is represented by dense (x, y) points in the robot frame

Coordinate convention matches the VLA modules:
- x = forward
- y = left
- negative y = right
- robot origin = (0, 0)

This module does NOT modify the occupancy grid and does NOT call OpenCV.
The next occupancy-checking module will convert ``filled_points_m`` into OGM
row/column cells and test whether any of them overlap occupied cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lookahead_path import LookaheadPathInfo


@dataclass(frozen=True)
class SafetyCorridorInfo:
    """Dense rectangular safety area in front of the robot."""

    filled_points_m: np.ndarray       # shape (N, 2), [forward_m, left_m]
    corners_m: np.ndarray             # shape (4, 2)
    length_m: float
    robot_width_m: float
    robot_length_m: float
    safety_margin_m: float
    effective_width_m: float
    half_width_m: float
    requested_spacing_m: float
    actual_x_spacing_m: float
    actual_y_spacing_m: float


def _axis_samples(start: float, end: float, max_spacing: float) -> np.ndarray:
    """Return samples including both ends, with spacing <= max_spacing."""
    distance = float(abs(end - start))
    if distance <= 1e-12:
        return np.array([float(start)], dtype=np.float64)

    intervals = max(1, int(np.ceil(distance / max_spacing)))
    return np.linspace(float(start), float(end), intervals + 1, dtype=np.float64)


def _symmetric_axis_samples(half_extent: float, max_spacing: float) -> np.ndarray:
    """Return symmetric samples that always include -edge, 0, and +edge."""
    positive = _axis_samples(0.0, float(half_extent), max_spacing)
    negative = -positive[:0:-1]
    return np.concatenate((negative, positive))


def build_safety_corridor(
    lookahead: LookaheadPathInfo,
    *,
    robot_width_m: float = 0.75,
    robot_length_m: float = 1.10,
    safety_margin_m: float = 0.20,
    corridor_length_m: float | None = None,
    sample_spacing_m: float = 0.05,
) -> SafetyCorridorInfo:
    """Create one filled rectangular safety zone in front of the robot.

    By default, the rectangle length comes directly from
    ``lookahead.target_total_length_m``. With the current Lookahead module this
    is 3.0 m.

    The rectangle is aligned with the robot frame:
        x in [0, corridor_length]
        y in [-effective_width/2, +effective_width/2]

    The physical robot length is stored because it is part of the robot
    geometry and will matter for later footprint/turn handling. The current
    simple forward rectangle intentionally keeps its total forward detection
    length at exactly ``corridor_length_m`` as agreed for this first version.
    """
    if robot_width_m <= 0:
        raise ValueError("robot_width_m must be > 0")
    if robot_length_m <= 0:
        raise ValueError("robot_length_m must be > 0")
    if safety_margin_m < 0:
        raise ValueError("safety_margin_m must be >= 0")
    if sample_spacing_m <= 0:
        raise ValueError("sample_spacing_m must be > 0")

    length_m = (
        float(lookahead.target_total_length_m)
        if corridor_length_m is None
        else float(corridor_length_m)
    )
    if length_m <= 0:
        raise ValueError("corridor_length_m must be > 0")

    effective_width_m = float(robot_width_m + 2.0 * safety_margin_m)
    half_width_m = 0.5 * effective_width_m

    x_samples = _axis_samples(0.0, length_m, float(sample_spacing_m))
    y_samples = _symmetric_axis_samples(half_width_m, float(sample_spacing_m))

    xx, yy = np.meshgrid(x_samples, y_samples, indexing="xy")
    filled_points_m = np.column_stack((xx.ravel(), yy.ravel()))

    corners_m = np.array(
        [
            [0.0, -half_width_m],
            [0.0, +half_width_m],
            [length_m, +half_width_m],
            [length_m, -half_width_m],
        ],
        dtype=np.float64,
    )

    actual_x_spacing_m = (
        float(np.max(np.diff(x_samples))) if len(x_samples) > 1 else 0.0
    )
    actual_y_spacing_m = (
        float(np.max(np.diff(y_samples))) if len(y_samples) > 1 else 0.0
    )

    return SafetyCorridorInfo(
        filled_points_m=filled_points_m,
        corners_m=corners_m,
        length_m=length_m,
        robot_width_m=float(robot_width_m),
        robot_length_m=float(robot_length_m),
        safety_margin_m=float(safety_margin_m),
        effective_width_m=effective_width_m,
        half_width_m=half_width_m,
        requested_spacing_m=float(sample_spacing_m),
        actual_x_spacing_m=actual_x_spacing_m,
        actual_y_spacing_m=actual_y_spacing_m,
    )
