"""Utilities for converting the SignWay VLA output into a robot-relative path.

VLA contract used here:
- output is 16 flat values: [x1, y1, ..., x8, y8]
- reshape to (8, 2)
- x = forward, y = left (negative y = right)
- values are meters
- each waypoint is an absolute future position relative to the robot's current pose
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class VlaPathInfo:
    """Clean geometric representation of one 8-waypoint VLA prediction."""

    waypoints: np.ndarray          # shape (8, 2), robot-relative meters
    path_points: np.ndarray        # shape (9, 2), starts with robot origin (0, 0)
    path_length_m: float
    final_heading_rad: float
    final_point_m: tuple[float, float]


def decode_vla_action(action: Iterable[float] | np.ndarray) -> np.ndarray:
    """Convert the flat 16-value VLA action into 8 (forward, left) waypoints."""
    arr = np.asarray(action, dtype=np.float64).reshape(-1)

    if arr.size != 16:
        raise ValueError(
            f"Expected exactly 16 VLA values (8 waypoints x 2), got {arr.size}."
        )
    if not np.isfinite(arr).all():
        raise ValueError("VLA action contains NaN or infinite values.")

    return arr.reshape(8, 2).copy()


def _final_nonzero_heading(path_points: np.ndarray, eps: float = 1e-12) -> float:
    """Return the heading of the last non-zero segment in the path."""
    deltas = np.diff(path_points, axis=0)
    for dx, dy in deltas[::-1]:
        if (dx * dx + dy * dy) > eps:
            return float(atan2(dy, dx))
    return 0.0


def build_vla_path(action: Iterable[float] | np.ndarray) -> VlaPathInfo:
    """Build geometric information from one VLA prediction.

    No fixed robot speed is assumed. The path length comes directly from the
    waypoint geometry produced by the VLA.
    """
    waypoints = decode_vla_action(action)
    origin = np.zeros((1, 2), dtype=np.float64)
    path_points = np.vstack((origin, waypoints))

    segment_vectors = np.diff(path_points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    path_length_m = float(segment_lengths.sum())

    final_heading_rad = _final_nonzero_heading(path_points)
    final_point_m = (float(waypoints[-1, 0]), float(waypoints[-1, 1]))

    return VlaPathInfo(
        waypoints=waypoints,
        path_points=path_points,
        path_length_m=path_length_m,
        final_heading_rad=final_heading_rad,
        final_point_m=final_point_m,
    )
