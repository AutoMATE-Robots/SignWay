"""Build a geometric Lookahead Path from the already-decoded VLA path.

This module is intentionally independent of ROS and the occupancy grid.
It consumes ``VlaPathInfo`` from ``vla_path.py`` and produces a dense path
that can later be checked against the occupancy grid.

Default design used for the first MSI version:
- total lookahead distance: 3.0 m from the robot along the path
- direction/curvature estimate: last 4 non-zero VLA segments
- dense sampling: at most 0.05 m (5 cm) between check points
- straight paths continue straight
- turning paths continue with approximately constant curvature

The extra Lookahead points are imaginary safety-check points only. They are
not commands sent to the robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin

import numpy as np

from vla_path import VlaPathInfo


@dataclass(frozen=True)
class LookaheadPathInfo:
    """Result of extending one VLA path for geometric safety checking."""

    vla_path_points: np.ndarray       # original path including robot origin
    extension_points: np.ndarray      # only imaginary points after VLA P8
    full_path_points: np.ndarray      # original VLA points + imaginary extension
    check_points: np.ndarray          # dense full path for later OGM checking
    target_total_length_m: float
    actual_total_length_m: float
    extension_length_m: float
    estimated_curvature_per_m: float
    final_heading_rad: float


def _nonzero_segments(path_points: np.ndarray, eps: float = 1e-9):
    deltas = np.diff(path_points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    mask = lengths > eps
    return deltas[mask], lengths[mask]


def _estimate_recent_curvature(
    path_points: np.ndarray,
    fit_segments: int = 4,
    straight_threshold_per_m: float = 0.05,
    max_abs_curvature_per_m: float = 2.5,
) -> tuple[float, float]:
    """Estimate final heading and recent curvature from the latest path segments.

    Curvature is the change in heading per meter of path. Positive means left,
    negative means right in the VLA coordinate convention.
    """
    deltas, lengths = _nonzero_segments(path_points)
    if len(deltas) == 0:
        return 0.0, 0.0

    headings = np.unwrap(np.array([atan2(dy, dx) for dx, dy in deltas], dtype=float))
    final_heading = float(headings[-1])

    count = min(int(fit_segments), len(deltas))
    recent_headings = headings[-count:]
    recent_lengths = lengths[-count:]

    if count < 2:
        return final_heading, 0.0

    # Place each segment heading at the midpoint of that segment's arc length,
    # then fit heading(s) ~= curvature * s + constant.
    starts = np.concatenate(([0.0], np.cumsum(recent_lengths[:-1])))
    mid_s = starts + 0.5 * recent_lengths
    curvature = float(np.polyfit(mid_s, recent_headings, 1)[0])

    if abs(curvature) < straight_threshold_per_m:
        curvature = 0.0

    curvature = float(np.clip(curvature, -max_abs_curvature_per_m, max_abs_curvature_per_m))
    return final_heading, curvature


def _advance_constant_curvature(
    x: float,
    y: float,
    heading: float,
    curvature: float,
    distance: float,
) -> tuple[float, float, float]:
    """Advance a pose by ``distance`` using constant planar curvature."""
    if abs(curvature) < 1e-12:
        return (
            x + distance * cos(heading),
            y + distance * sin(heading),
            heading,
        )

    next_heading = heading + curvature * distance
    next_x = x + (sin(next_heading) - sin(heading)) / curvature
    next_y = y + (-cos(next_heading) + cos(heading)) / curvature
    return next_x, next_y, next_heading


def _densify_polyline(points: np.ndarray, max_spacing_m: float) -> np.ndarray:
    """Interpolate a polyline so no neighboring samples are farther apart."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(points) == 1:
        return points.astype(np.float64, copy=True)

    dense = [points[0].astype(np.float64)]
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1e-12:
            continue

        pieces = max(1, int(np.ceil(length / max_spacing_m)))
        for j in range(1, pieces + 1):
            dense.append(start + delta * (j / pieces))

    return np.asarray(dense, dtype=np.float64)


def build_lookahead_path(
    vla_path: VlaPathInfo,
    *,
    total_lookahead_m: float = 3.0,
    sample_spacing_m: float = 0.05,
    fit_segments: int = 4,
    straight_threshold_per_m: float = 0.05,
    max_abs_curvature_per_m: float = 2.5,
) -> LookaheadPathInfo:
    """Extend the VLA path to a configurable total path length.

    ``total_lookahead_m`` is measured along the path from the robot origin.
    Example: if the VLA path itself is 0.4 m long and the target is 3.0 m,
    this module creates about 2.6 m of imaginary extension.
    """
    if total_lookahead_m <= 0:
        raise ValueError("total_lookahead_m must be > 0")
    if sample_spacing_m <= 0:
        raise ValueError("sample_spacing_m must be > 0")
    if fit_segments < 1:
        raise ValueError("fit_segments must be >= 1")
    if max_abs_curvature_per_m <= 0:
        raise ValueError("max_abs_curvature_per_m must be > 0")

    path_points = np.asarray(vla_path.path_points, dtype=np.float64)
    if path_points.ndim != 2 or path_points.shape[1] != 2:
        raise ValueError("vla_path.path_points must have shape (N, 2)")

    extension_length = max(0.0, float(total_lookahead_m) - float(vla_path.path_length_m))
    final_heading, curvature = _estimate_recent_curvature(
        path_points,
        fit_segments=fit_segments,
        straight_threshold_per_m=straight_threshold_per_m,
        max_abs_curvature_per_m=max_abs_curvature_per_m,
    )

    extension = []
    heading = final_heading
    x = float(path_points[-1, 0])
    y = float(path_points[-1, 1])

    remaining = extension_length
    while remaining > 1e-12:
        step = min(float(sample_spacing_m), remaining)
        x, y, heading = _advance_constant_curvature(x, y, heading, curvature, step)
        extension.append((x, y))
        remaining -= step

    extension_points = (
        np.asarray(extension, dtype=np.float64)
        if extension
        else np.empty((0, 2), dtype=np.float64)
    )

    if len(extension_points):
        full_path_points = np.vstack((path_points, extension_points))
    else:
        full_path_points = path_points.copy()

    # Densify the original VLA sections too, not just the extension, because
    # the next occupancy checker should inspect the whole path continuously.
    check_points = _densify_polyline(full_path_points, float(sample_spacing_m))

    actual_total_length = float(vla_path.path_length_m + extension_length)

    return LookaheadPathInfo(
        vla_path_points=path_points.copy(),
        extension_points=extension_points,
        full_path_points=full_path_points,
        check_points=check_points,
        target_total_length_m=float(total_lookahead_m),
        actual_total_length_m=actual_total_length,
        extension_length_m=float(extension_length),
        estimated_curvature_per_m=float(curvature),
        final_heading_rad=float(heading),
    )
