import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from camera_occupancy_node import backproject_depth, fit_floor_ransac, build_camera_occupancy


def test_backproject_depth_center_pixel_points_straight_forward():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    xyz, uv = backproject_depth(depth, fx=100.0, fy=100.0, cx=2.0, cy=2.0, step=1, max_depth=20.0)
    idx = np.where((uv[:, 0] == 2) & (uv[:, 1] == 2))[0][0]
    np.testing.assert_allclose(xyz[idx], [0.0, 0.0, 2.0], atol=1e-6)


def test_fit_floor_ransac_recovers_camera_floor_plane():
    rng = np.random.default_rng(4)
    x = rng.uniform(-2.0, 2.0, 500)
    z = rng.uniform(1.0, 8.0, 500)
    y = 0.05 * x + 0.03 * z + 1.2 + rng.normal(0.0, 0.005, 500)
    pts = np.column_stack((x, y, z)).astype(np.float32)
    coeff, inliers = fit_floor_ransac(pts, threshold=0.03, iterations=120, rng=np.random.default_rng(7))
    np.testing.assert_allclose(coeff, [0.05, 0.03, 1.2], atol=0.02)
    assert inliers.mean() > 0.95


def test_build_camera_occupancy_does_not_mark_behind_first_obstacle_free():
    # Floor evidence exists out to 8 m straight ahead, but an obstacle is at 3 m.
    floor_pts = np.array([
        [0.00, 1.2, 2.0], [0.01, 1.2, 4.0], [-0.01, 1.2, 6.0], [0.0, 1.2, 8.0]
    ], dtype=np.float32)
    obstacle_pts = np.array([[0.0, 0.5, 3.0]], dtype=np.float32)

    grid = build_camera_occupancy(
        floor_pts=floor_pts,
        obstacle_pts=obstacle_pts,
        lateral_half_m=2.0,
        forward_m=10.0,
        resolution=0.1,
        angle_resolution_deg=1.0,
        min_floor_points_per_bin=3,
        max_obstacle_connection_m=0.3,
    )

    center = grid.shape[1] // 2
    row_2m = int(2.0 / 0.1)
    row_3m = int(3.0 / 0.1)
    row_5m = int(5.0 / 0.1)

    assert grid[row_2m, center] == 0      # observed free before obstacle
    assert grid[row_3m, center] == 100    # occupied at obstacle
    assert grid[row_5m, center] == -1     # unknown behind obstacle, never free
