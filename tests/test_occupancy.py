"""Occupancy-from-depth tests with synthetic depth — no Habitat, no GPU.

Verifies the metric back-projection puts a wall at the right forward distance, and that the
safety backend detects a collision and reroutes around a partial obstacle.
"""
import numpy as np

from c4_safety import occupancy_replan as OR
from c4_safety.safety_occupancy import SafetyOccupancy, grid_to_occupancy


# small pinhole: 90 deg HFOV, 160x120
W, H = 160, 120
FX = (W / 2) / np.tan(np.deg2rad(90) / 2)      # = 80
K = np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]], float)
CAM_H = 1.0


def test_full_wall_lands_at_right_distance():
    depth = np.full((H, W), 2.0)               # flat wall 2 m ahead, fills the view
    g = OR.occupancy_from_depth(depth, K, CAM_H, grid=OR.Grid(x_max=4.0, y_half=2.0, res=0.05))
    assert g.occ.any()
    rows = np.where(g.occ.any(axis=1))[0]
    # occupied rows should cluster around forward = 2.0 m -> row ~ 2.0/0.05 = 40
    assert 36 <= rows.mean() <= 44


def test_partial_wall_reroutes():
    depth = np.full((H, W), 10.0)              # mostly open (out of range)
    depth[:, 70:90] = 1.2                      # a pillar dead ahead, centre columns
    g = OR.occupancy_from_depth(depth, K, CAM_H, grid=OR.Grid(x_max=4.0, y_half=2.0, res=0.05))
    assert g.occ.any()
    occ = grid_to_occupancy(g)
    safety = SafetyOccupancy()
    # a chunk is (8,4): [dx, dy, hx, hy] — the last two are the policy's own heading
    straight = np.column_stack([np.linspace(0.1, 2.0, 8), np.zeros(8),
                                np.ones(8), np.zeros(8)])
    used, path = safety.refine(occ, straight)
    assert used is False                       # straight chunk hits the pillar
    assert path is not None                    # but a safe route around it exists


def test_no_occupancy_passes_through():
    safety = SafetyOccupancy()
    # a chunk is (8,4): [dx, dy, hx, hy] — the last two are the policy's own heading
    straight = np.column_stack([np.linspace(0.1, 2.0, 8), np.zeros(8),
                                np.ones(8), np.zeros(8)])
    used, path = safety.refine(None, straight)
    assert used is True
    assert path.shape == (8, 4)
