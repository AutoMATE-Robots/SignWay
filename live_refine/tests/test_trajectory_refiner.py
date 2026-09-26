import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from trajectory_refiner import (
    GridSnapshot, RefinerConfig, TrajectoryRefiner, action_from_waypoints,
    compensate, densify_and_extend, make_snapshot, meters_to_cell,
    pack_occupancy_grid, unpack_occupancy_grid, waypoints_from_action,
)

CFG = RefinerConfig()
RES, LAT = CFG.resolution, CFG.lateral_half_m
ROWS, COLS = int(CFG.forward_m / RES), int(2 * LAT / RES)   # 160 x 100


def empty_grid():
    return np.zeros((ROWS, COLS), dtype=np.int16)


def straight_action(step=0.04, n=8):
    i = np.arange(1, n + 1) * step
    return action_from_waypoints(np.stack([i, np.zeros(n)], axis=1))


def test_meters_to_cell_y_sign():
    # 2 m ahead, 0.5 m LEFT must land LEFT of center (smaller col).
    pts = np.array([[2.0, 0.5]])
    cell = meters_to_cell(pts, CFG)[0]
    assert cell[0] == 20
    assert cell[1] == 45 and cell[1] < COLS // 2


def test_pack_unpack_roundtrip_and_orientation():
    g = empty_grid()
    g[30, 20] = 100          # 3 m ahead, col 20 -> y = 5 - 20.5*0.1 = +2.95 (left)
    data, w, h, origin = pack_occupancy_grid(g, RES, LAT)
    assert (w, h) == (ROWS, COLS) and origin == (0.0, -LAT)
    back = unpack_occupancy_grid(data, w, h)
    assert np.array_equal(back, g)
    # nav_msgs indexing: cell at map x=3.0, y=+2.95 -> mx=30, my=(2.95+5)/0.1=79
    assert data[79 * w + 30] == 100


def test_safe_passthrough_on_empty_grid():
    snap = make_snapshot(empty_grid(), stamp=100.0, resolution=RES)
    r = TrajectoryRefiner().refine(straight_action(), snap, now=100.1)
    assert r.reason == "safe" and r.safe and not r.modified
    assert np.allclose(r.action, r.raw_action)


def test_avoids_obstacle_ahead_right():
    g = empty_grid()
    g[12:18, 44:52] = 100    # blob 1.2-1.8 m ahead, straddling center-right
    snap = make_snapshot(g, stamp=0.0, resolution=RES)
    r = TrajectoryRefiner().refine(straight_action(), snap, now=0.1)
    assert r.reason == "refined" and r.modified
    raw_wp = waypoints_from_action(r.raw_action)
    new_wp = waypoints_from_action(r.action)
    assert r.min_clearance > r.min_clearance_raw
    assert not np.allclose(new_wp[:, 1], raw_wp[:, 1])   # actually steered
    assert np.isfinite(r.first_obstacle_m)


def test_blocked_slows_down():
    g = empty_grid()
    g[8:12, :] = 100         # full-width wall 0.8 m ahead — nothing is safe
    snap = make_snapshot(g, stamp=0.0, resolution=RES)
    r = TrajectoryRefiner().refine(straight_action(), snap, now=0.1)
    assert r.blocked and r.reason == "blocked"
    raw_wp = waypoints_from_action(r.raw_action)
    new_wp = waypoints_from_action(r.action)
    assert new_wp[-1, 0] < 0.6 * raw_wp[-1, 0]           # strongly slowed


def test_stale_grid_passthrough():
    snap = make_snapshot(empty_grid(), stamp=0.0, resolution=RES)
    r = TrajectoryRefiner().refine(straight_action(), snap, now=10.0)
    assert r.reason == "stale_grid" and not r.modified


def test_no_grid_passthrough():
    r = TrajectoryRefiner().refine(straight_action(), None)
    assert r.reason == "no_grid" and not r.modified


def test_error_failsafe_returns_raw():
    r = TrajectoryRefiner().refine(np.zeros(16), snap="not a snapshot")
    assert r.reason.startswith("error") and np.allclose(r.action, r.raw_action)


def test_odometry_compensation_geometry():
    # Robot advanced 0.5 m (no rotation) since grid capture: a point 0.5 m
    # ahead NOW is 1.0 m ahead in the grid frame.
    p = compensate(np.array([[0.5, 0.0]]), pose_grid=(0, 0, 0),
                   pose_now=(0.5, 0, 0))
    assert np.allclose(p, [[1.0, 0.0]])
    # 90-degree left rotation since capture: "ahead now" = "left in grid frame".
    p = compensate(np.array([[1.0, 0.0]]), pose_grid=(0, 0, 0),
                   pose_now=(0, 0, math.pi / 2))
    assert np.allclose(p, [[0.0, 1.0]], atol=1e-9)


def test_compensation_changes_safety_verdict():
    g = empty_grid()
    g[10:13, 46:54] = 100    # obstacle ~1.0-1.3 m ahead in GRID frame
    # Robot has moved 0.9 m forward since capture -> obstacle is ~0.1-0.4 m
    # ahead NOW; the raw path must not be judged "safe".
    snap = make_snapshot(g, stamp=0.0, resolution=RES, pose=(0.0, 0.0, 0.0))
    r = TrajectoryRefiner().refine(straight_action(), snap,
                                   pose_now=(0.9, 0.0, 0.0), now=0.3)
    assert r.reason in ("refined", "blocked")


def test_turning_path_uses_short_extension():
    cfg = RefinerConfig()
    i = np.arange(1, 9) * 0.04
    turn = np.stack([i, np.linspace(0.005, 0.12, 8)], axis=1)  # leftward arc
    snap = make_snapshot(empty_grid(), stamp=0.0, resolution=RES)
    r = TrajectoryRefiner(cfg).refine(action_from_waypoints(turn), snap, now=0.1)
    assert r.ext_len_used == cfg.ext_len_turning
    straight = TrajectoryRefiner(cfg).refine(straight_action(), snap, now=0.1)
    assert straight.ext_len_used == cfg.ext_len


def test_extension_reaches_fixed_metric_length():
    # Model underestimating speed (tiny spacing) must NOT shrink the horizon.
    tiny = straight_action(step=0.01)
    dense = densify_and_extend(waypoints_from_action(tiny), 2.0, 0.05)
    arc = np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()
    assert arc > 1.9


def test_raw_wins_when_already_optimal():
    # Symmetric corridor, walls at y = +/-1.5 m; straight down the middle IS
    # the optimum. Tighten d_safe so the contest is forced to run: any
    # rotation or lateral shift moves toward a wall, so raw must win.
    g = empty_grid()
    g[:, 33:35] = 100        # left wall  (y ~ +1.5)
    g[:, 65:67] = 100        # right wall (y ~ -1.6)
    cfg = RefinerConfig(d_safe=2.0)
    snap = make_snapshot(g, stamp=0.0, resolution=RES)
    r = TrajectoryRefiner(cfg).refine(straight_action(), snap, now=0.1)
    assert r.reason == "refined" and r.candidate == "raw"
    assert np.allclose(r.action, r.raw_action)


def test_log_dict_is_json_serializable():
    import json
    snap = make_snapshot(empty_grid(), stamp=0.0, resolution=RES)
    r = TrajectoryRefiner().refine(straight_action(), snap, now=0.1)
    json.dumps(r.log_dict())
