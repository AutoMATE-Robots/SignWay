"""Safety backend: occupancy-based collision veto + local replanner, implementing
signway_core.interfaces.Safety. Wraps occupancy_replan.py.

`grid_from_depth` builds the metric occupancy grid from a depth image + intrinsics (used by the
Habitat bridge, which has ground-truth depth). `SafetyOccupancy.refine/path_exists` operate on
whatever Grid the observation carries.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from signway_core.interfaces import Safety
from signway_core.types import Occupancy
from signway_backends import occupancy_replan as OR


def grid_from_depth(depth_m, K, cam_height, cam_pitch_deg=0.0,
                    x_max=4.0, y_half=2.0, res=0.05) -> "OR.Grid":
    grid = OR.Grid(x_max=x_max, y_half=y_half, res=res)
    return OR.occupancy_from_depth(depth_m, K, cam_height, cam_pitch_deg,
                                   grid=grid, max_range=x_max)


def grid_to_occupancy(grid: "OR.Grid", stamp: float = 0.0) -> Occupancy:
    """Convert an occupancy_replan.Grid into the core Occupancy message (for logging / the FSM)."""
    rr, cc = np.where(grid.occ)
    return Occupancy(res=grid.res, x_max=grid.x_max, y_half=grid.y_half,
                     cells=list(zip(rr.tolist(), cc.tolist())), stamp=stamp)


def occupancy_to_grid(occ: Optional[Occupancy]) -> Optional["OR.Grid"]:
    if occ is None:
        return None
    g = OR.Grid(x_max=occ.x_max, y_half=occ.y_half, res=occ.res)
    for r, c in occ.cells:
        if g.in_bounds(r, c):
            g.occ[r, c] = True
    return g


class SafetyOccupancy(Safety):
    def __init__(self, robot_radius: float = 0.20, lookahead: float = 2.5):
        self.robot_radius = robot_radius
        self.lookahead = lookahead

    def refine(self, occupancy: Optional[Occupancy], waypoints: np.ndarray):
        grid = occupancy_to_grid(occupancy)
        if grid is None:                       # no occupancy -> nothing to veto
            return (True, np.asarray(waypoints, float).reshape(-1, 2))
        used, path, _occ_inf, _sub = OR.refine(grid, waypoints,
                                               robot_radius=self.robot_radius,
                                               lookahead=self.lookahead)
        return (bool(used), None if path is None else np.asarray(path, float).reshape(-1, 2))

    def path_exists(self, occupancy: Optional[Occupancy], goal_robot) -> bool:
        grid = occupancy_to_grid(occupancy)
        if grid is None:
            return True
        # a straight probe toward the goal bearing, capped to the grid; replan decides feasibility
        fwd, left, _ = goal_robot
        n = float(np.hypot(fwd, left)) or 1.0
        reach = min(self.lookahead, grid.x_max - grid.res)
        probe = np.column_stack([np.linspace(0, fwd / n * reach, 8),
                                 np.linspace(0, left / n * reach, 8)])
        used, path, _i, _s = OR.refine(grid, probe, robot_radius=self.robot_radius,
                                       lookahead=self.lookahead)
        return used or (path is not None)
