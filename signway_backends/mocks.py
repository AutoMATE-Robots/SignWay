"""Mock backends for tests and the Phase 1 sim smoke run — no model, no GPU.

These implement the same interfaces as the real backends, so the FSM under test is byte-for-byte
the FSM that will run with OmniVLA and the real occupancy replanner.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from signway_core.interfaces import Policy, Safety
from signway_core.types import Occupancy


class MockPolicy(Policy):
    """Forward-biased arc that leans toward the goal bearing (stand-in for OmniVLA)."""

    def __init__(self, n: int = 8, step: float = 0.12):
        self.n, self.step = n, step

    def predict_waypoints(self, images, goal_robot: Tuple[float, float, float]) -> np.ndarray:
        fwd, left, _ = goal_robot
        bearing = float(np.clip(np.arctan2(left, max(fwd, 1e-3)), -0.6, 0.6))
        pts, x, y, th = [], 0.0, 0.0, 0.0
        for _ in range(self.n):
            th += 0.15 * bearing
            x += self.step * np.cos(th)
            y += self.step * np.sin(th)
            pts.append([x, y])
        return np.array(pts, float)


class MockSafety(Safety):
    """No obstacles by default. Set blocked=True to simulate 'no safe path' (for HALT tests)."""

    def __init__(self, blocked: bool = False):
        self.blocked = blocked

    def refine(self, occupancy: Optional[Occupancy], waypoints: np.ndarray):
        if self.blocked:
            return (False, None)
        return (True, np.asarray(waypoints, float).reshape(-1, 2))

    def path_exists(self, occupancy: Optional[Occupancy], goal_robot) -> bool:
        return not self.blocked
