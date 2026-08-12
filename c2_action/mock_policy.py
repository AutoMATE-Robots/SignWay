"""A fake action model. Outputs a simple forward curve that leans toward the goal, instead of
running the real 7B OmniVLA. Lets us test the whole pipeline with no GPU and no model."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from common.interfaces import Policy


class MockPolicy(Policy):
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
            pts.append([x, y, np.cos(th), np.sin(th)])   # [dx, dy, hx, hy] like OmniVLA
        return np.array(pts, float)