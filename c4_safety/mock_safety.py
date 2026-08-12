"""A fake safety layer. Either lets everything through, or blocks everything (blocked=True),
so we can test how the orchestrator reacts without building a real occupancy grid."""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.interfaces import Safety
from common.types import Occupancy


class MockSafety(Safety):
    def __init__(self, blocked: bool = False):
        self.blocked = blocked

    def refine(self, occupancy: Optional[Occupancy], waypoints: np.ndarray):
        if self.blocked:
            return (False, None)
        return (True, np.asarray(waypoints, float))

    def path_exists(self, occupancy: Optional[Occupancy], goal_robot) -> bool:
        return not self.blocked