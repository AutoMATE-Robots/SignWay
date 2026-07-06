"""Bridge interface: the sim seam. reset/get_obs/move_to/intrinsics/close. Both FakeBridge (for
testing the loop with no simulator) and HabitatBridge implement it, so run_sim.py is identical
across them — and an isaac_bridge.py would slot in the same way if that ever became necessary."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from signway_core.types import Observation, Pose


class Bridge(ABC):
    @abstractmethod
    def reset(self, start: Pose | None = None) -> Observation:
        ...

    @abstractmethod
    def get_obs(self) -> Observation:
        ...

    @abstractmethod
    def move_to(self, pose: Pose) -> Observation:
        """Collision-aware move toward `pose`; returns the observation at the pose actually reached."""
        ...

    @abstractmethod
    def intrinsics(self) -> np.ndarray:
        ...

    def close(self) -> None:
        pass
