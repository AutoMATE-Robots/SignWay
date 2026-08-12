"""Abstract backends. The core depends only on these interfaces; concrete implementations
(OmniVLA client, occupancy replanner, GroundingDINO, the VLM) live in the component folders and are
swapped in without touching the FSM. This is the 'swappable backend' rule applied system-wide."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from .types import Decision, Detection, Occupancy


class Policy(ABC):
    """Action model (e.g. OmniVLA). Returns a waypoint chunk toward a robot-frame goal."""

    @abstractmethod
    def predict_waypoints(self, images: list,
                          goal_robot: Tuple[float, float, float]) -> np.ndarray:
        """images: recent RGB frames. goal_robot: (forward, left, dtheta). Returns (N,2)."""
        ...


class Safety(ABC):
    """Occupancy-based collision veto + local replanner."""

    @abstractmethod
    def refine(self, occupancy: Optional[Occupancy],
               waypoints: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (used_policy, path).
        used_policy=True  -> waypoints are collision-free as-is (path == waypoints).
        used_policy=False -> path is a replanned route, or None if no safe path exists."""
        ...

    @abstractmethod
    def path_exists(self, occupancy: Optional[Occupancy],
                    goal_robot: Tuple[float, float, float]) -> bool:
        """Is there any collision-free route toward the goal bearing?"""
        ...


class Detector(ABC):   # Phase 2
    """Cheap always-on sign detector (e.g. GroundingDINO)."""

    @abstractmethod
    def detect(self, image) -> List[Detection]:
        ...


class Reasoner(ABC):   # Phase 2
    """The VLM. In ROS2 it is a cancelable action; here the sync interface is the contract."""

    @abstractmethod
    def reason(self, request: dict) -> Decision:
        ...
