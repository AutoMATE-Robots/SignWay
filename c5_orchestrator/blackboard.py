"""Shared runtime state owned by the FSM (orchestration spec §3).

The orchestrator is the single arbiter: it holds this state and updates it from each incoming
Observation. No separate blackboard process — keeps the system simple.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Set, Tuple

import numpy as np

from common.types import Decision, FSMState, Leg, Occupancy, Pose, Subgoal


@dataclass
class Blackboard:
    mission_goal: str = ""
    robot_pose: Pose = field(default_factory=Pose)
    subgoal: Optional[Subgoal] = None
    occupancy: Optional[Occupancy] = None
    detections: List = field(default_factory=list)
    last_decision: Optional[Decision] = None
    committed_signs: Set[str] = field(default_factory=set)
    state: FSMState = FSMState.DRIVE
    reason_budget: int = 30
    leg: Optional[Leg] = None
    pending_sign: object = None      # the Detection that triggered REASON
    pending_sign_key: str = ""       # its debounce key, computed while it was FRESH
    stamp: float = 0.0
    _progress: Deque[Tuple[float, Pose]] = field(default_factory=lambda: deque(maxlen=256))

    def update_from_obs(self, obs) -> None:
        self.robot_pose = obs.robot_pose
        self.occupancy = obs.occupancy
        self.detections = obs.detections
        self.stamp = obs.stamp
        self._progress.append((obs.stamp, obs.robot_pose))

    def recent_progress(self, window_s: float) -> float:
        """Straight-line displacement over the last `window_s` seconds. Returns +inf until we
        actually have a full window of history, so a stall is never declared prematurely."""
        if len(self._progress) < 2:
            return float("inf")
        now, cur = self._progress[-1]
        oldest_t = self._progress[0][0]
        if now - oldest_t < window_s:
            return float("inf")
        target = now - window_s
        ref = min(self._progress, key=lambda tp: abs(tp[0] - target))[1]
        return float(np.hypot(cur.x - ref.x, cur.y - ref.y))