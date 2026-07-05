"""The orchestration state machine (spec §5).

Phase 1 implements DRIVE / HALT / ARRIVED / FAILED with the SAFETY, SUBGOAL_REACHED and STALLED
triggers. REASON / MANEUVER and the sign/hazard triggers are added in Phase 2 — the enum and the
priority-ordered dispatch already accommodate them, so Phase 2 is additive, not a rewrite.

The FSM is pure Python: hand it a Policy and a Safety backend, feed it Observations via tick(),
and it returns a Command for the controller. Same object runs in sim and on the robot.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import triggers
from .blackboard import Blackboard
from .config import Params
from .geometry import world_goal_to_robot
from .interfaces import Detector, Policy, Reasoner, Safety
from .types import Command, FSMState, Observation, Pose, Subgoal


class FSM:
    def __init__(self, policy: Policy, safety: Safety, cfg: Optional[Params] = None,
                 mission_goal: str = "",
                 detector: Optional[Detector] = None,      # Phase 2
                 reasoner: Optional[Reasoner] = None):     # Phase 2
        self.policy = policy
        self.safety = safety
        self.detector = detector
        self.reasoner = reasoner
        self.cfg = cfg or Params()
        self.bb = Blackboard(mission_goal=mission_goal, reason_budget=self.cfg.reason_budget)
        self._obs: Optional[Observation] = None

    # ---- public API ----
    def set_subgoal(self, pose: Pose, stamp: float = 0.0) -> None:
        self.bb.subgoal = Subgoal(pose=pose, stamp=stamp)

    def tick(self, obs: Observation) -> Command:
        self._obs = obs
        self.bb.update_from_obs(obs)
        st = self.bb.state
        if st == FSMState.DRIVE:
            return self._drive()
        if st == FSMState.HALT:
            return self._halt()
        if st in (FSMState.ARRIVED, FSMState.FAILED):
            return Command(kind="stop", meta={"reason": st.value.lower()})
        # REASON / MANEUVER handled in Phase 2
        return Command(kind="stop", meta={"reason": "unhandled_state"})

    # ---- states ----
    def _drive(self) -> Command:
        bb, cfg = self.bb, self.cfg
        # triggers in priority order (Phase 1 subset of spec §5.1)
        if triggers.no_safe_path(bb, self.safety, cfg):
            bb.state = FSMState.HALT
            return Command(kind="stop", meta={"reason": "no_safe_path"})
        if triggers.subgoal_reached(bb, cfg):
            bb.state = FSMState.ARRIVED
            return Command(kind="stop", meta={"reason": "arrived"})
        if triggers.stalled(bb, cfg):
            bb.state = FSMState.HALT
            return Command(kind="stop", meta={"reason": "stalled"})
        if bb.subgoal is None:
            return Command(kind="stop", meta={"reason": "no_subgoal"})

        # normal execution: policy proposes a chunk, safety vets or replans it
        goal_robot = world_goal_to_robot(bb.subgoal.pose, bb.robot_pose, cfg.omni_range_m)
        wp = np.asarray(self.policy.predict_waypoints(self._two_images(), goal_robot),
                        float).reshape(-1, 2)
        used, path = self.safety.refine(bb.occupancy, wp)
        if not used and path is None:
            bb.state = FSMState.HALT
            return Command(kind="stop", meta={"reason": "no_safe_path"})
        return Command(kind="waypoints",
                       waypoints=(wp if used else path),
                       meta={"used_policy": bool(used)})

    def _halt(self) -> Command:
        bb, cfg = self.bb, self.cfg
        if bb.subgoal is not None:
            goal_robot = world_goal_to_robot(bb.subgoal.pose, bb.robot_pose, cfg.omni_range_m)
            if self.safety.path_exists(bb.occupancy, goal_robot):
                bb.state = FSMState.DRIVE
                return self._drive()
        return Command(kind="stop", meta={"reason": "halted"})

    # ---- helpers ----
    def _two_images(self):
        """OmniVLA expects 2 frames; duplicate the latest if only one is available."""
        imgs = list(self._obs.images) if self._obs else []
        if len(imgs) == 1:
            imgs = [imgs[0], imgs[0]]
        return imgs
