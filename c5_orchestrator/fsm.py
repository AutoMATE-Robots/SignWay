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

from c5_orchestrator import decision as decision_map
from c5_orchestrator import triggers
from c5_orchestrator.blackboard import Blackboard
from common.config import Params
from common.geometry import pose_ahead, world_goal_to_robot
from common.interfaces import Detector, Policy, Reasoner, Safety
from common.types import Command, FSMState, Leg, Observation, Pose, Subgoal


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
        self._maneuver_name: Optional[str] = None
        self._maneuver_issued = False

    # ---- public API ----
    def set_subgoal(self, pose: Pose, stamp: float = 0.0, source: str = "mission") -> None:
        self.bb.subgoal = Subgoal(pose=pose, stamp=stamp, source=source)

    def tick(self, obs: Observation) -> Command:
        self._obs = obs
        self.bb.update_from_obs(obs)
        # The detector is component 3's cheap always-on half: it runs every frame and only
        # answers "is a sign there, roughly how far". It never READS the sign — that is the
        # VLM's job, and the whole point is to invoke the VLM as rarely as possible.
        if self.detector is not None and obs.images:
            try:
                self.bb.detections = self.detector.detect(obs.images[-1])
            except Exception:
                self.bb.detections = []     # a blind detector must not stop the robot
        st = self.bb.state
        if st == FSMState.DRIVE:
            return self._drive()
        if st == FSMState.HALT:
            return self._halt()
        if st == FSMState.REASON:
            return self._reason()
        if st == FSMState.MANEUVER:
            return self._maneuver()
        if st in (FSMState.ARRIVED, FSMState.FAILED):
            return Command(kind="stop", meta={"reason": st.value.lower()})
        return Command(kind="stop", meta={"reason": "unhandled_state"})

    # ---- states ----
    def _drive(self) -> Command:
        bb, cfg = self.bb, self.cfg
        # triggers in priority order (Phase 1 subset of spec §5.1)
        if triggers.no_safe_path(bb, self.safety, cfg):
            bb.state = FSMState.HALT
            return Command(kind="stop", meta={"reason": "no_safe_path"})
        # T5 HAZARD then T1 SIGN_READABLE (spec §5.1 priority order)
        if self.reasoner is not None:
            det = triggers.hazard(bb, cfg) or triggers.sign_readable(bb, cfg)
            if det is not None:
                return self._enter_reason(det)
        # A leg ends when its instruction has been carried out: either the robot got within
        # d_arrive of the nudge, or it has travelled leg_max_m trying. Both must be checked
        # together — checking expiry first and arrival second let an expiring leg be mistaken
        # for the mission goal and end the run.
        if bb.leg is not None:
            if triggers.subgoal_reached(bb, cfg) or triggers.leg_expired(bb, cfg):
                bb.leg = None                       # re-arms sign reading
                self.set_subgoal(pose_ahead(bb.robot_pose, cfg.d_look, 0.0),
                                 bb.stamp, source="continue")
        elif triggers.subgoal_reached(bb, cfg):
            if bb.subgoal is not None and bb.subgoal.source == "mission":
                bb.state = FSMState.ARRIVED         # the destination we were actually given
                return Command(kind="stop", meta={"reason": "arrived"})
            # a "continue" subgoal is just a carrot: push it further ahead and keep driving
            self.set_subgoal(pose_ahead(bb.robot_pose, cfg.d_look, 0.0),
                             bb.stamp, source="continue")
        if triggers.stalled(bb, cfg):
            bb.state = FSMState.HALT
            return Command(kind="stop", meta={"reason": "stalled"})
        if bb.subgoal is None:
            return Command(kind="stop", meta={"reason": "no_subgoal"})

        # normal execution: policy proposes a chunk, safety vets or replans it
        goal_robot = world_goal_to_robot(bb.subgoal.pose, bb.robot_pose, cfg.omni_range_m)
        # (N,4): [dx, dy, hx, hy]. Do NOT reshape to (-1,2) — that silently interleaves the
        # heading columns into the positions and doubles the waypoint count.
        wp = np.atleast_2d(np.asarray(
            self.policy.predict_waypoints(self._two_images(), goal_robot), float))
        used, path = self.safety.refine(bb.occupancy, wp)
        if not used and path is None:
            bb.state = FSMState.HALT
            return Command(kind="stop", meta={"reason": "no_safe_path"})
        return Command(kind="waypoints",
                       waypoints=(wp if used else path),
                       meta={"used_policy": bool(used), "omni": wp,
                             "refined": (None if used else path)})

    def _halt(self) -> Command:
        bb, cfg = self.bb, self.cfg
        if bb.subgoal is not None:
            goal_robot = world_goal_to_robot(bb.subgoal.pose, bb.robot_pose, cfg.omni_range_m)
            if self.safety.path_exists(bb.occupancy, goal_robot):
                bb.state = FSMState.DRIVE
                return self._drive()
        return Command(kind="stop", meta={"reason": "halted"})

    # ---- Phase 2: reasoning ----
    def _enter_reason(self, det) -> Command:
        """A sign fired a trigger. Stop the robot and hand the frame to the slow model.
        (This is 'stop-and-reason': the simplest, safest synchronisation. Spec §6, Mode A.)"""
        bb = self.bb
        if bb.reason_budget <= 0:            # degraded default: conservative (spec §8)
            bb.state = FSMState.HALT
            return Command(kind="stop", meta={"reason": "reason_budget_exhausted"})
        bb.state = FSMState.REASON
        bb.pending_sign = det
        # Compute the debounce key NOW, while the detection matches the current pose. By the
        # time _reason() runs the robot has moved a step, and combining the stale detection
        # distance with the new pose places the sign somewhere it never was — so the committed
        # key never matches the next fresh one and the VLM fires on the same sign forever.
        bb.pending_sign_key = triggers.sign_key(det, bb)
        return Command(kind="stop", meta={"reason": "reasoning", "sign": det.label})

    def _reason(self) -> Command:
        bb, cfg = self.bb, self.cfg
        det = bb.pending_sign
        img = self._obs.images[-1] if self._obs and self._obs.images else None
        crop = None
        if img is not None and det is not None and hasattr(self.detector, "crop"):
            try:
                crop = self.detector.crop(img, det)
            except Exception:
                crop = None
        request = {
            "image": img,
            # The VLM gets the sign's pixels. Falls back to the whole frame, which still works
            # but costs more visual tokens to prefill — cropping is the cheapest latency win.
            "sign_image": crop if crop is not None else img,
            "sign_crop": getattr(det, "bbox", None),
            "label": getattr(det, "label", ""),
            "mission_goal": bb.mission_goal,
            "context": f"pose=({bb.robot_pose.x:.1f},{bb.robot_pose.y:.1f})",
        }
        try:
            dec = self.reasoner.reason(request)
        except Exception as e:               # reasoning must never crash the robot
            bb.state = FSMState.HALT
            bb.pending_sign = None
            bb.pending_sign_key = ""
            return Command(kind="stop", meta={"reason": "reasoner_error", "error": repr(e)})

        bb.reason_budget -= 1
        bb.last_decision = dec
        if det is not None:                  # debounce: don't re-read the same sign (spec §8)
            key = bb.pending_sign_key or triggers.sign_key(det, bb)
            bb.committed_signs.add(key)
            bb.leg = Leg(start_pose=bb.robot_pose, decision=dec, start_stamp=bb.stamp,
                         sign_key=key)
        bb.pending_sign = None
        bb.pending_sign_key = ""

        # apply() returns a POSE, not a Subgoal.
        state, subgoal_pose, maneuver = decision_map.apply(dec, bb.robot_pose, cfg)
        bb.state = state
        if subgoal_pose is not None:
            # source="sign": reaching this means the sign's instruction has been carried out,
            # not that we have arrived anywhere.
            bb.subgoal = Subgoal(pose=subgoal_pose, stamp=bb.stamp, source="sign")
        self._maneuver_name = maneuver

        if state == FSMState.MANEUVER:
            return self._maneuver()
        if state == FSMState.ARRIVED:
            # A destination sign said the goal is here. The mission is complete — and it ended
            # because the robot read a sign, never because it was handed the goal's coordinates.
            return Command(kind="stop", meta={"reason": "arrived",
                                              "decision": dec.type.value,
                                              "sign": getattr(det, "label", "")})
        if state == FSMState.HALT:
            return Command(kind="stop", meta={"reason": "decision_stop",
                                              "decision": dec.type.value})
        return self._drive()                 # decision set a new goal -> get moving

    def _maneuver(self) -> Command:
        """A turn-in-place the action model can't express. Issue it once, then resume driving."""
        if self._maneuver_issued:
            self._maneuver_issued = False
            self.bb.state = FSMState.DRIVE
            return self._drive()
        self._maneuver_issued = True
        self.bb.state = FSMState.MANEUVER
        return Command(kind="maneuver", maneuver=self._maneuver_name,
                       meta={"decision": (self.bb.last_decision.type.value
                                          if self.bb.last_decision else None)})

    # ---- helpers ----
    def _two_images(self):
        """OmniVLA expects 2 frames; duplicate the latest if only one is available."""
        imgs = list(self._obs.images) if self._obs else []
        if len(imgs) == 1:
            imgs = [imgs[0], imgs[0]]
        return imgs