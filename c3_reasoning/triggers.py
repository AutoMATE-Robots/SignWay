"""Trigger predicates evaluated in DRIVE (orchestration spec §5.1).

Phase 1 wires the three that need no perception or VLM: SAFETY, SUBGOAL_REACHED, STALLED.
HAZARD / SIGN_READABLE / DECISION_POINT arrive together with the REASON state in Phase 2 — they
are pure predicates over the blackboard too, so they slot in without restructuring.
"""
from __future__ import annotations

import numpy as np

from c5_orchestrator.blackboard import Blackboard
from common.config import Params
from common.geometry import dist, world_goal_to_robot
from common.interfaces import Safety


def subgoal_reached(bb: Blackboard, cfg: Params) -> bool:
    if bb.subgoal is None:
        return False
    return dist(bb.robot_pose, bb.subgoal.pose) < cfg.d_arrive


def stalled(bb: Blackboard, cfg: Params) -> bool:
    return bb.recent_progress(cfg.t_stall) < cfg.eps_progress


def no_safe_path(bb: Blackboard, safety: Safety, cfg: Params) -> bool:
    if bb.subgoal is None:
        return False
    goal_robot = world_goal_to_robot(bb.subgoal.pose, bb.robot_pose, cfg.omni_range_m)
    return not safety.path_exists(bb.occupancy, goal_robot)


# ── Phase 2 triggers (spec §5.1) ────────────────────────────────────────────────
def sign_key(det, bb: Blackboard) -> str:
    """Debounce identity for a sign: its label + where the SIGN is, coarsely binned.

    The sign's world position is estimated from the detection (distance + bearing) plus the
    robot's pose. Anchoring to the sign rather than the robot is the whole point: a physical
    sign must map to ONE key no matter where it is seen from. Keying on the robot's pose — which
    is what this did originally — made the same sign look new every couple of metres, so the VLM
    fired repeatedly on one sign. That directly breaks the "reason once, only when needed"
    claim the project rests on.
    """
    d = float(getattr(det, "est_distance_m", 0.0) or 0.0)
    b = float(getattr(det, "est_bearing_rad", 0.0) or 0.0)
    if not np.isfinite(d):
        # No depth -> the sign cannot be located, so fall back to anchoring on the robot. This
        # is the WEAK key (the same sign looks new every couple of metres), which is why leg
        # gating above is the primary guard and this is only a secondary one.
        return f"{det.label}@robot:{round(bb.robot_pose.x / 2.0)},{round(bb.robot_pose.y / 2.0)}"
    world_b = bb.robot_pose.yaw + b
    sx = bb.robot_pose.x + d * np.cos(world_b)
    sy = bb.robot_pose.y + d * np.sin(world_b)
    return f"{det.label}@{round(sx / 2.0)},{round(sy / 2.0)}"


def hazard(bb: Blackboard, cfg: Params):
    """T5: a hazard sign in view, confident. Returns the Detection or None."""
    for d in bb.detections:
        if d.hazard and d.conf >= cfg.tau_conf and sign_key(d, bb) not in bb.committed_signs:
            return d
    return None


def leg_expired(bb: Blackboard, cfg: Params) -> bool:
    """Has the robot travelled far enough under the current decision to look for a new sign?"""
    if bb.leg is None:
        return False
    d = np.hypot(bb.robot_pose.x - bb.leg.start_pose.x, bb.robot_pose.y - bb.leg.start_pose.y)
    return bool(d >= cfg.leg_max_m)


def sign_readable(bb: Blackboard, cfg: Params):
    """T1: a legible sign we have not already acted on. Fire the reasoner on every one.

    LEGIBILITY IS NOT DECIDED HERE. The detector already measured it (text height, blur,
    detector confidence) and only returns detections that passed. This function decides
    NOVELTY, which is a different question. Re-checking legibility here with a distance proxy
    duplicated that logic and could disagree with it — and did: the detector reports distance
    as inf when no depth is available, so a `est_distance_m > d_read` filter silently discarded
    every legible sign and the reasoner never fired.

    Two independent guards stop one sign costing many VLM calls:
      1. leg gating — while a decision is being carried out, new signs are ignored. This needs
         no distance and is the robust one.
      2. the committed-sign key — a positional guard for when several signs are in view.
    """
    if bb.leg is not None and not leg_expired(bb, cfg):
        return None                     # mid-decision: not listening for new instructions
    candidates = [d for d in bb.detections
                  if d.conf >= cfg.tau_conf and sign_key(d, bb) not in bb.committed_signs]
    if not candidates:
        return None
    # Nearest first. With no depth every distance is inf and min() keeps the first, which is
    # the right behaviour: read something rather than nothing.
    return min(candidates, key=lambda d: d.est_distance_m)