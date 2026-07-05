"""Trigger predicates evaluated in DRIVE (orchestration spec §5.1).

Phase 1 wires the three that need no perception or VLM: SAFETY, SUBGOAL_REACHED, STALLED.
HAZARD / SIGN_READABLE / DECISION_POINT arrive together with the REASON state in Phase 2 — they
are pure predicates over the blackboard too, so they slot in without restructuring.
"""
from __future__ import annotations

from .blackboard import Blackboard
from .config import Params
from .geometry import dist, world_goal_to_robot
from .interfaces import Safety


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
