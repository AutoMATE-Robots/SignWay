"""Decision -> subgoal / maneuver mapping (orchestration spec §7).

The VLM's only effect on the robot is here: it moves the goalpost the VLA is already chasing.
Turns become pose subgoals so OmniVLA still does the locomotion (keeping its learned corridor
following); only in-place rotation bypasses the policy, because that is exactly what a
forward-biased waypoint policy cannot express.
"""
from __future__ import annotations

import numpy as np

from common.config import Params
from common.geometry import pose_ahead
from common.types import Decision, DecisionType, FSMState, Pose


def apply(decision: Decision, robot: Pose, cfg: Params):
    """Return (new_state, subgoal_pose_or_None, maneuver_or_None)."""
    t = decision.type

    if t == DecisionType.STOP:
        return FSMState.HALT, None, None

    # The destination sign: the goal is HERE, so the run is over. No subgoal — the robot stops.
    # This is the only decision that ends a mission, and it comes from reading a sign, not from
    # a coordinate. Distinct from STOP, which is a safety/uncertainty halt the robot recovers from.
    if t == DecisionType.ARRIVED:
        return FSMState.ARRIVED, None, None

    # A u-turn is the ONE case that bypasses the policy. OmniVLA cannot reverse — a
    # forward-biased waypoint policy has no way to express "go backwards" — so this pivots in
    # place and then drives on the new heading. Everything else stays a pose subgoal, because
    # the whole design is that the VLM only ever moves the goalpost and the VLA does the driving.
    if t == DecisionType.U_TURN:
        turned = Pose(robot.x, robot.y, robot.yaw + np.pi)
        return FSMState.MANEUVER, pose_ahead(turned, cfg.d_look, 0.0), "u_turn"

    if t == DecisionType.CONTINUE:
        return FSMState.DRIVE, pose_ahead(robot, cfg.d_look, 0.0), None

    # Turns are a NUDGE: drop the goalpost off to the side and let the policy steer toward it.
    # The robot does not have to reach it — swinging the heading round is the point, after which
    # the corridor itself guides the robot. (Note the subgoal at d_look directly abeam can sit
    # inside the robot's turning circle, so it may orbit rather than arrive; raise max_turn or
    # d_look if you want it actually reached.)
    if t == DecisionType.TURN_LEFT:
        return FSMState.DRIVE, pose_ahead(robot, cfg.d_look, np.pi / 2), None

    if t == DecisionType.TURN_RIGHT:
        return FSMState.DRIVE, pose_ahead(robot, cfg.d_look, -np.pi / 2), None

    if t == DecisionType.GOTO:
        tgt = decision.target or {}
        bearing = float(tgt.get("bearing_rad", 0.0))
        dist = float(tgt.get("distance_m", cfg.d_look))
        return FSMState.DRIVE, pose_ahead(robot, dist, bearing), None

    # unknown decision -> be conservative
    return FSMState.HALT, None, None