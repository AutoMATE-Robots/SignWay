"""Phase 1 FSM tests — run in plain Python, no simulator or GPU.

Proves the drive loop: emit waypoints while driving, stop on arrival, HALT when safety blocks
and recover when it clears, and HALT on a stall.
"""
import numpy as np

from c5_orchestrator.fsm import FSM
from common.config import Params
from common.types import FSMState, Observation, Pose
from c2_action.mock_policy import MockPolicy
from c4_safety.mock_safety import MockSafety


def _obs(x, y, yaw=0.0, t=0.0):
    return Observation(images=[np.zeros((4, 4, 3), np.uint8)], robot_pose=Pose(x, y, yaw), stamp=t)


def test_drive_emits_waypoints():
    fsm = FSM(MockPolicy(), MockSafety(), Params())
    fsm.set_subgoal(Pose(5, 0, 0))
    cmd = fsm.tick(_obs(0, 0))
    assert fsm.bb.state == FSMState.DRIVE
    assert cmd.kind == "waypoints"
    assert cmd.waypoints.shape == (8, 4)   # [dx, dy, hx, hy]
    assert cmd.meta["used_policy"] is True


def test_arrival():
    fsm = FSM(MockPolicy(), MockSafety(), Params())
    fsm.set_subgoal(Pose(0.2, 0.0, 0.0))          # within d_arrive of the start
    cmd = fsm.tick(_obs(0, 0))
    assert fsm.bb.state == FSMState.ARRIVED
    assert cmd.kind == "stop"


def test_safety_halt_and_recover():
    fsm = FSM(MockPolicy(), MockSafety(blocked=True), Params())
    fsm.set_subgoal(Pose(5, 0, 0))
    cmd = fsm.tick(_obs(0, 0))
    assert fsm.bb.state == FSMState.HALT
    assert cmd.kind == "stop"
    fsm.safety.blocked = False                      # path clears
    cmd2 = fsm.tick(_obs(0, 0, t=0.1))
    assert fsm.bb.state == FSMState.DRIVE
    assert cmd2.kind == "waypoints"


def test_stall_halts():
    cfg = Params(t_stall=1.0, eps_progress=0.2)
    fsm = FSM(MockPolicy(), MockSafety(), cfg)
    fsm.set_subgoal(Pose(5, 0, 0))
    fsm.tick(_obs(0, 0, t=0.0))                     # not enough history yet
    fsm.tick(_obs(0, 0, t=0.6))                     # window not full -> no stall
    cmd = fsm.tick(_obs(0, 0, t=1.4))              # full window, no movement -> stall
    assert fsm.bb.state == FSMState.HALT
    assert cmd.meta.get("reason") == "stalled"


def test_no_subgoal_stops():
    fsm = FSM(MockPolicy(), MockSafety(), Params())
    cmd = fsm.tick(_obs(0, 0))
    assert cmd.kind == "stop"
    assert cmd.meta.get("reason") == "no_subgoal"
