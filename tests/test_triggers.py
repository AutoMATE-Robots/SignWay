"""Phase 1 trigger predicate tests."""
from c5_orchestrator import triggers
from c5_orchestrator.blackboard import Blackboard
from common.config import Params
from common.types import Pose, Subgoal
from c4_safety.mock_safety import MockSafety


def _bb(pose, subgoal=None):
    bb = Blackboard()
    bb.robot_pose = pose
    if subgoal is not None:
        bb.subgoal = Subgoal(pose=subgoal)
    return bb


def test_subgoal_reached():
    cfg = Params(d_arrive=0.5)
    assert triggers.subgoal_reached(_bb(Pose(0, 0), Pose(0.3, 0)), cfg)
    assert not triggers.subgoal_reached(_bb(Pose(0, 0), Pose(2, 0)), cfg)
    assert not triggers.subgoal_reached(_bb(Pose(0, 0)), cfg)   # no subgoal


def test_no_safe_path():
    cfg = Params()
    bb = _bb(Pose(0, 0), Pose(5, 0))
    assert triggers.no_safe_path(bb, MockSafety(blocked=True), cfg)
    assert not triggers.no_safe_path(bb, MockSafety(blocked=False), cfg)


def test_stalled_needs_full_window():
    cfg = Params(t_stall=1.0, eps_progress=0.2)
    bb = Blackboard()
    bb.update_from_obs(type("O", (), {"robot_pose": Pose(0, 0), "occupancy": None,
                                      "detections": [], "stamp": 0.0})())
    assert not triggers.stalled(bb, cfg)            # only one sample -> inf progress
