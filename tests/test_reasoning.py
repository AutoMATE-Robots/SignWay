"""The reasoning cycle (component 3 + the orchestrator's REASON state).

Proves: the robot drives, spots a sign, stops, asks the slow model, acts on the answer, marks
that sign as done so it doesn't re-read it, and resumes driving.

No VLM, no GPU, no simulator — mock detector + mock reasoner. When the real VLM is ported into
c3_reasoning/reasoner_vlm.py, these same tests cover it.
"""
import numpy as np

from c5_orchestrator.fsm import FSM
from common.config import Params
from common.types import Detection, FSMState, Observation, Pose
from c2_action.mock_policy import MockPolicy
from c4_safety.mock_safety import MockSafety
from c3_reasoning.mock_reasoning import MockDetector, MockReasoner


def _obs(x, y, yaw=0.0, t=0.0, dets=None):
    return Observation(images=[np.zeros((4, 4, 3), np.uint8)], robot_pose=Pose(x, y, yaw),
                       detections=dets or [], stamp=t)


def _sign(label, dist, conf=0.9, hazard=False):
    return Detection(id="s0", label=label, conf=conf, est_distance_m=dist, hazard=hazard)


def _fsm(reasoner=None):
    return FSM(MockPolicy(), MockSafety(), Params(), mission_goal="cafeteria",
               reasoner=reasoner or MockReasoner())


def test_no_sign_means_no_reasoning():
    """The whole thesis: if nothing needs reading, the slow model never runs."""
    r = MockReasoner()
    fsm = _fsm(r)
    fsm.set_subgoal(Pose(5, 0, 0))
    for k in range(5):
        cmd = fsm.tick(_obs(k * 0.3, 0, t=k * 0.1))
        assert cmd.kind == "waypoints"
    assert r.calls == 0                       # zero VLM calls on an empty corridor
    assert fsm.bb.state == FSMState.DRIVE


def test_sign_triggers_reasoning_and_turn():
    """Sign -> stop -> reason -> the goalpost moves LEFT -> the policy steers toward it.

    The VLM never drives motors. Its entire effect is relocating the subgoal; OmniVLA does the
    locomotion. That separation is the design.
    """
    r = MockReasoner()
    fsm = _fsm(r)
    fsm.set_subgoal(Pose(5, 0, 0))
    fsm.tick(_obs(0, 0))                                        # clear corridor -> DRIVE
    assert fsm.bb.state == FSMState.DRIVE

    cmd = fsm.tick(_obs(1, 0, t=0.1, dets=[_sign("cafeteria left", 2.0)]))
    assert fsm.bb.state == FSMState.REASON                      # sign in range -> stop & think
    assert cmd.kind == "stop"

    cmd = fsm.tick(_obs(1, 0, t=0.2, dets=[_sign("cafeteria left", 2.0)]))
    assert r.calls == 1                                          # the slow model ran once
    assert fsm.bb.last_decision.type.value == "turn_left"
    assert fsm.bb.state == FSMState.DRIVE                        # keep driving, new goalpost
    assert cmd.kind == "waypoints"
    assert fsm.bb.subgoal.pose.y > 0.5                           # the goal moved LEFT
    assert np.asarray(cmd.meta["omni"])[:, 1].sum() > 0          # and the policy steers left


def test_same_sign_is_not_read_twice():
    """Debounce: the sign stays in view for many frames but must only cost one VLM call."""
    r = MockReasoner()
    fsm = _fsm(r)
    fsm.set_subgoal(Pose(5, 0, 0))
    for k in range(6):
        fsm.tick(_obs(1, 0, t=k * 0.1, dets=[_sign("cafeteria left", 2.0)]))
    assert r.calls == 1


def test_distance_alone_no_longer_decides_whether_to_read():
    """A distant sign with large lettering IS readable, and the FSM trusts the detector.

    This test used to assert the opposite: distance > d_read => ignore. That distance proxy is
    exactly what the readability gate replaced — legibility is now MEASURED from the image
    (text height in pixels, blur, detector confidence) by c3_reasoning/readability.py, and a
    detection only reaches the FSM if it already passed. Re-checking it here with distance
    duplicated the decision and could contradict it.
    """
    r = MockReasoner()
    fsm = _fsm(r)
    fsm.set_subgoal(Pose(5, 0, 0))
    fsm.tick(_obs(0, 0, dets=[_sign("cafeteria left", 20.0)]))
    assert fsm.bb.state == FSMState.REASON       # legible per the detector -> worth reading


def test_low_confidence_sign_is_ignored():
    r = MockReasoner()
    fsm = _fsm(r)
    fsm.set_subgoal(Pose(5, 0, 0))
    fsm.tick(_obs(0, 0, dets=[_sign("cafeteria left", 2.0, conf=0.1)]))
    assert r.calls == 0
    assert fsm.bb.state == FSMState.DRIVE


def test_stop_sign_halts():
    fsm = _fsm()
    fsm.set_subgoal(Pose(5, 0, 0))
    fsm.tick(_obs(1, 0, dets=[_sign("stop", 1.5, hazard=True)]))
    cmd = fsm.tick(_obs(1, 0, t=0.1, dets=[_sign("stop", 1.5, hazard=True)]))
    assert fsm.bb.state == FSMState.HALT
    assert cmd.kind == "stop"


def test_u_turn_goes_through_maneuver():
    """'Staff only' -> turn around. The action model can't reverse, so this bypasses it."""
    fsm = _fsm()
    fsm.set_subgoal(Pose(5, 0, 0))
    fsm.tick(_obs(1, 0, dets=[_sign("staff only", 1.5)]))
    cmd = fsm.tick(_obs(1, 0, t=0.1, dets=[_sign("staff only", 1.5)]))
    assert fsm.bb.state == FSMState.MANEUVER
    assert cmd.kind == "maneuver" and cmd.maneuver == "u_turn"
    cmd2 = fsm.tick(_obs(1, 0, t=0.2))                            # harness did the turn
    assert fsm.bb.state == FSMState.DRIVE
    assert cmd2.kind == "waypoints"


def test_hazard_beats_a_normal_sign():
    """Priority: a hazard in view wins over an ordinary directory sign."""
    fsm = _fsm()
    fsm.set_subgoal(Pose(5, 0, 0))
    cmd = fsm.tick(_obs(1, 0, dets=[_sign("cafeteria left", 2.0),
                                    _sign("stop", 2.5, hazard=True)]))
    assert fsm.bb.state == FSMState.REASON
    assert cmd.meta["sign"] == "stop"


def test_reasoner_crash_does_not_crash_the_robot():
    class Broken(MockReasoner):
        def reason(self, request):
            raise RuntimeError("VLM died")
    fsm = _fsm(Broken())
    fsm.set_subgoal(Pose(5, 0, 0))
    fsm.tick(_obs(1, 0, dets=[_sign("cafeteria left", 2.0)]))
    cmd = fsm.tick(_obs(1, 0, t=0.1, dets=[_sign("cafeteria left", 2.0)]))
    assert fsm.bb.state == FSMState.HALT                          # stopped safely
    assert cmd.meta["reason"] == "reasoner_error"


def test_budget_exhausted_degrades_safely():
    cfg = Params(reason_budget=0)
    fsm = FSM(MockPolicy(), MockSafety(), cfg, mission_goal="cafeteria",
              reasoner=MockReasoner())
    fsm.set_subgoal(Pose(5, 0, 0))
    cmd = fsm.tick(_obs(1, 0, dets=[_sign("cafeteria left", 2.0)]))
    assert fsm.bb.state == FSMState.HALT
    assert cmd.meta["reason"] == "reason_budget_exhausted"


def test_detector_reports_nearby_signs_only():
    d = MockDetector(signs=[(2.0, 0.0, "cafeteria left", False),
                            (50.0, 0.0, "staff only", False)], visible_from=3.0)
    d.set_pose(0.0, 0.0, 0.0)
    found = d.detect(None)
    assert len(found) == 1 and found[0].label == "cafeteria left"


def test_one_sign_costs_exactly_one_vlm_call_while_driving_toward_it():
    """The thesis, as a test. A sign stays in view for many frames as the robot approaches it
    from different poses. It must be READ ONCE.

    This is the case the first Isaac run failed: the debounce key was anchored to the robot's
    pose, so every couple of metres the same sign looked new and the VLM fired again.
    """
    r = MockReasoner()
    d = MockDetector(signs=[(6.0, 0.0, "cafeteria left", False)], visible_from=5.0)
    fsm = FSM(MockPolicy(), MockSafety(), Params(), mission_goal="cafeteria",
              detector=d, reasoner=r)
    fsm.set_subgoal(Pose(10, 0, 0))
    for k in range(12):                       # drive from 0 to 5.5m, sign visible most of the way
        x = k * 0.5
        d.set_pose(x, 0.0, 0.0)
        fsm.tick(_obs(x, 0.0, t=k * 0.1))
    assert r.calls == 1, f"one sign must cost one VLM call, got {r.calls}"


def test_two_different_signs_cost_two_calls():
    r = MockReasoner()
    d = MockDetector(signs=[(3.0, 0.0, "cafeteria left", False),
                            (9.0, 0.0, "staff only", False)], visible_from=2.0)
    fsm = FSM(MockPolicy(), MockSafety(), Params(), mission_goal="cafeteria",
              detector=d, reasoner=r)
    fsm.set_subgoal(Pose(12, 0, 0))
    for k in range(24):
        x = k * 0.5
        d.set_pose(x, 0.0, 0.0)
        fsm.tick(_obs(x, 0.0, t=k * 0.1))
    assert r.calls == 2, f"two signs -> two calls, got {r.calls}"


def test_reasoner_is_deaf_while_carrying_out_a_decision():
    """One sign, one call — guarded by the leg, not by geometry.

    While a decision is being executed the robot ignores new signs. This is the guard that
    works with no depth: it needs no distance estimate, unlike the positional debounce key.
    """
    r = MockReasoner()
    d = MockDetector(signs=[(3.0, 0.0, "cafeteria left", False)], visible_from=6.0)
    fsm = FSM(MockPolicy(), MockSafety(), Params(), mission_goal="cafeteria",
              detector=d, reasoner=r)
    fsm.set_subgoal(Pose(10, 0, 0))
    for k in range(8):                       # 0 -> 1.4m: sign visible the whole way
        x = k * 0.2
        d.set_pose(x, 0.0, 0.0)
        fsm.tick(_obs(x, 0.0, t=k * 0.1))
    assert r.calls == 1
    assert fsm.bb.leg is not None            # still mid-decision


def test_an_unreachable_subgoal_does_not_silence_the_robot_forever():
    """A leg must expire on travel, or one nudge the robot can't reach makes it deaf for good.

    OmniVLA is documented to sail past goals rather than stop on them, so "subgoal never
    reached" is the normal case. Without expiry, sign reading would end after the first sign.
    """
    r = MockReasoner()
    cfg = Params(leg_max_m=3.0)
    d = MockDetector(signs=[(1.0, 0.0, "cafeteria left", False),
                            (9.0, 0.0, "staff only", False)], visible_from=2.0)
    fsm = FSM(MockPolicy(), MockSafety(), cfg, mission_goal="cafeteria", detector=d, reasoner=r)
    fsm.set_subgoal(Pose(12, 0, 0))
    for k in range(24):                      # drives straight past both, never reaching either
        x = k * 0.5                          # sign-derived subgoal
        d.set_pose(x, 0.0, 0.0)
        fsm.tick(_obs(x, 0.0, t=k * 0.1))
    assert r.calls == 2                      # the leg expired, so the second sign was read


def test_finishing_a_sign_leg_is_not_arriving():
    """Reaching the nudge a sign gave us must NOT end the mission.

    The two events are indistinguishable without provenance: both are "robot is within
    d_arrive of bb.subgoal". Marking sign-derived subgoals means only the MISSION goal can
    end the run — otherwise the first turn the robot completes looks like success.
    """
    r = MockReasoner()
    cfg = Params(d_arrive=0.5, d_look=1.0)
    d = MockDetector(signs=[(1.0, 0.0, "cafeteria left", False)], visible_from=1.5)
    fsm = FSM(MockPolicy(), MockSafety(), cfg, mission_goal="cafeteria", detector=d, reasoner=r)
    fsm.set_subgoal(Pose(20, 0, 0))                  # mission goal, far away
    for k in range(6):
        d.set_pose(0.5, 0.0, 0.0)
        fsm.tick(_obs(0.5, 0.0, t=k * 0.1))
    assert r.calls == 1
    assert fsm.bb.subgoal.source == "sign"           # goalpost moved by the sign
    # sit on top of the sign's subgoal: the leg completes, but the run must NOT end
    sg = fsm.bb.subgoal.pose
    for k in range(3):
        d.set_pose(sg.x, sg.y, 0.0)
        fsm.tick(_obs(sg.x, sg.y, t=1.0 + k * 0.1))
    assert fsm.bb.state != FSMState.ARRIVED
    assert fsm.bb.leg is None                        # instruction done -> listening again


def test_reaching_the_mission_goal_does_arrive():
    fsm = FSM(MockPolicy(), MockSafety(), Params(d_arrive=0.5))
    fsm.set_subgoal(Pose(1.0, 0.0, 0.0))             # source defaults to "mission"
    fsm.tick(_obs(0.9, 0.0))
    assert fsm.bb.state == FSMState.ARRIVED


def test_a_destination_sign_ends_the_mission():
    """The mapless finish: the robot arrives because it READ a sign naming the goal here,
    never because it was handed the goal's coordinates."""
    r = MockReasoner()
    fsm = _fsm(r)
    fsm.set_subgoal(Pose(50, 0, 0))                  # a carrot far off; the sign, not this, ends it
    fsm.tick(_obs(1, 0, dets=[_sign("cafeteria", 1.5)]))
    cmd = fsm.tick(_obs(1, 0, t=0.1, dets=[_sign("cafeteria", 1.5)]))
    assert fsm.bb.state == FSMState.ARRIVED
    assert cmd.kind == "stop" and cmd.meta.get("reason") == "arrived"
    assert r.calls == 1                              # arrival cost exactly one read


def test_a_chain_of_signs_composes_into_a_route():
    """Long horizon: turn left at sign 1, turn right at sign 2, arrive at sign 3 — three
    reads, one route, and only the last sign ends the run."""
    cfg = Params(d_arrive=0.5, d_look=1.0, leg_max_m=1.0)
    signs = [(3.0, 0.0, "cafeteria left", False),
             (3.0, 3.0, "cafeteria right", False),
             (6.0, 3.0, "cafeteria", False)]
    d = MockDetector(signs=signs, visible_from=1.2)
    r = MockReasoner()
    fsm = FSM(MockPolicy(), MockSafety(), cfg, mission_goal="cafeteria", detector=d, reasoner=r)
    fsm.set_subgoal(Pose(100, 0, 0))                 # never reached; the signs steer the whole way

    seen = []
    # walk the robot past each sign in turn; the detector is told where the robot is each tick
    waypoints = [(3.0, 0.2), (3.0, 3.0), (5.8, 3.0)]
    for (rx, ry) in waypoints:
        for k in range(4):
            d.set_pose(rx, ry, 0.0)
            fsm.tick(_obs(rx, ry, t=len(seen) * 0.1))
            seen.append(fsm.bb.last_decision.type.value if fsm.bb.last_decision else None)
            if fsm.bb.state == FSMState.ARRIVED:
                break
        if fsm.bb.state == FSMState.ARRIVED:
            break
    assert r.calls == 3                              # exactly one read per sign, no re-reads
    assert fsm.bb.state == FSMState.ARRIVED          # and the last one ended it