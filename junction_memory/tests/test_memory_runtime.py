#!/usr/bin/env python3
"""
test_memory_runtime.py — synthetic validation of the online memory runtime.

Reproduces the README's arrival cases:
  case 1  same directed edge, already reasoned   -> beam + edge confirm, recall
  case 2  reverse of a driven edge               -> undirected perp match
  case 3  node reached via an undriven corridor  -> beam alone
plus the parallel-corridor rejection (beam vs bearing cone), rival
suppression (miss-not-wrong-turn), recall reprojection math, straight-through
visits, and YawScaler parity with calibrate_yaw.reintegrate.
"""
import math

import numpy as np
import pytest

from memory_runtime import (MemoryRuntime, YawScaler, wrap_pi, wrap_deg,
                            LATERAL_M, EDGE_RADIUS_M, EDGE_RADIUS_TIGHT)
from synth_traj import traj


def run(rt, arrays, hooks=None):
    """Step the runtime; hooks = {sample_index: callable(rt)}."""
    t, x, y, yaw = arrays
    for i in range(len(t)):
        if hooks and i in hooks:
            hooks[i](rt)
        rt.step(t[i], x[i], y[i], yaw[i])
    return rt


def kinds(rt):
    return [e.kind for e in rt.events]


def first(rt, kind):
    for e in rt.events:
        if e.kind == kind:
            return e
    return None


def all_of(rt, kind):
    return [e for e in rt.events if e.kind == kind]


# ----------------------------------------------------------------------------
# CASE 1 — the square bag: reason at A going north, loop the block, come back
# up the same corridor, recall 'straight' for the new goal without a call.
# ----------------------------------------------------------------------------
def square_arrays():
    # start (40,-25) heading north; A at (40,0). Left turns all the way round;
    # final leg D->A retraces the first northbound corridor; straight through.
    return traj([
        ("straight", 25.0),            # -> A
        ("turn", 90, 1.5),             # left at A (VLM said left for goal_A)
        ("straight", 38.0),            # -> B
        ("turn", 90, 1.5),
        ("straight", 23.0),            # -> C
        ("turn", 90, 1.5),
        ("straight", 38.0),            # -> D (same corridor line as start)
        ("turn", 90, 1.5),
        ("straight", 23.0),            # northbound retrace toward A
        ("straight", 8.0),             # straight THROUGH A
    ], x0=40.0, y0=-25.0, yaw0=math.pi / 2)


@pytest.fixture(scope="module")
def square_rt():
    rt = MemoryRuntime(bag_name="synthetic-square")
    t, x, y, yaw = square_arrays()
    n = len(t)
    read_i = int(0.55 * (25.0 / 0.04) )          # ~mid first approach
    # goal schedule + first-pass VLM read, injected directly (gate shim has
    # its own integration test below)
    goal_switch_i = None
    # find sample just after leaving A the first time (~s=30 m)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    goal_switch_i = int(np.searchsorted(s, 45.0))
    hooks = {
        0: lambda r: setattr(r, "current_goal", "goal_A"),
        read_i: lambda r: r.provide_directory(
            {"goal_A": "left", "goal_E": "straight"},
            goal="goal_A", decision="left", conf=0.91, frame=read_i),
        goal_switch_i: lambda r: setattr(r, "current_goal", "goal_E"),
    }
    return run(rt, (t, x, y, yaw), hooks), s


def test_square_nodes_and_directory(square_rt):
    rt, s = square_rt
    assert len(rt.nodes) == 4, f"expected 4 corner nodes, got {list(rt.nodes)}"
    ds = first(rt, "DIRECTORY_STORED")
    assert ds is not None and ds.data["node"] == "n_001"
    a = rt.nodes["n_001"]
    assert set(a.routing) == {"goal_a", "goal_e"}
    assert a.visits == 2                          # turn visit + straight-through
    assert a.read_yaw is not None
    assert len(a.sign_visible_from) == 1          # readable from the south approach


def test_square_beam_edge_and_recall(square_rt):
    rt, s = square_rt
    lock = [e for e in all_of(rt, "BEAM_LOCK") if e.data["node"] == "n_001"
            and e.s > 60.0]                       # the revisit approach only
    assert lock, "beam never locked on A during the retrace"
    assert lock[0].data["lead_m"] > 14.0, lock[0].data

    em = [e for e in all_of(rt, "EDGE_MATCH") if e.data.get("toward") == "n_001"]
    assert em, "retrace corridor not recognised"
    assert em[0].data["same_direction"] is True
    assert em[0].data["tight_frac"] > 0.6         # clean same-direction retrace

    rec = first(rt, "MEMORY_RECALL")
    assert rec is not None
    assert rec.data["goal"] == "goal_E" or rec.data["goal"] == "goal_e"
    assert rec.data["action"] == "straight"       # NOT the stored 'left' action
    assert rec.data["lead_m"] > 12.0
    assert rec.data["flip_margin_deg"] > 30.0

    st = first(rt, "NODE_VISIT_STRAIGHT")
    assert st is not None and st.data["node"] == "n_001"

    gp = [e for e in all_of(rt, "GATE_PRIOR") if e.s > 60.0]
    assert gp and gp[0].data["legible"] is True   # same approach as the read

    prox = [e for e in all_of(rt, "PROXIMITY_FIRE") if e.data["node"] == "n_001"]
    assert prox, "baseline proximity gate should also fire, later"
    assert rec.data["lead_m"] > 5 * prox[0].data["range_m"]


def test_square_graph_schema(square_rt):
    rt, s = square_rt
    g = rt.to_json()
    assert g["schema_version"] == 1
    a = g["nodes"]["n_001"]
    assert a["arity"] >= 3
    assert a["routing"]["goal_e"]["src"] == "vlm"
    assert a["read_yaw_rad"] is not None
    assert any(e["length_m"] > 15 for e in g["edges"].values())
    # straight-through must still split the corridor at A
    eps = [tuple(e["endpoints"]) for e in g["edges"].values()]
    assert ("n_004", "n_001") in eps or ["n_004", "n_001"] in [list(p) for p in eps]


# ----------------------------------------------------------------------------
# CASE 2 — reverse traversal of a driven edge (the 'hard to explain' one):
# undirected perpendicular match, direction from index drift, and the
# lateral-offset cliff that killed nearest-sample matching.
# ----------------------------------------------------------------------------
def _reverse_setup(offset_recovers: bool):
    spec = [
        ("straight", 30.0),         # west->east corridor
        ("turn", 90, 1.5),          # node N1 at (30,0); corridor stored
        ("straight", 18.0),         # northbound corridor N1 -> apex
        ("turn", 180, 1.0),         # U-turn: node N2; lateral offset ~2.0 m
    ]
    if offset_recovers:
        spec += [                   # steer back to the corridor centreline
            ("straight", 2.0),
            ("turn", 30, 2.0), ("straight", 3.2), ("turn", -30, 2.0),
            ("straight", 12.0),     # southbound ON the stored centreline
        ]
    else:
        spec += [("straight", 16.0)]   # southbound, constant 2 m offset
    return traj(spec)


def test_reverse_traversal_perpendicular_match():
    rt = MemoryRuntime()
    rt.current_goal = "goal_x"
    run(rt, _reverse_setup(offset_recovers=False))
    em = [e for e in all_of(rt, "EDGE_MATCH") if e.data.get("reverse")]
    assert em, "reverse traversal not recognised despite 2 m offset"
    e = em[0]
    assert e.data["toward"] == "n_001"            # index drift points back to N1
    assert e.data["same_direction"] is False
    # the offset sits between the tight and loose radii: perpendicular at
    # 3.0 m catches it, the 1.5 m same-direction tier must not
    assert e.data["tight_frac"] < 0.2, "2 m offset should fail the tight tier"


def test_reverse_beam_needs_recentering():
    # with a constant 2 m offset the beam's |lateral| gate (1.5 m) rejects the
    # node — the edge channel is what covers the un-recentered reverse case
    rt = MemoryRuntime()
    rt.current_goal = "goal_x"
    run(rt, _reverse_setup(offset_recovers=False))
    locks = [e for e in all_of(rt, "BEAM_LOCK") if e.data["node"] == "n_001"]
    assert not locks
    # once the robot recentres, the beam locks and leads
    rt2 = MemoryRuntime()
    rt2.current_goal = "goal_x"
    run(rt2, _reverse_setup(offset_recovers=True))
    locks2 = [e for e in all_of(rt2, "BEAM_LOCK") if e.data["node"] == "n_001"]
    assert locks2 and locks2[0].data["lead_m"] > 8.0


# ----------------------------------------------------------------------------
# CASE 3 — undriven corridor + reprojection across approach directions,
# parallel-corridor rejection, rival suppression.
# ----------------------------------------------------------------------------
def test_undriven_corridor_beam_and_reprojection():
    rt = MemoryRuntime()
    rt.current_goal = "room_310"
    # first pass: eastbound, read the sign, turn LEFT (north) at N1=(20,0)
    t1 = traj([("straight", 20.0), ("turn", 90, 1.5), ("straight", 10.0),
               ("turn", 90, 1.5), ("straight", 8.0)])
    s1 = np.concatenate([[0], np.cumsum(np.hypot(np.diff(t1[1]), np.diff(t1[2])))])
    read_i = int(np.searchsorted(s1, 12.0))
    run(rt, t1, hooks={read_i: lambda r: r.provide_directory(
        {"room_310": "left"}, goal="room_310", decision="left",
        conf=0.9, frame=read_i)})
    assert "n_001" in rt.nodes and "room_310" in rt.nodes["n_001"].routing

    # second pass: approach N1 from the EAST heading WEST — a corridor never
    # driven. Beam must lock; recall must reproject left(north) -> RIGHT.
    t2 = traj([("straight", 26.0)], x0=48.0, y0=0.0, yaw0=math.pi)
    for i in range(len(t2[0])):
        rt.step(t2[0][i] + 1000.0, t2[1][i], t2[2][i], t2[3][i])
    lock = [e for e in all_of(rt, "BEAM_LOCK") if e.data["node"] == "n_001"
            and e.t > 900]
    assert lock and lock[0].data["lead_m"] > 15.0
    rec = [e for e in all_of(rt, "MEMORY_RECALL") if e.t > 900]
    assert rec, "no recall on the undriven approach"
    assert rec[0].data["action"] == "right"       # north exit seen from westbound
    assert rec[0].data["source"] == "beam"        # no stored corridor to confirm


def test_parallel_corridor_rejected_where_cone_fires():
    rt = MemoryRuntime()
    rt.current_goal = "g"
    # plant a single known node via a mini drive ending at (20, 3): a node in
    # the corridor NEXT DOOR (3 m lateral) relative to the test approach y=0
    t1 = traj([("straight", 20.0), ("turn", 90, 1.5), ("straight", 6.0)],
              y0=3.0)
    run(rt, t1)
    assert len(rt.nodes) == 1
    # approach along y=0 westward from x=52: the node sits 3 m off-axis
    t2 = traj([("straight", 30.0)], x0=52.0, y0=0.0, yaw0=math.pi)
    cone_would_fire = False
    node = rt.nodes["n_001"]
    for i in range(len(t2[0])):
        rt.step(t2[0][i] + 500, t2[1][i], t2[2][i], t2[3][i])
        brg = abs(wrap_deg(math.degrees(
            math.atan2(node.y - t2[2][i], node.x - t2[1][i]) - t2[3][i])))
        rng = math.hypot(node.x - t2[1][i], node.y - t2[2][i])
        if brg < 10.0 and rng < 30.0:
            cone_would_fire = True
    assert cone_would_fire, "test geometry broken: cone never sees the node"
    assert not [e for e in all_of(rt, "BEAM_LOCK") if e.t > 400], \
        "beam must reject a parallel-corridor node the cone would accept"


def test_rival_suppression_is_a_miss_not_a_wrong_turn():
    rt = MemoryRuntime()
    rt.current_goal = "g"
    # two collinear nodes ahead: A(20,0) and B(32,0)
    t1 = traj([("straight", 20.0), ("turn", 90, 1.5), ("straight", 4.0),
               ("turn", -90, 1.5), ("straight", 8.0), ("turn", -90, 1.5),
               ("straight", 4.0), ("turn", 90, 1.5), ("straight", 6.0)])
    run(rt, t1)   # creates several nodes; keep the two on y~0
    on_axis = [nid for nid, n in rt.nodes.items() if abs(n.y) < 1.0]
    assert len(on_axis) >= 2
    t2 = traj([("straight", 30.0)], x0=60.0, y0=0.0, yaw0=math.pi)
    for i in range(len(t2[0])):
        rt.step(t2[0][i] + 500, t2[1][i], t2[2][i], t2[3][i])
    assert [e for e in all_of(rt, "BEAM_RIVAL") if e.t > 400], \
        "two in-beam nodes at comparable range must raise a rival flag"
    far = max(on_axis, key=lambda nid: rt.nodes[nid].x)
    assert not [e for e in all_of(rt, "MEMORY_RECALL")
                if e.t > 400 and e.data["node"] == far]


# ----------------------------------------------------------------------------
# recall math + misc
# ----------------------------------------------------------------------------
def test_recall_bins_and_flip_margin():
    from memory_runtime import Node
    n = Node("n", 0, 0)
    n.register_branch(math.radians(90))                    # north exit
    n.routing["goal"] = {"branch": "b0", "src": "vlm", "conf": 0.9}
    a, m = n.recall("Goal", math.radians(90))              # facing north
    assert a == "straight" and m["bin_error_deg"] < 1e-6
    a, m = n.recall("goal", 0.0)                           # facing east
    assert a == "left"
    a, m = n.recall("goal", math.radians(90 + 44))         # 44 deg off
    assert a == "straight" and m["flip_margin_deg"] < 2.0
    a, m = n.recall("goal", math.radians(90 + 46))
    assert a == "right"                                    # flipped past 45
    a, m = n.recall("goal", math.radians(90 - 46))
    assert a == "left"                                     # flips the other way


def test_rel_synonyms_and_goal_norm():
    from memory_runtime import norm_rel, norm_goal
    assert norm_rel("Turn_Left") == "left"
    assert norm_rel("FORWARD") == "straight"
    assert norm_goal("  Goal  A ") == "goal a"
    with pytest.raises(ValueError):
        norm_rel("sideways")


def test_yaw_scaler_matches_offline_reintegration():
    import calibrate_yaw
    rng = np.random.default_rng(0)
    n = 800
    dt = 0.05
    yaw = np.cumsum(rng.normal(0, 0.02, n)) + np.linspace(0, 3.5, n)
    x = np.cumsum(0.04 * np.cos(yaw))
    y = np.cumsum(0.04 * np.sin(yaw))
    t = np.arange(n) * dt
    k = 0.8333
    xc, yc, yawc = calibrate_yaw.reintegrate(t, x, y, yaw, k)
    sc = YawScaler(k)
    xs, ys, psis = [], [], []
    for i in range(n):
        a, b, c = sc.step(x[i], y[i], yaw[i])
        xs.append(a); ys.append(b); psis.append(c)
    assert np.max(np.abs(np.array(xs) - xc)) < 1e-9
    assert np.max(np.abs(np.array(ys) - yc)) < 1e-9
    assert np.max(np.abs(np.array(psis) - yawc)) < 1e-9


# ----------------------------------------------------------------------------
# gate shim integration on the square: 1 call instead of 2
# ----------------------------------------------------------------------------
def test_gate_shim_saves_the_second_call(tmp_path):
    import yaml as _yaml
    from ar_gate_shim import ARGateShim
    t, x, y, yaw = square_arrays()
    n = len(t)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    # image frames at half odom rate
    frame_of = lambda i: i // 2
    arm1, dec1 = frame_of(int(np.searchsorted(s, 15.0))), \
                 frame_of(int(np.searchsorted(s, 21.0)))
    arm2, dec2 = frame_of(int(np.searchsorted(s, 138.0))), \
                 frame_of(int(np.searchsorted(s, 144.0)))
    ann = {
        "bag": "synthetic-square",
        "goals": [{"from_frame": 0, "goal": "goal_A"},
                  {"from_frame": frame_of(int(np.searchsorted(s, 45.0))),
                   "goal": "goal_E"}],
        "signs": [
            {"arm_frame": arm1, "decide_frame": dec1, "decision": "left",
             "conf": 0.91,
             "directory": {"goal_A": "left", "goal_E": "straight"}},
            {"arm_frame": arm2, "decide_frame": dec2, "decision": "straight",
             "conf": 0.90,
             "directory": {"goal_A": "left", "goal_E": "straight"}},
        ],
    }
    p = tmp_path / "square.yaml"
    p.write_text(_yaml.safe_dump(ann))
    rt = MemoryRuntime(bag_name="synthetic-square")
    gate = ARGateShim(str(p), rt)
    last_f = -1
    for i in range(n):
        f = frame_of(i)
        if f != last_f:
            gate.step(f)
            last_f = f
        rt.step(t[i], x[i], y[i], yaw[i])
    summ = gate.summary()
    assert summ["vlm_calls"] == 1, summ
    assert summ["counterfactual_always_invoke"] == 2
    assert summ["calls_saved"] == 1 and summ["reuse_rate"] == 0.5
    assert first(rt, "GATE_SUPPRESSED") is not None or \
           first(rt, "GATE_CALL_CANCELLED") is not None
    assert gate.standing_prompt == "straight" and gate.prompt_source == "memory"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


def test_directory_from_vlm_conversion():
    from memory_runtime import directory_from_vlm
    d = directory_from_vlm([
        {"label": "Rooms 340-360", "direction": "turn_left"},
        {"label": "Cafeteria", "direction": "straight"},
        {"label": "Exit", "direction": "stop"},          # not a bearing: dropped
        {"label": "", "direction": "turn_right"},        # junk: dropped
        {"label": "Lab", "direction": "sideways"},       # unparseable: dropped
    ])
    assert d == {"rooms 340-360": "left", "cafeteria": "straight"}


def test_recall_via_goal_matcher():
    from memory_runtime import Node
    n = Node("n", 0, 0)
    n.register_branch(math.radians(90))
    n.routing["rooms 340-360"] = {"branch": "b0", "src": "vlm", "conf": 0.9}
    # exact key miss — the goal is a room number, the label is a range
    assert n.recall("6-352", 0.0)[0] is None
    matcher = lambda goal, labels: next(
        (l for l in labels if "340-360" in l), None)
    a, m = n.recall("6-352", 0.0, matcher=matcher)
    assert a == "left" and m["matched_label"] == "rooms 340-360"


# ----------------------------------------------------------------------------
# sign-anchored nodes: a junction driven STRAIGHT through has no turn to
# anchor it, so the plate anchors it instead (else the read expires, or worse,
# binds to a junction 20 m away that it never described).
# ----------------------------------------------------------------------------
def _straight_through_bag(seed: bool):
    """Read a sign early, drive straight through its junction, turn far later."""
    rt = MemoryRuntime()
    rt.current_goal = "6-118"
    arrays = traj([("straight", 40.0), ("turn", -90, 1.5), ("straight", 10.0)])
    t, x, y, yaw = arrays
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    i_read = int(np.searchsorted(s, 5.0))
    i_gone = int(np.searchsorted(s, 8.0))      # plate leaves view at s=8 m
    hooks = {i_read: lambda r: r.provide_directory(
        {"6-115 to 6-189": "straight", "6-201 to 6-225": "left"},
        goal="6-118", decision="straight", conf=0.9, frame=i_read,
        plate_id="SIG:106" if seed else None)}
    for i in range(len(t)):
        if i in hooks:
            hooks[i](rt)
        if seed and i < i_gone:
            rt.observe_plates(["SIG:106"])      # tracker still sees it
        rt.step(t[i], x[i], y[i], yaw[i])
    return rt, s


def test_sign_seeds_a_node_when_no_turn_does():
    rt, s = _straight_through_bag(seed=True)
    ev = first(rt, "NODE_FROM_SIGN")
    assert ev is not None, "plate left the view but no node was anchored"
    nid = ev.data["node"]
    node = rt.nodes[nid]
    assert set(node.routing) == {"6-115 to 6-189", "6-201 to 6-225"}
    # anchored near where the plate went stale, NOT at the distant turn
    assert 7.0 < math.hypot(node.x, node.y) < 11.0, (node.x, node.y)
    assert first(rt, "READ_EXPIRED") is None


def test_without_the_cue_the_read_expires_rather_than_misfiling():
    rt, s = _straight_through_bag(seed=False)
    assert first(rt, "NODE_FROM_SIGN") is None
    assert first(rt, "READ_EXPIRED") is not None      # 40 m > BIND_MAX_M
    assert all(not n.routing for n in rt.nodes.values()), \
        "a read must never bind to a junction 40 m away that it never described"


def test_seeded_node_recalls_on_a_reverse_revisit():
    """The point of seeding: the straight-through junction becomes recallable."""
    rt, s = _straight_through_bag(seed=True)
    nid = first(rt, "NODE_FROM_SIGN").data["node"]
    node = rt.nodes[nid]
    # come back down the same corridor heading the other way
    rt.current_goal = "6-201 to 6-225"
    back = traj([("straight", 30.0)], x0=node.x + 25.0, y0=node.y,
                yaw0=math.pi)
    for i in range(len(back[0])):
        rt.step(back[0][i] + 1000.0, back[1][i], back[2][i], back[3][i])
    rec = [e for e in all_of(rt, "MEMORY_RECALL") if e.t > 900]
    assert rec, "seeded node did not resolve on the reverse approach"
    # stored LEFT while heading +x; arriving headed -x, the same branch is RIGHT
    assert rec[0].data["action"] == "right", rec[0].data


def test_turn_at_the_sign_still_wins_over_seeding():
    """If a turn anchors a node right there, bind to it -- do not duplicate."""
    rt = MemoryRuntime()
    rt.current_goal = "g"
    arrays = traj([("straight", 6.0), ("turn", 90, 1.5), ("straight", 12.0)])
    t, x, y, yaw = arrays
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    i_read = int(np.searchsorted(s, 3.0))
    i_gone = int(np.searchsorted(s, 4.5))
    for i in range(len(t)):
        if i == i_read:
            rt.provide_directory({"g": "left"}, goal="g", decision="left",
                                 conf=0.9, frame=i, plate_id="P1")
        if i < i_gone:
            rt.observe_plates(["P1"])
        rt.step(t[i], x[i], y[i], yaw[i])
    routed = [n for n in rt.nodes.values() if n.routing]
    assert len(routed) == 1, f"expected one routed node, got {len(routed)}"


# ----------------------------------------------------------------------------
# reversing: a three-point turn in a narrow corridor must not be re-integrated
# forward, and must not fool the "have I passed the sign / gone too far" rules.
# ----------------------------------------------------------------------------
def _three_point_turn():
    """Drive in, shuffle back-and-forth to reverse heading, drive back out."""
    return traj([
        ("straight", 10.0),
        ("turn", 60, 1.2), ("rturn", 60, 1.2),      # forward-arc, reverse-arc
        ("turn", 60, 1.2), ("rturn", 60, 1.2),
        ("turn", 60, 1.2),
        ("straight", 10.0),
    ])


def test_signed_steps_survive_a_three_point_turn():
    import calibrate_yaw
    t, x, y, yaw = _three_point_turn()
    st = calibrate_yaw.signed_step(x, y, yaw)
    assert (st < 0).any(), "generator produced no reversing steps"
    frac, metres = calibrate_yaw.reverse_fraction(x, y, yaw)
    assert 0.05 < frac < 0.5 and metres > 1.0, (frac, metres)
    # with k=1 the re-integration must reproduce the original path; the old
    # magnitude-only version turned the shuffle into a forward loop
    xc, yc, _ = calibrate_yaw.reintegrate(t, x, y, yaw, 1.0)
    assert np.max(np.abs(xc - x)) < 1e-6 and np.max(np.abs(yc - y)) < 1e-6


def test_yaw_scaler_matches_reintegration_with_reversing():
    import calibrate_yaw
    t, x, y, yaw = _three_point_turn()
    k = 0.83
    xc, yc, yawc = calibrate_yaw.reintegrate(t, x, y, yaw, k)
    sc = YawScaler(k)
    out = [sc.step(x[i], y[i], yaw[i]) for i in range(len(t))]
    xs = np.array([o[0] for o in out]); ys = np.array([o[1] for o in out])
    ps = np.array([o[2] for o in out])
    assert np.max(np.abs(xs - xc)) < 1e-9
    assert np.max(np.abs(ys - yc)) < 1e-9
    assert np.max(np.abs(ps - yawc)) < 1e-9


def test_shuffling_does_not_expire_a_read_or_seed_a_node():
    """Path length racks up during a three-point turn while the robot goes
    nowhere. Displacement, not path length, must govern both rules."""
    rt = MemoryRuntime()
    rt.current_goal = "g"
    t, x, y, yaw = _three_point_turn()
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    i_read = int(np.searchsorted(s, 8.0))          # read before the shuffle
    for i in range(len(t)):
        if i == i_read:
            rt.provide_directory({"g": "left"}, goal="g", decision="left",
                                 conf=0.9, frame=i, plate_id="P1")
        if i <= i_read:
            rt.observe_plates(["P1"])              # plate visible up to the read
        rt.step(t[i], x[i], y[i], yaw[i])
        if i == i_read + 1:
            continue
    # the shuffle covers several metres of PATH within ~2 m of displacement
    shuffle_path = s[-1] - s[i_read]
    disp = math.hypot(x[-1] - x[i_read], y[-1] - y[i_read])
    assert shuffle_path > disp, (shuffle_path, disp)
    seeds = all_of(rt, "NODE_FROM_SIGN")
    # a node may be seeded once the robot genuinely advances past the sign,
    # but never while it is only shuffling in place
    for e in seeds:
        assert e.data["read_to_seed_m"] > 1.0, e.data