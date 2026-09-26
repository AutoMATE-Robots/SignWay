#!/usr/bin/env python3
"""
odom_memory_probe.py
--------------------
Diagnostic for odom-only junction association on a single loop bag.

This is NOT a memory system. It is a measurement tool that answers:
  1. Does the odom trajectory close on a full loop, and by how much?
  2. Would a pose-proximity gate correctly fire at the revisited junction?
  3. At what radius, and does that radius also produce spurious fires?
  4. Does the stored sign directory reproject to the correct action on arrival?

Standalone: no ROS install, no Qwen, no OmniVLA. Needs numpy + matplotlib,
plus `rosbags` only if reading an MCAP directly.

    pip install rosbags numpy matplotlib

Usage:
    python odom_memory_probe.py --bag /path/to/rosbag2_dir
    python odom_memory_probe.py --bag bag_dir --dump-csv odom.csv   # cache once
    python odom_memory_probe.py --csv odom.csv                      # fast iterate
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# CONFIG — tune these, they are the whole experiment
# ----------------------------------------------------------------------------

TURN_RATE_THRESH = 0.25   # rad/s; above this counts as "turning"
MIN_TURN_DEG     = 45.0   # discard wobbles smaller than this
TURN_MERGE_S     = 1.0    # merge turn spans separated by less than this
SMOOTH_S         = 0.3    # yaw-rate smoothing window (seconds)

LOCKOUT_M        = 10.0   # must travel this far from a node before it can re-fire
RADII_SWEEP      = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# The sign directory read at the FIRST visit to A. Directions are robot-relative
# at the moment of reading. Edit to match what your VLM actually returned.
SIGN_DIRECTORY_REL = {
    "goal_A": "left",
    "goal_E": "straight",
}

# Which goal the robot is chasing on the SECOND pass, and what it should do.
SECOND_PASS_GOAL     = "goal_E"
SECOND_PASS_EXPECTED = "straight"

REL_OFFSETS_DEG = {"straight": 0.0, "left": 90.0, "right": -90.0, "back": 180.0}
ACTION_BINS_DEG = {"straight": 0.0, "left": 90.0, "right": -90.0, "back": 180.0}


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------

def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def _guess_typestore(bag_path):
    """Older rosbag2 does not embed type definitions. Infer the distro."""
    import sqlite3, glob, re
    from rosbags.typesys import Stores, get_typestore

    name = None
    # ROS 2 db3 keeps the distro in a `schema` table
    for db in glob.glob(str(Path(bag_path) / "*.db3")):
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            row = c.execute("SELECT ros_distro FROM schema").fetchone()
            c.close()
            if row and row[0]:
                name = str(row[0]).strip().upper()
                break
        except Exception:
            pass
    # metadata.yaml sometimes carries it too
    if name is None:
        try:
            txt = (Path(bag_path) / "metadata.yaml").read_text()
            m = re.search(r"ros_distro:\s*(\w+)", txt)
            if m:
                name = m.group(1).upper()
        except Exception:
            pass

    key = f"ROS2_{name}" if name else None
    if key and hasattr(Stores, key):
        print(f"[info] bag has no embedded typedefs; using typestore {key}")
        return get_typestore(getattr(Stores, key))
    print(f"[info] bag has no embedded typedefs; distro {name!r} unrecognised, "
          f"defaulting to ROS2_HUMBLE (Odometry is identical across distros)")
    return get_typestore(Stores.ROS2_HUMBLE)


def load_odom_bag(bag_path: str, topic: str | None = None):
    """Read nav_msgs/Odometry out of a ROS 2 bag (mcap or db3)."""
    from pathlib import Path as _P
    from rosbags.highlevel import AnyReader

    def _open():
        try:
            r = AnyReader([_P(bag_path)])
            r.open()
            return r
        except Exception as e:
            if "type definitions" not in str(e):
                raise
            r = AnyReader([_P(bag_path)],
                          default_typestore=_guess_typestore(bag_path))
            r.open()
            return r

    rows = []
    reader = _open()
    try:
        conns = [c for c in reader.connections
                 if c.msgtype == "nav_msgs/msg/Odometry"
                 and (topic is None or c.topic == topic)]
        if not conns:
            avail = sorted({(c.topic, c.msgtype) for c in reader.connections})
            raise SystemExit(
                "No nav_msgs/msg/Odometry connection found.\nTopics in bag:\n  "
                + "\n  ".join(f"{t}  [{m}]" for t, m in avail)
            )
        picked = sorted({c.topic for c in conns})
        if len(picked) > 1:
            print(f"[warn] multiple odom topics {picked}; using {picked[0]}")
            conns = [c for c in conns if c.topic == picked[0]]
        print(f"[info] reading odom from {conns[0].topic} ({conns[0].msgcount} msgs)")

        for conn, tns, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            rows.append((tns * 1e-9, p.x, p.y, quat_to_yaw(q.x, q.y, q.z, q.w)))
    finally:
        reader.close()

    if not rows:
        raise SystemExit("Odometry topic present but contained no messages.")
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def load_odom_csv(path: str):
    a = np.loadtxt(path, delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def dump_csv(path, t, x, y, yaw):
    np.savetxt(path, np.column_stack([t, x, y, yaw]),
               delimiter=",", header="t,x,y,yaw", comments="")
    print(f"[info] wrote {path}  ({len(t)} samples)")


# ----------------------------------------------------------------------------
# derived signals
# ----------------------------------------------------------------------------

def derive(t, x, y, yaw, curv_baseline_m=1.0):
    """
    Arc length, unwrapped yaw, smoothed yaw rate, and CURVATURE.

    Curvature (rad per metre) is the reliable turn signal. Yaw rate (rad per
    second) misses wide arcs: an Ackermann base rounding a corner on a 3 m
    radius at 0.4 m/s only reaches 0.13 rad/s -- below any threshold that also
    rejects corridor wobble. Curvature is speed-invariant, so the same corner
    reads 0.33 rad/m however slowly it is driven.

    Measured over a fixed distance baseline rather than per sample, which
    suppresses heading noise on straight sections.
    """
    d = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(d)])

    yaw_u = np.unwrap(yaw)
    dt = np.diff(t)
    dt[dt <= 0] = 1e-6
    rate = np.concatenate([[0.0], np.diff(yaw_u) / dt])

    hz = 1.0 / max(np.median(dt), 1e-6)
    w = max(1, int(round(SMOOTH_S * hz)))
    if w > 1:
        rate = np.convolve(rate, np.ones(w) / w, mode="same")

    j = np.clip(np.searchsorted(s, s + curv_baseline_m), 0, len(s) - 1)
    dsb = s[j] - s
    dpsi = yaw_u[j] - yaw_u
    kappa = np.where(dsb > 1e-3, np.abs(dpsi) / np.maximum(dsb, 1e-9), 0.0)
    # in-place rotation advances no distance -> defer to the rate channel
    kappa = np.where((dsb <= 1e-3) & (np.abs(rate) > 1e-3), np.inf, kappa)

    return s, yaw_u, rate, kappa, hz


# ----------------------------------------------------------------------------
# turn detection  ->  junction anchors
# ----------------------------------------------------------------------------

@dataclass
class TurnEvent:
    i0: int
    i1: int
    anchor_i: int          # pose at half the cumulative heading change
    t: float
    x: float
    y: float
    entry_yaw: float       # heading before the turn
    exit_yaw: float        # heading after the turn
    turn_deg: float        # signed


def detect_turns(t, x, y, yaw_u, rate, kappa, rate_thresh=None,
                 curv_thresh=0.10, min_turn_deg=None) -> list[TurnEvent]:
    rate_thresh = TURN_RATE_THRESH if rate_thresh is None else rate_thresh
    min_turn_deg = MIN_TURN_DEG if min_turn_deg is None else min_turn_deg

    active = (kappa > curv_thresh) | (np.abs(rate) > rate_thresh)
    if not active.any():
        return []

    edges = np.diff(active.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if active[0]:
        starts.insert(0, 0)
    if active[-1]:
        ends.append(len(active) - 1)

    merged = []
    for s0, s1 in zip(starts, ends):
        if merged and t[s0] - t[merged[-1][1]] < TURN_MERGE_S:
            merged[-1] = (merged[-1][0], s1)
        else:
            merged.append((s0, s1))

    events = []
    for i0, i1 in merged:
        total = math.degrees(yaw_u[i1] - yaw_u[i0])
        if abs(total) < min_turn_deg:
            continue
        half = yaw_u[i0] + 0.5 * (yaw_u[i1] - yaw_u[i0])
        ai = i0 + int(np.argmin(np.abs(yaw_u[i0:i1 + 1] - half)))
        events.append(TurnEvent(
            i0=i0, i1=i1, anchor_i=ai, t=float(t[ai]),
            x=float(x[ai]), y=float(y[ai]),
            entry_yaw=float(yaw_u[i0]), exit_yaw=float(yaw_u[i1]),
            turn_deg=float(total),
        ))
    return events


# ----------------------------------------------------------------------------
# the memory (deliberately minimal — pose + yaw + directory, nothing else)
# ----------------------------------------------------------------------------

@dataclass
class JunctionNode:
    node_id: str
    x: float
    y: float
    read_yaw: float                     # continuous-valued, radians, odom frame
    directory_abs: dict = field(default_factory=dict)   # label -> abs yaw (rad)
    created_t: float = 0.0

    @staticmethod
    def from_read(node_id, x, y, read_yaw, directory_rel, t=0.0):
        d = {lbl: float(wrap_pi(read_yaw + math.radians(REL_OFFSETS_DEG[rel])))
             for lbl, rel in directory_rel.items()}
        return JunctionNode(node_id, x, y, float(read_yaw), d, float(t))

    def recall(self, goal: str, yaw_now: float):
        """Reproject a stored absolute exit heading into an egocentric action."""
        if goal not in self.directory_abs:
            return None, None
        rel = math.degrees(float(wrap_pi(self.directory_abs[goal] - yaw_now)))
        action = min(ACTION_BINS_DEG,
                     key=lambda k: abs(float(wrap_deg(rel - ACTION_BINS_DEG[k]))))
        margin = abs(float(wrap_deg(rel - ACTION_BINS_DEG[action])))
        return action, {"rel_deg": rel, "bin_error_deg": margin}


# ----------------------------------------------------------------------------
# proximity gate — the thing actually under test
# ----------------------------------------------------------------------------

def gate_fires(node, t, x, y, s, radius, lockout_m, anchor_i=0):
    """
    Walk the trajectory. Report every entry into the radius of `node`,
    after the robot has first travelled `lockout_m` away from it.

    Returns list of dicts, one per firing episode.
    """
    d = np.hypot(x - node.x, y - node.y)
    inside = d < radius

    # suppress the departure episode: robot starts inside its own node
    armed = False
    fires, i = [], int(anchor_i)      # start from where the node was created
    n = len(d)
    while i < n:
        if not armed:
            if d[i] > lockout_m:
                armed = True
            i += 1
            continue
        if inside[i]:
            j = i
            while j < n and inside[j]:
                j += 1
            seg = slice(i, j)
            k = i + int(np.argmin(d[seg]))
            fires.append({
                "arm_window_m": float(s[j - 1] - s[i]),
                "t_enter": float(t[i]),
                "t_exit": float(t[j - 1]),
                "t_closest": float(t[k]),
                "closest_m": float(d[k]),
                "idx_closest": int(k),
                "dwell_s": float(t[j - 1] - t[i]),
                "path_m_since_start": float(s[k]),
            })
            i = j
            # re-disarm so we need to leave again before re-firing
            armed = False
        else:
            i += 1
    return fires


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag")
    ap.add_argument("--csv")
    ap.add_argument("--topic", default=None)
    ap.add_argument("--dump-csv", default=None)
    ap.add_argument("--first-turn-index", type=int, default=0,
                    help="which detected turn is junction A (0 = first)")
    ap.add_argument("--turn-curv-thresh", type=float, default=0.10,
                    help="rad/m; 0.10 == turning radius under 10 m")
    ap.add_argument("--turn-rate-thresh", type=float, default=TURN_RATE_THRESH)
    ap.add_argument("--min-turn-deg", type=float, default=MIN_TURN_DEG)
    ap.add_argument("--node-at-s", type=float, default=None,
                    help="place node A manually at this arc length (m), "
                         "bypassing turn detection")
    ap.add_argument("--plot", default="odom_probe.png")
    ap.add_argument("--json", default="odom_probe.json")
    args = ap.parse_args()

    if args.csv:
        t, x, y, yaw = load_odom_csv(args.csv)
    elif args.bag:
        t, x, y, yaw = load_odom_bag(args.bag, args.topic)
        if args.dump_csv:
            dump_csv(args.dump_csv, t, x, y, yaw)
    else:
        raise SystemExit("need --bag or --csv")

    s, yaw_u, rate, kappa, hz = derive(t, x, y, yaw)
    turns = detect_turns(t, x, y, yaw_u, rate, kappa,
                         rate_thresh=args.turn_rate_thresh,
                         curv_thresh=args.turn_curv_thresh,
                         min_turn_deg=args.min_turn_deg)

    out = {}
    W = 78

    print("=" * W)
    print("ODOM-ONLY JUNCTION ASSOCIATION PROBE")
    print("=" * W)
    print(f"samples          {len(t)}")
    print(f"duration         {t[-1] - t[0]:.1f} s   (~{hz:.1f} Hz)")
    print(f"path length      {s[-1]:.2f} m")
    print(f"net yaw change   {math.degrees(yaw_u[-1] - yaw_u[0]):+.2f} deg")

    # ---- loop closure ------------------------------------------------------
    gap = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    yaw_resid = float(wrap_deg(math.degrees(yaw_u[-1] - yaw_u[0])))
    print("\n" + "-" * W)
    print("RAW START-vs-END  (meaningful ONLY if the bag starts and stops")
    print("                   at the same physical spot — usually it does not)")
    print("-" * W)
    print(f"position gap     {gap:.2f} m   ({100.0 * gap / max(s[-1], 1e-6):.2f}% of path)")
    print(f"yaw residual     {yaw_resid:+.2f} deg   (0 if the loop truly closed)")
    print(f"                 {'OK — under half a 90deg bin' if abs(yaw_resid) < 45 else 'PROBLEM — action bins will flip'}")
    out["loop"] = {"path_m": float(s[-1]), "pos_gap_m": gap,
                   "yaw_resid_deg": yaw_resid,
                   "drift_pct": 100.0 * gap / max(s[-1], 1e-6)}

    # ---- turns -------------------------------------------------------------
    print("\n" + "-" * W)
    print(f"DETECTED TURNS  (curvature>{args.turn_curv_thresh} rad/m OR "
          f"rate>{args.turn_rate_thresh} rad/s, |turn|>{args.min_turn_deg} deg)")
    print("-" * W)
    print(f"{'#':>2} {'t(s)':>8} {'s(m)':>8} {'x':>8} {'y':>8} {'turn':>9} {'entry->exit hdg':>20}")
    for i, e in enumerate(turns):
        print(f"{i:>2} {e.t - t[0]:>8.1f} {s[e.anchor_i]:>8.1f} {e.x:>8.2f} {e.y:>8.2f} "
              f"{e.turn_deg:>+8.1f}d "
              f"{math.degrees(wrap_pi(e.entry_yaw)):>8.1f} ->{math.degrees(wrap_pi(e.exit_yaw)):>8.1f}")
    out["turns"] = [asdict(e) for e in turns]

    if not turns and args.node_at_s is None:
        raise SystemExit("\nNo turns detected — lower --turn-curv-thresh and rerun.")

    # ---- build the node from the first visit -------------------------------
    if args.node_at_s is not None:
        ai = int(np.argmin(np.abs(s - args.node_at_s)))
        a = TurnEvent(i0=ai, i1=ai, anchor_i=ai, t=float(t[ai]),
                      x=float(x[ai]), y=float(y[ai]),
                      entry_yaw=float(yaw_u[ai]), exit_yaw=float(yaw_u[ai]),
                      turn_deg=0.0)
        print(f"\n[info] node A placed MANUALLY at s={s[ai]:.1f} m")
    else:
        a = turns[args.first_turn_index]
    node = JunctionNode.from_read(
        "A", a.x, a.y, read_yaw=a.entry_yaw,
        directory_rel=SIGN_DIRECTORY_REL, t=a.t,
    )
    print("\n" + "-" * W)
    print("NODE A  (%s)" % ("manual --node-at-s" if args.node_at_s is not None
                             else "from turn #%d" % args.first_turn_index))
    print("-" * W)
    print(f"anchor           ({node.x:.2f}, {node.y:.2f})")
    print(f"read heading     {math.degrees(wrap_pi(node.read_yaw)):.1f} deg")
    for lbl, ab in node.directory_abs.items():
        print(f"  {lbl:<10} rel={SIGN_DIRECTORY_REL[lbl]:<9} -> abs {math.degrees(wrap_pi(ab)):+7.1f} deg")

    # ---- how close does the return leg actually get? -----------------------
    d_all = np.hypot(x - node.x, y - node.y)
    # departure must be measured FROM THE ANCHOR, not from the start of the bag
    tail = d_all[a.anchor_i:]
    left = np.where(tail > LOCKOUT_M)[0]
    if len(left) == 0:
        raise SystemExit(f"\nRobot never travelled {LOCKOUT_M} m away from node A "
                         f"after creating it — no revisit in this bag.")
    after = a.anchor_i + int(left[0])
    k = after + int(np.argmin(d_all[after:]))
    closest = float(d_all[k])

    print("\n" + "-" * W)
    print("RETURN LEG — closest approach to node A")
    print("-" * W)
    print(f"closest distance {closest:.2f} m   at t+{t[k] - t[0]:.1f}s, s={s[k]:.1f}m")
    print(f"heading there    {math.degrees(wrap_pi(yaw_u[k])):.1f} deg")
    print(f"heading vs read  {float(wrap_deg(math.degrees(yaw_u[k] - node.read_yaw))):+.1f} deg")
    print("\n  >>> THIS NUMBER IS THE RESULT. Any radius gate must exceed")
    print(f"  >>> {closest:.2f} m to fire at all, and must stay below your")
    print("  >>> junction spacing to avoid firing on the wrong junction.")
    out["return"] = {"closest_m": closest, "t_rel": float(t[k] - t[0]),
                     "heading_deg": math.degrees(wrap_pi(yaw_u[k])),
                     "heading_vs_read_deg": float(wrap_deg(math.degrees(yaw_u[k] - node.read_yaw)))}

    # ---- radius sweep ------------------------------------------------------
    print("\n" + "-" * W)
    print("RADIUS SWEEP  (lockout = %.1f m)" % LOCKOUT_M)
    print("-" * W)
    # how far is A from the nearest OTHER detected junction? that bounds r.
    others = [e for i, e in enumerate(turns) if i != args.first_turn_index]
    if others:
        nn = min(math.hypot(e.x - node.x, e.y - node.y) for e in others)
        print(f"nearest other detected junction is {nn:.1f} m from A")
        print(f"  -> radius must stay well under {nn:.1f} m or A will alias onto it")
        out["nearest_other_junction_m"] = nn
    else:
        nn = float("inf")

    print()
    print(f"{'r(m)':>6} {'fires':>6} {'closest':>9} {'arm window':>11}  verdict")
    sweep = {}
    for r in RADII_SWEEP:
        f = gate_fires(node, t, x, y, s, r, LOCKOUT_M, anchor_i=a.anchor_i)
        if r >= LOCKOUT_M:
            note = "  [r >= lockout: degenerate, ignore]"
        elif r >= 0.5 * nn:
            note = "  [too close to junction spacing]"
        else:
            note = ""
        if not f:
            print(f"{r:>6.1f} {0:>6} {'-':>9} {'-':>11}  MISS — never fires{note}")
            sweep[r] = f
            continue
        c, win = f[0]["closest_m"], f[0]["arm_window_m"]
        if len(f) > 1:
            verdict = f"AMBIGUOUS — {len(f)} fires"
        elif win > 5.0:
            verdict = "fires, but window too wide to time an action"
        else:
            verdict = "USABLE"
        print(f"{r:>6.1f} {len(f):>6} {c:>8.2f}m {win:>10.1f}m  {verdict}{note}")
        sweep[r] = f
    out["sweep"] = {str(r): v for r, v in sweep.items()}

    # ---- reprojection at the fired pose ------------------------------------
    print("\n" + "-" * W)
    print("RECALL TEST  (goal = %s, expected = %s)" % (SECOND_PASS_GOAL, SECOND_PASS_EXPECTED))
    print("-" * W)
    action, meta = node.recall(SECOND_PASS_GOAL, float(yaw_u[k]))
    if action is None:
        print(f"  MISS — '{SECOND_PASS_GOAL}' not in stored directory.")
        print("  The first-pass VLM call did not capture the whole plate.")
        print("  This is a PROMPT problem, not a memory problem.")
    else:
        ok = action == SECOND_PASS_EXPECTED
        print(f"  recalled action  {action}   {'CORRECT' if ok else 'WRONG'}")
        print(f"  relative bearing {meta['rel_deg']:+.1f} deg")
        print(f"  bin error        {meta['bin_error_deg']:.1f} deg "
              f"(flips at 45 deg — margin {45.0 - meta['bin_error_deg']:.1f} deg)")
    out["recall"] = {"goal": SECOND_PASS_GOAL, "expected": SECOND_PASS_EXPECTED,
                     "action": action, **(meta or {})}

    # ---- the detection gap -------------------------------------------------
    print("\n" + "-" * W)
    print("STRAIGHT-THROUGH CHECK")
    print("-" * W)
    turn_near = [i for i, e in enumerate(turns)
                 if abs(e.t - t[k]) < 5.0 and e.anchor_i > after]
    if turn_near:
        print(f"  A turn was detected near the revisit (turn #{turn_near[0]}).")
        print("  Turn-triggered junction detection would have caught this.")
    else:
        print("  NO turn detected at the revisit — the robot drove straight through.")
        print("  A turn-onset junction detector sees NOTHING here.")
        print("  Association MUST be driven by proximity polling, not by turn events.")
    out["straight_through"] = not bool(turn_near)

    # ---- plot --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, y, lw=1.2, color="#444", label="odom trajectory")
        ax.scatter([x[0]], [y[0]], s=90, marker="o", c="green", zorder=5, label="start")
        ax.scatter([x[-1]], [y[-1]], s=90, marker="X", c="red", zorder=5, label="end")
        for i, e in enumerate(turns):
            ax.scatter([e.x], [e.y], s=45, c="#1f77b4", zorder=4)
            ax.annotate(f"T{i}", (e.x, e.y), textcoords="offset points",
                        xytext=(6, 6), fontsize=9, color="#1f77b4")
        ax.scatter([node.x], [node.y], s=200, facecolors="none",
                   edgecolors="darkorange", lw=2.2, zorder=6, label="node A")
        ax.scatter([x[k]], [y[k]], s=110, marker="*", c="darkorange",
                   zorder=6, label=f"closest revisit ({closest:.2f} m)")
        for r in (2.0, 5.0):
            ax.add_patch(plt.Circle((node.x, node.y), r, fill=False,
                                    ls="--", lw=0.9, color="darkorange", alpha=0.5))
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"odom loop — closure gap {gap:.2f} m, yaw resid {yaw_resid:+.1f}°")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"\n[info] wrote {args.plot}")
    except Exception as ex:
        print(f"[warn] plot failed: {ex}")

    Path(args.json).write_text(json.dumps(out, indent=2, default=_jsonable))
    print(f"[info] wrote {args.json}")
    print("=" * W)


if __name__ == "__main__":
    main()
