#!/usr/bin/env python3
"""
node_approach_probe.py
----------------------
Arrival at a known junction along a corridor that has NEVER been driven.

Edge matching cannot help: there is no stored polyline. But while odom is
still consistent, the robot's pose and the node's pose are both known, so the
node can be recognised WITHOUT recognising the corridor.

Naive version is junction proximity (fire inside r metres), which is too late
for the reasoning gate. Better: project forward along the current heading and
ask whether a known node lies in a narrow cone ahead. That fires as soon as
the robot is pointed down the corridor at the node -- restoring most of the
lead distance that edge matching would have given, with no stored corridor.

    approaching(B)  ==  |bearing to B| < cone
                        AND range < max_range
                        AND range decreasing
                        AND no rival node at comparable range

Once B is reached and confirmed, the new corridor is written into the graph,
so the SECOND arrival from that direction gets full edge matching.
"""
from __future__ import annotations

import argparse
import math

import numpy as np


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def approach_signal(x, y, yaw, nodes, target, cone_deg, max_range,
                    decreasing_win=10, lateral_m=None):
    """
    Per-sample: is `target` (index into nodes) the node being approached?
    Conservative -- a rival node at comparable range suppresses the signal.
    """
    tx, ty = nodes[target]
    rng = np.hypot(x - tx, y - ty)
    brg = np.abs(wrap_deg(np.degrees(np.arctan2(ty - y, tx - x) - yaw)))

    if lateral_m is not None:
        # LATERAL OFFSET gate, not a bearing cone. Bearing shrinks with range,
        # so a node in the next corridor over looks straight ahead from far
        # enough away and a fixed cone cannot reject it. Perpendicular offset
        # from the robot's forward axis is range-invariant and can.
        lat = rng * np.abs(np.sin(np.radians(brg)))
        fwd = rng * np.cos(np.radians(brg))
        ok = (lat < lateral_m) & (fwd > 0) & (rng < max_range)
    else:
        ok = (brg < cone_deg) & (rng < max_range)

    # range must be closing, measured over a short window
    dec = np.zeros_like(ok)
    for i in range(decreasing_win, len(rng)):
        dec[i] = rng[i] < rng[i - decreasing_win]
    ok &= dec

    # rival check: any other node also in the cone at a comparable range
    for k, (nx_, ny_) in enumerate(nodes):
        if k == target:
            continue
        r2 = np.hypot(x - nx_, y - ny_)
        b2 = np.abs(wrap_deg(np.degrees(np.arctan2(ny_ - y, nx_ - x) - yaw)))
        rival = (b2 < cone_deg) & (r2 < 1.5 * rng)
        ok &= ~rival
    return ok, rng, brg


def longest_run(mask):
    best = cur = bi = ci = 0
    for k, v in enumerate(mask):
        if v:
            if cur == 0:
                ci = k
            cur += 1
            if cur > best:
                best, bi = cur, ci
        else:
            cur = 0
    return bi, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target-turn", type=int, default=0,
                    help="which detected turn is the known node")
    ap.add_argument("--cone-deg", type=float, default=20.0)
    ap.add_argument("--lateral-m", type=float, default=1.5,
                    help="corridor half-width gate (preferred); "
                         "set 0 to fall back to a bearing cone")
    ap.add_argument("--max-range", type=float, default=30.0)
    ap.add_argument("--proximity-radius", type=float, default=2.0)
    ap.add_argument("--lockout", type=float, default=10.0)
    ap.add_argument("--turn-curv-thresh", type=float, default=0.10)
    ap.add_argument("--plot", default="node_approach.png")
    args = ap.parse_args()

    import odom_memory_probe as P
    a = np.loadtxt(args.csv, delimiter=",", skiprows=1)
    t, x, y, yaw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    s, yaw_u, rate, kappa, hz = P.derive(t, x, y, yaw)
    turns = P.detect_turns(t, x, y, yaw_u, rate, kappa,
                           curv_thresh=args.turn_curv_thresh)
    nodes = [(e.x, e.y) for e in turns]
    tgt = args.target_turn
    anchor_i = turns[tgt].anchor_i

    W = 76
    print("=" * W)
    print("APPROACH TO A KNOWN NODE VIA AN UNDRIVEN CORRIDOR")
    print("=" * W)
    for k, e in enumerate(turns):
        mark = "   <-- known node" if k == tgt else ""
        print(f"  turn {k}: ({e.x:6.1f}, {e.y:6.1f})  turn {e.turn_deg:+6.1f} deg{mark}")

    # only consider the return, after the robot has left the node
    tail = np.hypot(x[anchor_i:] - nodes[tgt][0], y[anchor_i:] - nodes[tgt][1])
    left = np.where(tail > args.lockout)[0]
    if len(left) == 0:
        raise SystemExit("Robot never left the node — no revisit.")
    after = anchor_i + int(left[0])

    lat_gate = args.lateral_m if args.lateral_m > 0 else None
    ok, rng, brg = approach_signal(x, y, yaw, nodes, tgt, args.cone_deg,
                                   args.max_range, lateral_m=lat_gate)
    ok[:after] = False

    bi, blen = longest_run(ok)
    prox = (rng < args.proximity_radius)
    prox[:after] = False

    print("\n" + "-" * W)
    print("LEAD DISTANCE")
    print("-" * W)
    if blen > 5:
        print(f"cone approach fires   {rng[bi]:6.1f} m out  "
              f"(sustained {s[bi+blen-1]-s[bi]:.1f} m of travel)")
    else:
        print("cone approach         NEVER FIRES")
    if prox.any():
        k = int(np.argmax(prox))
        print(f"proximity gate fires  {rng[k]:6.1f} m out  "
              f"(radius {args.proximity_radius} m)")
        if blen > 5:
            print(f"                      -> cone is {rng[bi]/max(rng[k],1e-6):.0f}x earlier")
    else:
        print(f"proximity gate        NEVER FIRES (closest {rng[after:].min():.2f} m)")
    print(f"\nclosest approach on the return: {rng[after:].min():.2f} m")

    print("\n" + "-" * W)
    print("LATERAL GATE SWEEP")
    print("-" * W)
    print(f"{'lat(m)':>6} {'fires at':>10} {'sustained':>11}  note")
    for c in (0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        o, r2, _ = approach_signal(x, y, yaw, nodes, tgt, args.cone_deg,
                                   args.max_range, lateral_m=c)
        o[:after] = False
        b2, l2 = longest_run(o)
        if l2 > 5:
            note = "wider than a corridor — parallel-corridor risk" if c >= 3.0 else ""
            print(f"{c:>6.2f} {r2[b2]:>9.1f} m {s[b2+l2-1]-s[b2]:>10.1f} m  {note}")
        else:
            print(f"{c:>6.2f} {'-':>9} {'-':>10}   too tight")

    # ---------------------------------------------------------------- plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lat_all = rng * np.abs(np.sin(np.radians(brg)))
        fwd_all = rng * np.cos(np.radians(brg))
        tx, ty = nodes[tgt]
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))

        # --- 1. where the gate is active, in the plane
        ax[0].plot(x, y, lw=0.9, color="#ccc", label="full run")
        ax[0].plot(x[after:], y[after:], lw=1.6, color="#444", label="return leg")
        if blen > 5:
            ax[0].plot(x[bi:bi+blen], y[bi:bi+blen], lw=3.4, color="#2ca02c",
                       alpha=.85, label="gate active")
            ax[0].scatter([x[bi]], [y[bi]], s=150, marker="*", c="green",
                          zorder=6, label=f"fires ({rng[bi]:.0f} m out)")
        ax[0].scatter([tx], [ty], s=190, facecolors="none", edgecolors="darkorange",
                      lw=2.5, zorder=6, label="known node")
        for r_ in (args.proximity_radius,):
            ax[0].add_patch(plt.Circle((tx, ty), r_, fill=False, ls=":",
                                       color="darkorange", alpha=.8))
        for k, (nx_, ny_) in enumerate(nodes):
            if k != tgt:
                ax[0].scatter([nx_], [ny_], s=45, c="#1f77b4", zorder=4)
        ax[0].set_aspect("equal"); ax[0].grid(alpha=.3)
        ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
        ax[0].legend(fontsize=8, loc="best")
        ax[0].set_title("where the gate opens")

        # --- 2. THE diagnostic: lateral offset vs range
        m = (fwd_all > 0) & (rng < args.max_range)
        m[:after] = False
        ax[1].plot(rng[m], lat_all[m], lw=1.7, color="#d62728")
        if lat_gate:
            ax[1].axhline(lat_gate, ls="--", c="gray",
                          label=f"lateral gate {lat_gate} m")
            ax[1].axhspan(0, lat_gate, color="#2ca02c", alpha=.08)
        ax[1].axvline(args.proximity_radius, ls=":", c="darkorange",
                      label="proximity gate")
        if blen > 5:
            ax[1].axvline(rng[bi], ls="-", c="green", alpha=.6,
                          label=f"fires at {rng[bi]:.0f} m")
        ax[1].invert_xaxis()
        ax[1].set_xlabel("range to node (m)   [robot approaches ->]")
        ax[1].set_ylabel("lateral offset from forward axis (m)")
        ax[1].set_ylim(0, max(6, float(np.nanmax(lat_all[m])) * 1.05) if m.any() else 6)
        ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
        ax[1].set_title("inside the band = pointed at the node")

        # --- 3. how much lead each gate width buys
        gs, leads = [], []
        for c in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
            o, r2, _ = approach_signal(x, y, yaw, nodes, tgt, args.cone_deg,
                                       args.max_range, lateral_m=c)
            o[:after] = False
            b2, l2 = longest_run(o)
            gs.append(c); leads.append(r2[b2] if l2 > 5 else 0.0)
        ax[2].plot(gs, leads, "o-", color="#1f77b4")
        if lat_gate:
            ax[2].axvline(lat_gate, ls="--", c="gray", label="current setting")
        ax[2].axvspan(3.0, 5.2, color="#d62728", alpha=.10)
        ax[2].text(4.0, max(leads) * .12 if max(leads) else 1,
                   "wider than\na corridor", ha="center", fontsize=8, color="#a00")
        ax[2].axhline(args.proximity_radius, ls=":", c="darkorange",
                      label="proximity gate lead")
        ax[2].set_xlabel("lateral gate width (m)"); ax[2].set_ylabel("lead distance (m)")
        ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
        ax[2].set_title("lead is flat — tighten it for free")

        fig.tight_layout(); fig.savefig(args.plot, dpi=140)
        print(f"\nwrote {args.plot}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")
    print("=" * W)


if __name__ == "__main__":
    main()
