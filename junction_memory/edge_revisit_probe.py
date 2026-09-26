#!/usr/bin/env python3
"""
edge_revisit_probe.py
---------------------
Measures how early a DIRECTED EDGE traversal can be recognized, versus how
early a junction-proximity gate would fire.

Why this matters. The adaptive-reasoning gate fires when evidence about the
sign becomes sufficient -- typically several metres before the junction. A
junction-proximity gate with a 2 m radius resolves memory AFTER that, so the
VLM call has already been dispatched and the memory saves nothing. Recognising
the directed edge on entry resolves memory tens of metres earlier, before the
gate can arm.

DIRECTED, not undirected: the stored routing table is allocentric and the
egocentric action depends on approach heading, so traversing an edge backwards
must not count as the same thing. Heading agreement is required.

    python edge_revisit_probe.py --csv odom_calibrated.csv
"""
from __future__ import annotations

import argparse
import math

import numpy as np

HEADING_TOL_DEG = 50.0   # rejects the reverse traversal (180 deg away)
MATCH_RADIUS_M = 2.5     # lateral tolerance for "same edge"


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def load(csv):
    a = np.loadtxt(csv, delimiter=",", skiprows=1)
    t, x, y, yaw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return t, x, y, np.unwrap(yaw), s


def segments(anchors, n):
    """Split the run at turn anchors -> one directed traversal per segment."""
    bounds = [0] + list(anchors) + [n - 1]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def match_profile(qx, qy, qpsi, rx, ry, rpsi):
    """
    For each query sample, distance to the nearest reference sample whose
    heading agrees. Heading disagreement -> inf, so reverse traversals of the
    same corridor never match.
    """
    out = np.full(len(qx), np.inf)
    for i in range(len(qx)):
        dpsi = np.abs(wrap_deg(np.degrees(rpsi - qpsi[i])))
        ok = dpsi < HEADING_TOL_DEG
        if not ok.any():
            continue
        out[i] = np.min(np.hypot(rx[ok] - qx[i], ry[ok] - qy[i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--turn-curv-thresh", type=float, default=0.10)
    ap.add_argument("--radius", type=float, default=MATCH_RADIUS_M)
    ap.add_argument("--junction-radius", type=float, default=2.0,
                    help="radius a junction-proximity gate would use")
    ap.add_argument("--plot", default="edge_revisit.png")
    args = ap.parse_args()

    import odom_memory_probe as P
    t, x, y, yaw, s = load(args.csv)
    _, yaw_u, rate, kappa, hz = P.derive(t, x, y, yaw)
    turns = P.detect_turns(t, x, y, yaw_u, rate, kappa,
                           curv_thresh=args.turn_curv_thresh)
    anchors = [e.anchor_i for e in turns]
    segs = segments(anchors, len(t))

    W = 76
    print("=" * W)
    print("DIRECTED EDGE REVISIT")
    print("=" * W)
    print(f"{'seg':>4} {'s range (m)':>16} {'len':>7} {'mean heading':>14}")
    for k, (i0, i1) in enumerate(segs):
        h = math.degrees(np.arctan2(np.sin(yaw[i0:i1]).mean(),
                                    np.cos(yaw[i0:i1]).mean()))
        print(f"{k:>4} {s[i0]:>7.1f} - {s[i1]:>6.1f} {s[i1]-s[i0]:>7.1f} {h:>13.1f}")

    # pairwise directed match
    print(f"\nfraction of segment ROW matched by segment COL "
          f"(within {args.radius} m, heading within {HEADING_TOL_DEG:.0f} deg)")
    n = len(segs)
    hdr = "     " + "".join(f"{j:>8}" for j in range(n))
    print(hdr)
    M = np.zeros((n, n))
    D = {}
    for i, (a0, a1) in enumerate(segs):
        row = f"{i:>4} "
        for j, (b0, b1) in enumerate(segs):
            if i == j:
                row += f"{'-':>8}"
                continue
            d = match_profile(x[a0:a1], y[a0:a1], yaw[a0:a1],
                              x[b0:b1], y[b0:b1], yaw[b0:b1])
            D[(i, j)] = d
            f = float((d < args.radius).mean())
            M[i, j] = f
            row += f"{f:>8.2f}"
        print(row)

    # A retrace must come AFTER the segment it retraces. Without this the
    # argmax can pick (earlier, later) and then take the "junction" from the
    # end of the LAST segment -- which is the end of the bag, not a junction.
    Mo = np.where(np.tril(np.ones_like(M), -1) > 0, M, -1.0)
    i, j = np.unravel_index(np.argmax(Mo), Mo.shape)
    if Mo[i, j] < 0.2:
        raise SystemExit("No directed edge revisit found in this bag.")
    print(f"\nstrongest revisit: segment {i} (later) retraces segment {j} (earlier), "
          f"{M[i, j]*100:.0f}% of samples")

    # how early is it recognised, relative to arriving at the junction?
    a0, a1 = segs[i]
    b0, b1 = segs[j]
    d = match_profile(x[a0:a1], y[a0:a1], yaw[a0:a1],
                      x[b0:b1], y[b0:b1], yaw[b0:b1])
    ss = s[a0:a1]

    # the reference segment ends at a turn anchor -- that is the node approached
    jx, jy = x[b1], y[b1]
    print(f"node approached: end of segment {j} at ({jx:.1f}, {jy:.1f})")
    dist_to_j = np.hypot(x[a0:a1] - jx, y[a0:a1] - jy)

    inside = d < args.radius
    if not inside.any():
        raise SystemExit("Segment matched in aggregate but never within radius.")
    first = int(np.argmax(inside))
    lead_edge = float(dist_to_j[first])

    jn = dist_to_j < args.junction_radius
    fires = bool(jn.any())
    lead_junction = float(dist_to_j[int(np.argmax(jn))]) if fires else float("nan")

    print("\n" + "-" * W)
    print("LEAD DISTANCE  (how far before the junction memory resolves)")
    print("-" * W)
    print(f"directed-edge match      {lead_edge:6.1f} m before the junction")
    if fires:
        print(f"junction proximity gate  {lead_junction:6.1f} m "
              f"(radius {args.junction_radius} m)")
        print(f"                         -> edge match is "
              f"{lead_edge/max(lead_junction,1e-6):.0f}x earlier")
    else:
        print(f"junction proximity gate     NEVER FIRES — the retrace stays "
              f"outside {args.junction_radius} m")
        print("                         -> junction-only memory would miss this "
              "revisit entirely")
    print(f"\nmedian lateral error along the retrace: {np.median(d[inside]):.2f} m")

    # ---- radius sweep: lead distance vs false-match exposure ----------------
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

    print("\n" + "-" * W)
    print("MATCH RADIUS SWEEP")
    print("-" * W)
    print(f"{'radius':>7} {'first':>8} {'sustained':>10} {'plateau':>9} "
          f"{'matched':>8} {'worst FP':>9}  verdict")
    print(f"{'(m)':>7} {'(m out)':>8} {'(m out)':>10} {'(m)':>9} "
          f"{'(%)':>8} {'(%)':>9}")
    for r in (0.75, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0):
        inside_r = d < r
        if not inside_r.any():
            print(f"{r:>7.2f} {'-':>8} {'-':>10} {'-':>9} {0:>8.0f} "
                  f"{'-':>9}  NO MATCH")
            continue
        first_r = int(np.argmax(inside_r))
        b_i, b_len = longest_run(inside_r)
        plateau = float(abs(ss[min(b_i + b_len - 1, len(ss) - 1)] - ss[b_i]))
        # worst false positive: best-scoring pair that is NOT the true revisit
        # exclude BOTH orderings of the true revisit -- (j,i) is the same
        # event seen backwards, not a false positive
        fp = max((float((dd < r).mean()) for (pi, pj), dd in D.items()
                  if {pi, pj} != {i, j}), default=0.0)
        matched = 100 * float(inside_r.mean())
        if fp > 0.5:
            verdict = "UNSAFE — aliases onto another segment"
        elif fp > 0.25:
            verdict = "marginal"
        else:
            verdict = "safe"
        print(f"{r:>7.2f} {dist_to_j[first_r]:>8.1f} {dist_to_j[b_i]:>10.1f} "
              f"{plateau:>9.1f} {matched:>8.0f} {100*fp:>9.0f}  {verdict}")
    print("\n'first' = first sample under threshold (can be a lucky frame).")
    print("'sustained' = start of the longest unbroken run — the number to trust.")
    print("'worst FP' = best score from a pair that is NOT the true revisit;")
    print("             it must stay well below 'matched' for the gate to be safe.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
        ax[0].plot(x, y, lw=0.8, color="#bbb")
        ax[0].plot(x[b0:b1], y[b0:b1], lw=2.5, color="#1f77b4",
                   label=f"seg {j} — first pass (turns at A)")
        ax[0].plot(x[a0:a1], y[a0:a1], lw=2.5, ls="--", color="#d62728",
                   label=f"seg {i} — retrace (straight through A)")
        ax[0].scatter([jx], [jy], s=160, facecolors="none", edgecolors="darkorange",
                      lw=2.5, label="junction")
        ax[0].scatter([x[a0 + first]], [y[a0 + first]], s=110, marker="*",
                      c="green", zorder=6, label=f"recognised ({lead_edge:.0f} m out)")
        ax[0].set_aspect("equal"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
        ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")

        ax[1].plot(dist_to_j, np.minimum(d, 10), lw=1.6, color="#d62728")
        ax[1].axhline(args.radius, ls="--", c="gray", label=f"match radius {args.radius} m")
        ax[1].axvline(args.junction_radius, ls=":", c="darkorange",
                      label=f"junction gate fires here")
        ax[1].invert_xaxis()
        ax[1].set_xlabel("distance to junction (m)  [robot moves left to right]")
        ax[1].set_ylabel("lateral error to stored edge (m)")
        ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
        ax[1].set_title("edge recognised far earlier than junction proximity")
        fig.tight_layout(); fig.savefig(args.plot, dpi=140)
        print(f"\nwrote {args.plot}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")
    print("=" * W)


if __name__ == "__main__":
    main()
