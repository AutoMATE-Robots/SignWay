#!/usr/bin/env python3
"""make_map.py — the top-down view: what the robot did and why.

The camera video shows what the robot SAW. This shows what it DECIDED: where it drove, where it
spotted a sign, where the goalpost jumped to, and the line it then took to get there. Side by
side, the two tell the whole story.

    # one overview image of the whole run
    python tools/make_map.py --log isaac_out/run.jsonl --out isaac_out/map.png

    # one map per frame, for stitching next to the camera video
    python tools/make_map.py --log isaac_out/run.jsonl --frames isaac_out --animate

Run OUTSIDE the Isaac container (needs matplotlib).
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load(path):
    meta, recs = {}, []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("type") == "meta":
            meta = d
        elif "pose" in d:
            recs.append(d)
    return meta, recs


def draw(ax, recs, upto=None, meta=None):
    """Draw the run up to record `upto` (None = all). Everything is in world metres."""
    n = len(recs) if upto is None else min(upto + 1, len(recs))
    live = recs[:n]
    if not live:
        return

    xs = [r["pose"][0] for r in live]
    ys = [r["pose"][1] for r in live]

    # the trail so far
    ax.plot(xs, ys, "-", color="#4a90d9", lw=2, zorder=3, label="robot path")
    ax.plot(xs[0], ys[0], "o", color="#888", ms=9, zorder=4, label="start")

    # every frame where the sign was SEEN but skipped as unreadable: small hollow markers.
    # Side by side with the single gold star, this is the contribution as a picture — the sign
    # was in view for ages and cost nothing until it was worth reading.
    skipped = [r for r in live if (r.get("gate") and not r["gate"].get("legible")
                                   and r["gate"].get("h_px", 0) > 0)]
    if skipped:
        ax.plot([r["pose"][0] for r in skipped], [r["pose"][1] for r in skipped],
                "o", mfc="none", mec="#c8a33a", ms=7, mew=1.2, zorder=4,
                label="sign seen, too small to read")

    # where a sign was READ — the whole point of the project is that this is rare
    for r in live:
        if r.get("sign"):
            ax.plot(r["pose"][0], r["pose"][1], "*", color="#e2b93b", ms=22,
                    markeredgecolor="#7a5c00", zorder=6)
            note = f' READ: "{r["sign"]}"'
            v = r.get("vlm")
            if v:
                note += f'\n {v["decision"]} ({v["latency_s"]}s, conf {v["confidence"]})'
            ax.annotate(note, (r["pose"][0], r["pose"][1]), fontsize=9, color="#7a5c00",
                        zorder=6, xytext=(8, -22), textcoords="offset points")

    # every distinct subgoal the run has had; the last one is the live one
    seen, subs = set(), []
    for r in live:
        sg = r.get("subgoal")
        if sg is None:
            continue
        key = (round(sg[0], 2), round(sg[1], 2))
        if key not in seen:
            seen.add(key)
            subs.append((sg, r))
    for i, (sg, r) in enumerate(subs):
        last = (i == len(subs) - 1)
        ax.plot(sg[0], sg[1], "X", ms=14 if last else 9,
                color="#d9534f" if last else "#e0a3a1", zorder=5)
        if last:
            ax.annotate(" subgoal", (sg[0], sg[1]), fontsize=9, color="#d9534f",
                        xytext=(8, 6), textcoords="offset points", zorder=5)

    cur = live[-1]
    # the line the robot is currently trying to close: here -> subgoal
    if cur.get("subgoal"):
        sg = cur["subgoal"]
        ax.plot([cur["pose"][0], sg[0]], [cur["pose"][1], sg[1]], "--",
                color="#d9534f", lw=1.4, alpha=0.8, zorder=4)

    # the robot: a dot plus a nose showing heading
    import numpy as np
    px, py, pyaw = cur["pose"]
    ax.plot(px, py, "o", color="#1b5e20", ms=11, zorder=7)
    ax.plot([px, px + 0.6 * np.cos(pyaw)], [py, py + 0.6 * np.sin(pyaw)],
            "-", color="#1b5e20", lw=3, zorder=7)

    state = cur.get("state", "")
    dec = cur.get("decision")
    ax.set_title(f"step {cur['i']}   state={state}" + (f"   last decision: {dec}" if dec else ""),
                 fontsize=11)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")


def bounds(recs, pad=1.5):
    xs = [r["pose"][0] for r in recs] + [r["subgoal"][0] for r in recs if r.get("subgoal")]
    ys = [r["pose"][1] for r in recs] + [r["subgoal"][1] for r in recs if r.get("subgoal")]
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=None, help="overview png")
    ap.add_argument("--animate", action="store_true", help="one map per frame -> map_XXXX.png")
    ap.add_argument("--frames", default=None, help="folder to write map frames into")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta, recs = load(args.log)
    if not recs:
        raise SystemExit(f"no records in {args.log}")
    x0, x1, y0, y1 = bounds(recs)

    if args.animate:
        out_dir = args.frames or os.path.dirname(os.path.abspath(args.log))
        for i in range(len(recs)):
            fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
            draw(ax, recs, upto=i, meta=meta)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"map_{recs[i]['i']:04d}.png"))
            plt.close(fig)
        print(f"[map] wrote {len(recs)} map frames to {out_dir}")
        return

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.log)), "map.png")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    draw(ax, recs, meta=meta)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    print(f"[map] wrote {out}")


if __name__ == "__main__":
    main()