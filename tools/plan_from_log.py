"""Draw a top-down floorplan from a run's logged occupancy — no USD, no Isaac needed.

Every frame's occ_cells are real wall/obstacle detections the robot's depth camera made, in a
robot-centric grid. Transformed into world coordinates and accumulated over a whole run, they
trace out the corridors the robot actually drove past. This is ground truth we already have on
disk — no need for the USD library, which only loads inside Isaac's full app startup.

    python tools/plan_from_log.py --log isaac_out/run.jsonl --out plan.png \
        --mark -25 4 start --mark -10 4 junction
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="plan.png")
    ap.add_argument("--mark", nargs=3, action="append", metavar=("X", "Y", "LABEL"), default=[])
    ap.add_argument("--res", type=float, default=0.05, help="grid cell size (m); from log meta")
    ap.add_argument("--x-max", type=float, default=4.0, help="grid forward extent (m)")
    ap.add_argument("--y-half", type=float, default=2.0, help="grid half-width (m)")
    args = ap.parse_args()

    meta, recs = {}, []
    for line in open(args.log):
        d = json.loads(line)
        if d.get("type") == "meta":
            meta = d
        elif "pose" in d:
            recs.append(d)
    g = meta.get("grid", {})
    res = g.get("res", args.res)
    x_max = g.get("x_max", args.x_max)
    y_half = g.get("y_half", args.y_half)

    # Each occ cell is (row, col) in a robot-centric grid: col spans [-y_half, y_half] to the
    # left, row spans [0, x_max] forward. Rotate by the robot's yaw and add its position to get
    # world coordinates.
    pts = []
    for r in recs:
        cells = r.get("occ_cells") or []
        if not cells:
            continue
        px, py, yaw = r["pose"]
        c, s = np.cos(yaw), np.sin(yaw)
        for (row, col) in cells:
            fwd = row * res                      # metres ahead
            left = y_half - col * res            # metres to the left
            wx = px + fwd * c - left * s
            wy = py + fwd * s + left * c
            pts.append((wx, wy))

    if not pts:
        raise SystemExit("no occ_cells in the log — was the detector/occupancy on?")
    pts = np.array(pts)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8), dpi=110)
    ax.plot(pts[:, 0], pts[:, 1], ".", ms=2, color="#4a6a8a", alpha=0.3)
    xs = [r["pose"][0] for r in recs]
    ys = [r["pose"][1] for r in recs]
    ax.plot(xs, ys, "-", color="#4a90d9", lw=1.5, label="robot path")

    for (x, y, label) in args.mark:
        x, y = float(x), float(y)
        ax.plot(x, y, "o", color="#d9534f", ms=12, zorder=5)
        ax.annotate(f" {label} ({x:.1f},{y:.1f})", (x, y), fontsize=11, color="#7a1f1c",
                    xytext=(8, 4), textcoords="offset points", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("walls seen during the run (from logged occupancy)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"[plan] {len(pts)} wall points from {len(recs)} frames -> {args.out}")


if __name__ == "__main__":
    main()
