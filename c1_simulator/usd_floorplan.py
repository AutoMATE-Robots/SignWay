"""Read the scene's geometry straight from the USD and draw a top-down floorplan.

WHY: placing signs by driving the sim and eyeballing frames is slow trial-and-error. The USD
already contains every wall, door and prop with exact world coordinates — it IS the map. This
walks the stage, projects every mesh's world-space bounding box onto the floor, and renders a
top-down plan you can read sign coordinates straight off. Same idea as Real2USD (arXiv:2510.10778),
which hands the hospital USD to an LLM to pick navigation waypoints inside the walls.

Runs inside the Isaac container (it needs pxr, the USD library).

    /isaac-sim/python.sh -m c1_simulator.usd_floorplan \
        --scene /assets/.../hospital.usd --out plan.png \
        --mark -25 4 start --mark -10 4 junction --mark -4.3 4 sign
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", default="floorplan.png")
    ap.add_argument("--z-min", type=float, default=0.1, help="ignore geometry below this (floor)")
    ap.add_argument("--z-max", type=float, default=2.2, help="ignore geometry above this (ceiling)")
    ap.add_argument("--mark", nargs=3, action="append", metavar=("X", "Y", "LABEL"),
                    default=[], help="annotate a point, repeatable")
    args = ap.parse_args()

    from pxr import Usd, UsdGeom, Gf
    import numpy as np

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        raise SystemExit(f"could not open {args.scene}")

    xform_cache = UsdGeom.XformCache()
    boxes = []                       # (x0, y0, x1, y1) footprints at mid-height
    zc = (args.z_min + args.z_max) / 2.0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        try:
            b = img.ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
            rng = b.ComputeAlignedBox()
            lo, hi = rng.GetMin(), rng.GetMax()
        except Exception:
            continue
        # keep only geometry that straddles robot height — walls/desks/chairs, not floor/ceiling
        if hi[2] < args.z_min or lo[2] > args.z_max:
            continue
        boxes.append((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))

    if not boxes:
        raise SystemExit("no wall-height geometry found — adjust --z-min/--z-max")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    xs = [b[0] for b in boxes] + [b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[3] for b in boxes]
    fig, ax = plt.subplots(figsize=(16, 10), dpi=110)
    for (x0, y0, x1, y1) in boxes:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#4a6a8a",
                               edgecolor="none", alpha=0.35))

    for (x, y, label) in args.mark:
        x, y = float(x), float(y)
        ax.plot(x, y, "o", color="#d9534f", ms=12, zorder=5)
        ax.annotate(f" {label} ({x:.1f},{y:.1f})", (x, y), fontsize=11, color="#7a1f1c",
                    zorder=5, xytext=(8, 4), textcoords="offset points")

    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title(f"{args.scene.split('/')[-1]} — top-down, geometry {args.z_min}-{args.z_max}m")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"[floorplan] {len(boxes)} meshes -> {args.out}", flush=True)
    print(f"[floorplan] extent x {min(xs):.1f}..{max(xs):.1f}  y {min(ys):.1f}..{max(ys):.1f}",
          flush=True)


if __name__ == "__main__":
    main()