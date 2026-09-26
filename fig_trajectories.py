#!/usr/bin/env python3
"""SignWay paper figure: predicted vs ground-truth TRAJECTORIES.

At each frame the policy outputs an 8-waypoint path in the robot's current
body frame. This plots that path directly against the path the human actually
drove over the same 8 steps -- no reduction to a scalar, no chaining of
displacements. One panel per selected moment, all panels on a shared scale so
magnitudes are comparable across them.

Frame convention: x forward (up the panel), y left (+ left, plotted leftwards).
Each panel starts at the robot (0,0).

Panels are chosen at the moments that carry the claim -- approach under the
standing prompt, turn onset, peak of the turn, exit -- so the sequence reads
left to right as the decision being held and then executed.

Usage:
  python fig_trajectories.py --sweep $S --ckpt v8@9k --bag e2 --out figures/fig_traj
  python fig_trajectories.py --sweep $S --ckpt v8@9k \
      --panels "e2:300,e2:340,e2:380,e3:200" --out figures/fig_traj
"""
import argparse
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COL2 = 7.16
PRED = "#C8102E"
GT = "#000000"


def load_series(path):
    d = np.load(path, allow_pickle=True)
    return d.item() if hasattr(d, "item") and d.dtype == object else dict(d)


def _step_label(n):
    n = int(n)
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


def derive_labels(payload, path):
    ckpt = str(payload.get("checkpoint", "") or "")
    ver = re.search(r"oft_ckpts_(v\d+)", ckpt)
    step = re.search(r"--(\d+)_chkpt", ckpt)
    if ver and step:
        cl = f"{ver.group(1)}@{_step_label(step.group(1))}"
    elif ver:
        cl = ver.group(1)
    elif ckpt:
        cl = os.path.basename(ckpt.rstrip("/"))[:24]
    else:
        cl = os.path.basename(os.path.dirname(os.path.dirname(path)))
    bag = str(payload.get("bag", "") or "")
    base = os.path.basename(bag.rstrip("/"))
    bl = base.rsplit("-", 1)[-1] if base else os.path.basename(
        os.path.dirname(path))
    return cl, bl


def _sort_key(label):
    return (re.sub(r"\d+", "", label),
            [int(x) for x in re.findall(r"\d+", label)])


def discover(sweep_dirs):
    found = {}
    for root_dir in sweep_dirs:
        for dirpath, _, files in os.walk(root_dir):
            if "series.npy" in files:
                p = os.path.join(dirpath, "series.npy")
                try:
                    found.setdefault(derive_labels(load_series(p), p), p)
                except Exception as e:  # noqa: BLE001
                    print(f"  [skip] {p}: {e}")
    return found


def auto_panels(d, n):
    """Pick the moments that carry the claim, as (index, caption) pairs."""
    lat = np.asarray(d["actual_lat"])
    a = np.abs(lat)
    peak = a.max()
    frames = np.asarray(d["frames"])
    if peak * 100.0 < 2.0:                       # straight bag
        idx = np.linspace(0, len(lat) - 9, n).astype(int)
        return [(i, f"frame {frames[i]}") for i in idx]

    side = "left" if lat[np.argmax(a)] > 0 else "right"
    inturn = np.where(a > 0.30 * peak)[0]
    t0, t1 = int(inturn[0]), int(inturn[-1])
    pk = (t0 + t1) // 2          # middle of the turn window: stable when the
                                 # turn plateaus, unlike argmax
    lo = 0
    f = d.get("flip_frame", None)
    if f not in (None, "", "None"):
        try:
            lo = int(np.searchsorted(frames, int(f)))
        except (TypeError, ValueError):
            lo = 0
    picks = [(max(0, (lo + t0) // 2), "approach\n(prompt held)"),
             (t0, "turn onset"),
             (pk, f"mid-turn ({side})"),
             (min(len(lat) - 9, (t1 + len(lat)) // 2), "after turn")]
    if n >= 5:
        picks.insert(0, (max(0, lo // 2), "before prompt"))
    return picks[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", default=[])
    ap.add_argument("--run", action="append", default=[],
                    help="LABEL:BAG:PATH/to/series.npy")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--bag", default=None,
                    help="bag to auto-select panels from")
    ap.add_argument("--panels", default=None,
                    help="explicit BAG:RAWFRAME list, comma separated")
    ap.add_argument("--n-panels", type=int, default=4)
    ap.add_argument("--out", default="fig_trajectories")
    args = ap.parse_args()

    if not args.sweep and not args.run:
        ap.error("give --sweep DIR or --run LABEL:BAG:PATH")
    paths = discover(args.sweep) if args.sweep else {}
    for spec in args.run:
        label, bag, path = spec.split(":", 2)
        paths[(label, bag)] = path
    if not paths:
        ap.error("no series.npy found")

    ckpts = sorted({c for c, _ in paths}, key=_sort_key)
    bags = sorted({b for _, b in paths}, key=_sort_key)
    if args.list:
        print(f"\ncheckpoints: {ckpts}\nbags:        {bags}\n")
        for (c, b), p in sorted(paths.items(), key=lambda kv: kv[0]):
            print(f"   {c:>10s}  {b:<4s}  {p}")
        return

    ckpt = args.ckpt or (ckpts[0] if len(ckpts) == 1 else None)
    if ckpt is None:
        ap.error(f"several checkpoints present, pick one with --ckpt: {ckpts}")
    if ckpt not in ckpts:
        ap.error(f"--ckpt not found: {ckpt}. Available: {ckpts}")

    panels = []          # (series dict, index, caption)
    if args.panels:
        for spec in args.panels.split(","):
            b, raw = spec.strip().split(":")
            if (ckpt, b) not in paths:
                ap.error(f"no series for {ckpt}/{b}. Bags: {bags}")
            d = load_series(paths[(ckpt, b)])
            i = int(np.searchsorted(np.asarray(d["frames"]), int(raw)))
            i = min(max(i, 0), len(d["pred_traj"]) - 1)
            panels.append((d, i, f"{b} · frame {raw}"))
    else:
        b = args.bag or bags[0]
        if (ckpt, b) not in paths:
            ap.error(f"no series for {ckpt}/{b}. Bags: {bags}")
        d = load_series(paths[(ckpt, b)])
        panels = [(d, i, cap) for i, cap in auto_panels(d, args.n_panels)]
        print(f"[panels] {b}: " +
              ", ".join(f"{d['frames'][i]}({cap.splitlines()[0]})"
                        for d, i, cap in panels))

    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(COL2, COL2 / n * 1.55 + 0.9))
    axes = np.atleast_1d(axes)

    allx, ally = [], []
    for ax, (d, i, cap) in zip(axes, panels):
        gt = np.vstack([[0, 0], np.asarray(d["actual_traj"])[i]])
        pr = np.vstack([[0, 0], np.asarray(d["pred_traj"])[i]])
        ax.plot(gt[:, 1], gt[:, 0], color=GT, ls=(0, (4, 2)), lw=1.3,
                marker="o", ms=2.2, mfc=GT, mec="none", zorder=3)
        ax.plot(pr[:, 1], pr[:, 0], color=PRED, lw=1.3, marker="o", ms=2.2,
                mfc=PRED, mec="none", zorder=4)
        ax.plot(0, 0, marker="s", ms=3.4, color="#333333", zorder=5)
        ax.set_title(cap, pad=3)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_axisbelow(True)
        allx += [gt[:, 1], pr[:, 1]]
        ally += [gt[:, 0], pr[:, 0]]

    X = np.concatenate(allx)
    Y = np.concatenate(ally)
    hx = max(0.5 * (X.max() - X.min()) * 1.25, 0.18)
    cx = 0.5 * (X.max() + X.min())
    y0, y1 = Y.min() - 0.05, Y.max() + 0.08
    for j, ax in enumerate(axes):
        ax.set_xlim(cx + hx, cx - hx)            # inverted: +y (left) on left
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("y [m]", labelpad=1)
        if j == 0:
            ax.set_ylabel("x [m]  (forward)")
        else:
            ax.tick_params(labelleft=False)

    handles = [Line2D([], [], color=GT, ls=(0, (4, 2)), lw=1.3,
                      label="Ground truth (odometry)"),
               Line2D([], [], color=PRED, lw=1.3, label=f"{ckpt} prediction")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), columnspacing=1.8, handlelength=2.0)
    fig.subplots_adjust(top=0.76, bottom=0.15, left=0.08, right=0.99,
                        wspace=0.12)

    for ext in ("pdf", "png"):
        p = f"{args.out}.{ext}"
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    for d, i, cap in panels:
        e = np.linalg.norm(np.asarray(d["pred_traj"])[i]
                           - np.asarray(d["actual_traj"])[i], axis=1)
        lat = abs(float(d["pred_traj"][i][-1][1] - d["actual_traj"][i][-1][1]))
        print(f"  frame {d['frames'][i]:>4}: endpoint err {e[-1] * 100:.1f} cm "
              f"(lateral {lat * 100:.1f} cm)")


if __name__ == "__main__":
    main()
