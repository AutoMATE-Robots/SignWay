#!/usr/bin/env python3
"""
fig_longhorizon.py -- figures for a multi-junction (square-loop) evaluation.

Three figures, and they show what single-junction anchors cannot:

  fig_square_bev        the WHOLE loop, dead-reckoned, with the policy's
                        predicted trajectories sprouting along it, coloured by
                        the standing prompt and numbered at each junction. One
                        panel per model when comparing.

  fig_square_timeline   lateral offset across the entire episode, models
                        overlaid, each junction shaded. Shows patience between
                        corners and magnitude at each one, in a single read.

  fig_square_junctions  per-junction bars: approach drift (patience) and peak
                        ratio (magnitude). Answers "does it degrade over four
                        corners" directly.

    python signway_dataset/fig_longhorizon.py \
        --series v10_6000=$SCRATCH/eval_square_v10_6000/series.npy \
                 v11_10000=$SCRATCH/eval_square_v11_10000/series.npy \
        --segments $SCRATCH/square_segments.json --out ~/SignWay/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(_HERE))
from make_figures import (COL1, COL2, C, PROMPT_C, apply_style,  # noqa: E402
                          dead_reckon, save)

FLAT_TOL = 0.01


def load(spec):
    """'name=path' -> (name, series dict)"""
    name, _, path = spec.partition("=")
    if not path:
        name, path = Path(spec).parent.name, spec
    return name, np.load(path, allow_pickle=True).item()


def junction_stats(fr, pl, al, segments):
    out = []
    for k, s in enumerate(segments, 1):
        m = (fr >= s["legible"]) & (fr < s["turn_done"])
        if not m.any():
            out.append(dict(k=k, approach=np.nan, ratio=np.nan))
            continue
        p, a = pl[m], al[m]
        flat = np.abs(a) < FLAT_TOL
        app = np.abs(p[flat]).mean() * 100 if flat.any() else np.nan
        pk_a, pk_p = np.abs(a).max(), np.abs(p).max()
        out.append(dict(k=k, approach=app,
                        ratio=(pk_p / pk_a) if pk_a > 0.02 else np.nan,
                        actual_cm=pk_a * 100, pred_cm=pk_p * 100))
    return out


# --------------------------------------------------------------------------- #
def fig_bev(models, segments, out_dir, every=12):
    import matplotlib.pyplot as plt

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(COL1 * 1.15 * n, 3.3))
    axes = np.atleast_1d(axes)
    for ax, (name, d) in zip(axes, models):
        fr = np.asarray(d["frames"], float)
        A = np.asarray(d["actual_traj"], float).reshape(len(fr), -1, 2)
        P = np.asarray(d["pred_traj"], float).reshape(A.shape)
        pr = np.asarray([str(x) for x in d["prompts"]])
        pos, th = dead_reckon(A)

        ax.plot(pos[:, 0], pos[:, 1], color="#9A9A9A", lw=2.6, zorder=2,
                solid_capstyle="round")
        for i in range(0, len(pos), every):
            c, s_ = np.cos(th[i]), np.sin(th[i])
            R = np.array([[c, -s_], [s_, c]])
            w = pos[i] + P[i] @ R.T
            ax.plot(np.r_[pos[i, 0], w[:, 0]], np.r_[pos[i, 1], w[:, 1]],
                    color=PROMPT_C.get(pr[i], "#888888"), lw=0.9, alpha=0.9,
                    zorder=3)
        # junction markers
        for j, s in enumerate(segments, 1):
            k = int(np.argmin(np.abs(fr - s["legible"])))
            ax.plot(pos[k, 0], pos[k, 1], marker="*", ms=9, color="#B8860B",
                    mec="white", mew=0.4, zorder=5)
            ax.annotate(f"J{j}", (pos[k, 0], pos[k, 1]), xytext=(5, 5),
                        textcoords="offset points", fontsize=6.5,
                        color="#8A6508")
        ax.plot(pos[0, 0], pos[0, 1], "o", ms=5, color="#2B2B2B", zorder=6)
        ax.annotate("start", (pos[0, 0], pos[0, 1]), xytext=(4, -10),
                    textcoords="offset points", fontsize=6, color="#2B2B2B")

        ax.set_aspect("equal")
        ax.text(0.5, -0.02, name, transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5)
        # scale bar
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        bx, by = x1 - 3.0, y1 - 0.06 * (y1 - y0)
        ax.plot([bx, bx + 2.0], [by, by], color="#444444", lw=1.5)
        ax.annotate("2 m", (bx + 1.0, by), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=6,
                    color="#444444")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    import matplotlib.pyplot as plt
    handles = [plt.Line2D([0], [0], color="#9A9A9A", lw=2.6,
                          label="executed path (dead-reckoned odometry)")]
    for k_, lab in (("straight", "prediction | straight"),
                    ("turn_left", "prediction | turn_left"),
                    ("turn_right", "prediction | turn_right")):
        handles.append(plt.Line2D([0], [0], color=PROMPT_C[k_], lw=1.2,
                                  label=lab))
    handles.append(plt.Line2D([0], [0], marker="*", color="#B8860B", ls="",
                              ms=8, mec="white", label="junction (sign read)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.09),
               ncol=5, columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout()
    save(fig, out_dir, "fig_square_bev")
    plt.close(fig)


def fig_timeline(models, segments, out_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(COL2, 2.3))
    ref = models[0][1]
    fr = np.asarray(ref["frames"], float)
    al = np.asarray(ref["actual_lat"], float) * 100
    for j, s in enumerate(segments, 1):
        ax.axvspan(s["legible"], s["turn_done"], color=C["prompt"], zorder=0,
                   label="turn prompt standing" if j == 1 else None)
        ax.annotate(f"J{j}", ((s["legible"] + s["turn_done"]) / 2, 0),
                    xytext=(0, 2), textcoords="offset points", ha="center",
                    va="bottom", fontsize=6.5, color="#8A6508")
    ax.axhline(0, color="#999999", lw=0.5, zorder=1)
    ax.plot(fr, al, color=C["actual"], lw=1.3, zorder=3, label="ground truth")
    cols = [C["pred"], C["pred_alt"], "#6BA46B", "#9B7BC4"]
    for i, (name, d) in enumerate(models):
        ax.plot(np.asarray(d["frames"], float),
                np.asarray(d["pred_lat"], float) * 100,
                color=cols[i % 4], ls="--", lw=1.1, zorder=4, label=name)
    ax.set_xlabel("frame")
    ax.set_ylabel("lateral offset  [cm]\n(+ left, − right)")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.005)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4,
              columnspacing=1.3)
    fig.tight_layout()
    save(fig, out_dir, "fig_square_timeline")
    plt.close(fig)


def fig_junctions(models, segments, out_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.0))
    x = np.arange(len(segments))
    w = 0.8 / max(1, len(models))
    cols = [C["pred"], C["pred_alt"], "#6BA46B", "#9B7BC4"]
    for i, (name, d) in enumerate(models):
        st = junction_stats(np.asarray(d["frames"], float),
                            np.asarray(d["pred_lat"], float),
                            np.asarray(d["actual_lat"], float), segments)
        off = i * w - 0.4 + w / 2
        for ax, key, ttl in ((axes[0], "approach",
                              "approach drift per junction (cm)\nlower = more patient"),
                             (axes[1], "ratio",
                              "turn magnitude ratio\n1.0 = exact")):
            vals = [s[key] for s in st]
            bars = ax.bar(x + off, vals, w * 0.9, color=cols[i % 4],
                          label=name if ax is axes[0] else None,
                          edgecolor="white", linewidth=0.4)
            for b, v in zip(bars, vals):
                if v == v:
                    ax.annotate(f"{v:.2f}",
                                (b.get_x() + b.get_width() / 2, v),
                                ha="center", va="bottom", fontsize=5.2,
                                color="#444444", xytext=(0, 1),
                                textcoords="offset points")
            ax.set_title(ttl, fontsize=7.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"J{k+1}" for k in range(len(segments))])
            ax.grid(axis="y", zorder=0)
            ax.set_axisbelow(True)
    axes[1].axhline(1.0, color="#999999", lw=0.7, ls=":")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.11),
               ncol=len(models), columnspacing=1.3)
    fig.tight_layout()
    save(fig, out_dir, "fig_square_junctions")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", required=True,
                    help="name=path/to/series.npy (repeatable)")
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--every", type=int, default=12,
                    help="draw a prediction whisker every N steps in the BEV")
    args = ap.parse_args()

    apply_style()
    segs = json.loads(Path(args.segments).read_text())
    segments = sorted(segs["segments"] if isinstance(segs, dict) else segs,
                      key=lambda s: s["legible"])
    models = [load(s) for s in args.series]
    print(f"{len(models)} model(s), {len(segments)} junctions")

    out = Path(args.out)
    fig_bev(models, segments, out, every=args.every)
    fig_timeline(models, segments, out)
    fig_junctions(models, segments, out)

    print("\nper-junction summary")
    for name, d in models:
        st = junction_stats(np.asarray(d["frames"], float),
                            np.asarray(d["pred_lat"], float),
                            np.asarray(d["actual_lat"], float), segments)
        print(f"  {name}: " + "  ".join(
            f'J{s["k"]} drift {s["approach"]:.2f}cm ratio {s["ratio"]:.2f}'
            for s in st))


if __name__ == "__main__":
    main()
