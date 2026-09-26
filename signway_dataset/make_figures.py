#!/usr/bin/env python3
"""
make_figures.py -- publication figures for the trajectory-policy section.

Three figures, each tied to one claim:

  fig_timing        decision-vs-timing. Predicted vs actual lateral for the frozen
                    anchors. The PATIENCE REGION -- turn prompt standing, robot
                    still straight -- is shaded distinctly, because that region is
                    where the paper's claim lives and where v9 failed.

  fig_leakage       the FLIP_ZERO ablation: a leaky checkpoint beside a patient
                    one on the same bag, annotated. This is the figure that
                    explains why prompt relabelling was abandoned.

  fig_metrics       lateral metric suite across versions. Forward error is on its
                    own axis with a caption note -- never pooled.

Style targets IEEE two-column: 3.5 in single column, 7.16 in double, 300 dpi,
PDF (vector, for LaTeX) plus PNG (for slides). No chartjunk: y-grid only, no top
or right spines, frameless legends.

    python signway_dataset/make_figures.py --sweep $SCRATCH/eval_sweep_20260821_1856 \
        --ship v10_6000 --leaky v9_9000 --baseline v8_9000 --out figures/
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
COL1, COL2 = 3.5, 7.16          # IEEE column widths, inches
C = dict(
    actual="#2B2B2B",           # ground truth: near-black, solid
    pred="#3B7DD8",             # prediction: blue
    pred_alt="#D8703B",         # second model in a comparison: orange
    leak="#C4453B",             # the failure being highlighted
    prompt="#F2E4D4",           # standing-turn shading
    patience="#D9E8F5",         # the patience region (the claim)
    grid="#D8D8D8",
)


def apply_style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "axes.edgecolor": "#444444",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.handlelength": 1.6,
        "lines.linewidth": 1.3, "lines.solid_capstyle": "round",
        "grid.color": C["grid"], "grid.linewidth": 0.5, "grid.alpha": 0.8,
    })


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    print(f"  {out_dir / name}.pdf / .png")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_sweep(sweep: Path):
    """(ckpt, bag) -> series dict."""
    out = {}
    for f in glob.glob(str(sweep / "**" / "series.npy"), recursive=True):
        m = re.match(r"eval_(.+?)_(rosbag2-[\w-]+)$", Path(f).parent.name)
        if not m:
            continue
        try:
            out[(m.group(1), m.group(2).replace("rosbag2-", ""))] = np.load(
                f, allow_pickle=True).item()
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {f}: {e}")
    return out


def unpack(d):
    fr = np.asarray(d["frames"], float)
    pl = np.asarray(d["pred_lat"], float) * 100      # cm
    al = np.asarray(d["actual_lat"], float) * 100
    pr = np.asarray([str(p) for p in d["prompts"]])
    turn = np.char.startswith(pr, "turn")
    return fr, pl, al, turn


def contiguous_runs(mask, fr, min_len=3):
    """[(start_frame, end_frame)] for each contiguous True run.

    Shading min(mask)..max(mask) would span the gaps too -- for the patience
    mask that swallows the turn itself, which is exactly the region it is
    meant to exclude.
    """
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    return [(fr[g[0]], fr[g[-1]]) for g in groups if len(g) >= min_len]


def shade_regions(ax, fr, al, turn, flat_tol_cm=1.0, label=True):
    """Standing-turn window, with the patience sub-regions called out."""
    first = True
    for a, b in contiguous_runs(turn, fr):
        ax.axvspan(a, b, color=C["prompt"], zorder=0,
                   label="turn prompt standing" if (label and first) else None)
        first = False
    first = True
    for a, b in contiguous_runs(turn & (np.abs(al) < flat_tol_cm), fr):
        ax.axvspan(a, b, color=C["patience"], zorder=0,
                   label="patience region\n(prompt set, still straight)"
                   if (label and first) else None)
        first = False


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_timing(series, ckpt, bags, out_dir, sharey=True):
    """sharey: one scale across panels, so the straight panel's millimetre noise
    is not blown up into an apparent failure by auto-scaling."""
    import matplotlib.pyplot as plt

    bags = [b for b in bags if (ckpt, b) in series]
    if not bags:
        print("  [fig_timing] no matching bags; skipped")
        return
    fig, axes = plt.subplots(1, len(bags), figsize=(COL2, 1.9), sharey=sharey)
    axes = np.atleast_1d(axes)
    titles = {"keller-e1": "(a) right turn", "keller-e2": "(b) left turn",
              "keller-e3": "(c) straight"}
    for ax, bag in zip(axes, bags):
        fr, pl, al, turn = unpack(series[(ckpt, bag)])
        shade_regions(ax, fr, al, turn, label=(ax is axes[0]))
        ax.axhline(0, color="#999999", lw=0.5, zorder=1)
        ax.plot(fr, al, color=C["actual"], zorder=3,
                label="ground truth (odometry)" if ax is axes[0] else None)
        ax.plot(fr, pl, color=C["pred"], ls="--", zorder=4,
                label="our VLA" if ax is axes[0] else None)
        ax.set_title(titles.get(bag, bag))
        ax.set_xlabel("frame")
        ax.grid(axis="y", zorder=0)
        ax.margins(x=0.01)
    axes[0].set_ylabel("lateral offset  [cm]\n(+ left, − right)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4,
               columnspacing=1.4)
    fig.tight_layout()
    save(fig, out_dir, "fig_timing")
    plt.close(fig)


def fig_leakage(series, leaky, clean, bag, out_dir, leaky_label=None,
                clean_label=None):
    import matplotlib.pyplot as plt

    if (leaky, bag) not in series or (clean, bag) not in series:
        print("  [fig_leakage] missing a checkpoint; skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.1), sharey=True, sharex=True)
    for ax, (ck, lab, col) in zip(axes, [
            (leaky, leaky_label or f"{leaky}  (prompt relabelling)", C["leak"]),
            (clean, clean_label or f"{clean}  (annotations respected)", C["pred"])]):
        fr, pl, al, turn = unpack(series[(ck, bag)])
        shade_regions(ax, fr, al, turn, label=(ax is axes[0]))
        ax.axhline(0, color="#999999", lw=0.5, zorder=1)
        ax.plot(fr, al, color=C["actual"], zorder=3,
                label="ground truth" if ax is axes[0] else None)
        ax.plot(fr, pl, color=col, ls="--", zorder=4,
                label="prediction" if ax is axes[0] else None)
        ax.set_title(lab)
        ax.set_xlabel("frame")
        ax.grid(axis="y", zorder=0)

        pat = turn & (np.abs(al) < 1.0)
        runs = contiguous_runs(pat, fr)
        if pat.any() and runs:
            drift = np.abs(pl[pat])
            a, b = runs[0]                       # the APPROACH run, pre-turn
            mid = (a + b) / 2
            k = int(np.argmin(np.abs(fr - mid)))
            ax.annotate(f"approach drift\nmean {drift.mean():.2f} cm, "
                        f"max {drift.max():.2f} cm",
                        xy=(mid, pl[k]),
                        xytext=(0.05, 0.78), textcoords="axes fraction",
                        fontsize=6.5, color=col,
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.7,
                                        shrinkB=2))
    axes[0].set_ylabel("lateral offset  [cm]")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4,
               columnspacing=1.4)
    fig.tight_layout()
    save(fig, out_dir, "fig_leakage")
    plt.close(fig)


def fig_metrics(series, ckpts, bags, out_dir, labels=None):
    """Lateral suite as grouped bars; forward on its own panel, never pooled."""
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from traj_metrics import metrics_from_series

    ckpts = [c for c in ckpts if any((c, b) in series for b in bags)]
    if not ckpts:
        print("  [fig_metrics] nothing to plot; skipped")
        return
    labels = labels or {c: c for c in ckpts}

    panels = [("approach_mean_cm", "approach drift (cm)\nlower = more patient", False),
              ("peak_ratio", "turn magnitude ratio\n1.0 = exact", True),
              ("lat_mae_cm", "lateral MAE (cm)", False),
              ("fwd_mae_cm", "forward MAE (cm)\n(speed — see caption)", False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(COL2, 2.0))
    x = np.arange(len(ckpts))
    w = 0.8 / max(1, len(bags))
    cols = [C["pred"], C["pred_alt"], "#6BA46B", "#9B7BC4"]

    for ax, (key, title, unit_line) in zip(axes, panels):
        for j, bag in enumerate(bags):
            vals = []
            for c in ckpts:
                d = series.get((c, bag))
                vals.append(metrics_from_series(d)[key] if d else np.nan)
            bars = ax.bar(x + j * w - 0.4 + w / 2, vals, w * 0.92,
                          color=cols[j % len(cols)],
                          label=bag if ax is axes[0] else None,
                          edgecolor="white", linewidth=0.4)
            for b, v in zip(bars, vals):
                if v == v:
                    ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                                ha="center", va="bottom", fontsize=5.2,
                                color="#444444", xytext=(0, 1),
                                textcoords="offset points")
        if unit_line:
            ax.axhline(1.0, color="#999999", lw=0.7, ls=":")
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] for c in ckpts], rotation=20, ha="right")
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=len(bags), columnspacing=1.4)
    fig.tight_layout()
    save(fig, out_dir, "fig_metrics")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# BEV overlay -- the figure VLA/nav papers lead with
# --------------------------------------------------------------------------- #
PROMPT_C = {"straight": "#6BA46B", "turn_left": "#4C9BE8",
            "turn_right": "#E8994C", "stop": "#C4453B"}


def dead_reckon(actual_traj):
    """Global (pos, heading) chained from the per-frame first waypoint.

    wp1 is the displacement to the next strided frame in the CURRENT robot
    frame. CHORD-vs-ARC: on a constant-curvature step the displacement chord
    bisects the turn, so its direction is only HALF the heading change --
    heading must advance by 2*atan2(left, fwd), not atan2(left, fwd).

    Using the chord angle directly (the original bug) drew every 90 degree
    corner as roughly 45 degrees and left square loops visibly open. Affects the
    BEV figure only; no metric ever used this function.

    Still dead-reckoning: error accumulates over long episodes, so a loop may
    not close exactly. Caption accordingly.
    """
    n = len(actual_traj)
    pos = np.zeros((n, 2))
    th = np.zeros(n)
    for i in range(n - 1):
        fwd, left = actual_traj[i, 0]
        c, s_ = np.cos(th[i]), np.sin(th[i])
        pos[i + 1] = pos[i] + np.array([c * fwd - s_ * left,
                                        s_ * fwd + c * left])
        if abs(fwd) + abs(left) > 1e-6:
            th[i + 1] = th[i] + 2.0 * np.arctan2(left, fwd)
        else:
            th[i + 1] = th[i]
    return pos, th


def fig_bev(series, ckpt, bags, out_dir, every=8, window_m=(3.5, 2.0)):
    """Executed path with predicted-trajectory whiskers, coloured by the standing
    prompt, CROPPED to the junction window -- a 0.5 m prediction is invisible
    against a 40 m corridor, so the figure shows from window_m[0] metres before
    the prompt flip to window_m[1] past the end of the turn. A star marks the
    flip (the sign read). The claim in one picture: whiskers stay straight after
    the star, bend only at the junction."""
    import matplotlib.pyplot as plt

    bags = [b for b in bags if (ckpt, b) in series]
    if not bags:
        print("  [fig_bev] no matching bags; skipped")
        return
    fig, axes = plt.subplots(1, len(bags), figsize=(COL2, 2.6))
    axes = np.atleast_1d(axes)
    titles = {"keller-e1": "(a) right turn", "keller-e2": "(b) left turn",
              "keller-e3": "(c) straight"}
    for ax, bag in zip(axes, bags):
        d = series[(ckpt, bag)]
        A = np.asarray(d["actual_traj"], float).reshape(len(d["frames"]), -1, 2)
        P = np.asarray(d["pred_traj"], float).reshape(A.shape)
        pr = np.asarray([str(x) for x in d["prompts"]])
        pos, th = dead_reckon(A)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]

        flips = np.nonzero(pr[1:] != pr[:-1])[0]
        if flips.size:                                   # turn bag
            f = flips[0] + 1
            dth = np.abs(np.diff(th))
            turning = np.nonzero(dth > np.deg2rad(0.5))[0]
            t0 = turning[0] if turning.size else len(pos) - 1
            end = turning[-1] + 1 if turning.size else len(pos) - 1
            # window around the JUNCTION, not the whole approach: a 0.5 m
            # prediction whisker vanishes against a 15 m corridor
            m = (arc >= arc[t0] - window_m[0]) & (arc <= arc[end] + window_m[1])
        else:                                            # straight bag
            f = None
            mid = arc[-1] / 2
            m = np.abs(arc - mid) <= (window_m[0] + window_m[1])
        idx = np.nonzero(m)[0]

        # path UNDER the whiskers: accurate predictions lie on the path, so
        # drawing the path on top makes them invisible
        ax.plot(pos[idx, 0], pos[idx, 1], color="#8C8C8C", lw=3.0, zorder=2,
                solid_capstyle="round")
        for i in idx[::every]:
            c, s_ = np.cos(th[i]), np.sin(th[i])
            R = np.array([[c, -s_], [s_, c]])
            w = pos[i] + P[i] @ R.T
            ax.plot(np.r_[pos[i, 0], w[:, 0]], np.r_[pos[i, 1], w[:, 1]],
                    color=PROMPT_C.get(pr[i], "#888888"), lw=1.2, alpha=0.95,
                    zorder=3)
            ax.plot(pos[i, 0], pos[i, 1], "o", ms=1.8, color="#3A3A3A",
                    zorder=4)
        if f is not None:
            if m[f]:
                ax.plot(pos[f, 0], pos[f, 1], marker="*", ms=12,
                        color="#B8860B", mec="white", mew=0.5, zorder=5)
                ax.text(0.98, 0.96, "$\\bigstar$ sign read here",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=6.2, color="#8A6508")
            else:
                # the sign was read before this window: say how far back
                back = arc[idx[0]] - arc[f]
                ax.text(0.98, 0.96, f"$\\bigstar$ sign read {back:.1f} m earlier",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=6.2, color="#8A6508")

        xr = [pos[idx, 0].min(), pos[idx, 0].max()]
        yr = [pos[idx, 1].min(), pos[idx, 1].max()]
        # minimum extents: a straight bag otherwise collapses to a strip with
        # no room for the scale bar or panel label
        for r, mn in ((xr, 3.0), (yr, 2.6)):
            if r[1] - r[0] < mn:
                c_ = (r[0] + r[1]) / 2
                r[0], r[1] = c_ - mn / 2, c_ + mn / 2
        padx = 0.10 * (xr[1] - xr[0])
        pady = 0.14 * (yr[1] - yr[0])
        ax.set_xlim(xr[0] - padx, xr[1] + padx)
        ax.set_ylim(yr[0] - pady, yr[1] + pady)
        ax.set_aspect("equal")

        # 1 m scale bar, bottom-right corner (bottom-left holds the panel label
        # on straight bags whose path hugs the bottom)
        bx = ax.get_xlim()[1] - 1.45
        by = ax.get_ylim()[0] + 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.plot([bx, bx + 1.0], [by, by], color="#444444", lw=1.5)
        ax.annotate("1 m", (bx + 0.5, by), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=6,
                    color="#444444")

        # panel label inside the axes: consistent placement across panels of
        # different aspect (set_title floats at differing heights otherwise)
        ax.text(0.02, 0.96, titles.get(bag, bag), transform=ax.transAxes,
                ha="left", va="top", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    handles = [plt.Line2D([0], [0], color="#8C8C8C", lw=3.0,
                          label="executed path (odometry)")]
    for k, lab in (("straight", "our VLA (prompt: straight)"),
                   ("turn_left", "our VLA (prompt: turn left)"),
                   ("turn_right", "our VLA (prompt: turn right)")):
        handles.append(plt.Line2D([0], [0], color=PROMPT_C[k], lw=1.2,
                                  label=lab))
    handles.append(plt.Line2D([0], [0], marker="*", color="#B8860B", ls="",
                              ms=9, mec="white", label="sign read (prompt set)"))
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.10), ncol=5, columnspacing=1.0,
               handletextpad=0.5)
    fig.tight_layout()
    save(fig, out_dir, "fig_bev")
    plt.close(fig)


def fig_horizon(series, ckpts, bags, out_dir, labels=None):
    """Error vs prediction horizon (waypoint 1..8): how far ahead the policy
    sees. Lateral and forward on separate panels -- never pooled."""
    import matplotlib.pyplot as plt

    labels = labels or {c: c for c in ckpts}
    fig, axes = plt.subplots(1, 2, figsize=(COL1 * 1.6, 2.0))
    styles = [dict(color=C["pred"]), dict(color=C["pred_alt"]),
              dict(color="#6BA46B"), dict(color="#9B7BC4")]
    for ci, ck in enumerate(ckpts):
        L, F = [], []
        for bag in bags:
            d = series.get((ck, bag))
            if d is None:
                continue
            A = np.asarray(d["actual_traj"], float).reshape(
                len(d["frames"]), -1, 2)
            P = np.asarray(d["pred_traj"], float).reshape(A.shape)
            L.append(np.abs(P[:, :, 1] - A[:, :, 1]).mean(0))
            F.append(np.abs(P[:, :, 0] - A[:, :, 0]).mean(0))
        if not L:
            continue
        k = np.arange(1, len(L[0]) + 1)
        axes[0].plot(k, np.mean(L, 0) * 100, marker="o", ms=3,
                     label=labels[ck], **styles[ci % 4])
        axes[1].plot(k, np.mean(F, 0) * 100, marker="o", ms=3,
                     label=labels[ck], **styles[ci % 4])
    axes[0].set_title("lateral error vs horizon")
    axes[1].set_title("forward error vs horizon\n(speed — see caption)",
                      fontsize=7.5)
    for ax in axes:
        ax.set_xlabel("waypoint index (~0.1 s each)")
        ax.set_ylabel("mean abs. error  [cm]")
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(range(1, 9))
    axes[0].legend()
    fig.tight_layout()
    save(fig, out_dir, "fig_horizon")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--ship", default="v10_6000", help="checkpoint for fig_timing")
    ap.add_argument("--leaky", default="v9_9000")
    ap.add_argument("--baseline", default="v8_9000")
    ap.add_argument("--leak-bag", default="keller-e2")
    ap.add_argument("--bags", default="keller-e1,keller-e2,keller-e3")
    ap.add_argument("--compare", default=None,
                    help="comma list for fig_metrics; default ship,leaky,baseline")
    args = ap.parse_args()

    apply_style()
    series = load_sweep(Path(args.sweep))
    if not series:
        raise SystemExit(f"no series.npy under {args.sweep}")
    print(f"loaded {len(series)} (ckpt, bag) series")
    bags = args.bags.split(",")
    cmp_ckpts = (args.compare.split(",") if args.compare
                 else [args.baseline, args.leaky, args.ship])

    out = Path(args.out)
    fig_bev(series, args.ship, bags, out)
    fig_timing(series, args.ship, bags, out)
    fig_horizon(series, cmp_ckpts, bags, out)
    fig_leakage(series, args.leaky, args.ship, args.leak_bag, out)
    fig_metrics(series, cmp_ckpts, bags, out)
    print("\nCaption note for fig_metrics: forward MAE is shown on a separate "
          "axis and is NOT combined with the lateral columns -- it is dominated "
          "by unobservable driver speed (0.45-1.48 m/s across bags, no "
          "proprioception input) and is near-identical in behaviour across "
          "versions.")


if __name__ == "__main__":
    main()