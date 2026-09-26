#!/usr/bin/env python3
"""
make_figures_publication.py

Publication-quality figures for the SignWay trajectory-policy evaluation.

Preserves the original figure semantics:
  * fig_timing   : decision timing / patience at the junction
  * fig_leakage  : prompt-relabelling leakage ablation
  * fig_metrics  : lateral metric suite across checkpoints
  * fig_bev      : bird's-eye-view trajectory overlay
  * fig_horizon  : error vs prediction horizon

Main visual changes relative to the original script:
  * consistent IEEE-sized typography and spacing
  * stronger visual hierarchy between ground truth and prediction
  * restrained event-region shading
  * fixed, symmetric timing axes across panels
  * panel labels and compact shared legends
  * aggregate + per-bag metric display instead of dense grouped bars
  * uncertainty bands across bags in horizon plots
  * 600 dpi raster export plus vector PDF

Example:
    python signway_dataset/make_figures_publication.py \
        --sweep $SCRATCH/eval_sweep_20260821_1856 \
        --ship v10_6000 \
        --leaky v9_9000 \
        --baseline v8_9000 \
        --compare v8_9000,v10_3000,v10_6000,v10_9000,v10_12000 \
        --metrics-csv $SCRATCH/eval_sweep_20260821_1856/metrics_lateral.csv \
        --out figures_pub/
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Publication style
# --------------------------------------------------------------------------- #
COL1, COL2 = 3.50, 7.16  # IEEE single- and double-column widths [in]

C = {
    "actual": "#222222",
    "pred": "#2F6BBD",
    "pred_alt": "#D97732",
    "leak": "#B43C3C",
    "baseline": "#707780",
    "grid": "#D9DEE5",
    "zero": "#9AA1A9",
    "prompt": "#F3E8D8",
    "patience": "#DDEAF5",
    "mean_fill": "#E9EDF2",
    "mean_edge": "#5D6875",
    "text_muted": "#5D6670",
}

BAG_STYLE = {
    "keller-e1": ("o", "#2F6BBD", "E1 · right"),
    "keller-e2": ("s", "#D97732", "E2 · left"),
    "keller-e3": ("^", "#4F8B68", "E3 · straight"),
    "keller-t1": ("D", "#8A6BB7", "T1 · train ref."),
}

PROMPT_C = {
    "straight": "#5B8F69",
    "turn_left": "#3C84C6",
    "turn_right": "#DD8A38",
    "stop": "#B43C3C",
}


def apply_style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.6,
        "axes.labelsize": 7.8,
        "axes.titlesize": 8.2,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.65,
        "axes.edgecolor": "#50555B",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "lines.linewidth": 1.35,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "grid.color": C["grid"],
        "grid.linewidth": 0.45,
        "grid.alpha": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=600, facecolor="white")
    print(f"  {pdf} / {png}")


def clean_axis(ax, grid=True):
    if grid:
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(pad=2)


def panel_label(ax, text: str):
    # Put the panel letter just outside the plotting area so it never covers data.
    ax.text(
        -0.10,
        1.035,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        clip_on=False,
        zorder=20,
    )


def short_ckpt(ckpt: str) -> str:
    m = re.fullmatch(r"(v\d+)_(\d+)", ckpt)
    if not m:
        return ckpt.replace("_", "\n")
    step = int(m.group(2))
    step_s = f"{step // 1000}k" if step % 1000 == 0 else str(step)
    return f"{m.group(1)}\n{step_s}"


def legend_above(fig, handles, labels, ncol, y=1.02):
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        borderaxespad=0,
    )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_sweep(sweep: Path):
    """Load {(checkpoint, bag): series_dict}."""
    out = {}
    for f in glob.glob(str(sweep / "**" / "series.npy"), recursive=True):
        m = re.match(r"eval_(.+?)_(rosbag2-[\w-]+)$", Path(f).parent.name)
        if not m:
            continue
        try:
            out[(m.group(1), m.group(2).replace("rosbag2-", ""))] = np.load(
                f, allow_pickle=True
            ).item()
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {f}: {e}")
    return out


def unpack(d):
    fr = np.asarray(d["frames"], float)
    pl = np.asarray(d["pred_lat"], float) * 100.0
    al = np.asarray(d["actual_lat"], float) * 100.0
    pr = np.asarray([str(p) for p in d["prompts"]])
    turn = np.char.startswith(pr, "turn")
    return fr, pl, al, turn


def contiguous_runs(mask, fr, min_len=3):
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    return [(fr[g[0]], fr[g[-1]]) for g in groups if len(g) >= min_len]


def shade_regions(ax, fr, al, turn, flat_tol_cm=1.0, label=True):
    """Shade turn-prompt interval and, within it, the pre-turn patience region."""
    first = True
    for a, b in contiguous_runs(turn, fr):
        ax.axvspan(
            a,
            b,
            color=C["prompt"],
            alpha=0.48,
            ec="none",
            zorder=0,
            label="turn instruction active" if (label and first) else None,
        )
        first = False

    first = True
    patience = turn & (np.abs(al) < flat_tol_cm)
    for a, b in contiguous_runs(patience, fr):
        ax.axvspan(
            a,
            b,
            color=C["patience"],
            alpha=0.85,
            ec="none",
            zorder=0.2,
            label="patience region" if (label and first) else None,
        )
        first = False


def read_metrics_csv(path: Path):
    """Return {(ckpt, bag): row} from metrics_lateral.csv."""
    rows = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if k in {"ckpt", "bag"}:
                    continue
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = np.nan
            rows[(r["ckpt"], r["bag"])] = out
    return rows


def build_metrics(series, ckpts, bags, metrics_csv: Path | None = None):
    if metrics_csv is not None and metrics_csv.exists():
        return read_metrics_csv(metrics_csv)

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from traj_metrics import metrics_from_series

    rows = {}
    for ck in ckpts:
        for bag in bags:
            d = series.get((ck, bag))
            if d is not None:
                rows[(ck, bag)] = metrics_from_series(d)
    return rows


# --------------------------------------------------------------------------- #
# 1) Timing / patience figure
# --------------------------------------------------------------------------- #
def fig_timing(series, ckpt, bags, out_dir, sharey=True):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    bags = [b for b in bags if (ckpt, b) in series]
    if not bags:
        print("  [fig_timing] no matching bags; skipped")
        return

    titles = {
        "keller-e1": "Right turn",
        "keller-e2": "Left turn",
        "keller-e3": "Straight",
    }

    fig, axes = plt.subplots(
        1,
        len(bags),
        figsize=(COL2, 2.02),
        sharey=sharey,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    # One common symmetric scale prevents the straight sequence from looking
    # artificially noisy because of auto-scaling.
    yabs = 0.0
    for bag in bags:
        _, pl, al, _ = unpack(series[(ckpt, bag)])
        yabs = max(yabs, float(np.nanmax(np.abs(np.r_[pl, al]))))
    ylim = max(5.0, np.ceil(yabs * 1.10 / 2.0) * 2.0)

    for i, (ax, bag) in enumerate(zip(axes, bags)):
        fr, pl, al, turn = unpack(series[(ckpt, bag)])
        shade_regions(ax, fr, al, turn, label=(i == 0))

        ax.axhline(0, color=C["zero"], lw=0.65, zorder=1)
        ax.plot(
            fr,
            al,
            color=C["actual"],
            lw=1.65,
            zorder=4,
            label="odometry" if i == 0 else None,
        )
        ax.plot(
            fr,
            pl,
            color=C["pred"],
            lw=1.55,
            ls=(0, (4, 2.2)),
            zorder=5,
            label="VLA prediction" if i == 0 else None,
        )

        # Mark the moment the prompt first changes to a turn instruction.
        starts = np.nonzero(turn & ~np.r_[False, turn[:-1]])[0]
        if starts.size:
            x0 = fr[starts[0]]
            ax.axvline(x0, color=C["text_muted"], lw=0.7, ls=(0, (1.5, 2.0)), zorder=2)

        ax.set_title(titles.get(bag, bag), pad=5)
        ax.set_xlabel("Frame")
        ax.set_ylim(-ylim, ylim)
        ax.set_xlim(fr.min(), fr.max())
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        clean_axis(ax)
        panel_label(ax, f"({chr(97 + i)})")

    axes[0].set_ylabel("Lateral offset (cm)\n$+$ left, $-$ right")

    h, l = axes[0].get_legend_handles_labels()
    # Re-order lines first, regions second.
    order = [i for i, x in enumerate(l) if x in {"odometry", "VLA prediction"}] + [
        i for i, x in enumerate(l) if x not in {"odometry", "VLA prediction"}
    ]
    legend_above(fig, [h[i] for i in order], [l[i] for i in order], ncol=len(order), y=1.01)

    save(fig, out_dir, "fig_timing")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2) Leakage ablation
# --------------------------------------------------------------------------- #
def fig_leakage(
    series,
    leaky,
    clean,
    bag,
    out_dir,
    leaky_label=None,
    clean_label=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    if (leaky, bag) not in series or (clean, bag) not in series:
        print("  [fig_leakage] missing a checkpoint; skipped")
        return

    fig, axes = plt.subplots(
        1, 2, figsize=(COL2, 2.18), sharey=True, sharex=True, constrained_layout=True
    )

    configs = [
        (leaky, leaky_label or "Prompt-relabeled model", C["leak"]),
        (clean, clean_label or "Annotation-respecting model", C["pred"]),
    ]

    for i, (ax, (ck, title, col)) in enumerate(zip(axes, configs)):
        fr, pl, al, turn = unpack(series[(ck, bag)])
        shade_regions(ax, fr, al, turn, label=(i == 0))
        ax.axhline(0, color=C["zero"], lw=0.65, zorder=1)

        ax.plot(
            fr,
            al,
            color=C["actual"],
            lw=1.65,
            label="odometry" if i == 0 else None,
            zorder=4,
        )
        ax.plot(
            fr,
            pl,
            color=col,
            lw=1.60,
            ls=(0, (4, 2.2)),
            label="prediction" if i == 0 else None,
            zorder=5,
        )

        pat = turn & (np.abs(al) < 1.0)
        if pat.any():
            drift = np.abs(pl[pat])
            txt = f"Approach drift\nmean  {drift.mean():.2f} cm\nmax    {drift.max():.2f} cm"
            ax.text(
                0.04,
                0.93,
                txt,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
                color=col,
                linespacing=1.15,
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor="white",
                    edgecolor=col,
                    linewidth=0.7,
                    alpha=0.94,
                ),
                zorder=10,
            )

        ax.set_title(title, pad=5)
        ax.set_xlabel("Frame")
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        clean_axis(ax)
        panel_label(ax, f"({chr(97 + i)})")

    axes[0].set_ylabel("Lateral offset (cm)")
    h, l = axes[0].get_legend_handles_labels()
    legend_above(fig, h, l, ncol=len(l), y=1.01)

    save(fig, out_dir, "fig_leakage")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3) Metric suite
# --------------------------------------------------------------------------- #
def fig_metrics(series, ckpts, bags, out_dir, labels=None, metrics_csv=None):
    """
    Four small multiples. Each checkpoint is summarized by a mean bar; individual
    bag values are overlaid as markers. This makes variance visible without the
    clutter of a grouped bar chart or dozens of value labels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    metrics = build_metrics(series, ckpts, bags, metrics_csv)
    ckpts = [c for c in ckpts if any((c, b) in metrics for b in bags)]
    if not ckpts:
        print("  [fig_metrics] nothing to plot; skipped")
        return

    labels = labels or {c: short_ckpt(c) for c in ckpts}

    panels = [
        ("approach_mean_cm", "Approach drift", "Mean |lateral| (cm)", None),
        ("peak_ratio", "Turn magnitude", "Predicted / actual peak", 1.0),
        ("lat_mae_cm", "Lateral accuracy", "MAE (cm)", None),
        ("fwd_mae_cm", "Forward accuracy", "MAE (cm)", None),
    ]

    fig, axes = plt.subplots(
        1, 4, figsize=(COL2, 2.22), constrained_layout=True
    )
    x = np.arange(len(ckpts), dtype=float)
    offsets = np.linspace(-0.15, 0.15, max(1, len(bags)))

    for pi, (ax, (key, title, ylabel, target)) in enumerate(zip(axes, panels)):
        means = []
        mins = []
        maxs = []
        raw = []

        for ck in ckpts:
            vals = np.asarray(
                [metrics.get((ck, b), {}).get(key, np.nan) for b in bags], dtype=float
            )
            vals = vals[np.isfinite(vals)]
            raw.append(vals)
            means.append(np.nan if vals.size == 0 else float(vals.mean()))
            mins.append(np.nan if vals.size == 0 else float(vals.min()))
            maxs.append(np.nan if vals.size == 0 else float(vals.max()))

        means = np.asarray(means)
        mins = np.asarray(mins)
        maxs = np.asarray(maxs)

        # Aggregate checkpoint summary.
        bars = ax.bar(
            x,
            means,
            width=0.58,
            color=C["mean_fill"],
            edgecolor=C["mean_edge"],
            linewidth=0.75,
            zorder=2,
        )

        # Min-to-max whisker across evaluation bags.
        valid = np.isfinite(mins) & np.isfinite(maxs)
        ax.vlines(x[valid], mins[valid], maxs[valid], color=C["mean_edge"], lw=1.0, zorder=3)

        # Individual bag values.
        for bi, bag in enumerate(bags):
            marker, col, _ = BAG_STYLE.get(bag, ("o", C["baseline"], bag))
            vals = np.asarray(
                [metrics.get((ck, bag), {}).get(key, np.nan) for ck in ckpts], dtype=float
            )
            m = np.isfinite(vals)
            ax.scatter(
                x[m] + offsets[bi],
                vals[m],
                s=20,
                marker=marker,
                facecolor=col,
                edgecolor="white",
                linewidth=0.45,
                zorder=5,
            )

        # Small mean labels only — one number per checkpoint, not per bag.
        for b, v in zip(bars, means):
            if np.isfinite(v):
                ax.annotate(
                    f"{v:.2f}",
                    (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=5.7,
                    color=C["text_muted"],
                )

        if target is not None:
            ax.axhline(target, color=C["zero"], lw=0.8, ls=(0, (2, 2)), zorder=1)
            ax.text(
                0.98,
                target,
                "ideal = 1",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=5.8,
                color=C["text_muted"],
            )

        ax.set_title(title, pad=5)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(c, short_ckpt(c)) for c in ckpts])
        ax.yaxis.set_major_locator(MaxNLocator(5))
        clean_axis(ax)
        panel_label(ax, f"({chr(97 + pi)})")

    # Bag legend only once, below the figure; checkpoint names are the x-axis.
    handles = []
    for bag in bags:
        marker, col, lab = BAG_STYLE.get(bag, ("o", C["baseline"], bag))
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                ls="",
                ms=5,
                markerfacecolor=col,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=lab,
            )
        )
    handles.append(
        Line2D([0], [0], color=C["mean_edge"], lw=5, alpha=0.45, label="mean across bags")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(5, len(handles)),
        frameon=False,
        borderaxespad=0,
    )

    save(fig, out_dir, "fig_metrics")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4) BEV trajectory overlay
# --------------------------------------------------------------------------- #
def dead_reckon(actual_traj):
    """Global (position, heading) chained from the per-frame first waypoint."""
    n = len(actual_traj)
    pos = np.zeros((n, 2))
    th = np.zeros(n)
    for i in range(n - 1):
        fwd, left = actual_traj[i, 0]
        c, s = np.cos(th[i]), np.sin(th[i])
        pos[i + 1] = pos[i] + np.array([c * fwd - s * left, s * fwd + c * left])
        if abs(fwd) + abs(left) > 1e-6:
            th[i + 1] = th[i] + np.arctan2(left, fwd)
        else:
            th[i + 1] = th[i]
    return pos, th


def fig_bev(series, ckpt, bags, out_dir, every=8, window_m=(3.5, 2.0)):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    bags = [b for b in bags if (ckpt, b) in series]
    if not bags:
        print("  [fig_bev] no matching bags; skipped")
        return

    titles = {
        "keller-e1": "Right turn",
        "keller-e2": "Left turn",
        "keller-e3": "Straight",
    }

    fig, axes = plt.subplots(
        1, len(bags), figsize=(COL2, 2.55), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ai, (ax, bag) in enumerate(zip(axes, bags)):
        d = series[(ckpt, bag)]
        A = np.asarray(d["actual_traj"], float).reshape(len(d["frames"]), -1, 2)
        P = np.asarray(d["pred_traj"], float).reshape(A.shape)
        pr = np.asarray([str(x) for x in d["prompts"]])
        pos, th = dead_reckon(A)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]

        flips = np.nonzero(pr[1:] != pr[:-1])[0]
        if flips.size:
            f = flips[0] + 1
            dth = np.abs(np.diff(th))
            turning = np.nonzero(dth > np.deg2rad(0.5))[0]
            t0 = turning[0] if turning.size else len(pos) - 1
            end = turning[-1] + 1 if turning.size else len(pos) - 1
            m = (arc >= arc[t0] - window_m[0]) & (arc <= arc[end] + window_m[1])
        else:
            f = None
            mid = arc[-1] / 2.0
            m = np.abs(arc - mid) <= (window_m[0] + window_m[1])

        idx = np.nonzero(m)[0]
        if idx.size == 0:
            continue

        # Broad pale underlay then dark centerline gives the executed path a
        # clean, map-like visual while keeping the prediction whiskers visible.
        path = ax.plot(
            pos[idx, 0],
            pos[idx, 1],
            color=C["actual"],
            lw=2.5,
            zorder=2,
        )[0]
        path.set_path_effects([pe.Stroke(linewidth=4.3, foreground="white"), pe.Normal()])

        for i in idx[::every]:
            c, s = np.cos(th[i]), np.sin(th[i])
            R = np.array([[c, -s], [s, c]])
            w = pos[i] + P[i] @ R.T
            col = PROMPT_C.get(pr[i], C["baseline"])
            line = ax.plot(
                np.r_[pos[i, 0], w[:, 0]],
                np.r_[pos[i, 1], w[:, 1]],
                color=col,
                lw=1.25,
                alpha=0.95,
                zorder=3,
            )[0]
            line.set_path_effects([pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
            ax.plot(pos[i, 0], pos[i, 1], "o", ms=1.7, color=C["actual"], zorder=4)

        if f is not None:
            if m[f]:
                ax.plot(
                    pos[f, 0],
                    pos[f, 1],
                    marker="*",
                    ms=10,
                    color="#B78313",
                    mec="white",
                    mew=0.7,
                    zorder=6,
                )
                read_note = "sign read"
            else:
                back = arc[idx[0]] - arc[f]
                read_note = f"sign read {back:.1f} m earlier"
            ax.text(
                0.98,
                0.04,
                read_note,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.9,
                color=C["text_muted"],
            )

        xr = [pos[idx, 0].min(), pos[idx, 0].max()]
        yr = [pos[idx, 1].min(), pos[idx, 1].max()]
        for r, mn in ((xr, 3.0), (yr, 2.6)):
            if r[1] - r[0] < mn:
                cc = (r[0] + r[1]) / 2.0
                r[0], r[1] = cc - mn / 2.0, cc + mn / 2.0

        padx = 0.10 * (xr[1] - xr[0])
        pady = 0.13 * (yr[1] - yr[0])
        ax.set_xlim(xr[0] - padx, xr[1] + padx)
        ax.set_ylim(yr[0] - pady, yr[1] + pady)
        ax.set_aspect("equal")

        # Scale bar.
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        bx = x1 - 1.35
        by = y0 + 0.07 * (y1 - y0)
        ax.plot([bx, bx + 1.0], [by, by], color=C["actual"], lw=1.4, zorder=8)
        ax.text(bx + 0.5, by + 0.03 * (y1 - y0), "1 m", ha="center", va="bottom", fontsize=5.9)

        ax.text(
            0.03,
            0.97,
            f"({chr(97 + ai)})  {titles.get(bag, bag)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="semibold",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    handles = [Line2D([0], [0], color=C["actual"], lw=2.5, label="executed path")]
    for k, lab in (
        ("straight", "prediction · straight"),
        ("turn_left", "prediction · left"),
        ("turn_right", "prediction · right"),
    ):
        handles.append(Line2D([0], [0], color=PROMPT_C[k], lw=1.5, label=lab))
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="#B78313",
            ls="",
            ms=8,
            mec="white",
            label="sign read",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=False,
        borderaxespad=0,
    )

    save(fig, out_dir, "fig_bev")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5) Prediction-horizon figure
# --------------------------------------------------------------------------- #
def fig_horizon(series, ckpts, bags, out_dir, labels=None):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    labels = labels or {c: short_ckpt(c).replace("\n", " @ ") for c in ckpts}
    fig, axes = plt.subplots(
        1, 2, figsize=(COL1 * 1.72, 2.08), constrained_layout=True
    )

    palette = [C["pred"], C["pred_alt"], "#4F8B68", "#8A6BB7", "#707780"]

    for ci, ck in enumerate(ckpts):
        L, F = [], []
        for bag in bags:
            d = series.get((ck, bag))
            if d is None:
                continue
            A = np.asarray(d["actual_traj"], float).reshape(len(d["frames"]), -1, 2)
            P = np.asarray(d["pred_traj"], float).reshape(A.shape)
            L.append(np.abs(P[:, :, 1] - A[:, :, 1]).mean(0) * 100.0)
            F.append(np.abs(P[:, :, 0] - A[:, :, 0]).mean(0) * 100.0)

        if not L:
            continue

        L = np.asarray(L)
        F = np.asarray(F)
        k = np.arange(1, L.shape[1] + 1)
        col = palette[ci % len(palette)]

        for ax, arr in zip(axes, (L, F)):
            mean = arr.mean(0)
            lo = arr.min(0)
            hi = arr.max(0)
            ax.fill_between(k, lo, hi, color=col, alpha=0.10, linewidth=0, zorder=1)
            ax.plot(
                k,
                mean,
                marker="o",
                ms=3.0,
                mfc="white",
                mew=0.8,
                color=col,
                lw=1.45,
                label=labels.get(ck, ck),
                zorder=3,
            )

    axes[0].set_title("Lateral error", pad=5)
    axes[1].set_title("Forward error", pad=5)
    for i, ax in enumerate(axes):
        ax.set_xlabel("Prediction waypoint")
        ax.set_ylabel("Mean absolute error (cm)")
        ax.set_xticks(range(1, 9))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        clean_axis(ax)
        panel_label(ax, f"({chr(97 + i)})")

    h, l = axes[0].get_legend_handles_labels()
    legend_above(fig, h, l, ncol=min(4, len(l)), y=1.01)

    save(fig, out_dir, "fig_horizon")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", default="figures_pub")
    ap.add_argument("--ship", default="v10_6000", help="checkpoint for timing + BEV")
    ap.add_argument("--leaky", default="v9_9000")
    ap.add_argument("--baseline", default="v8_9000")
    ap.add_argument("--leak-bag", default="keller-e2")
    ap.add_argument("--bags", default="keller-e1,keller-e2,keller-e3")
    ap.add_argument(
        "--compare",
        default=None,
        help="comma-separated checkpoints for metric/horizon plots",
    )
    ap.add_argument(
        "--metrics-csv",
        default=None,
        help="optional metrics_lateral.csv; if omitted, auto-detect under --sweep",
    )
    args = ap.parse_args()

    apply_style()
    sweep = Path(args.sweep)
    series = load_sweep(sweep)
    if not series:
        raise SystemExit(f"no series.npy under {sweep}")

    print(f"loaded {len(series)} (checkpoint, bag) series")
    bags = [x.strip() for x in args.bags.split(",") if x.strip()]
    cmp_ckpts = (
        [x.strip() for x in args.compare.split(",") if x.strip()]
        if args.compare
        else [args.baseline, args.leaky, args.ship]
    )

    if args.metrics_csv:
        metrics_csv = Path(args.metrics_csv)
    else:
        candidate = sweep / "metrics_lateral.csv"
        metrics_csv = candidate if candidate.exists() else None

    out = Path(args.out)

    fig_bev(series, args.ship, bags, out)
    fig_timing(series, args.ship, bags, out)
    fig_horizon(series, cmp_ckpts, bags, out)
    fig_leakage(series, args.leaky, args.ship, args.leak_bag, out)
    fig_metrics(series, cmp_ckpts, bags, out, metrics_csv=metrics_csv)

    print("\nPublication notes:")
    print("  • PDF is vector; PNG is exported at 600 dpi.")
    print("  • Timing panels share one symmetric y-scale.")
    print("  • Metric bars are checkpoint means; markers are individual bags.")
    print("  • Horizon bands show the min–max spread across bags.")
    print("  • Forward MAE remains visually separate from lateral metrics.")


if __name__ == "__main__":
    main()
