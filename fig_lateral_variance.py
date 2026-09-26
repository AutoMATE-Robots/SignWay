#!/usr/bin/env python3
"""SignWay paper figure: endpoint lateral vs frame, with error-variance bands
and per-phase error distributions.

Built on the view that works (the decision-vs-timing plots): centimetres on the
y-axis, frames on the x-axis. No world-frame reconstruction, so no odometry and
no accumulated heading error -- the quantity plotted is exactly the quantity
evaluated.

Layout, one column per condition:
  Row 1  endpoint lateral (cm): ground truth (black) vs model (colour) with a
         shaded +/- variance band. Prompt-flip marked; turn window shaded.
  Row 2  |lateral error| box plots split by PHASE (approach / turn), because
         pooling the whole bag mixes two behaviours that fail differently:
         patience on the approach, magnitude calibration in the turn.
  Row 3  (--with-forward) same for forward error, which is speed-dominated and
         must be read separately.

Band definition (stated in the legend, never implicit):
  --band ckpt    spread across checkpoints (min-max). Needs >= 2 checkpoints.
  --band smooth  +/-1 std of the raw series about its rolling mean, i.e. the
                 within-run temporal jitter the smoothing hides.
  default: ckpt when >= 2 checkpoints are plotted, else smooth.

Usage:
  python fig_lateral_variance.py --sweep $S --list
  python fig_lateral_variance.py --sweep $S --ckpts "v10@6k,v11@30k" \
      --bags "e1,e2,e3,t1" --out figures/fig_variance
"""
import argparse
import os
import re
from collections import OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COL2 = 7.16                      # IEEE two-column full text width, inches
COLORS = ["#C8102E", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E"]
GT = "#000000"
TURN_FRAC = 0.30                 # |gt| above this fraction of peak = "turn"


# ---------------------------------------------------------------- discovery
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
    nums = [int(x) for x in re.findall(r"\d+", label)]
    return (re.sub(r"\d+", "", label), nums)


def discover(sweep_dirs):
    found = {}
    for root_dir in sweep_dirs:
        for dirpath, _, files in os.walk(root_dir):
            if "series.npy" not in files:
                continue
            path = os.path.join(dirpath, "series.npy")
            try:
                found.setdefault(derive_labels(load_series(path), path), path)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {path}: {e}")
    return found


# ---------------------------------------------------------------- analysis
def roll(y, w):
    """Centred rolling mean, edge-padded so length is preserved."""
    if w <= 1:
        return np.asarray(y, dtype=float)
    w = int(w) | 1                                   # force odd
    pad = w // 2
    yp = np.pad(np.asarray(y, dtype=float), pad, mode="edge")
    return np.convolve(yp, np.ones(w) / w, mode="valid")


def phases(gt_cm, frames, flip_frame):
    """Split a bag into named frame masks by what the human actually did.

    Turn window is threshold-relative (|gt| > 30% of peak) rather than taken
    from the annotation, so it tracks the driven turn rather than the label.
    """
    a = np.abs(gt_cm)
    peak = a.max()
    if peak < 2.0:                                   # a real turn is 10-20 cm
        return [("straight", np.ones(len(gt_cm), dtype=bool))], None
    inturn = a > TURN_FRAC * peak
    idx = np.where(inturn)[0]
    t0, t1 = idx[0], idx[-1]
    approach = np.zeros(len(gt_cm), dtype=bool)
    lo = 0
    if flip_frame not in (None, "", "None"):
        try:
            lo = int(np.searchsorted(frames, int(flip_frame)))
        except (TypeError, ValueError):
            lo = 0
    approach[lo:t0] = True
    turn = np.zeros(len(gt_cm), dtype=bool)
    turn[t0:t1 + 1] = True
    return [("approach", approach), ("turn", turn)], (t0, t1)


def flip_index(d):
    f = d.get("flip_frame", None)
    if f in (None, "", "None"):
        return None
    try:
        return int(np.searchsorted(np.asarray(d["frames"]), int(f)))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- plotting
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", default=[])
    ap.add_argument("--run", action="append", default=[],
                    help="LABEL:BAG:PATH/to/series.npy")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ckpts", default=None)
    ap.add_argument("--bags", default=None)
    ap.add_argument("--title", action="append", default=[],
                    help='BAG="panel title"')
    ap.add_argument("--smooth", type=int, default=9,
                    help="rolling-mean window in steps (disclosed on the axis)")
    ap.add_argument("--band", choices=["ckpt", "smooth", "none"], default=None)
    ap.add_argument("--straight-ylim", type=float, default=2.0,
                    help="half-range in cm for straight panels (default 2)")
    ap.add_argument("--with-longitudinal", "--with-forward", dest="with_long",
                    action="store_true",
                    help="add a longitudinal-error row (speed-dominated)")
    ap.add_argument("--out", default="fig_variance")
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
    for name, sel, pool in (("--ckpts", args.ckpts, ckpts),
                            ("--bags", args.bags, bags)):
        if sel:
            want = [s.strip() for s in sel.split(",")]
            miss = [w for w in want if w not in pool]
            if miss:
                ap.error(f"{name} not found: {miss}. Available: {pool}")
            if name == "--ckpts":
                ckpts = want
            else:
                bags = want

    if args.list:
        print(f"\ncheckpoints: {ckpts}\nbags:        {bags}\n")
        for (c, b), p in sorted(paths.items(), key=lambda kv: kv[0]):
            print(f"   {c:>10s}  {b:<4s}  {p}")
        return

    runs = OrderedDict()
    for c in ckpts:
        for b in bags:
            if (c, b) in paths:
                runs[(c, b)] = load_series(paths[(c, b)])
    if not runs:
        ap.error("filters left nothing to plot")

    band = args.band or ("ckpt" if len(ckpts) >= 2 else "smooth")
    if band == "ckpt" and len(ckpts) < 2:
        print("[band] only one checkpoint; falling back to --band smooth")
        band = "smooth"
    titles = dict(s.split("=", 1) for s in args.title)
    color = {c: COLORS[i % len(COLORS)] for i, c in enumerate(ckpts)}

    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    rows = 2 + int(args.with_long)
    heights = [2.3, 1.15] + ([1.15] if args.with_long else [])
    ncol = len(bags)
    fig, axes = plt.subplots(
        rows, ncol, figsize=(COL2, 1.55 * sum(heights)),
        gridspec_kw={"height_ratios": heights, "hspace": 0.30, "wspace": 0.26})
    axes = np.atleast_2d(axes)
    if ncol == 1:
        axes = axes.reshape(rows, 1)

    # ---- row 1: lateral series + variance band --------------------------
    turn_axes, straight_axes = [], []
    for j, bag in enumerate(bags):
        ax = axes[0, j]
        ref = next(runs[(c, bag)] for c in ckpts if (c, bag) in runs)
        frames = np.asarray(ref["frames"])
        gt = roll(np.asarray(ref["actual_lat"]) * 100.0, args.smooth)
        ph, window = phases(gt, frames, ref.get("flip_frame"))

        if window:                                   # shade the driven turn
            ax.axvspan(frames[window[0]], frames[window[1]],
                       color="#000000", alpha=0.05, lw=0, zorder=0)
        fi = flip_index(ref) if window else None
        if fi is not None and 0 < fi < len(frames):
            ax.axvline(frames[fi], color="#777777", ls=(0, (3, 2)), lw=0.7,
                       zorder=1)
            ax.annotate("prompt flip", xy=(frames[fi], 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(2, -7), textcoords="offset points",
                        fontsize=6, color="#777777", ha="left")

        stack = []
        for c in ckpts:
            d = runs.get((c, bag))
            if d is None:
                continue
            stack.append(np.asarray(d["pred_lat"]) * 100.0)
        stack = np.vstack(stack)
        sm = np.vstack([roll(s, args.smooth) for s in stack])

        if band == "ckpt":
            ax.fill_between(frames, sm.min(axis=0), sm.max(axis=0),
                            color=COLORS[0], alpha=0.22, lw=0, zorder=2)
        elif band == "smooth":
            resid = stack[0] - sm[0]
            sd = np.sqrt(roll(resid ** 2, args.smooth))
            ax.fill_between(frames, sm[0] - sd, sm[0] + sd,
                            color=COLORS[0], alpha=0.22, lw=0, zorder=2)
        for i, c in enumerate(ckpts):
            if (c, bag) in runs:
                ax.plot(frames, sm[i], color=color[c], lw=1.0, zorder=3)
        ax.plot(frames, gt, color=GT, lw=1.3, zorder=4)

        ax.axhline(0, color="#BBBBBB", lw=0.5, zorder=1)
        ax.set_title(titles.get(bag, bag), pad=3)
        ax.set_xlabel("frame", labelpad=1)
        if j == 0:
            ax.set_ylabel("lateral (cross-track) [cm]\n+ = left,  \u2212 = right")
        ax.grid(alpha=0.2, lw=0.4)
        ax.set_axisbelow(True)
        (straight_axes if len(ph) == 1 else turn_axes).append(ax)

    # turn panels share a scale; straight panels keep their own and say so
    if turn_axes:
        lo = min(a.get_ylim()[0] for a in turn_axes)
        hi = max(a.get_ylim()[1] for a in turn_axes)
        for a in turn_axes:
            a.set_ylim(lo, hi)
    for a in straight_axes:
        a.set_ylim(-args.straight_ylim, args.straight_ylim)

    # ---- rows 2+: per-phase error distributions -------------------------
    err_rows = [("pred_lat", "actual_lat", "|lateral error| [cm]")]
    if args.with_long:
        err_rows.append(("pred_fwd", "actual_fwd",
                         "|longitudinal error| [cm]"))

    for r, (pk, akey, ylabel) in enumerate(err_rows, start=1):
        for j, bag in enumerate(bags):
            ax = axes[r, j]
            ref = next(runs[(c, bag)] for c in ckpts if (c, bag) in runs)
            gt_lat = roll(np.asarray(ref["actual_lat"]) * 100.0, args.smooth)
            ph, _ = phases(gt_lat, np.asarray(ref["frames"]),
                           ref.get("flip_frame"))
            data, cols, centres, names = [], [], [], []
            n_ck = len([c for c in ckpts if (c, bag) in runs])
            w = 0.8 / max(1, n_ck)
            for p, (pname, mask) in enumerate(ph):
                k = 0
                for c in ckpts:
                    d = runs.get((c, bag))
                    if d is None:
                        continue
                    e = np.abs(np.asarray(d[pk]) - np.asarray(d[akey])) * 100.0
                    data.append(e[mask])
                    cols.append(color[c])
                    centres.append(p + k * w - 0.4 + w / 2)
                    k += 1
                names.append(pname)
            bp = ax.boxplot(
                data, positions=centres, widths=w * 0.85, patch_artist=True,
                showmeans=True, showfliers=False, whis=(5, 95),
                medianprops=dict(color="black", lw=0.9),
                meanprops=dict(marker="D", markersize=2.6,
                               markerfacecolor="#EEEEEE",
                               markeredgecolor="black", markeredgewidth=0.4),
                boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5),
                capprops=dict(lw=0.5))
            for patch, fc in zip(bp["boxes"], cols):
                patch.set_facecolor(fc)
                patch.set_alpha(0.45)
                patch.set_edgecolor(fc)
            ax.set_xticks(np.arange(len(names)))
            ax.set_xticklabels(names)
            ax.set_xlim(-0.6, len(names) - 0.4)
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.2, lw=0.4)
            ax.set_axisbelow(True)
        lo = min(axes[r, j].get_ylim()[0] for j in range(ncol))
        hi = max(axes[r, j].get_ylim()[1] for j in range(ncol))
        for j in range(ncol):
            axes[r, j].set_ylim(lo, hi)

    band_txt = {"ckpt": "band: min\u2013max across checkpoints",
                "smooth": f"band: \u00b11 SD about the w={args.smooth} mean",
                "none": ""}[band]
    handles = [Line2D([], [], color=GT, lw=1.3, label="Ground truth")]
    handles += [Line2D([], [], color=color[c], lw=1.0, label=c) for c in ckpts]
    if band_txt:
        handles.append(Patch(facecolor=COLORS[0], alpha=0.22, label=band_txt))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.5, handlelength=1.8)
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.10, right=0.99)

    for ext in ("pdf", "png"):
        p = f"{args.out}.{ext}"
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    print(f"smoothing w={args.smooth} steps; {band_txt or 'no band'}")


if __name__ == "__main__":
    main()