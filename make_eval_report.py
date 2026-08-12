#!/usr/bin/env python3
"""
make_eval_report.py — aggregate a Pepper-VLA eval sweep into meeting-ready output.

Reads:  <root>/eval_<ckpt>_<bag>/ dirs produced by eval_sweep.sbatch
Writes: <root>/report/
    cmp_<bag>_lateral_cm.png        all checkpoints + ground truth, raw, shared ylim
    cmp_<bag>_lateral_cm_smooth.png same, disclosed rolling mean (labeled on plot)
    montage_<bag>.png               fallback: existing timing_test.png stacked
    summary.md                      per-(ckpt, bag) metrics table

Metrics come straight from series.npy (mean_l1 / mean_endpoint saved by the patched
eval_openloop.py); if a run predates the patch, falls back to grepping the run log
for the printed "mean L1 traj error" / "mean endpoint error" lines, and its figures
degrade to a PNG montage.
CPU-only; reruns in seconds on a login node:
    python make_eval_report.py --root $SCRATCH/eval_sweep_<stamp>
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR_RE = re.compile(r"eval_(?P<ckpt>[A-Za-z0-9]+_[0-9]+)_(?P<bag>rosbag2-.+)$")
SMOOTH_W = 9  # rolling-mean window (steps @10Hz) — disclosed on the plot


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_series(outdir):
    """Return the series.npy dict (pred/actual in metres) or None."""
    f = os.path.join(outdir, "series.npy")
    if not os.path.isfile(f):
        return None
    try:
        obj = np.load(f, allow_pickle=True)
        d = obj.item() if obj.dtype == object else None
    except Exception:
        return None
    if not isinstance(d, dict) or "pred_lat" not in d or "actual_lat" not in d:
        return None
    d["pred_lat"] = np.asarray(d["pred_lat"], dtype=float)
    d["actual_lat"] = np.asarray(d["actual_lat"], dtype=float)
    return d


def grep_metrics(logpath):
    """Fallback: pull mean L1 / endpoint error from the eval's stdout log."""
    l1 = ep = None
    if logpath and os.path.isfile(logpath):
        txt = open(logpath, errors="replace").read()
        m = re.search(r"mean L1 traj error\s*:\s*([0-9]*\.?[0-9]+)", txt)
        if m:
            l1 = float(m.group(1))
        m = re.search(r"mean endpoint error\s*:\s*([0-9]*\.?[0-9]+)", txt)
        if m:
            ep = float(m.group(1))
    return l1, ep


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rolling(x, w):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def flip_index(series, n):
    """Index into the (strided) series where the prompt flips, from raw flip_frame."""
    flip = series.get("flip_frame")
    if flip in (None, 0):
        return None
    frames = series.get("frames")
    if frames is not None and len(frames):
        i = int(np.searchsorted(np.asarray(frames), int(flip)))
        return i if 0 < i < n else None
    stride = int(series.get("stride", 2) or 2)
    i = int(flip) // stride
    return i if 0 < i < n else None


def metrics_for(series, bag):
    """Resting lean, turn peak vs actual, onset delta — the meeting numbers."""
    p, a = series["pred_lat"], series["actual_lat"]
    n = min(len(p), len(a))
    p, a = p[:n], a[:n]

    out = {}
    is_straight = "e3" in bag or np.max(np.abs(a)) < 0.05

    if is_straight:
        out["resting_mean_cm"] = 100 * float(np.mean(p))
        out["resting_maxabs_cm"] = 100 * float(np.max(np.abs(p)))
        return out

    # approach region = before flip (or first 40% as fallback)
    fi = flip_index(series, n)
    cut = fi if fi else int(0.4 * n)
    if cut > 5:
        out["resting_mean_cm"] = 100 * float(np.mean(p[:cut]))

    # turn peaks: signed extreme, sign taken from ground truth
    sgn = 1.0 if a[np.argmax(np.abs(a))] >= 0 else -1.0
    a_pk = float(np.max(sgn * a))
    p_pk = float(np.max(sgn * p))
    out["actual_peak_cm"] = 100 * sgn * a_pk
    out["pred_peak_cm"] = 100 * sgn * p_pk
    out["peak_ratio"] = (p_pk / a_pk) if a_pk > 1e-6 else float("nan")

    # onset: first sustained crossing of 25% of the actual peak, after approach
    thr = 0.25 * a_pk

    def onset(x):
        y = sgn * x
        for i in range(cut, n - 3):
            if np.all(y[i:i + 3] > thr):
                return i
        return None

    oa, op = onset(a), onset(p)
    if oa is not None:
        out["onset_actual_step"] = oa
    if op is not None:
        out["onset_pred_step"] = op
    if oa is not None and op is not None:
        out["onset_delta_steps"] = op - oa  # + = late, - = early (@10Hz: 1 step = 0.1s)
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot_bag(bag, runs, outdir, smooth=False):
    """One figure per bag: every checkpoint's pred + ground truth, cm.
    X-axis = raw frame index (matches timing_test.png) when available."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    gt_done = False
    lo, hi = 0.0, 0.0
    for name in sorted(runs):
        s = runs[name]
        p = 100 * s["pred_lat"]
        a = 100 * s["actual_lat"]
        x = s.get("frames")
        x = np.asarray(x) if x is not None and len(x) == len(p) else np.arange(len(p))
        if smooth:
            p, a = rolling(p, SMOOTH_W), rolling(a, SMOOTH_W)
        if not gt_done:
            ax.plot(x, a, color="black", lw=1.8, label="ground truth")
            gt_done = True
            lo, hi = min(lo, a.min()), max(hi, a.max())
        ax.plot(x, p, lw=1.2, label=name)
        lo, hi = min(lo, p.min()), max(hi, p.max())
    pad = 0.06 * max(hi - lo, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.axhline(0, color="gray", lw=0.5)
    title = f"{bag} — endpoint lateral (cm)"
    if smooth:
        title += f"  [rolling mean, w={SMOOTH_W} steps]"
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel("lateral (cm)   + = left, − = right")
    ax.legend(fontsize=8)
    fig.tight_layout()
    suffix = "_smooth" if smooth else ""
    path = os.path.join(outdir, f"cmp_{bag}_lateral_cm{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def montage_bag(bag, png_paths, outdir):
    """Fallback when raw series are missing: stack existing timing_test.png."""
    imgs = [(n, plt.imread(p)) for n, p in sorted(png_paths.items())
            if os.path.isfile(p)]
    if not imgs:
        return None
    fig, axes = plt.subplots(len(imgs), 1, figsize=(10, 3.6 * len(imgs)))
    if len(imgs) == 1:
        axes = [axes]
    for ax, (name, im) in zip(axes, imgs):
        ax.imshow(im)
        ax.set_title(f"{bag} — {name}", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    path = os.path.join(outdir, f"montage_{bag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="eval_sweep results root")
    args = ap.parse_args()

    report = os.path.join(args.root, "report")
    os.makedirs(report, exist_ok=True)
    log_dir = os.path.join(args.root, "logs")

    by_bag_series = {}   # bag -> {ckpt: series}
    by_bag_pngs = {}     # bag -> {ckpt: timing_test.png}
    rows = []            # (ckpt, bag, metrics, mean_l1, mean_ep)

    for d in sorted(glob.glob(os.path.join(args.root, "eval_*"))):
        if not os.path.isdir(d):
            continue
        m = DIR_RE.search(os.path.basename(d))
        if not m:
            continue
        ckpt, bag = m.group("ckpt"), m.group("bag")

        s = load_series(d)
        if s is not None:
            mean_l1 = s.get("mean_l1")
            mean_ep = s.get("mean_endpoint")
            by_bag_series.setdefault(bag, {})[ckpt] = s
            rows.append((ckpt, bag, metrics_for(s, bag), mean_l1, mean_ep))
        else:
            mean_l1, mean_ep = grep_metrics(
                os.path.join(log_dir, f"{ckpt}_{bag}.log"))
            rows.append((ckpt, bag, {}, mean_l1, mean_ep))
        png = os.path.join(d, "timing_test.png")
        if os.path.isfile(png):
            by_bag_pngs.setdefault(bag, {})[ckpt] = png

    if not rows:
        print(f"No eval_* dirs found under {args.root}", file=sys.stderr)
        sys.exit(1)

    # figures
    made = []
    for bag, runs in by_bag_series.items():
        made.append(plot_bag(bag, runs, report, smooth=False))
        made.append(plot_bag(bag, runs, report, smooth=True))
    for bag, pngs in by_bag_pngs.items():
        if bag not in by_bag_series:  # montage only where raw data is missing
            p = montage_bag(bag, pngs, report)
            if p:
                made.append(p)

    # summary table
    cols = ["ckpt", "bag", "mean_L1_m", "mean_EP_m", "resting_mean_cm",
            "resting_maxabs_cm", "pred_peak_cm", "actual_peak_cm",
            "peak_ratio", "onset_delta_steps"]
    lines = ["# Eval sweep summary", "",
             "| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for ckpt, bag, mt, l1, ep in sorted(rows, key=lambda r: (r[1], r[0])):
        vals = [ckpt, bag,
                f"{l1:.4f}" if l1 is not None else "—",
                f"{ep:.4f}" if ep is not None else "—"]
        for k in cols[4:]:
            v = mt.get(k)
            if v is None:
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["",
              "Notes: resting/peak in cm; onset_delta in 10 Hz steps "
              "(+ = model turns late, 1 step = 0.1 s). Reference bands: v4 e3 "
              "resting +0.2…+0.7 cm; v5 lean failure +1.8 cm. "
              "Smoothed figures use a disclosed rolling mean (w=%d)." % SMOOTH_W]

    summary_path = os.path.join(report, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote {summary_path}")
    for p in made:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()