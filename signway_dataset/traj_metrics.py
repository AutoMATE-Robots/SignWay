#!/usr/bin/env python3
"""
traj_metrics.py -- metrics that measure what the paper actually claims.

WHY THIS EXISTS: the old summary pooled forward and lateral error into one
`mean_L1_m`. Measured on the v10 sweep, **91-96% of that number was FORWARD
error** -- so every cross-version comparison was ranking models on their ability
to guess the human driver's speed, which is not observable from a single image
(no proprioception; collection speeds span 0.45-1.48 m/s). The lateral errors,
which are what decision-vs-timing is about, were nearly identical across v8, v9
and v10 and were invisible under the pooled number.

The suite, one metric per claim:

  direction_acc   WHICH way. Sign agreement between predicted and actual lateral
                  over the turn window, weighted by |actual| so near-zero frames
                  (where sign is meaningless) do not dominate.

  approach_mean   WHEN, part 1: PATIENCE. Mean/max |predicted lateral| over frames
  approach_max    where a turn prompt is standing but the robot is still going
                  straight (|actual| < flat_tol). No threshold, no smoothing --
                  this is the metric that exposed v9's prompt leakage after
                  `onset_delta` mis-described it as "turns early".

  onset_lag       WHEN, part 2: TIMING. Difference in frames between where the
                  predicted and actual curves cross 25% of THEIR OWN peak.
                  Relative, so a model with a bigger turn is not scored as early
                  merely for being bigger -- the flaw in the absolute-threshold
                  onset_delta.

  peak_ratio      magnitude fidelity: predicted peak / actual peak. 1.0 is exact.

  resting_mean    false turns: |lateral| on straight bags (no turn prompt).
  resting_max

  fwd_mae         forward error, REPORTED SEPARATELY and never summed with the
                  lateral terms. Expect it to dominate; it is a property of the
                  data, not of the policy.

Reads the series.npy dumps written by eval_openloop.save_series:
    keys: frames, prompts, pred_lat, actual_lat, pred_fwd, actual_fwd,
          pred_traj, actual_traj    (lateral/forward in METRES)

    python signway_dataset/traj_metrics.py --sweep $SCRATCH/eval_sweep_20260821_1856
    python signway_dataset/traj_metrics.py --sweep <dir> --csv out.csv --md out.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np

FLAT_TOL = 0.01      # metres: below this the robot is "still going straight"
ONSET_FRAC = 0.25    # fraction of a curve's OWN peak that counts as onset
MIN_TURN = 0.02      # metres: a peak below this is not a turn at all


# --------------------------------------------------------------------------- #
# pure metric functions (unit-testable, no I/O)
# --------------------------------------------------------------------------- #
def direction_accuracy(pred_lat, actual_lat, turn_mask, min_mag=0.01):
    """Sign agreement over the turn, weighted by |actual| (WHICH way)."""
    p, a = pred_lat[turn_mask], actual_lat[turn_mask]
    m = np.abs(a) >= min_mag
    if not m.any():
        return float("nan")
    w = np.abs(a[m])
    agree = (np.sign(p[m]) == np.sign(a[m])).astype(float)
    return float((agree * w).sum() / w.sum())


def approach_patience(pred_lat, actual_lat, turn_mask, flat_tol=FLAT_TOL):
    """|predicted lateral| while a turn prompt stands and truth is still flat.

    THE patience metric. Threshold-free: no smoothing, no crossing detection, so
    a model cannot score well by turning sharply nor badly by turning strongly.
    """
    m = turn_mask & (np.abs(actual_lat) < flat_tol)
    if not m.any():
        return float("nan"), float("nan"), 0
    v = np.abs(pred_lat[m])
    return float(v.mean()), float(v.max()), int(m.sum())


def onset_frame(lat, frames, mask, frac=ONSET_FRAC, min_turn=MIN_TURN):
    """Frame where |lat| first reaches `frac` of its own peak within `mask`."""
    if not mask.any():
        return None
    l, f = np.abs(np.asarray(lat)[mask]), np.asarray(frames)[mask]
    pk = l.max()
    if pk < min_turn:
        return None
    hits = np.nonzero(l >= frac * pk)[0]
    return int(f[hits[0]]) if hits.size else None


def onset_lag(pred_lat, actual_lat, frames, turn_mask, frac=ONSET_FRAC):
    """predicted onset - actual onset, in frames. Negative = model turns early.

    Relative threshold on each curve's own peak, so magnitude differences do not
    masquerade as timing differences (the bug in absolute-threshold onset_delta).
    """
    pf = onset_frame(pred_lat, frames, turn_mask, frac)
    af = onset_frame(actual_lat, frames, turn_mask, frac)
    if pf is None or af is None:
        return float("nan")
    return float(pf - af)


def peak_ratio(pred_lat, actual_lat, turn_mask):
    if not turn_mask.any():
        return float("nan"), float("nan"), float("nan")
    pp = np.abs(pred_lat[turn_mask]).max()
    ap = np.abs(actual_lat[turn_mask]).max()
    if ap < MIN_TURN:
        return float("nan"), float(pp), float(ap)
    return float(pp / ap), float(pp), float(ap)


def resting(pred_lat, turn_mask):
    """|predicted lateral| where NO turn prompt stands (false-turn tendency)."""
    m = ~turn_mask
    if not m.any():
        return float("nan"), float("nan")
    v = np.abs(pred_lat[m])
    return float(v.mean()), float(v.max())


def metrics_from_series(d: dict) -> dict:
    fr = np.asarray(d["frames"], float)
    pl = np.asarray(d["pred_lat"], float)
    al = np.asarray(d["actual_lat"], float)
    pf = np.asarray(d["pred_fwd"], float)
    af = np.asarray(d["actual_fwd"], float)
    pr = np.asarray([str(p) for p in d["prompts"]])
    turn = np.char.startswith(pr, "turn")

    ap_mean, ap_max, ap_n = approach_patience(pl, al, turn)
    ratio, pk_p, pk_a = peak_ratio(pl, al, turn)
    rest_mean, rest_max = resting(pl, turn)
    return dict(
        n_frames=len(fr),
        direction_acc=direction_accuracy(pl, al, turn),
        approach_mean_cm=ap_mean * 100 if ap_mean == ap_mean else float("nan"),
        approach_max_cm=ap_max * 100 if ap_max == ap_max else float("nan"),
        approach_n=ap_n,
        onset_lag_frames=onset_lag(pl, al, fr, turn),
        peak_ratio=ratio,
        pred_peak_cm=pk_p * 100, actual_peak_cm=pk_a * 100,
        resting_mean_cm=rest_mean * 100 if rest_mean == rest_mean else float("nan"),
        resting_max_cm=rest_max * 100 if rest_max == rest_max else float("nan"),
        lat_mae_cm=float(np.abs(pl - al).mean()) * 100,
        fwd_mae_cm=float(np.abs(pf - af).mean()) * 100,
    )


# --------------------------------------------------------------------------- #
# sweep reader
# --------------------------------------------------------------------------- #
def scan_sweep(sweep_dir: Path):
    """Find every series.npy and infer (ckpt, bag) from the directory name
    `eval_<ckpt>_<bag>`."""
    rows = []
    for f in sorted(glob.glob(str(sweep_dir / "**" / "series.npy"), recursive=True)):
        name = Path(f).parent.name
        m = re.match(r"eval_(.+?)_(rosbag2-[\w-]+)$", name)
        ckpt, bag = (m.group(1), m.group(2)) if m else (name, "?")
        try:
            d = np.load(f, allow_pickle=True).item()
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: {e}")
            continue
        r = dict(ckpt=ckpt, bag=bag.replace("rosbag2-", ""))
        r.update(metrics_from_series(d))
        rows.append(r)
    return rows


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--sort", default="bag,ckpt")
    args = ap.parse_args()

    rows = scan_sweep(Path(args.sweep))
    if not rows:
        raise SystemExit(f"no series.npy under {args.sweep}")
    keys = args.sort.split(",")
    rows.sort(key=lambda r: tuple(str(r.get(k, "")) for k in keys))

    hdr = ["ckpt", "bag", "dir_acc", "approach_mean", "approach_max",
           "onset_lag", "peak_ratio", "resting_mean", "lat_mae", "| fwd_mae"]
    line = ("| {ckpt} | {bag} | {da} | {am} | {ax} | {ol} | {pr} | {rm} | {lm} "
            "| {fm} |")
    out = ["## Lateral-first trajectory metrics",
           "",
           "Forward error is reported **separately** and is not part of any other "
           "column: it is dominated by unobservable driver speed (0.45-1.48 m/s "
           "across bags, no proprioception input), so pooling it hides the "
           "lateral behaviour the system is judged on.",
           "",
           "| ckpt | bag | dir acc | approach mean (cm) | approach max (cm) | "
           "onset lag (frames) | peak ratio | resting mean (cm) | lat MAE (cm) | "
           "fwd MAE (cm) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(line.format(
            ckpt=r["ckpt"], bag=r["bag"], da=fmt(r["direction_acc"], 3),
            am=fmt(r["approach_mean_cm"]), ax=fmt(r["approach_max_cm"]),
            ol=fmt(r["onset_lag_frames"], 0), pr=fmt(r["peak_ratio"], 3),
            rm=fmt(r["resting_mean_cm"]), lm=fmt(r["lat_mae_cm"]),
            fm=fmt(r["fwd_mae_cm"])))
    out += ["",
            "**Reading it.** `approach mean/max` = predicted lateral while a turn "
            "prompt stands and the robot is still straight -> **patience** "
            "(lower is better; this is the metric that exposed v9's prompt "
            "leakage). `onset lag` = crossing of 25% of each curve's OWN peak, "
            "negative = early. `dir acc` = magnitude-weighted sign agreement. "
            "`resting` = lateral with no turn prompt (false turns)."]
    text = "\n".join(out)
    print(text)

    if args.md:
        Path(args.md).write_text(text)
        print(f"\n[md] {args.md}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[csv] {args.csv}")


if __name__ == "__main__":
    main()