"""
action_snr.py — why the action parameterisation decides whether a turn survives
L1 regression.  CPU only, reads the built RLDS dataset; no checkpoint needed.

For each parameterisation we ask how loud the turn is relative to the spread the
regressor already has to model:

    SNR(d) = | mean(a_d | turn frames) - mean(a_d | straight frames) | / std(a_d)

evaluated on the same frames, for
  (i)  cumulative waypoints  a = [x1,y1,...,x8,y8]   (deployed; we report wp8 lateral)
  (ii) per-step displacement a = [dx,dy]             (v2; recomputed from the same
       trajectories, so the comparison is exact rather than historical)

An L1 head minimises loss by predicting the conditional median; when SNR << 1 the
per-prompt median (~straight) is already competitive and the turn is not learned.

  python action_snr.py --data $SCRATCH/rlds_v6 --dataset signway --out $SCRATCH/vla_eval/snr.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", default="signway")
    ap.add_argument("--turn-th", type=float, default=0.03, help="|wp8 lateral| [m] marking a turn frame")
    ap.add_argument("--out", default="snr.csv")
    a = ap.parse_args()
    sys.argv.append("--signway")                      # constants auto-detect
    import tensorflow_datasets as tfds

    ds = tfds.builder_from_directory(a.data).as_dataset(split="train")
    A = []
    for ep in tfds.as_numpy(ds):
        for st in ep["steps"]:
            A.append(np.asarray(st["action"], np.float64))
    A = np.stack(A)                                    # (N,16)
    wp = A.reshape(len(A), 8, 2)
    lat8 = wp[:, -1, 1]                                # endpoint lateral
    turn = np.abs(lat8) > a.turn_th
    print(f"{len(A)} steps | turn frames: {turn.sum()} ({100*turn.mean():.1f}%)")

    # per-step displacement recomputed from the same waypoints: wp1 is the next frame
    step_dx, step_dy = wp[:, 0, 0], wp[:, 0, 1]

    rows = []
    def add(name, v):
        """Three quantities, each answering a different question:
        magnitude  - how big is the turn in metres (what the notes call 'louder')
        snr        - turn magnitude against the NOISE FLOOR measured on straight
                     frames, i.e. can the turn be distinguished from odometry jitter
        shortcut   - excess L1 loss (normalised units) incurred by a predictor that
                     ignores the image and outputs the per-prompt median.  This is
                     the quantity an L1 head actually trades off: small => the
                     lazy constant policy is nearly optimal and the turn is dropped.
        """
        if turn.sum() < 5 or (~turn).sum() < 5:
            return
        mag = np.abs(v[turn]).mean()
        noise = np.abs(v[~turn] - np.median(v[~turn])).mean()      # straight-frame jitter
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)         # Q99 normalisation
        vn = 2 * (v - lo) / max(hi - lo, 1e-12) - 1
        const = np.median(vn)                                       # the lazy predictor
        shortcut = np.abs(vn - const).mean()
        rows.append({"representation": name, "turn_magnitude_m": mag,
                     "straight_noise_m": noise, "snr_vs_noise": mag / (noise + 1e-12),
                     "excess_L1_norm": shortcut,
                     "turn_frac": float(turn.mean())})
    add("cumulative waypoint (wp8 lateral)", lat8)
    add("cumulative waypoint (wp4 lateral)", wp[:, 3, 1])
    add("per-step displacement (dy)", step_dy)
    add("per-step displacement (dx, forward)", step_dx)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n{'representation':38s} {'|turn| m':>9s} {'noise m':>9s} {'SNR':>7s} {'excess L1':>10s}")
    for r in rows:
        print(f"{r['representation']:38s} {r['turn_magnitude_m']:9.4f} {r['straight_noise_m']:9.4f} "
              f"{r['snr_vs_noise']:7.2f} {r['excess_L1_norm']:10.4f}")
    g = lambda k, pre: next(r[k] for r in rows if r["representation"].startswith(pre))
    print(f"\nturn magnitude : waypoint {g('turn_magnitude_m','cumulative waypoint (wp8'):.4f} m "
          f"vs per-step {g('turn_magnitude_m','per-step displacement (dy'):.4f} m "
          f"({g('turn_magnitude_m','cumulative waypoint (wp8')/max(g('turn_magnitude_m','per-step displacement (dy'),1e-12):.0f}x)")
    print(f"SNR vs noise   : {g('snr_vs_noise','cumulative waypoint (wp8'):.2f} vs "
          f"{g('snr_vs_noise','per-step displacement (dy'):.2f}")
    print(f"excess L1 of the image-ignoring predictor: "
          f"{g('excess_L1_norm','cumulative waypoint (wp8'):.4f} vs "
          f"{g('excess_L1_norm','per-step displacement (dy'):.4f} (normalised units)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
