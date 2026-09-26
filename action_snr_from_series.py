#!/usr/bin/env python3
"""action_snr.py's analysis, computed from eval series.npy instead of RLDS.

The RLDS build was purged from scratch and the source bags are no longer local,
so the dataset cannot be rebuilt tonight. But the quantity action_snr.py measures
is a property of the ACTION REPRESENTATION, not of a particular split, and every
series.npy already stores the ground-truth action for each frame:

    actual_traj[i]  = (8, 2) cumulative waypoints  == the deployed action
    actual_traj[i][0] = per-step displacement      == the v2 representation

Identical maths to action_snr.py, different source. State in the paper that the
statistic was computed over the held-out evaluation bags rather than the training
split -- it is a claim about the parameterisation, and the bags are real
trajectories from the same robot and corridors.

  python action_snr_from_series.py --sweep $S --ckpt v8@9k --out snr.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_path import discover, load_series, _sort_key            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--bags", default=None)
    ap.add_argument("--turn-th", type=float, default=0.03,
                    help="|wp8 lateral| [m] marking a turn frame")
    ap.add_argument("--out", default="snr.csv")
    a = ap.parse_args()

    paths = discover(a.sweep)
    ckpts = sorted({c for c, _ in paths}, key=_sort_key)
    ckpt = a.ckpt or (ckpts[0] if len(ckpts) == 1 else None)
    if ckpt is None:
        ap.error(f"several checkpoints; pick one with --ckpt: {ckpts}")
    bags = ([b.strip() for b in a.bags.split(",")] if a.bags
            else sorted({b for c, b in paths if c == ckpt}, key=_sort_key))

    chunks = []
    for b in bags:
        if (ckpt, b) not in paths:
            print(f"  [skip] {b}")
            continue
        at = np.asarray(load_series(paths[(ckpt, b)])["actual_traj"],
                        dtype=np.float64)
        chunks.append(at)
        print(f"  {b}: {len(at)} steps")
    if not chunks:
        raise SystemExit("no series found")
    wp = np.concatenate(chunks, axis=0)                 # (N, 8, 2)

    lat8 = wp[:, -1, 1]
    turn = np.abs(lat8) > a.turn_th
    print(f"\n{len(wp)} steps | turn frames: {turn.sum()} "
          f"({100 * turn.mean():.1f}%)")

    rows = []

    def add(name, v):
        if turn.sum() < 5 or (~turn).sum() < 5:
            return
        mag = np.abs(v[turn]).mean()
        noise = np.abs(v[~turn] - np.median(v[~turn])).mean()
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        vn = 2 * (v - lo) / max(hi - lo, 1e-12) - 1
        shortcut = np.abs(vn - np.median(vn)).mean()
        rows.append({"representation": name, "turn_magnitude_m": mag,
                     "straight_noise_m": noise,
                     "snr_vs_noise": mag / (noise + 1e-12),
                     "excess_L1_norm": shortcut,
                     "turn_frac": float(turn.mean())})

    add("cumulative waypoint (wp8 lateral)", lat8)
    add("cumulative waypoint (wp4 lateral)", wp[:, 3, 1])
    add("per-step displacement (dy)", wp[:, 0, 1])
    add("per-step displacement (dx, forward)", wp[:, 0, 0])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'representation':38s} {'|turn| m':>9s} {'noise m':>9s} "
          f"{'SNR':>7s} {'excess L1':>10s}")
    for r in rows:
        print(f"{r['representation']:38s} {r['turn_magnitude_m']:9.4f} "
              f"{r['straight_noise_m']:9.4f} {r['snr_vs_noise']:7.2f} "
              f"{r['excess_L1_norm']:10.4f}")

    def g(k, pre):
        return next(r[k] for r in rows if r["representation"].startswith(pre))

    wpm = g("turn_magnitude_m", "cumulative waypoint (wp8")
    psm = g("turn_magnitude_m", "per-step displacement (dy")
    print(f"\nturn magnitude : waypoint {wpm:.4f} m vs per-step {psm:.4f} m "
          f"({wpm / max(psm, 1e-12):.0f}x)")
    print(f"SNR vs noise   : {g('snr_vs_noise','cumulative waypoint (wp8'):.2f}"
          f" vs {g('snr_vs_noise','per-step displacement (dy'):.2f}")
    print(f"excess L1 of the image-ignoring predictor: "
          f"{g('excess_L1_norm','cumulative waypoint (wp8'):.4f} vs "
          f"{g('excess_L1_norm','per-step displacement (dy'):.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
