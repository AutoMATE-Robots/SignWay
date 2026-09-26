#!/usr/bin/env python3
"""Export eval series.npy to CSV so the figure can be rebuilt anywhere.

  python export_csv.py --sweep $S --ckpt v8@9k --bags e2,e3,17_27_33 --out csv/
  python export_csv.py --sweep $S --ckpt v8@9k --full --out csv/   # + all 8 wps

Per-frame columns (one file per bag):
  frame            raw frame index in the bag (native fps)
  t_s              seconds from bag start (dataset runs at 10 Hz)
  s_m              distance travelled along the reference path, metres
  ref_x, ref_y     reference (human) pose in world metres, x forward, y left
  ref_yaw_deg      reference heading, degrees
  vla_x, vla_y     VLA predicted endpoint, anchored at that frame's reference
                   pose and transformed to world metres
  lat_err_cm       lateral error, robot frame (VLA endpoint - human endpoint)
  lon_err_cm       longitudinal error, robot frame
  phase            approach / turn / straight
With --full, also wp1_lat_cm .. wp8_lat_cm and wp1_lon_cm .. wp8_lon_cm:
per-waypoint errors across the 0.8 s prediction horizon.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_path import (discover, load_series, reference_path, endpoint_locus,
                      phases, _sort_key)                      # noqa: E402

HZ = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--bags", default=None)
    ap.add_argument("--full", action="store_true",
                    help="also write per-waypoint errors")
    ap.add_argument("--out", default="csv")
    args = ap.parse_args()

    paths = discover(args.sweep)
    ckpts = sorted({c for c, _ in paths}, key=_sort_key)
    ckpt = args.ckpt or (ckpts[0] if len(ckpts) == 1 else None)
    if ckpt is None:
        ap.error(f"pick one with --ckpt: {ckpts}")
    bags = ([b.strip() for b in args.bags.split(",")] if args.bags
            else sorted({b for c, b in paths if c == ckpt}, key=_sort_key))

    os.makedirs(args.out, exist_ok=True)
    for bag in bags:
        if (ckpt, bag) not in paths:
            print(f"  [skip] no series for {ckpt}/{bag}")
            continue
        d = load_series(paths[(ckpt, bag)])
        ref, yaws = reference_path(np.asarray(d["actual_traj"]))
        loc = endpoint_locus(np.asarray(d["pred_traj"]), ref, yaws)
        frames = np.asarray(d["frames"])
        pt = np.asarray(d["pred_traj"])
        at = np.asarray(d["actual_traj"])
        step = np.linalg.norm(np.diff(ref, axis=0), axis=1)
        sdist = np.concatenate([[0.0], np.cumsum(step)])
        ph = phases(d)
        lab = np.array(["" for _ in frames], dtype=object)
        for name, mask in ph:
            lab[mask] = name

        head = ["frame", "t_s", "s_m", "ref_x", "ref_y", "ref_yaw_deg",
                "vla_x", "vla_y", "lat_err_cm", "lon_err_cm", "phase"]
        if args.full:
            head += [f"wp{k}_lat_cm" for k in range(1, pt.shape[1] + 1)]
            head += [f"wp{k}_lon_cm" for k in range(1, pt.shape[1] + 1)]
        p = os.path.join(args.out, f"{bag}.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            for i in range(len(frames)):
                row = [int(frames[i]), round(i / HZ, 3), round(sdist[i], 4),
                       round(ref[i, 0], 4), round(ref[i, 1], 4),
                       round(np.degrees(yaws[i]), 3),
                       round(loc[i, 0], 4), round(loc[i, 1], 4),
                       round((pt[i, -1, 1] - at[i, -1, 1]) * 100, 3),
                       round((pt[i, -1, 0] - at[i, -1, 0]) * 100, 3),
                       lab[i] or "n/a"]
                if args.full:
                    row += [round((pt[i, k, 1] - at[i, k, 1]) * 100, 3)
                            for k in range(pt.shape[1])]
                    row += [round((pt[i, k, 0] - at[i, k, 0]) * 100, 3)
                            for k in range(pt.shape[1])]
                w.writerow(row)
        print(f"wrote {p}  ({len(frames)} rows, phases: "
              f"{', '.join(n for n, _ in ph)})")


if __name__ == "__main__":
    main()
