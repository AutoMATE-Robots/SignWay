#!/usr/bin/env python3
# one-off: post-ingest cache integrity check (truncation, silent poisoning), used 2026-08-08 before annotating, kept for provenance
r"""
validate_caches.py -- run AFTER ingest_r2.sh, BEFORE annotating or building.

The ingest can only catch download and read failures. These are the problems it
cannot see, each of which silently poisons training if it reaches the dataset:

  1. TRUNCATED CACHE     interrupted write -> npz won't load (or loads short)
  2. DUPLICATE BAGS      same recording uploaded to two day folders -> that
                         trajectory gets double weight in training
  3. BAD ODOM            frozen (robot "never moves") or jumpy (teleport) odometry.
                         Frames still look fine, so nothing upstream complains, but
                         every waypoint derived from it is wrong.
  4. TIME PROBLEMS       non-monotonic timestamps (clock jump), or odom that does
                         not cover the frame time range (first frames extrapolated)
  5. FPS / STRIDE        bags at different frame rates need different stride to land
                         at the same dataset Hz. Groups them so the BAGS list is right.
  6. TOO SHORT           fewer frames than horizon*stride -> contributes ZERO steps,
                         silently absent from training

RUN:
    python validate_caches.py --cache-dir $SCRATCH/bag_cache
    python validate_caches.py --cache-dir $SCRATCH/bag_cache --horizon 8 --stride 3
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3, help="planned stride, for the length check")
    ap.add_argument("--max-jump", type=float, default=0.5,
                    help="metres between consecutive frames that counts as an odom jump")
    a = ap.parse_args()

    files = sorted(Path(a.cache_dir).glob("*.npz"))
    partials = sorted(Path(a.cache_dir).glob("*.npz.partial"))
    if not files:
        print(f"no caches in {a.cache_dir}")
        sys.exit(1)
    print(f"{len(files)} caches in {a.cache_dir}")
    if partials:
        print(f"WARNING: {len(partials)} .partial files -- interrupted writes, "
              f"delete them and re-run ingest for those bags")

    bad, warn = [], []
    sigs = defaultdict(list)          # duplicate detection
    fps_groups = defaultdict(list)
    rows = []

    for f in files:
        try:
            d = np.load(f)
            t, imgs, poses = d["times"], d["images"], d["poses"]
            fps = float(d["fps"]) if "fps" in d else 0.0
        except Exception as e:
            bad.append(f"{f.name}: UNREADABLE ({type(e).__name__}) -- delete and re-ingest")
            continue

        n = len(t)
        if n != len(imgs) or n != len(poses):
            bad.append(f"{f.name}: length mismatch t={n} img={len(imgs)} pose={len(poses)}")
            continue

        # --- 6. too short to yield any step ---
        need = a.stride * a.horizon + 1
        if n < need:
            bad.append(f"{f.name}: only {n} frames, needs >{need} at stride {a.stride} "
                       f"-> would contribute ZERO steps")

        # --- 4. time problems ---
        dt = np.diff(t)
        if n > 1 and dt.min() <= 0:
            bad.append(f"{f.name}: non-monotonic timestamps (min dt={dt.min():.4f}s)")
        if n > 1 and dt.max() > 1.0:
            warn.append(f"{f.name}: {int((dt > 1.0).sum())} gap(s) >1s in the frame stream "
                        f"(max {dt.max():.1f}s) -- dropped frames?")

        # --- 3. bad odom ---
        xy = poses[:, :2]
        step = np.hypot(*np.diff(xy, axis=0).T) if n > 1 else np.zeros(1)
        total = float(step.sum())
        if total < 0.5:
            bad.append(f"{f.name}: odom moves only {total:.2f}m over {n} frames "
                       f"-> FROZEN odometry, waypoints would be ~zero")
        njump = int((step > a.max_jump).sum())
        if njump:
            bad.append(f"{f.name}: {njump} odom jump(s) >{a.max_jump}m between frames "
                       f"(max {step.max():.2f}m) -> teleport/glitch")
        yaw = poses[:, 2]
        dyaw = np.abs(np.arctan2(np.sin(np.diff(yaw)), np.cos(np.diff(yaw))))
        if n > 1 and float(dyaw.max()) > 1.0:
            warn.append(f"{f.name}: yaw jumps >1 rad between frames -- check wraparound")

        # --- 2. duplicate signature: start time + frame count + first-frame content ---
        sig = (round(float(t[0]), 3), n, int(imgs[0].astype(np.int64).sum()))
        sigs[sig].append(f.name)

        # --- 5. fps grouping ---
        fps_groups[round(fps)].append(f.name)

        rows.append((f.name, n, fps, total))

    # ---- duplicates ----
    dupes = {k: v for k, v in sigs.items() if len(v) > 1}

    # ---- report ----
    print()
    print("fps groups (each needs its own stride to reach the same dataset Hz):")
    for fps, names in sorted(fps_groups.items()):
        s3 = fps / 3.0
        s2 = fps / 2.0
        print(f"  ~{fps} fps : {len(names):3d} bags   stride 2 -> {s2:.1f}Hz | stride 3 -> {s3:.1f}Hz")

    dur = [n / f for _, n, f, _ in rows if f > 0]
    dist = [d for *_, d in rows]
    if dur:
        print(f"\nduration : min {min(dur):.0f}s  median {np.median(dur):.0f}s  max {max(dur):.0f}s")
        print(f"path len : min {min(dist):.1f}m  median {np.median(dist):.1f}m  max {max(dist):.1f}m")

    if dupes:
        print(f"\nDUPLICATES ({len(dupes)} groups) -- same recording cached twice, "
              f"would double-weight in training:")
        for names in dupes.values():
            print("   " + "  ==  ".join(names))

    if warn:
        print(f"\nWARNINGS ({len(warn)}):")
        for w in warn:
            print("   " + w)

    if bad:
        print(f"\nPROBLEMS ({len(bad)}) -- exclude these from the BAGS list:")
        for b in bad:
            print("   " + b)

    print()
    if bad or dupes:
        print("*** NOT CLEAN -- resolve the above before building ***")
        sys.exit(1)
    print(f"*** {len(rows)} caches clean ***")


if __name__ == "__main__":
    main()
