"""
check_flip.py — is the annotated `flip_frame` when the sign really becomes usable?

Motivation (Aug 2026): the 25 new approaches carry flip→junction margins of
~2.8 s median, versus ~7.2 s on the older Keller set — even though the robot
was driven SLOWER, which should have made margins longer.  Suspicion: the new
rows reused the VLA `legible_frame` (a prompt-scheduling label) rather than
"first frame a reader could extract the direction".

This script answers that with machine evidence instead of re-judging by eye.
From each bag's E1 dump it reports the first processed frame at which:
  t_R      any plate's relevance crosses --r-min (the goal's sign is FOUND),
  t_stable that holds for --stable consecutive processed frames (no flukes),
  t_q      R·ℓ crosses --tau (the gate's own firing condition),
and compares those to the annotated flip.  Interpretation:

  t_stable EARLIER than annotated flip  → annotation is conservative; the
      usable margin is larger than the CSV implies (re-annotate or redefine).
  t_stable ≈ annotated flip             → annotation is right and the tight
      margins are a real property of these corridors (a finding, not a bug).

ℓ is uncalibrated at time of writing, so treat t_q as indicative; t_R and
t_stable depend only on OCR + relevance and are solid.

Usage:
  python -m adaptive_reasoning.replay.check_flip --dumps $SCRATCH/ar_replay \
      --csv adaptive_reasoning/annotations/ar_extra-aug-20.csv
  # single bag:
  python -m adaptive_reasoning.replay.check_flip --dumps $SCRATCH/ar_replay \
      --csv ... --only rosbag2_2026_08_20-17_12_19 --verbose
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import numpy as np


def first_cross(series: np.ndarray, thr: float, stable: int = 1) -> int | None:
    """First index where series >= thr for `stable` consecutive samples."""
    ok = series >= thr
    if stable <= 1:
        idx = np.flatnonzero(ok)
        return int(idx[0]) if idx.size else None
    run = 0
    for i, v in enumerate(ok):
        run = run + 1 if v else 0
        if run >= stable:
            return i - stable + 1
    return None


def analyze(npz_path: Path, r_min: float, tau: float, stable: int, verbose: bool):
    d = np.load(npz_path, allow_pickle=True)
    if "junction_frame" not in d:
        return None
    T = len(d["frame_idx"])
    fps = float(d["fps"]) if "fps" in d else 10.0
    junction = int(d["junction_frame"])
    flip = int(d["flip_frame"]) if "flip_frame" in d else None

    best_R = np.zeros(T)
    best_q = np.zeros(T)
    labels: list[str] = []
    i = 0
    while f"plate{i}_id" in d:
        R, ell = d[f"plate{i}_R"], d[f"plate{i}_ell"]
        best_R = np.maximum(best_R, R)
        best_q = np.maximum(best_q, R * ell)
        if R.max() >= r_min:
            labels.append(str(d[f"plate{i}_label"])[:38])
        i += 1

    t_R = first_cross(best_R, r_min, 1)
    t_stable = first_cross(best_R, r_min, stable)
    t_q = first_cross(best_q, tau, 1)
    if verbose:
        print(f"    relevant tracks: {labels[:4]}")
    return {
        "bag": npz_path.stem, "T": T, "fps": fps, "junction": junction, "flip": flip,
        "t_R": t_R, "t_stable": t_stable, "t_q": t_q,
        "ann_margin": (junction - flip) / fps if flip is not None else None,
        "ocr_margin": (junction - t_stable) / fps if t_stable is not None else None,
        "q_margin": (junction - t_q) / fps if t_q is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--r-min", type=float, default=0.9, help="relevance counting as 'found'")
    ap.add_argument("--tau", type=float, default=0.55)
    ap.add_argument("--stable", type=int, default=3, help="consecutive frames required")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    rows = {r["bag"]: r for r in csv.DictReader(open(a.csv))}
    out = []
    for bag, row in rows.items():
        if a.only and bag != a.only:
            continue
        f = Path(a.dumps) / f"{bag}.npz"
        if not f.exists():
            continue
        if a.verbose:
            print(f"{bag}  goal={row['goal']}")
        res = analyze(f, a.r_min, a.tau, a.stable, a.verbose)
        if res:
            res["goal"] = row["goal"]
            res["building"] = row.get("building", "")
            out.append(res)

    if not out:
        raise SystemExit("no dumps matched")

    print(f"\n{'bag':34s} {'bld':7s} {'ann':>6s} {'ocr':>6s} {'q':>6s}  {'Δ(ocr-ann)':>10s}")
    for r in sorted(out, key=lambda r: r["bag"]):
        am = f"{r['ann_margin']:.1f}" if r["ann_margin"] is not None else "-"
        om = f"{r['ocr_margin']:.1f}" if r["ocr_margin"] is not None else "none"
        qm = f"{r['q_margin']:.1f}" if r["q_margin"] is not None else "none"
        dl = (f"{r['ocr_margin'] - r['ann_margin']:+.1f}s"
              if None not in (r["ocr_margin"], r["ann_margin"]) else "")
        print(f"{r['bag'][:34]:34s} {r['building'][:7]:7s} {am:>6s} {om:>6s} {qm:>6s}  {dl:>10s}")

    def med(key, subset=None):
        vals = [r[key] for r in (subset or out) if r[key] is not None]
        return st.median(vals) if vals else float("nan")

    print(f"\nmedian annotated margin : {med('ann_margin'):.1f}s")
    print(f"median OCR-found margin : {med('ocr_margin'):.1f}s   "
          f"(sign FOUND this early, {a.stable} consecutive frames, R>={a.r_min})")
    print(f"median q>tau margin     : {med('q_margin'):.1f}s   (uncalibrated ℓ — indicative)")
    found = sum(r["ocr_margin"] is not None for r in out)
    print(f"goal's sign found by OCR in {found}/{len(out)} approaches")
    for b in sorted({r["building"] for r in out if r["building"]}):
        sub = [r for r in out if r["building"] == b]
        print(f"  {b:8s} n={len(sub):2d}  ann {med('ann_margin', sub):.1f}s  "
              f"ocr {med('ocr_margin', sub):.1f}s")


if __name__ == "__main__":
    main()
