#!/usr/bin/env python3
# one-off: built the annotation team's CSV sheet from the .npz caches, used 2026-08-08, kept for provenance
r"""
make_annotation_sheet.py -- build the sheet the annotation team fills in.

Reads every .npz cache and writes a CSV with the bag facts already filled in
(frames, fps, duration, the stride this bag needs) plus EMPTY columns for the
annotators: flip_frame, decision, turn_done_frame, notes.

Why stride is per-bag: the dataset must land at ~10 Hz. The 20 fps bags need
stride 2; the ~31 fps bags need stride 3. Getting this wrong for a bag silently
distorts every waypoint magnitude in it -- the model would see that bag as if the
robot moved at 1.5x speed.

RUN:
    python make_annotation_sheet.py --cache-dir $SCRATCH/bag_cache \
        --out $SCRATCH/annotation_sheet.csv
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-hz", type=float, default=10.0)
    a = ap.parse_args()

    files = sorted(Path(a.cache_dir).glob("*.npz"))
    if not files:
        print("no caches found"); return

    def sort_key(p):
        s = p.stem.replace("rosbag2-keller-", "")
        return (0 if s.startswith("t") else 1, int(s[1:]) if s[1:].isdigit() else 0, s)

    rows = []
    strides = Counter()
    for f in sorted(files, key=sort_key):
        d = np.load(f)
        t = d["times"]
        n = len(t)
        fps = float(d["fps"]) if "fps" in d else (
            1.0 / float(np.median(np.diff(t))) if n > 1 else 0.0)
        stride = max(1, int(round(fps / a.target_hz)))
        strides[stride] += 1
        rows.append({
            "bag": f.stem,
            "frames": n,
            "fps": round(fps, 1),
            "duration_s": round(n / fps, 1) if fps else 0,
            "stride": stride,
            "flip_frame": "",
            "decision": "",
            "turn_done_frame": "",
            "notes": "",
        })

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} bags -> {a.out}")
    print("\nstride groups (target {:.0f} Hz):".format(a.target_hz))
    for s, c in sorted(strides.items()):
        ex = [r["bag"] for r in rows if r["stride"] == s][:3]
        f_ex = [r["fps"] for r in rows if r["stride"] == s][:1]
        print(f"  stride {s}: {c:3d} bags  (~{f_ex[0]} fps)  e.g. {', '.join(ex)}")

    dur = [r["duration_s"] for r in rows]
    print(f"\nduration: min {min(dur):.0f}s  median {np.median(dur):.0f}s  max {max(dur):.0f}s")
    print(f"total footage: {sum(dur)/60:.0f} min")


if __name__ == "__main__":
    main()
