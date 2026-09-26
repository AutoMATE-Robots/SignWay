"""
gen_memory_pairs.py — build memory_pairs.csv automatically from the replay dumps.

Same-sign pairs: for every plate with >=2 stored crops, take the FIRST and LAST
stored crop (typically the farthest and nearest views of the same physical sign).
Different-sign pairs: random cross-plate crop pairs within the same building.
No human pairing needed.  Output feeds memory_recognition.py.

  python -m adaptive_reasoning.replay.gen_memory_pairs --dumps $SCRATCH/ar_replay \\
      --out heatmap/memory_pairs.csv [--max-diff 25]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def building(bag: str) -> str:
    parts = bag.split("-")
    return parts[1] if len(parts) > 2 else bag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-diff", type=int, default=25)
    ap.add_argument("--min-gap-frames", type=int, default=30,
                    help="same-sign pair needs this many frames between the two crops")
    a = ap.parse_args()
    crops = defaultdict(list)                     # (bag, plate_id) -> [(frame, path)]
    for f in sorted(Path(a.dumps).glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("crop") and Path(r["crop"]).exists():
                crops[(f.stem, r["plate_id"])].append((int(r["frame"]), r["crop"]))
    same, plates = [], []
    for (bag, pid), lst in crops.items():
        lst.sort()
        plates.append((bag, pid, lst[-1][1]))
        if len(lst) >= 2 and lst[-1][0] - lst[0][0] >= a.min_gap_frames:
            same.append({"first_image": lst[0][1], "image": lst[-1][1], "same_sign": 1, "goal": ""})
    rng = random.Random(0)
    diff = []
    by_b = defaultdict(list)
    for bag, pid, path in plates:
        by_b[building(bag)].append((pid, path))
    for b, lst in by_b.items():
        rng.shuffle(lst)
        for (p1, c1), (p2, c2) in zip(lst[::2], lst[1::2]):
            if p1 != p2:
                diff.append({"first_image": c1, "image": c2, "same_sign": 0, "goal": ""})
    diff = diff[: a.max_diff]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["first_image", "image", "same_sign", "goal"])
        w.writeheader(); w.writerows(same + diff)
    print(f"{len(same)} same-sign pairs (first vs last stored crop), {len(diff)} different-sign pairs -> {out}")
    print("run: python -m adaptive_reasoning.replay.memory_recognition --csv", out, "--images-root / --out $SCRATCH/ar_eval/memory")


if __name__ == "__main__":
    main()
