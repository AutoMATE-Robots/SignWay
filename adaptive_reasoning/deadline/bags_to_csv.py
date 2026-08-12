#!/usr/bin/env python3
"""
bags_to_csv.py -- regenerate the annotations CSV from the BAGS list in
tfds_builder.py, keeping ONE source of truth for bag annotations.

Parses the builder file with `ast` (no import, so tensorflow_datasets is never
loaded) and writes bag,split,flip_frame,decision,turn_done_frame,stride rows.

    python -m adaptive_reasoning.deadline.bags_to_csv \
        --builder ~/SignWay/signway_dataset/tfds_builder.py \
        --out ~/SignWay/data/annotations_v8.csv
"""
from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter
from pathlib import Path


def extract_bags(builder_path: Path):
    tree = ast.parse(builder_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "BAGS":
                    return ast.literal_eval(node.value)
    raise SystemExit(f"no BAGS assignment found in {builder_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builder", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bags = extract_bags(Path(args.builder).expanduser())
    with open(Path(args.out).expanduser(), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bag", "split", "flip_frame", "decision",
                    "turn_done_frame", "stride"])
        for split, bag, flip, dec, td, st in bags:
            w.writerow([bag, split, flip, dec, "" if td is None else td, st])

    c = Counter(b[3] for b in bags)
    flip0 = sum(1 for b in bags if b[2] == 0 and b[3] != "straight")
    print(f"{len(bags)} rows -> {args.out}")
    print(f"decisions: {dict(c)} | flip=0 turn bags (short approaches): {flip0}")


if __name__ == "__main__":
    main()
