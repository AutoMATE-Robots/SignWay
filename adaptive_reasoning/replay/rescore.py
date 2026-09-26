"""
rescore.py — recompute relevance for existing dumps without touching a bag.

Why: the dump stores every plate's OCR text per frame, and relevance is pure
text math.  So when the parser improves (Sept 10: "Rooms 43-58" was read as a
floor code), or an annotation's goal changes, R can be regenerated from the
jsonl in seconds — no raws, no OCR, no GPU.  Crops, φ, ℓ and scene grids are
untouched.

Rewrites <bag>.jsonl (R, R_struct fields) and <bag>.npz (plate{i}_R,
plate{i}_R_struct arrays; everything else copied).  Backs up the originals to
<bag>.jsonl.bak / <bag>.npz.bak on first run.

Usage:
  python -m adaptive_reasoning.replay.rescore --dumps $SCRATCH/ar_replay \
      --csv adaptive_reasoning/annotations/ar_extra-sep-4.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import warnings
from pathlib import Path

import numpy as np

from ..evidence.relevance import Goal, relevance_lines


def rescore_bag(dumps: Path, bag: str, goal_text: str) -> tuple[int, int, int]:
    jl, nz = dumps / f"{bag}.jsonl", dumps / f"{bag}.npz"
    if not (jl.exists() and nz.exists()):
        return 0, 0, 0
    for f in (jl, nz):
        bak = f.with_suffix(f.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(f, bak)
    goal = Goal.parse(goal_text)
    recs = [json.loads(l) for l in open(jl)]
    changed = 0
    for r in recs:
        lines = [s.strip() for s in r["text"].split("|")]
        rel = relevance_lines(lines, goal)
        r_new = float(rel.score)
        s_new = float(max((ln.struct for ln in rel.lines), default=0.0))
        if abs(r_new - r["R"]) > 1e-9 or abs(s_new - r.get("R_struct", 0.0)) > 1e-9:
            changed += 1
        r["R"], r["R_struct"] = r_new, s_new
    with open(jl, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    d = dict(np.load(nz, allow_pickle=True))
    grid = {int(f): j for j, f in enumerate(d["frame_idx"])}
    pid_to_i, i = {}, 0
    while f"plate{i}_id" in d:
        pid_to_i[str(d[f"plate{i}_id"])] = i
        i += 1
    for k in pid_to_i.values():
        d[f"plate{k}_R"] = np.zeros_like(d[f"plate{k}_R"])
        d[f"plate{k}_R_struct"] = np.zeros_like(d[f"plate{k}_R_struct"])
    miss = 0
    for r in recs:
        k = pid_to_i.get(r["plate_id"]); j = grid.get(int(r["frame"]))
        if k is None or j is None:
            miss += 1; continue
        d[f"plate{k}_R"][j] = r["R"]
        d[f"plate{k}_R_struct"][j] = r["R_struct"]
    d["goal"] = np.array(goal_text)
    np.savez(nz, **d)
    return len(recs), changed, miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    tot = tot_changed = 0
    for row in csv.DictReader(open(a.csv)):
        if a.only and row["bag"] != a.only:
            continue
        n, ch, miss = rescore_bag(Path(a.dumps), row["bag"], row["goal"])
        if n:
            tot += n; tot_changed += ch
            print(f"{row['bag']:24s} goal={row['goal']!r:34s} records={n:5d} changed={ch:4d}"
                  + (f"  (unmapped {miss})" if miss else ""))
    print(f"\nrescored {tot} records, {tot_changed} changed")


if __name__ == "__main__":
    main()
