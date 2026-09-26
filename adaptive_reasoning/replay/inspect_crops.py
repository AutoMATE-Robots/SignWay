"""
inspect_crops.py — look at exactly the crops the reasoner is being asked about.

Builds one contact sheet per bag from the GOAL-RELEVANT stored crops (R ≥ 0.5),
sorted by legibility, labelled with frame and ℓ.  Use it to answer "are our
crops the whole sign, or a fragment?" without opening 200 files.

  python -m adaptive_reasoning.replay.inspect_crops --dumps $SCRATCH/ar_replay \\
      --csv adaptive_reasoning/annotations/ar_extra-sep-4.csv --out figs/crop_sheets \\
      --top 12 [--only rosbag2-meche-1]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


def sheet(dumps: Path, bag: str, out: Path, top: int, cols: int = 4, cell: int = 420) -> int:
    f = dumps / f"{bag}.jsonl"
    if not f.exists():
        return 0
    recs = [json.loads(l) for l in open(f)]
    recs = [r for r in recs if r.get("crop") and r["R"] >= 0.5]
    if not recs:
        return 0
    recs.sort(key=lambda r: -r["ell"])
    seen, picked = set(), []
    for r in recs:
        if r["frame"] in seen:
            continue
        picked.append(r); seen.add(r["frame"])
        if len(picked) >= top:
            break
    rows = (len(picked) + cols - 1) // cols
    im = Image.new("RGB", (cols * cell, rows * (cell + 22)), (24, 24, 24))
    d = ImageDraw.Draw(im)
    for i, r in enumerate(picked):
        c = Image.open(dumps / f"{bag}_crops" / r["crop"]).convert("RGB")
        c.thumbnail((cell - 8, cell - 8))
        x, y = (i % cols) * cell, (i // cols) * (cell + 22)
        im.paste(c, (x + 4, y + 4))
        d.text((x + 6, y + cell + 2), f"f{r['frame']}  l={r['ell']:.2f}  {r['text'][:38]}",
               fill=(210, 210, 210))
    out.mkdir(parents=True, exist_ok=True)
    im.save(out / f"{bag}_goalcrops.jpg", quality=88)
    return len(picked)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True); ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="figs/crop_sheets")
    ap.add_argument("--top", type=int, default=12); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    for row in csv.DictReader(open(a.csv)):
        if a.only and row["bag"] != a.only:
            continue
        n = sheet(Path(a.dumps), row["bag"], Path(a.out), a.top)
        if n:
            print(f"{row['bag']:24s} goal={row['goal']!r:28s} {n} crops -> {a.out}/{row['bag']}_goalcrops.jpg")


if __name__ == "__main__":
    main()
