"""
memory_recognition.py — does the semantic memory recognise a sign it has seen?

Pairs CSV (columns: first_image, image, same_sign, goal[optional]):
  first_image  first encounter (enrolled into memory)
  image        a second view — of the SAME sign (same_sign=1: different distance/angle/
               time) or of a DIFFERENT sign (same_sign=0: the confusion test)
  goal         optional; if given, the goal-relevant plate is used, else the most
               directory-like plate in the frame

For every pair: docTR on both frames, the chosen plate is enrolled in a Memory,
then "recognised" = Memory.is_novel(second view) is False — i.e. the DEPLOYED
rule (digit-signature overlap >= 0.5), so the number is what the robot would do.
  recall    = recognised / same_sign pairs        (a miss costs one extra VLM call)
  precision = 1 - false matches / different pairs (a false match suppresses a NEEDED call)
Writes results.csv with every pair and the two OCR texts, so misses can be inspected.

  python -m adaptive_reasoning.replay.memory_recognition --csv heatmap/memory_pairs.csv \\
      --images-root figs --out $SCRATCH/ar_eval/memory
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from ..evidence.buffer import stable_plate_id
from ..evidence.detect import DetectConfig, detect
from ..evidence.relevance import Goal, relevance_lines
from ..memory import Memory, MemoryUpdate


def pick_plate(plates, goal_text: str | None):
    if not plates:
        return None
    if goal_text:
        goal = Goal.parse(goal_text)
        scored = [(relevance_lines([l.text for l in p.lines], goal).score, p) for p in plates]
        if max(s for s, _ in scored) >= 0.5:
            return max(scored, key=lambda t: t[0])[1]
    def rank(p):
        ranges = len(re.findall(r"\d[\d-]*\s*(?:to|-|–)\s*\d[\d-]*", p.text, flags=re.I))
        area = ((p.box[2] - p.box[0]) * (p.box[3] - p.box[1])) if p.box else 0.0
        return (ranges, len(p.lines), area)
    return max(plates, key=rank)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images-root", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from PIL import Image
    from ..evidence.detect import DocTREngine
    engine, cfg = DocTREngine(), DetectConfig()
    root, out = Path(a.images_root), Path(a.out); out.mkdir(parents=True, exist_ok=True)

    det = {}
    def plates_of(rel):
        if rel not in det:
            img = np.asarray(Image.open(root / rel).convert("RGB"))
            det[rel] = detect(img, engine, cfg)
        return det[rel]

    rows, res = list(csv.DictReader(open(a.csv))), []
    for r in rows:
        same = r["same_sign"].strip() == "1"
        p1 = pick_plate(plates_of(r["first_image"].strip()), r.get("goal", "").strip() or None)
        p2 = pick_plate(plates_of(r["image"].strip()), r.get("goal", "").strip() or None)
        id1 = stable_plate_id(p1.text) if p1 else None
        id2 = stable_plate_id(p2.text) if p2 else None
        mem = Memory(r.get("goal", "") or "test")          # the deployed novelty rule
        if id1:
            mem.apply(MemoryUpdate(id1, "vlm", None, 0))
        recognised = bool(id1 and id2 and not mem.is_novel(id2))
        res.append({"first_image": r["first_image"], "image": r["image"], "same_sign": int(same),
                    "recognised": int(recognised), "id_first": id1, "id_second": id2,
                    "text_first": (p1.text if p1 else "")[:80], "text_second": (p2.text if p2 else "")[:80],
                    "outcome": ("hit" if recognised else "miss") if same else ("FALSE MATCH" if recognised else "ok")})
    same_rows = [x for x in res if x["same_sign"]]; diff_rows = [x for x in res if not x["same_sign"]]
    recall = sum(x["recognised"] for x in same_rows) / max(len(same_rows), 1)
    fm = sum(x["recognised"] for x in diff_rows)
    precision = 1 - fm / max(len(diff_rows), 1)
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    summary = {"same_sign_pairs": len(same_rows), "recognised": sum(x["recognised"] for x in same_rows),
               "recall": recall, "different_sign_pairs": len(diff_rows), "false_matches": fm, "precision": precision}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"same-sign pairs: {len(same_rows)}  recognised {summary['recognised']}  → recall {100*recall:.0f}%")
    print(f"different-sign pairs: {len(diff_rows)}  false matches {fm}  → precision {100*precision:.0f}%")
    for x in res:
        if x["outcome"] in ("miss", "FALSE MATCH"):
            print(f"  {x['outcome']:11s} {x['first_image'][:24]:24s} vs {x['image'][:24]:24s} | '{x['text_first'][:35]}' | '{x['text_second'][:35]}'")
    print(f"wrote {out}/results.csv, summary.json")


if __name__ == "__main__":
    main()
