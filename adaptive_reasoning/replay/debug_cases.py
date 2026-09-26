"""
debug_cases.py — pull the evidence for heatmap cases into one folder.

For each case (default: the FAILING ones) of one strategy and condition it
copies the payload folder (img0.jpg = exactly what the model saw, prompt.txt,
answer.json) into <out>/<image>__<goal>/ and writes <out>/SUMMARY.txt with,
per case: expected, got, chosen plate text, relevance, and the model's
reasoning.  Open the folder in VS Code and look at img0.jpg next to the answer.

  python -m adaptive_reasoning.replay.debug_cases --ledger $SCRATCH/ar_eval/heatmap_v2/ledger.jsonl \\
      --strategy signway --condition numeric_sign --out ~/SignWay/figs/debug_numeric [--all]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--strategy", default="signway")
    ap.add_argument("--condition", default="numeric_sign")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="include correct cases too")
    a = ap.parse_args()

    rows = {}
    for line in open(a.ledger):
        r = json.loads(line)
        if r.get("eval") == f"heatmap_{a.strategy}" and r.get("condition") == a.condition:
            rows[(r["image"], r["goal"])] = r          # last row per case wins
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines, n_bad = [], 0
    for (img, goal), r in sorted(rows.items()):
        got = r["direction"] if r["applicable"] else "not_applicable"
        ok = got == r["expected"]
        if ok and not a.all:
            continue
        n_bad += not ok
        d = out / f"{Path(img).stem}__{goal.replace(' ', '_')}"
        d.mkdir(exist_ok=True)
        pd = r.get("payload_dir")
        if pd and Path(pd).exists():
            for f in Path(pd).iterdir():
                shutil.copy(f, d / f.name)
        (d / "case.json").write_text(json.dumps(r, indent=2))
        raw = (r.get("raw") or "").replace("\n", " ")[:300]
        lines.append(f"{'OK ' if ok else 'XX '}{img:26s} goal={goal:10s} exp={r['expected']:11s} got={got:14s}\n"
                     f"     plate: {r.get('plate','')[:90]}  (R={r.get('R')})\n"
                     f"     hint : {(r.get('hint') or '')[:110]}\n"
                     f"     model: {raw}\n")
    (out / "SUMMARY.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n{n_bad} failing cases → {out}/  (open img0.jpg next to case.json)")


if __name__ == "__main__":
    main()
