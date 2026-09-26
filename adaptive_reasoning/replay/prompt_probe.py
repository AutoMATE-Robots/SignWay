"""
prompt_probe.py — can the reasoner do the task at all, and with which prompt?

Takes the K most legible stored crops of the GOAL sign per approach (the
ceiling case: near, sharp), asks each prompt variant, grades against the
annotated decision.  Cheap (a few hundred local calls), decisive: if no variant
reads a room range on a clean crop, the model is the bottleneck, not the gate.

Variants:
  current   the deployed prompt
  rules     + explicit instructions: numeric ranges include every number
            between; match the goal to a line; read THAT line's arrow;
            arrow legend
  reason    rules + a short 'reasoning' field BEFORE the JSON answer
            (small models do markedly better when allowed to think in text)

  python -m adaptive_reasoning.replay.prompt_probe --dumps $SCRATCH/ar_replay \\
      --csv adaptive_reasoning/annotations/ar_extra-sep-4.csv --out $SCRATCH/ar_eval/probe \\
      --model Qwen/Qwen2.5-VL-7B-Instruct --base-url http://localhost:8000/v1 --k 3
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..reasoner import PROMPT, PROMPT_JSON_ONLY, OpenAICompatVLM, Reasoner, VLMAnswer, structural_hint

PROMPT_RULES = """You are the sign-reading module of an indoor robot.
Goal: "{goal}"
Recognized text on the candidate sign (may contain OCR errors):
{plate_texts}
Previously used signs/decisions for this goal: {memory_summary}

Rules:
1. The goal is a room number or a place name.
2. A line like "Rooms 43-58" means EVERY room from 43 to 58, so goal 49 IS on
   that line. "5-117 to 5-196" includes 5-182. Check each line this way.
3. Find the ONE line that contains the goal (as a number inside its range, or
   as a name). If no line contains it, answer applicable=false.
4. Read the arrow printed next to THAT line in the image: ← = turn_left,
   → = turn_right, ↑ = straight, ↗/↖ = straight (then turn later).
5. Answer with JSON ONLY:
{{"applicable": true/false, "direction": "turn_left"|"turn_right"|"straight"|"stop"|null,
  "confidence": 0.0-1.0, "summary": "<which line matched and which arrow it has>"}}"""

PROMPT_REASON = PROMPT_RULES.replace(
    "5. Answer with JSON ONLY:",
    "5. First write 2-4 short lines of reasoning: list each sign line with its arrow, "
    "and state which line contains the goal. Then, on a new line, output the JSON:")

VARIANTS = {"json_only": PROMPT_JSON_ONLY, "rules": PROMPT_RULES, "reason": PROMPT_REASON,
            "deployed": PROMPT}   # deployed = reason + parser hint


def pick_crops(dumps: Path, bag: str, k: int, how: str = "area") -> list[dict]:
    """Which stored crops to test.
      ell    most legible by the PLACEHOLDER ℓ — measured Sept 11 to rank a
             clipped text band above whole-sign crops, so not a good ceiling test
      area   largest payload box (closest/biggest view of the sign) — default
      last   nearest to the junction (latest frames)"""
    f = dumps / f"{bag}.jsonl"
    if not f.exists():
        return []
    recs = [json.loads(l) for l in open(f)]
    recs = [r for r in recs if r.get("crop") and r["R"] >= 0.5]
    if how == "ell":
        recs.sort(key=lambda r: -r["ell"])
    elif how == "last":
        recs.sort(key=lambda r: -r["frame"])
    else:
        def area(r):
            b = r.get("box")
            return (b[2] - b[0]) * (b[3] - b[1]) if b else 0.0
        recs.sort(key=lambda r: -area(r))
    out, seen = [], set()
    for r in recs:                                           # spread over distinct frames
        if r["frame"] in seen:
            continue
        out.append(r); seen.add(r["frame"])
        if len(out) >= k:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True); ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--reasoning", default="none"); ap.add_argument("--rpm", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=3, help="crops per approach")
    ap.add_argument("--pick", default="area", choices=["area", "ell", "last"],
                    help="how to choose them (default: largest sign view)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    reasoning = None if a.reasoning.lower() in ("none", "off", "") else a.reasoning
    client = OpenAICompatVLM(model=a.model, base_url=a.base_url, reasoning_effort=reasoning, rpm=a.rpm)
    from PIL import Image
    rows = list(csv.DictReader(open(a.csv)))
    results = []
    tally = {v: Counter() for v in a.variants}
    for row in rows:
        crops = pick_crops(Path(a.dumps), row["bag"], a.k, a.pick)
        if not crops:
            continue
        for v in a.variants:
            r_ = Reasoner(client, model_tag=f"{a.model}:{a.reasoning}:probe_{v}", cache_dir=out / "vlm_cache",
                          payload_dir=out / "payloads" / v / row["bag"], ledger_path=out / "ledger.jsonl",
                          eval_tag=f"probe_{v}", prompt_template=VARIANTS[v])
            for c in crops:
                img = np.asarray(Image.open(Path(a.dumps) / f"{row['bag']}_crops" / c["crop"]).convert("RGB"))
                ans = r_.ask(row["goal"], [c["text"]], [img], meta={"bag": row["bag"], "frame": c["frame"],
                                                                     "expected": row["decision"], "variant": v},
                             hint=structural_hint(c["text"], row["goal"]))
                got = ans.direction if ans.applicable else "not_applicable"
                ok = got == row["decision"]
                tally[v][("correct" if ok else "wrong")] += 1
                tally[v][f"got={got}"] += 1
                results.append({"bag": row["bag"], "goal": row["goal"], "expected": row["decision"],
                                "variant": v, "frame": c["frame"], "ell": c["ell"], "got": got,
                                "correct": ok, "summary": ans.summary[:120]})
        print(f"{row['bag']:22s} " + "  ".join(
            f"{v}:{sum(1 for x in results if x['bag']==row['bag'] and x['variant']==v and x['correct'])}/{len(crops)}"
            for v in a.variants))
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    print("\nvariant   accuracy   answer distribution")
    for v in a.variants:
        n = tally[v]["correct"] + tally[v]["wrong"]
        dist = {k[4:]: n_ for k, n_ in tally[v].items() if k.startswith("got=")}
        print(f"{v:9s} {tally[v]['correct']}/{n} = {100*tally[v]['correct']/max(n,1):.0f}%   {dist}")
    print(f"wrote {out}/results.csv (+ ledger, payloads)")


if __name__ == "__main__":
    main()
