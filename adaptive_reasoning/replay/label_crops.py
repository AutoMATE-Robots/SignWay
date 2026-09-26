"""
label_crops.py — the ONE VLM pass everything else replays from.

Walks every stored crop in the E1 dumps and asks the reasoner, with the exact
deployed payload (the stored crop IS the payload crop) and the exact deployed
prompt.  Every answer is cached by content and written to an audit ledger, so:

  * the τ sweep and every baseline gate score from cache with zero live calls
    (a fire snaps to the nearest stored crop, ≤0.5 s away — stated in the paper);
  * the crop-level robustness heatmap is a regrouping of the same rows;
  * "what was sent, what came back, for which experiment" is answerable for
    every single call: ledger.jsonl + payloads/<eval>/<bag>/callNNNN/.

Two eval tags:
  crop_pass          — each stored crop asked with its OWN approach's goal
  crop_pass_foreign  — goal-sign crops asked with a goal from ANOTHER approach
                       (expected answer: not applicable) → the "goal-irrelevant"
                       heatmap column and a hallucination control

Resumable: re-running only re-asks cache misses; the ledger records cached rows
too (with cached=true) so a complete ledger can always be regenerated.

Usage (server up, `ar` env):
  python -m adaptive_reasoning.replay.label_crops \
      --dumps $SCRATCH/ar_replay --csv adaptive_reasoning/annotations/ar_extra-sep-4.csv \
      --out $SCRATCH/ar_eval --model Qwen/Qwen2.5-VL-7B-Instruct \
      --base-url http://localhost:8000/v1 --limit 20        # smoke first
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..reasoner import FakeVLM, OpenAICompatVLM, Reasoner, structural_hint


def load_rows(csv_path: str) -> dict[str, dict]:
    return {r["bag"]: r for r in csv.DictReader(open(csv_path))}


def crop_records(dumps: Path, bag: str) -> list[dict]:
    f = dumps / f"{bag}.jsonl"
    if not f.exists():
        return []
    out = []
    for line in open(f):
        r = json.loads(line)
        if r.get("crop"):
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="eval root, e.g. $SCRATCH/ar_eval")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--rpm", type=float, default=0.0)
    ap.add_argument("--only", default=None, help="single bag")
    ap.add_argument("--limit", type=int, default=0, help="max asks (smoke test); 0 = all")
    ap.add_argument("--min-R", type=float, default=0.0,
                    help="skip crops below this relevance (0 = ask about everything, "
                         "which eager baselines need)")
    ap.add_argument("--foreign-goals", type=int, default=1,
                    help="per goal-sign crop, ask with N goals from other approaches")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="FakeVLM transport (pipeline test)")
    a = ap.parse_args()

    out = Path(a.out)
    rows = load_rows(a.csv)
    if a.only:
        rows = {a.only: rows[a.only]}
    reasoning = None if a.reasoning.lower() in ("none", "off", "") else a.reasoning
    if a.fake:
        client = FakeVLM({})
    else:
        client = OpenAICompatVLM(model=a.model, base_url=a.base_url,
                                 reasoning_effort=reasoning, rpm=a.rpm)
    ledger = out / "ledger.jsonl"
    rng = random.Random(a.seed)
    all_goals = sorted({r["goal"] for r in rows.values()})

    n_asked = n_cached = n_failed = 0
    lat: list[float] = []
    correct = Counter(); total = Counter()
    t0 = time.monotonic()
    done = False
    for bag, row in rows.items():
        recs = crop_records(Path(a.dumps), bag)
        if not recs:
            print(f"SKIP {bag}: no dump/crops")
            continue
        recs = [r for r in recs if r["R"] >= a.min_R]
        crops_dir = Path(a.dumps) / f"{bag}_crops"
        print(f"{bag}: {len(recs)} crops  goal={row['goal']}  decision={row['decision']}")
        for tag_goal in ("own", "foreign"):
            eval_tag = "crop_pass" if tag_goal == "own" else "crop_pass_foreign"
            reasoner = Reasoner(client, model_tag=f"{a.model}:{a.reasoning}",
                                cache_dir=out / "vlm_cache",
                                payload_dir=out / "payloads" / eval_tag / bag,
                                ledger_path=ledger, eval_tag=eval_tag)
            for r in recs:
                if tag_goal == "foreign":
                    if r["R"] < 0.5 or a.foreign_goals <= 0:
                        continue
                    goals = rng.sample([g for g in all_goals if g != row["goal"]],
                                       min(a.foreign_goals, len(all_goals) - 1))
                else:
                    goals = [row["goal"]]
                from PIL import Image
                img = np.asarray(Image.open(crops_dir / r["crop"]).convert("RGB"))
                for g in goals:
                    expected = row["decision"] if tag_goal == "own" and r["R"] >= 0.5 else "not_applicable"
                    meta = {"bag": bag, "building": row.get("building", ""),
                            "plate_id": r["plate_id"], "frame": r["frame"], "crop": r["crop"],
                            "R": r["R"], "R_struct": r.get("R_struct"), "ell": r["ell"],
                            "phi": r.get("phi"), "text": r["text"],
                            "annotated_decision": row["decision"], "expected": expected,
                            "sign_type": row.get("sign_type", ""),
                            "temporary_present": row.get("temporary_present", ""),
                            "semantic_only": row.get("semantic_only", "")}
                    try:
                        ans = reasoner.ask(g, [r["text"]], [img], scene=None, meta=meta,
                                           hint=structural_hint(r["text"], g))
                    except Exception as e:                      # noqa: BLE001
                        # never let one bad request end a multi-hour pass; the crop
                        # stays unanswered and the coverage guard will flag it
                        n_failed += 1
                        print(f"  ASK FAILED {bag} {r['plate_id']} f{r['frame']}: "
                              f"{type(e).__name__}: {str(e)[:90]}", flush=True)
                        continue
                    n_asked += 1
                    n_cached += ans.cached
                    if not ans.cached:
                        lat.append(ans.latency_s)
                    got = ans.direction if ans.applicable else "not_applicable"
                    key = f"{eval_tag}"
                    total[key] += 1
                    correct[key] += (got == expected)
                    if a.limit and n_asked >= a.limit:
                        done = True
                        break
                if done:
                    break
            if done:
                break
        if done:
            break

    dt = time.monotonic() - t0
    print(f"\nasked {n_asked} ({n_cached} cached) in {dt/60:.1f} min")
    if lat:
        ls = sorted(lat)
        print(f"live latency: median {st.median(ls):.2f}s  p90 {ls[int(0.9*len(ls))-1]:.2f}s  max {max(ls):.2f}s")
    for k in total:
        print(f"{k:18s} accuracy vs expected: {correct[k]}/{total[k]} = {100*correct[k]/max(total[k],1):.1f}%")
    print(f"ledger: {ledger}")
    print(f"payloads: {out/'payloads'}/<eval>/<bag>/callNNNN/{{prompt.txt,img0.jpg,answer.json}}")
    (out / "label_crops_summary.json").write_text(json.dumps({
        "asked": n_asked, "cached": n_cached, "minutes": dt / 60,
        "latency_median": st.median(lat) if lat else None,
        "accuracy": {k: correct[k] / max(total[k], 1) for k in total},
        "model": a.model, "csv": a.csv, "dumps": a.dumps}, indent=2))


if __name__ == "__main__":
    main()