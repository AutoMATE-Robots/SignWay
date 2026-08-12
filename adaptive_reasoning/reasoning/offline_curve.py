#!/usr/bin/env python3
"""
offline_curve.py -- THE headline experiment, entirely offline from existing bags.

For each annotated turn approach, at each invoke-distance bucket, call the VLM
in two modes and score against the ground-truth decision:
  * single : the single best crop available AT that distance
  * buffer : the best-K buffer accumulated from arming UP TO that distance
Output: accuracy-vs-invoke-distance CSV + the figure (two curves, deadline band).

This produces (1) the paper figure, (2) training data for the v1 sufficiency
predictor, (3) validation that the evidence score predicts VLM correctness.
Responses are cached to JSONL per (vlm, bag, mode, bucket) -- reruns resume free.

    python -m adaptive_reasoning.reasoning.offline_curve \
        --labels $SCRATCH/deadline_labels/deadline_labels_all.csv \
        --bag-root /users/1/munda057/SignWay/ros2_bags \
        --out $SCRATCH/offline_curve --vlm dry \
        [--backend fake|doctr|paddle] [--buckets 1,2,3,4,6,8,10,12] [--goal "room 305"]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_reasoning.config import GateConfig, bootstrap  # noqa: E402
from adaptive_reasoning.evidence.scorer import (  # noqa: E402
    BestKBuffer, EvidenceScorer, make_backend)
from adaptive_reasoning.reasoning.vlm_client import make_vlm  # noqa: E402


def load_turn_bags(labels_path: Path):
    """bag -> dict(decision, frames: {frame_index: (d_m, junction_ahead)})"""
    bags = defaultdict(lambda: dict(decision=None, frames={}))
    with open(labels_path) as f:
        for r in csv.DictReader(f):
            if r["decision"] in ("straight", "stop"):
                continue
            b = bags[r["bag"]]
            b["decision"] = r["decision"]
            b["split"] = r["split"]
            b["frames"][int(r["frame_index"])] = (
                float(r["d_m"]), int(r["junction_ahead"]))
    return {k: v for k, v in bags.items() if v["frames"]}


def frame_at_distance(frames_lbl: dict, d_target: float):
    """Frame index whose labeled d is closest to d_target (junction_ahead only)."""
    cand = [(abs(d - d_target), fi) for fi, (d, ah) in frames_lbl.items() if ah]
    return min(cand)[1] if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--bag-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vlm", default="dry", help="dry | anthropic:<model> | openai:<model>[@base_url]")
    ap.add_argument("--backend", default="fake", help="fake | doctr | paddle")
    ap.add_argument("--buckets", default="1,2,3,4,6,8,10,12", help="invoke distances (m)")
    ap.add_argument("--goal", default="room 305")
    ap.add_argument("--score-stride", type=int, default=3, help="score every Nth frame")
    ap.add_argument("--max-bags", type=int, default=0)
    ap.add_argument("--deadline-band", default=None,
                    help="optional 'lo,hi' metres to shade (e.g. from v*(L+margin))")
    ap.add_argument("--image-topic", default="/c1/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    args = ap.parse_args()

    bootstrap()
    from bag_to_episode import read_bag

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / f"cache_{args.vlm.replace(':', '_').replace('/', '_')}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            e = json.loads(line)
            cache[e["key"]] = e
    cache_f = open(cache_path, "a")

    vlm = make_vlm(args.vlm)
    cfg = GateConfig()
    buckets = [float(b) for b in args.buckets.split(",")]
    bags = load_turn_bags(Path(args.labels))
    names = sorted(bags)[: args.max_bags or None]
    print(f"{len(names)} turn approaches, buckets {buckets}, vlm={vlm.name}")

    results = []
    for bag in names:
        info = bags[bag]
        frames, _ = read_bag(Path(args.bag_root) / bag, args.image_topic,
                             args.odom_topic, args.ros_distro)
        backend = make_backend(args.backend) if args.backend != "fake" else \
            make_backend("fake", start_frame=min(
                fi for fi, (d, ah) in info["frames"].items() if ah), ramp=120)
        scorer = EvidenceScorer(backend)

        # score the whole approach once; reuse per bucket
        scored = {}
        for fi in sorted(info["frames"]):
            if fi % args.score_stride or fi >= len(frames):
                continue
            scored[fi] = scorer.score_frame(frames[fi][1], fi)

        for d_b in buckets:
            fi_b = frame_at_distance(info["frames"], d_b)
            if fi_b is None:
                continue
            # buffered evidence up to this frame; single = best item AT ~this frame
            buf = BestKBuffer(cfg.buffer_k, cfg.min_frame_gap)
            single_best = None
            for fi in sorted(scored):
                if fi > fi_b:
                    break
                for it in scored[fi]:
                    buf.add(it)
                    if abs(fi - fi_b) <= args.score_stride:
                        if single_best is None or it.score > single_best.score:
                            single_best = it
            for mode, crops in (("single", [single_best.crop] if single_best else []),
                                ("buffer", buf.crops())):
                if not crops:
                    continue
                key = f"{vlm.name}|{bag}|{mode}|{d_b}"
                if key in cache:
                    r = cache[key]["reading"]
                    lat = cache[key]["latency_s"]
                else:
                    r = vlm.timed_read(crops, args.goal)
                    lat = r.pop("latency_s")
                    e = dict(key=key, reading=r, latency_s=lat)
                    cache_f.write(json.dumps(e) + "\n")
                    cache_f.flush()
                    cache[key] = e
                correct = int(r["decision"] == info["decision"])
                results.append(dict(bag=bag, split=info.get("split", "train"),
                                    mode=mode, d_bucket=d_b, correct=correct,
                                    decision=r["decision"], gt=info["decision"],
                                    confidence=r.get("confidence", 0),
                                    best_score=(single_best.score if mode == "single"
                                                else buf.best_score),
                                    n_crops=len(crops), latency_s=lat))
                print(f"  {bag} {mode:>6} @{d_b:>4.1f}m -> {r['decision']:<12} "
                      f"({'OK' if correct else 'X'})  score={results[-1]['best_score']:.2f}")

    # ---- write results + figure ---------------------------------------------
    res_path = out / "curve_results.csv"
    with open(res_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode, style in (("single", "--o"), ("buffer", "-s")):
        xs, ys, ns = [], [], []
        for d_b in buckets:
            rs = [r for r in results if r["mode"] == mode and r["d_bucket"] == d_b]
            if rs:
                xs.append(d_b)
                ys.append(np.mean([r["correct"] for r in rs]))
                ns.append(len(rs))
        ax.plot(xs, ys, style, label=f"{mode} (n≈{int(np.mean(ns)) if ns else 0}/pt)")
    if args.deadline_band:
        lo, hi = (float(x) for x in args.deadline_band.split(","))
        ax.axvspan(lo, hi, color="#E86A6A", alpha=0.15,
                   label="deadline region (v·(L+margin))")
    ax.set_xlabel("invoke distance from junction (m)   ← later / closer")
    ax.set_ylabel("VLM decision accuracy")
    ax.set_ylim(0, 1.02)
    ax.invert_xaxis()
    ax.set_title(f"Accuracy vs invoke distance — {vlm.name}\n"
                 "single frame vs accumulated best-K evidence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = out / "accuracy_vs_invoke_distance.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nresults -> {res_path}\nfigure  -> {fig_path}")


if __name__ == "__main__":
    main()
