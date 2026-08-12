#!/usr/bin/env python3
"""
run_scorer.py -- sanity-check the evidence score on one bag.

Plots best evidence score vs frame (should RISE through the approach as the sign
becomes legible) with flip/onset markers, and dumps the top-N crops as PNGs for
eyeballing. This is the hour-of-work de-risk step before anything else depends
on the score.

    python -m adaptive_reasoning.evidence.run_scorer \
        --bag /users/1/munda057/SignWay/ros2_bags/rosbag2-keller-t1 \
        --backend doctr --out $SCRATCH/score_check_t1 \
        [--labels $SCRATCH/deadline_labels/deadline_labels_all.csv] [--stride 3]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_reasoning.config import bootstrap  # noqa: E402
from adaptive_reasoning.evidence.scorer import EvidenceScorer, make_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--backend", default="doctr", help="doctr | paddle | fake")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--top-crops", type=int, default=8)
    ap.add_argument("--image-topic", default="/c1/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    args = ap.parse_args()

    bootstrap()
    from bag_to_episode import read_bag

    frames, _ = read_bag(Path(args.bag), args.image_topic,
                         args.odom_topic, args.ros_distro)
    backend = make_backend(args.backend) if args.backend != "fake" \
        else make_backend("fake", start_frame=len(frames) // 3, ramp=len(frames) // 3)
    scorer = EvidenceScorer(backend)

    onset = flip = None
    if args.labels:
        name = Path(args.bag).name
        with open(args.labels) as f:
            for r in csv.DictReader(f):
                if r["bag"] == name:
                    onset = int(r["onset_frame"])
                    break

    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    xs, ys, all_items = [], [], []
    for fi in range(0, len(frames), args.stride):
        items = scorer.score_frame(frames[fi][1], fi)
        best = max((i.score for i in items), default=0.0)
        xs.append(fi)
        ys.append(best)
        all_items += items

    all_items.sort(key=lambda i: -i.score)
    from PIL import Image

    for k, it in enumerate(all_items[: args.top_crops]):
        if it.crop is not None:
            Image.fromarray(it.crop).save(
                out / "crops" / f"{k:02d}_f{it.frame_idx}_s{it.score:.2f}_"
                f"c{it.ocr_conf:.2f}.png")
            print(f"  top{k}: frame {it.frame_idx}  score {it.score:.3f}  "
                  f"ocr {it.ocr_conf:.2f}  area {it.area_frac:.4f}  "
                  f"sharp {it.sharpness:.2f}  text={it.text[:40]!r}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(xs, ys, color="#4C9BE8")
    if onset is not None and onset >= 0:
        ax.axvline(onset, color="k", lw=1.5, label="turn onset (odom)")
        ax.legend()
    ax.set_xlabel("frame")
    ax.set_ylabel("best evidence score")
    ax.set_title(f"Evidence score over approach — {Path(args.bag).name} "
                 f"[{args.backend}]  (should rise toward the junction)")
    fig.tight_layout()
    fig.savefig(out / "score_vs_frame.png", dpi=140)
    print(f"plot -> {out / 'score_vs_frame.png'}")


if __name__ == "__main__":
    main()
