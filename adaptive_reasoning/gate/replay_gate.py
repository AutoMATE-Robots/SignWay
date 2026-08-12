#!/usr/bin/env python3
"""
replay_gate.py -- run the gate over a labeled bag as if live, no robot needed.

Answers: WHEN would the gate have fired, and with how much lead over the true
junction? Uses labeled d (as a stand-in oracle) or a trained deadline checkpoint.
Emits a per-frame trace CSV and a diagnostic figure: evidence score, d, the
deadline countdown, fire marker vs true onset.

    python -m adaptive_reasoning.gate.replay_gate \
        --bag /users/1/munda057/SignWay/ros2_bags/rosbag2-keller-e1 \
        --labels $SCRATCH/deadline_labels/deadline_labels_all.csv \
        --out $SCRATCH/gate_replay_e1 [--backend fake|doctr|paddle] \
        [--deadline-ckpt $SCRATCH/deadline_model/deadline_head.pt] [--vlm dry:turn_right]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from adaptive_reasoning.config import GateConfig, bootstrap  # noqa: E402
from adaptive_reasoning.evidence.scorer import EvidenceScorer, make_backend  # noqa: E402
from adaptive_reasoning.gate.gate import Gate  # noqa: E402
from adaptive_reasoning.reasoning.vlm_client import make_vlm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="fake")
    ap.add_argument("--vlm", default="dry")
    ap.add_argument("--deadline-ckpt", default=None,
                    help="use the trained estimator instead of label-oracle d")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--image-topic", default="/c1/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    args = ap.parse_args()

    bootstrap()
    from bag_to_episode import read_bag

    bag_name = Path(args.bag).name
    lbl = {}
    onset = -1
    with open(args.labels) as f:
        for r in csv.DictReader(f):
            if r["bag"] != bag_name:
                continue
            lbl[int(r["frame_index"])] = r
            onset = int(r["onset_frame"])
    if not lbl:
        raise SystemExit(f"{bag_name} not found in {args.labels}")

    frames, _ = read_bag(Path(args.bag), args.image_topic,
                         args.odom_topic, args.ros_distro)
    est = None
    if args.deadline_ckpt:
        from adaptive_reasoning.deadline.model import DeadlineEstimator
        est = DeadlineEstimator(args.deadline_ckpt)

    if args.backend == "fake":
        ahead = [fi for fi, r in lbl.items() if r["junction_ahead"] == "1"]
        backend = make_backend("fake", start_frame=min(ahead) if ahead else 0, ramp=120)
    else:
        backend = make_backend(args.backend)
    scorer = EvidenceScorer(backend)
    gate = Gate(GateConfig())
    vlm = make_vlm(args.vlm)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trace = []
    fire_frame = None
    for fi in sorted(lbl):
        if fi % args.stride or fi >= len(frames):
            continue
        r = lbl[fi]
        rgb = frames[fi][1]
        items = scorer.score_frame(rgb, fi)
        if est is not None:
            pred = est.predict(rgb)
            d_q10, p_j = pred["d_q10"], pred["p_junction"]
        else:  # label-oracle mode: d_q10 = true d (perfect estimator baseline)
            d_true = float(r["d_m"])
            p_j = 1.0 if r["junction_ahead"] == "1" else 0.0
            d_q10 = d_true if p_j > 0.5 else None
        v = float(r["speed_mps"])
        o = gate.step(fi, items, d_q10, p_j, v)
        trace.append(dict(frame=fi, state=o.state, best_score=o.best_score,
                          d_q10=d_q10 if d_q10 is not None else "",
                          ttj=o.time_to_junction_s or "", fire=int(o.fire),
                          fire_reason=o.fire_reason or "",
                          slow_factor=o.slow_factor, v=v))
        if o.fire and fire_frame is None:
            fire_frame = fi
            reading = vlm.timed_read(scorer.score_frame(rgb, fi) and
                                     [i.crop for i in gate.buffer.items], "room 305")
            gate.report_vlm_latency(reading["latency_s"])
            gate.resolve(reading["decision"], "vlm")
            print(f"FIRED at frame {fi} ({o.fire_reason}); "
                  f"VLM -> {reading['decision']} in {reading['latency_s']:.2f}s")

    with open(out / "gate_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        w.writeheader()
        w.writerows(trace)

    lead = ((onset - fire_frame) if fire_frame is not None and onset >= 0 else None)
    print(f"onset frame {onset}  fire frame {fire_frame}  "
          f"lead {lead} frames" + ("" if lead is None else f" (~{lead/20:.1f}s @20fps)"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = [t["frame"] for t in trace]
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(fr, [t["best_score"] for t in trace], color="#4C9BE8",
             label="best evidence score")
    ax1.axhline(gate.cfg.tau_sufficiency, color="#4C9BE8", ls=":",
                label="tau_sufficiency")
    ax1.set_ylabel("evidence score")
    ax1.set_xlabel("frame")
    ax2 = ax1.twinx()
    dd = [float(t["d_q10"]) if t["d_q10"] != "" else np.nan for t in trace]
    ax2.plot(fr, dd, color="#5FB37A", label="d_q10 (m)")
    ax2.set_ylabel("distance to junction (m)")
    if onset >= 0:
        ax1.axvline(onset, color="k", lw=1.5, label="true onset")
    if fire_frame is not None:
        ax1.axvline(fire_frame, color="#E86A6A", lw=2, ls="--", label="gate FIRE")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax1.set_title(f"Gate replay — {bag_name}")
    fig.tight_layout()
    fig.savefig(out / "gate_replay.png", dpi=140)
    print(f"trace + figure -> {out}/")


if __name__ == "__main__":
    main()
