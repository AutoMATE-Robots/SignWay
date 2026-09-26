"""
decoder_variance.py — is the decoder using the image, or just the prompt?

Runs N frames x 3 prompts through a checkpoint and decomposes the variance of the
predicted endpoint lateral:

    Var(y) = Var_prompt + Var_image + residual
    Var_prompt = Var_p( mean_i y[i,p] ),  Var_image = Var_i( mean_p y[i,p] )

A prompt-only lookup table (the FiLM/high-lr failure) puts ~all variance in the
prompt term.  A decoder that conditions on the scene puts a growing share in the
image term as the junction approaches — which is what we report per distance bin.

  python decoder_variance.py --bag ros2_bags/rosbag2-keller-e1 --flip-frame 190 \
      --decision turn_right --turn-done-frame 522 --stride 2 --junction 432 \
      --checkpoint "$CKPT" --tag v6 --out $SCRATCH/vla_eval/var_v6_e1.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

PROMPTS = ["straight", "turn_left", "turn_right"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True); ap.add_argument("--flip-frame", type=int, required=True)
    ap.add_argument("--decision", required=True); ap.add_argument("--turn-done-frame", type=int, default=10**9)
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--junction", type=int, required=True)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--tag", default="ckpt")
    ap.add_argument("--every", type=int, default=5); ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--image-topic", default="/c1/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pathlib import Path
    from eval_openloop import load_policy, predict
    from bag_to_episode import build_steps, read_bag

    policy = load_policy(a.checkpoint)
    frames, odom = read_bag(Path(a.bag), a.image_topic, a.odom_topic,
                            a.ros_distro)
    steps = build_steps(frames, odom, a.flip_frame, a.decision,
                        turn_done_frame=a.turn_done_frame, stride=a.stride)
    rows = []
    for i, st in enumerate(steps):
        if i % a.every:
            continue
        y = {}
        for p in PROMPTS:
            wp = np.asarray(predict(policy, st["image"], p, a.horizon))
            y[p] = float(wp[-1, 1])
        rf = int(st["frame_index"])
        rows.append({"tag": a.tag, "raw_frame": rf,
                     "t_to_junction": (rf - a.junction) / (a.hz * a.stride), **y})
    if not rows:
        raise SystemExit("no steps evaluated — check --every and the bag path")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["tag", "raw_frame", "t_to_junction"] + PROMPTS)
        w.writeheader(); w.writerows(rows)

    # Prompt-response amplitude per image: how far apart do the two turn prompts
    # drive the SAME frame?  A prompt-only lookup table gives a CONSTANT amplitude
    # (the image cannot modulate it); a scene-conditioned decoder gives an amplitude
    # that grows as the junction comes into view.
    Y = np.array([[r[p] for p in PROMPTS] for r in rows])          # (frames, prompts)
    t = np.array([r["t_to_junction"] for r in rows])
    A = (Y[:, PROMPTS.index("turn_left")] - Y[:, PROMPTS.index("turn_right")]) / 2.0
    cv = A.std() / (np.abs(A).mean() + 1e-12)
    far, near = t < -8, (t > -3) & (t < 3)
    rho = np.corrcoef(-t[t < 3], A[t < 3])[0, 1] if (t < 3).sum() > 5 else np.nan
    print(f"[{a.tag}] frames={len(rows)}")
    print(f"  prompt-response amplitude A = (y_left - y_right)/2")
    print(f"    mean |A|            : {np.abs(A).mean():.4f} m")
    print(f"    coeff. of variation : {cv:.2f}      (0 = constant = lookup table)")
    if far.any() and near.any():
        print(f"    A far (>8 s out)    : {A[far].mean():.4f} m   (n={far.sum()})")
        print(f"    A at junction (+-3s): {A[near].mean():.4f} m   (n={near.sum()})")
        print(f"    modulation ratio    : {abs(A[near].mean())/max(abs(A[far].mean()),1e-6):.1f}x")
    print(f"    corr(A, proximity)  : {rho:+.2f}   (lookup table ~ 0)")
    for lo, hi in ((-30, -10), (-10, -4), (-4, 0), (0, 6)):
        m = (t >= lo) & (t < hi)
        if m.sum() > 3:
            print(f"    {lo:>4}..{hi:<3}s  n={m.sum():3d}  mean A={A[m].mean():+.4f} m")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
