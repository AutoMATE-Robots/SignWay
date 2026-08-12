#!/usr/bin/env python3
"""
make_trajectory_video.py -- the showpiece visualization.

For every frame of the bag, draw the camera image with the model's PREDICTED trajectory and the
ACTUAL (odometry) trajectory overlaid as arrows/paths, plus a live mini-plot of turn-deflection.
Stitches all frames into an MP4 (or GIF) that plays like a video of the robot "thinking".

Two panels per frame:
  LEFT  : the forward camera image with both trajectories drawn in a top-down inset (bird's eye),
          predicted = blue, actual = black. An arrow shows the immediate next move.
  RIGHT : a scrolling deflection plot (predicted vs actual) with a moving cursor at the current
          frame -- so you watch the turn decision unfold in time.

Run AFTER eval (needs the same checkpoint). Reuses eval_openloop's load_policy/predict.

    PYTHONPATH=~/SignWay/tools:~/SignWay/signway_dataset:$PYTHONPATH \
    python ~/SignWay/signway_dataset/make_trajectory_video.py \
      --bag /users/1/munda057/SignWay/ros2_bags/rosbag2_keller_c9 \
      --flip-frame 0 --decision turn_right --turn-done-frame 82 \
      --checkpoint "/scratch.global/.../...100_chkpt" \
      --out signway_video --fps 10

Output: signway_video/frames/*.png  and  signway_video/trajectory.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bag_to_episode import build_steps, read_bag
from eval_openloop import load_policy, predict, _attach_gt_trajectory


def draw_frame(img, pred_traj, actual_traj, defl_hist_pred, defl_hist_actual,
               frame_idx, prompt, all_frames, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(2, 2, width_ratios=[1.3, 1], height_ratios=[3, 1], hspace=0.35, wspace=0.25)

    # --- LEFT: camera image with trajectory inset ---
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(f"frame {frame_idx}   ·   prompt: {prompt}", fontsize=13)

    # bird's-eye inset (top-down trajectory), placed in the lower-right of the image
    h, w = img.shape[:2]
    inset = ax_img.inset_axes([0.62, 0.05, 0.36, 0.42])
    def draw_path(ax, traj, color, label, lw):
        p = np.vstack([[0, 0], traj])
        ax.plot(-p[:, 1], p[:, 0], color=color, lw=lw, label=label, solid_capstyle="round")
        # arrow for the immediate next step
        if len(traj) >= 1:
            ax.annotate("", xy=(-traj[0][1], traj[0][0]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    draw_path(inset, actual_traj, "#111111", "actual", 2.5)
    draw_path(inset, pred_traj, "#2a78d6", "predicted", 2.5)
    inset.scatter([0], [0], marker="^", c="k", s=60, zorder=5)
    inset.set_xlim(-0.6, 0.6); inset.set_ylim(-0.1, 1.6)
    inset.set_facecolor((1, 1, 1, 0.85))
    inset.set_xticks([]); inset.set_yticks([])
    inset.set_title("bird's-eye: next 8 steps", fontsize=8)
    inset.legend(fontsize=7, loc="upper left")

    # --- RIGHT TOP: deflection over time with moving cursor ---
    ax_t = fig.add_subplot(gs[0, 1])
    xs = list(range(len(all_frames)))
    ax_t.plot(xs[:len(defl_hist_actual)], defl_hist_actual, "k-", lw=2, label="actual")
    ax_t.plot(xs[:len(defl_hist_pred)], defl_hist_pred, color="#2a78d6", lw=2, ls="--", label="predicted")
    ax_t.axvline(frame_idx, color="#E8994C", lw=1.5)
    ax_t.axhline(0, color="#999", lw=0.5)
    ax_t.set_xlim(0, len(all_frames))
    ax_t.set_ylabel("deflection (m)\n− = right", fontsize=9)
    ax_t.set_title("turn decision over time", fontsize=10)
    ax_t.legend(fontsize=8, loc="lower right")
    ax_t.tick_params(labelsize=8)

    # --- RIGHT BOTTOM: a compact status readout ---
    ax_s = fig.add_subplot(gs[1, 1]); ax_s.axis("off")
    d = pred_traj[-1][1]
    turn = "RIGHT" if d < -0.01 else ("LEFT" if d > 0.01 else "straight")
    ax_s.text(0.0, 0.7, f"predicted move: {turn}", fontsize=12, color="#2a78d6", weight="bold")
    ax_s.text(0.0, 0.35, f"prompt (standing decision): {prompt}", fontsize=10, color="#333")
    ax_s.text(0.0, 0.0, f"endpoint deflection: {d:+.3f} m", fontsize=10, color="#555")

    fig.savefig(out_path, dpi=95, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--flip-frame", type=int, required=True)
    ap.add_argument("--decision", required=True, choices=["turn_left", "turn_right", "stop"])
    ap.add_argument("--turn-done-frame", type=int, default=None)
    ap.add_argument("--image-topic", default="/c1/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="signway_video")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    frames, odom = read_bag(Path(args.bag), args.image_topic, args.odom_topic, args.ros_distro)
    steps = build_steps(frames, odom, args.flip_frame, args.decision,
                        args.turn_done_frame, args.horizon, args.stride)
    _attach_gt_trajectory(steps, args.horizon)
    print(f"{len(steps)} frames")

    print("loading checkpoint...")
    policy = load_policy(args.checkpoint)

    # run predictions + accumulate deflection history
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    defl_pred, defl_actual = [], []
    preds = []
    for i, s in enumerate(steps):
        p = predict(policy, s["image"], s["prompt"], args.horizon)
        preds.append(p)
        defl_pred.append(float(p[-1][1]))
        defl_actual.append(float(s["trajectory"][-1][1]))
        if i % 10 == 0:
            print(f"  predicted {i}/{len(steps)}")

    print("rendering frames...")
    for i, s in enumerate(steps):
        out_path = os.path.join(args.out, "frames", f"f{i:04d}.png")
        draw_frame(s["image"], preds[i], s["trajectory"],
                   defl_pred[:i+1], defl_actual[:i+1],
                   i, s["prompt"], steps, out_path)
        if i % 10 == 0:
            print(f"  rendered {i}/{len(steps)}")

    # stitch to mp4 (ffmpeg) with a gif fallback
    mp4 = os.path.join(args.out, "trajectory.mp4")
    frames_glob = os.path.join(args.out, "frames", "f%04d.png")
    rc = os.system(f"ffmpeg -y -framerate {args.fps} -i {frames_glob} "
                   f"-c:v libx264 -pix_fmt yuv420p -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {mp4} 2>/dev/null")
    if rc == 0 and os.path.exists(mp4):
        print(f"\nVIDEO -> {mp4}")
    else:
        # fallback: imageio gif
        try:
            import imageio.v2 as imageio
            imgs = [imageio.imread(os.path.join(args.out, "frames", f"f{i:04d}.png"))
                    for i in range(len(steps))]
            gif = os.path.join(args.out, "trajectory.gif")
            imageio.mimsave(gif, imgs, fps=args.fps)
            print(f"\nffmpeg unavailable; GIF -> {gif}")
        except Exception as e:
            print(f"\nframes are in {args.out}/frames/ -- stitch manually (ffmpeg/imageio failed: {e})")


if __name__ == "__main__":
    main()
