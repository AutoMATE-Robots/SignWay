#!/usr/bin/env python3
"""
inspect_dataset.py -- read the BUILT RLDS dataset and verify it's correct before trusting training.

Two things it checks, straight from the on-disk dataset (not the bag):
  1. The PROMPT at every frame -- prints a per-frame table so you can confirm the schedule
     (straight before the sign, decision through the turn, straight after) is right.
  2. The ACTION -- confirms each step is a single (2,) displacement (not the whole chunk), and
     reconstructs the full trajectory by cumulatively chaining the single-step displacements,
     so you can see the path the model will learn.

Usage:
    python inspect_dataset.py --data-dir /scratch.global/$USER/rlds --plot
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def load_steps(data_dir, split="train"):
    import tensorflow_datasets as tfds
    ds = tfds.load("signway_dataset", data_dir=data_dir, split=split)
    episodes = []
    for ep in ds:
        steps = []
        for s in ep["steps"]:
            steps.append({
                "action": s["action"].numpy(),
                "prompt": s["language_instruction"].numpy().decode(),
                "image": s["observation"]["image"].numpy(),
                "is_first": bool(s["is_first"].numpy()),
                "is_last": bool(s["is_last"].numpy()),
            })
        episodes.append(steps)
    return episodes


def print_prompt_table(steps, every=4):
    """Per-frame prompt + action, so you can eyeball the schedule is right."""
    print(f"\n{'frame':>5} | {'prompt':11} | {'action (fwd, left)':>20} | dir")
    print("-" * 52)
    changes = []
    prev = None
    for i, s in enumerate(steps):
        p = s["prompt"]
        if p != prev:
            changes.append((i, p))
            prev = p
        if i % every == 0 or p != steps[max(0, i - 1)]["prompt"]:
            wp = s["action"].reshape(-1, 2)
            # wp[-1] = endpoint of the 0.8s trajectory: lateral offset here is the TURN signal
            d = "LEFT" if wp[-1][1] > 0.02 else ("RIGHT" if wp[-1][1] < -0.02 else "straight")
            print(f"{i:>5} | {p:11} | wp1=({wp[0][0]:+.3f},{wp[0][1]:+.3f}) wp8=({wp[-1][0]:+.3f},{wp[-1][1]:+.3f}) | {d}")
    print("\nPrompt changes at frames:")
    for i, p in changes:
        print(f"  frame {i:3d} -> '{p}'")
    from collections import Counter
    c = Counter(s["prompt"] for s in steps)
    print("Prompt counts:", ", ".join(f"{k}={v}" for k, v in c.items()))


def reconstruct_path(steps):
    """Chain single-step displacements into a global path (what the robot actually did).
    Each action is a (fwd,left) step in that frame's robot frame; integrate with heading."""
    x, y, th = 0.0, 0.0, 0.0
    xs, ys, prompts = [x], [y], []
    for s in steps:
        wp = np.asarray(s["action"], dtype=np.float32).reshape(-1, 2)
        fwd, left = float(wp[0][0]), float(wp[0][1])
        # advance in world frame using current heading
        x += fwd * np.cos(th) - left * np.sin(th)
        y += fwd * np.sin(th) + left * np.cos(th)
        # update heading from the step's turn (atan2 of the displacement)
        th += np.arctan2(left, fwd) if (abs(fwd) + abs(left)) > 1e-6 else 0.0
        xs.append(x); ys.append(y); prompts.append(s["prompt"])
    return np.array(xs), np.array(ys), prompts


def plot_overview(steps, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, ys, prompts = reconstruct_path(steps)
    colors = {"straight": "#5FB37A", "turn_left": "#4C9BE8",
              "turn_right": "#E8994C", "stop": "#E86A6A"}
    fig, ax = plt.subplots(figsize=(9, 9))
    for i in range(len(prompts)):
        ax.plot(ys[i:i+2], xs[i:i+2], color=colors.get(prompts[i], "#888"), lw=3, solid_capstyle="round")
    ax.scatter([ys[0]], [xs[0]], marker="^", s=200, c="k", zorder=5, label="start")
    ax.scatter([ys[-1]], [xs[-1]], marker="s", s=120, c="#333", zorder=5, label="end")
    ax.invert_xaxis()  # physical right on the right of the screen
    handles = [plt.Line2D([], [], color=c, lw=3, label=k) for k, c in colors.items()]
    ax.legend(handles=handles + [
        plt.Line2D([], [], marker="^", color="k", lw=0, label="start"),
    ], fontsize=9)
    ax.set_xlabel("left  <--  0  -->  right   [m]"); ax.set_ylabel("forward  [m]")
    ax.set_title("Reconstructed path from the RLDS dataset\n(color = prompt at each frame)")
    ax.set_aspect("equal"); fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/scratch/rlds"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="dataset_inspect")
    args = ap.parse_args()

    episodes = load_steps(args.data_dir, args.split)
    print(f"{len(episodes)} episode(s)")
    for ei, steps in enumerate(episodes):
        print(f"\n===== EPISODE {ei}: {len(steps)} steps =====")
        # verify action shape
        a0 = steps[0]["action"]
        print(f"action shape per step: {a0.shape}  (want (16,) = 8 waypoints x 2)")
        assert a0.shape == (16,), f"ACTION IS {a0.shape}, EXPECTED (16,) -- rebuild the dataset!"
        print_prompt_table(steps, args.every)
        if args.plot:
            os.makedirs(args.out, exist_ok=True)
            p = plot_overview(steps, os.path.join(args.out, f"episode_{ei}.png"))
            print(f"\nplot -> {p}")


if __name__ == "__main__":
    main()