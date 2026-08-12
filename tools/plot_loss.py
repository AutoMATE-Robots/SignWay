#!/usr/bin/env python3
"""
plot_loss.py -- turn the training log's [LOSS] lines into a loss-curve PNG.

Requires the loss-print patch in finetune.py (prints "[LOSS] step=N loss=X" each step).

    python tools/plot_loss.py --log /scratch.global/$USER/oft_train.log --out loss_curve.png
"""
import argparse
import re

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="loss_curve.png")
    ap.add_argument("--smooth", type=int, default=10, help="moving-average window")
    args = ap.parse_args()

    steps, losses = [], []
    pat = re.compile(r"\[LOSS\] step=(\d+) loss=([0-9.eE+-]+)")
    for line in open(args.log, errors="ignore"):
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    if not steps:
        raise SystemExit("no [LOSS] lines found -- is the loss-print patch applied?")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color="#B8CCE8", lw=1, label="per-step L1 loss")
    if len(losses) >= args.smooth:
        k = args.smooth
        sm = np.convolve(losses, np.ones(k) / k, mode="valid")
        ax.plot(steps[k - 1:], sm, color="#2a78d6", lw=2.2, label=f"smoothed (window {k})")
    ax.set_xlabel("training step")
    ax.set_ylabel("L1 loss (normalised)")
    ax.set_title(f"Training loss -- {len(steps)} steps, "
                 f"start {losses[0]:.4f} -> last {losses[-1]:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"loss curve -> {args.out}  ({len(steps)} points, "
          f"first {losses[0]:.4f}, last {losses[-1]:.4f}, min {min(losses):.4f})")


if __name__ == "__main__":
    main()
