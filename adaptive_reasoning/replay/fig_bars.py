"""
fig_bars.py — grouped bars per invocation policy: decision accuracy (left axis)
and VLM calls per approach (right axis), value labels on every bar, lines
across bar tops, dashed grid.  Reads eval_gates' summary.json.

  python -m adaptive_reasoning.replay.fig_bars --summary $SCRATCH/ar_eval/gates32/summary.json \\
      --gates iros signscene periodic5s always ours --out ~/SignWay/figs/ar_bars_32b
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from .eval_gates import LABEL

RED_L, RED_D = "#F6C4C4", "#F08A8A"
BLU_L, BLU_D = "#C3CEE3", "#5C6FB3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--gates", nargs="+", default=["iros", "signscene", "periodic5s", "always", "ours"])
    ap.add_argument("--out", default="figs/ar_bars")
    ap.add_argument("--labels", nargs="+", default=None, help="override x tick labels, same order as --gates")
    a = ap.parse_args()

    S = json.load(open(a.summary))["gates"]
    gates = [g for g in a.gates if g in S]
    acc = np.array([S[g]["acc"] for g in gates])
    calls = np.array([S[g]["calls"] for g in gates])
    names = a.labels or [LABEL[g].replace("†", "").replace(" (abl.)", "") for g in gates]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax2 = ax.twinx()
    x = np.arange(len(gates)); w = 0.30

    b1 = ax.bar(x - w / 2 - 0.02, acc, w, color=RED_D, edgecolor="none", label="Decision accuracy (%)", zorder=3)
    b2 = ax2.bar(x + w / 2 + 0.02, calls, w, color=BLU_D, edgecolor="none", label="VLM calls / approach", zorder=3)
    ax.plot(x - w / 2 - 0.02, acc, "-o", color=RED_D, ms=3, lw=1.3, zorder=4, alpha=0.9)
    ax2.plot(x + w / 2 + 0.02, calls, "-s", color=BLU_D, ms=3, lw=1.3, zorder=4, alpha=0.9)
    for xi, v in zip(x - w / 2 - 0.02, acc):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#333")
    for xi, v in zip(x + w / 2 + 0.02, calls):
        ax2.text(xi, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_ylabel("Decision accuracy (%)"); ax.set_ylim(0, 100)
    ax2.set_ylabel("VLM calls / approach"); ax2.set_ylim(0, max(7.0, calls.max() * 1.25))
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_xlabel("Reasoning-invocation policy")
    ax.grid(axis="y", ls="--", lw=0.6, alpha=0.6, zorder=0)
    for s in ("top",):
        ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    handles = [b1, b2]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.12), fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out + ".pdf", bbox_inches="tight"); fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight")
    np.savez(a.out + "_data.npz", gates=np.array(gates), acc=acc, calls=calls)
    print(f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
