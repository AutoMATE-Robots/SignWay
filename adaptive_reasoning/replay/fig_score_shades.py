"""
fig_score_shades.py — per-dataset stacked score bars, one colour per method,
metric segments as shades (darkest bottom = SR → lightest top = WT).

Score = equal-weight mean of four metrics, each contributing up to 25 points
(weights stated in the caption; per-metric values printed in the segments):
  SR  success rate (%)
  T   time-to-goal efficiency = SCT (%)  [needs duration_s in per_approach.csv]
  ST' 100 * (1 - ST / worst ST in the panel)   (stranded time, normalised)
  WT' 100 - wrong-commit rate (%)
Datasets: (a) temporary sign  (b) numeric range  (c) arrow + multi-text
(named-goal proxy, stated)  (d) revisited signage (n/a until route bags).

  python -m adaptive_reasoning.replay.fig_score_shades \\
      --per-approach $SCRATCH/ar_eval/gates32/per_approach.csv \\
      --annotations adaptive_reasoning/annotations/ar_extra-sep-4.csv \\
      --out ~/SignWay/figs/ar_score_shades
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np

from . import figstyle
from .figstyle import PAGE_W

METHODS = ["signnav", "signscene", "iros", "relevance", "ours"]
LABEL = {"signnav": "SignNav-style‡", "signscene": "SignScene-style†", "iros": "IROS-style†",
         "relevance": "Relevance-only (abl.)", "ours": "SignWay (ours)"}
BASE = {"signnav": "#6b8e4e", "signscene": "#3f6fb5", "iros": "#b0752b",
        "relevance": "#c96f3f", "ours": "#0f766e"}
DATASETS = [("temporary", "(a) Temporary sign"), ("numeric", "(b) Numeric range"),
            ("compound", "(c) Arrow + multi-text\n(named-goal proxy)"), ("revisited", "(d) Revisited signage")]
METRICS = ["SR", "T", "ST", "WT"]


def shades(base, n=4):
    rgb = np.array(mc.to_rgb(base))
    return [tuple(rgb + (1 - rgb) * f) for f in (0.0, 0.25, 0.5, 0.72)]


def in_dataset(a: dict, key: str) -> bool:
    if key == "temporary":
        # a CONFLICTING notice (changes the correct action), not mere presence;
        # falls back to temporary_present for annotation files without the column
        return a.get("temp_conflicting", a.get("temporary_present", "0")) == "1"
    if key == "numeric":
        return a.get("semantic_only") != "1"
    if key == "compound":
        return a.get("semantic_only") == "1"
    return False                                   # revisited: no data yet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-approach", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", default="figs/ar_score_shades")
    ap.add_argument("--methods", nargs="+", default=METHODS, help="row order; subset of METHODS")
    ap.add_argument("--datasets", nargs="+", default=[k for k, _ in DATASETS],
                    help="which panels to draw, e.g. --datasets numeric compound")
    a = ap.parse_args()
    ann = {r["bag"]: r for r in csv.DictReader(open(a.annotations))}
    by = defaultdict(list)
    for r in csv.DictReader(open(a.per_approach)):
        if r["bag"] in ann:
            by[r["gate"]].append(r)
    methods = [m for m in a.methods if m in by]

    def metrics(R, worst_st):
        S = np.array([int(r["correct"]) for r in R], float)
        st = np.array([float(r["overhead_s"]) for r in R])
        t = np.array([float(r.get("duration_s") or 0) for r in R])
        dec = np.array([int(r.get("decided", 1)) for r in R], float)
        return {"SR": 100 * S.mean(),
                "T": 100 * float(np.mean(S * t / np.maximum(t + st, 1e-6))) if t.any() else np.nan,
                "ST": 100 * (1 - st.mean() / worst_st) if worst_st > 0 else 100.0,
                "WT": 100 - 100 * float(np.mean(dec * (1 - S)))}

    figstyle.use()
    ds = [(k, t) for k, t in DATASETS if k in a.datasets]
    fig, axes = plt.subplots(1, len(ds), figsize=(PAGE_W * (0.26 * len(ds) + 0.06), 2.35), sharey=True,
                             squeeze=False, gridspec_kw={"wspace": 0.08})
    axes = axes[0]
    x = np.arange(len(methods))
    printed = {}
    for ax, (key, title) in zip(axes, ds):
        sel = {m: [r for r in by[m] if in_dataset(ann[r["bag"]], key)] for m in methods}
        ns = {m: len(sel[m]) for m in methods}
        if not any(ns.values()):
            ax.text(0.5, 0.5, "requires chained\nroute recordings", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6.5, color="#666")
            ax.set_title(title, fontsize=6.6); ax.set_xticks([]); continue
        worst_st = max(np.mean([float(r["overhead_s"]) for r in sel[m]]) for m in methods if sel[m])
        M = {m: metrics(sel[m], worst_st) for m in methods if sel[m]}
        printed[key] = M
        bottoms = np.zeros(len(methods))
        for k, met in enumerate(METRICS):
            seg = np.array([(0 if np.isnan(M[m][met]) else M[m][met]) / 4 if m in M else 0 for m in methods])
            ax.bar(x, seg, 0.58, bottom=bottoms, color=[shades(BASE[m])[k] for m in methods],
                   edgecolor="white", lw=0.5)
            for xi, (b, s_, m) in enumerate(zip(bottoms, seg, methods)):
                if s_ > 6:
                    ax.text(xi, b + s_ / 2, f"{s_*4:.0f}", ha="center", va="center", fontsize=4.6,
                            color="white" if k < 2 else "#333")
            bottoms += seg
        for xi, (tot, m) in enumerate(zip(bottoms, methods)):
            ax.text(xi, tot + 1.5, f"{tot:.0f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
        ax.set_title(title + f"  (n={max(ns.values())})", fontsize=6.6)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in methods], rotation=38, ha="right", fontsize=5.6)
        ax.set_ylim(0, 112); ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0].set_ylabel("score (equal-weight mean of 4 metrics)", fontsize=6)
    greys = shades("#555555")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=g) for g in greys],
               ["SR success rate (bottom, darkest)", "T time-to-goal eff. (SCT)",
                "ST low stranded time (norm.)", "WT low wrong turns (top, lightest)"],
               loc="lower center", ncol=4, fontsize=5.6, frameon=False, bbox_to_anchor=(0.5, -0.14))
    fig.text(0.01, -0.05, "‡SignNav-style: arrow-following without text grounding on our stack; "
                          "action execution assumed identical across methods.", fontsize=5, color="#555")
    figstyle.save(fig, a.out)
    for key, M in printed.items():
        print(f"\n[{key}]")
        for m, v in M.items():
            print(f"  {LABEL[m]:18s} " + "  ".join(f"{k}={v[k]:5.1f}" for k in METRICS))
    print(f"\nwrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()