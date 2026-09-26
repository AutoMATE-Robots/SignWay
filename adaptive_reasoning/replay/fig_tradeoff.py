"""
fig_tradeoff.py — the headline: VLM calls per approach (x) vs decision
accuracy (y), one point per trigger policy with 95% bootstrap CIs on both
axes.  Ours is drawn with its τ sweep as a curve, so the reader sees a
tunable operating range against fixed baselines.

Two panels by default: y = in-time accuracy (correct AND before the junction —
the metric that matters for navigation) and y = decision accuracy (any time).
Reads eval_gates per_approach.csv files.

  python -m adaptive_reasoning.replay.fig_tradeoff \\
      --per-approach $SCRATCH/ar_eval/gates32/per_approach.csv \\
      --tau-sweep "$SCRATCH/ar_eval/gates32_tau*/per_approach.csv" \\
      --out ~/SignWay/figs/ar_tradeoff_32b
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from . import figstyle
from .figstyle import C, COL_W
from .eval_gates import GATES, LABEL

MARK = {"always": "s", "continuous05": "D", "periodic5s": "v", "reactive": "^",
        "iros": "P", "signscene": "X", "relevance": "o", "ours": "*"}


def boot(x, n=10000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load(path):
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[r["gate"]].append(r)
    return by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-approach", required=True)
    ap.add_argument("--tau-sweep", default=None, help="glob of per_approach.csv from eval_gates --gates ours at several --tau")
    ap.add_argument("--out", default="figs/ar_tradeoff")
    ap.add_argument("--gates", nargs="+", default=GATES)
    ap.add_argument("--tau-star", default=None, help="operating τ (annotated on the curve)")
    ap.add_argument("--clean", action="store_true",
                    help="no error bars, no τ sweep, labelled points (the paper's main-text version)")
    a = ap.parse_args()

    by = load(a.per_approach)
    gates = [g for g in a.gates if g in by]
    n = len(by[gates[0]])
    stats = {}
    for g in gates:
        R = by[g]
        stats[g] = {"calls": boot([float(r["calls"]) for r in R]),
                    "in_time": boot([100 * int(r["in_time"]) for r in R]),
                    "acc": boot([100 * int(r["correct"]) for r in R])}

    sweep = []
    if a.tau_sweep:
        for f in sorted(glob.glob(a.tau_sweep)):
            m = re.search(r"tau([0-9.]+)", f)
            if not m:
                continue
            rows = [r for r in csv.DictReader(open(f)) if r["gate"] == "ours"]
            if rows:
                sweep.append((float(m.group(1)), np.mean([float(r["calls"]) for r in rows]),
                              100 * np.mean([int(r["in_time"]) for r in rows]),
                              100 * np.mean([int(r["correct"]) for r in rows])))
        sweep.sort()

    figstyle.use()
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2.05, 2.55), gridspec_kw={"wspace": 0.28})
    tau_star = float(a.tau_star) if a.tau_star else None
    for ax, key, ylab in ((axes[0], "in_time", "in-time decision accuracy [%]"),
                          (axes[1], "acc", "decision accuracy (any time) [%]")):
        drawn: dict = {}; labels_at: dict = {}
        if sweep and not a.clean:
            xs = [s[1] for s in sweep]; ys = [s[2] if key == "in_time" else s[3] for s in sweep]
            ax.plot(xs, ys, "-", color=C["ours_lt"], lw=1.2, zorder=1, label="ours, τ sweep")
            # label only the ends and τ* to keep it legible; the full sweep is in the table
            for t, x, yi, ya in (sweep[0], sweep[-1]):
                ax.annotate(f"τ={t:g}", (x, yi if key == "in_time" else ya), textcoords="offset points",
                            xytext=(4, -3 if t == sweep[0][0] else 3), fontsize=5, color=C["ours"])
        for g in gates:
            s = stats[g]; x, xl, xh = s["calls"]; y, yl, yh = s[key]
            col = C["ours"] if g == "ours" else (C["accent"] if g == "relevance" else C["baseline"])
            if a.clean:
                # policies that behave identically (e.g. always/continuous/reactive collapse
                # under one-pending-call serialisation) land on the same point: one marker,
                # one merged label, instead of three labels on top of each other
                keyxy = (round(x, 2), round(y, 1))
                if keyxy in drawn:
                    drawn[keyxy].append(LABEL[g].replace(" (abl.)", "").replace("†", ""))
                    continue
                drawn[keyxy] = [LABEL[g].replace(" (abl.)", "").replace("†", "")]
                ax.plot(x, y, MARK[g], ms=10 if g == "ours" else 6, color=col, label=LABEL[g], zorder=3)
                labels_at[keyxy] = (x, y, col, g)
            else:
                ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]],
                            fmt=MARK[g], ms=9 if g == "ours" else 5, color=col, ecolor=col,
                            elinewidth=0.6, capsize=1.5, label=LABEL[g], zorder=3)
        if a.clean:                                    # relabel legend handles with merged names
            for h in ax.get_legend_handles_labels()[0]:
                pass
            for keyxy, names in drawn.items():
                x, y, col, g = labels_at[keyxy]
                for line in ax.get_lines():
                    if line.get_label() == LABEL[g]:
                        line.set_label(" / ".join(names))
        ax.set_xlabel("VLM calls / approach")
        ax.set_ylabel(ylab)
        ax.set_xlim(left=0); ax.set_ylim(0, 100)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_title(f"n = {n} held-out approaches" + ("" if a.clean else " · 95% bootstrap CIs"),
                      loc="left", fontsize=7)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4 if a.clean else 5, fontsize=6 if a.clean else 5.6,
               handlelength=1.0, frameon=False, bbox_to_anchor=(0.5, -0.2 if not a.clean else -0.16))
    figstyle.save(fig, a.out, arrays={"gates": np.array(gates),
                                      "stats": np.array([[*stats[g]["calls"], *stats[g]["in_time"], *stats[g]["acc"]] for g in gates]),
                                      "sweep": np.array(sweep) if sweep else np.zeros((0, 4))})
    print(f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
