"""
fig_overhead.py — "Adaptive reasoning efficiency": semantic decision overhead
per approach, stacked into VLM inference (successful calls × L), wasted-call
overhead (not-applicable / parse calls × L), and — drawn separately, hatched —
cheap-layer compute (OCR/tracking; overlaps driving, not decision latency).
Error bars: 95% bootstrap CI on total overhead.  Above each bar: accuracy and
calls per approach.  Reads eval_gates.py's per_approach.csv.

  python -m adaptive_reasoning.replay.fig_overhead --per-approach $SCRATCH/ar_eval/gates/per_approach.csv \\
      --out $SCRATCH/ar_eval/figs/ar_overhead [--no-cheap]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from . import figstyle
from .figstyle import C, COL_W
from .eval_gates import GATES, LABEL


def bootstrap_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return 0.0, 0.0
    m = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-approach", required=True)
    ap.add_argument("--out", default="figs/ar_overhead")
    ap.add_argument("--gates", nargs="+", default=GATES)
    ap.add_argument("--with-cheap", action="store_true",
                    help="ALSO stack cheap-layer compute (default off: it overlaps driving and is "
                         "not decision latency; report it in the table/text instead)")
    a = ap.parse_args()

    by = defaultdict(list)
    for r in csv.DictReader(open(a.per_approach)):
        by[r["gate"]].append(r)
    gates = [g for g in a.gates if g in by]
    vlm = [np.mean([float(r["vlm_s"]) for r in by[g]]) for g in gates]
    wasted = [np.mean([float(r["wasted_s"]) for r in by[g]]) for g in gates]
    cheap = [np.mean([float(r["cheap_s"]) for r in by[g]]) if a.with_cheap else 0.0 for g in gates]
    total = [np.array([float(r["overhead_s"]) for r in by[g]]) for g in gates]
    cis = [bootstrap_ci(t) for t in total]
    acc = [100 * np.mean([int(r["correct"]) for r in by[g]]) for g in gates]
    calls = [np.mean([float(r["calls"]) for r in by[g]]) for g in gates]
    n = len(by[gates[0]])

    figstyle.use()
    fig, ax = plt.subplots(figsize=(COL_W * 2.05, 2.4))
    x = np.arange(len(gates))
    if a.with_cheap:
        ax.bar(x, cheap, color="none", edgecolor=C["baseline2"], hatch="////", lw=0.6,
               label="cheap-layer compute (overlaps driving)")
    ax.bar(x, vlm, bottom=cheap, color=C["ours"], label="VLM inference (successful calls)")
    ax.bar(x, wasted, bottom=np.array(cheap) + np.array(vlm), color=C["accent"],
           label="wasted / parse calls")
    tops = np.array(cheap) + np.array(vlm) + np.array(wasted)
    err = np.array([[max(np.mean(t) - lo, 0), max(hi - np.mean(t), 0)] for t, (lo, hi) in zip(total, cis)]).T
    ax.errorbar(x, tops, yerr=err,
                fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
    for i, g in enumerate(gates):
        ax.text(x[i], tops[i] + err[1][i] + 0.4, f"acc {acc[i]:.0f}%\n{calls[i]:.1f} calls",
                ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[g] for g in gates], rotation=20, ha="right")
    ax.set_ylabel("semantic decision overhead / approach [s]")
    ax.set_title(f"n = {n} held-out approaches; error bars = 95% bootstrap CI on total overhead",
                 loc="left", fontsize=7)
    ax.legend(loc="upper right", fontsize=6)
    ax.set_ylim(0, max(tops + err[1]) * 1.35)
    figstyle.save(fig, a.out, arrays={"gates": np.array(gates), "vlm_s": np.array(vlm),
                                      "wasted_s": np.array(wasted), "cheap_s": np.array(cheap),
                                      "acc": np.array(acc), "calls": np.array(calls),
                                      "ci": np.array(cis)})
    print(f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
