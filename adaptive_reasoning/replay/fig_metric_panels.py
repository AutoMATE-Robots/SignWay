"""
fig_metric_panels.py — four horizontal-bar panels, one per navigation metric,
one bar per invocation policy, with a target line (the style of a per-metric
bar figure).  All from eval_gates' per_approach.csv, so every number is real.

  (a) SR   success rate                 = mean(correct)                            target 100
  (b) T    time-to-goal efficiency      = mean(correct * t*/(t* + stranded))       target 100
           (SCT, Yokoyama et al. 2021; t* = recorded driving time; robot stops
            during reasoning, so completion = driving + stranded)
  (c) ST   stranded time [s/approach]   = mean(calls * latency)                    target 0
  (d) WT   wrong turns [%]              = mean(decided and not correct)            target 0
  Distance efficiency is NOT computable from replay (path = recorded path on
  success, failure on a wrong turn) — it needs end-to-end runs; omitted here.

  python -m adaptive_reasoning.replay.fig_metric_panels --per-approach $SCRATCH/ar_eval/gates32/per_approach.csv \\
      --out ~/SignWay/figs/ar_metric_panels [--category numeric|named|ambiguous --annotations ar_extra-sep-4.csv]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from . import figstyle
from .figstyle import C, PAGE_W

ORDER = ["always", "periodic5s", "iros", "signscene", "ours"]
LABEL = {"always": "Always-invoke", "periodic5s": "Periodic 5 s", "iros": "IROS-style†",
         "signscene": "SignScene-style†", "relevance": "Relevance-only", "ours": "SignWay (ours)"}
COLOR = {"always": "#5c6b73", "periodic5s": "#8a5aa3", "iros": "#b0752b", "signscene": "#3f6fb5",
         "relevance": C["accent"], "ours": C["ours"]}


def in_category(a: dict, cat: str) -> bool:
    if cat == "ambiguous":
        return a.get("temp_conflicting", a.get("temporary_present", "0")) == "1"
    if cat == "named":
        return a.get("semantic_only") == "1"
    if cat == "numeric":
        return a.get("semantic_only") != "1"
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-approach", required=True)
    ap.add_argument("--out", default="figs/ar_metric_panels")
    ap.add_argument("--gates", nargs="+", default=ORDER)
    ap.add_argument("--category", default=None, choices=[None, "numeric", "named", "ambiguous"])
    ap.add_argument("--annotations", default=None)
    a = ap.parse_args()

    ann = {r["bag"]: r for r in csv.DictReader(open(a.annotations))} if a.annotations else {}
    by = defaultdict(list)
    for r in csv.DictReader(open(a.per_approach)):
        if a.category and not in_category(ann.get(r["bag"], {}), a.category):
            continue
        by[r["gate"]].append(r)
    gates = [g for g in a.gates if g in by]
    n = len(by[gates[0]])

    def metrics(R):
        S = np.array([int(r["correct"]) for r in R], float)
        st = np.array([float(r["overhead_s"]) for r in R])
        t = np.array([float(r.get("duration_s") or 0) for r in R])
        dec = np.array([int(r.get("decided", 1)) for r in R])
        sr = 100 * S.mean()
        T = 100 * float(np.mean(S * t / np.maximum(t + st, 1e-6))) if t.any() else float("nan")
        return {"SR": sr, "T": T, "ST": float(st.mean()), "WT": 100 * float(np.mean(dec * (1 - S)))}

    M = {g: metrics(by[g]) for g in gates}
    have_T = not np.isnan(next(iter(M.values()))["T"])
    have_WT = any("decided" in r for r in next(iter(by.values())))
    panels = [("SR", "Success rate (%)", 100, 100)]
    if have_T:
        panels.append(("T", "Time-to-goal\nefficiency, SCT (%)", 100, 100))
    panels.append(("ST", "Stranded time\n(s / approach)", 0, None))
    if have_WT:
        panels.append(("WT", "Wrong turns (%)", 0, None))

    figstyle.use()
    fig, axes = plt.subplots(1, len(panels), figsize=(PAGE_W * (0.26 * len(panels)), 0.30 * len(gates) + 0.9),
                             sharey=True, gridspec_kw={"wspace": 0.10})
    axes = np.atleast_1d(axes)
    y = np.arange(len(gates))[::-1]
    for k, (ax, (key, xlabel, target, xmax)) in enumerate(zip(axes, panels)):
        vals = [M[g][key] for g in gates]
        ax.barh(y, vals, 0.52, color=[COLOR.get(g, "#888888") for g in gates])
        hi = xmax if xmax else max(vals) * 1.35 + 1e-6
        for yi, v in zip(y, vals):
            ax.text(v + hi * 0.02, yi, f"{v:.1f}", va="center", fontsize=6)
        ax.axvline(target, ls="--", lw=0.8, color="black")
        ax.set_xlim(0, hi)
        ax.set_xlabel(xlabel, fontsize=6.5)
        ax.set_title(f"({'abcd'[k]})", loc="left", fontsize=7, fontweight="bold")
        ax.text(1.0, 1.02, f"Target: {target}{'%' if key != 'ST' else ' s'}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABEL.get(g, g) for g in gates], fontsize=6.5)
    cat = f" · {a.category} goals" if a.category else ""
    fig.suptitle(f"n = {n} held-out approaches{cat}", fontsize=7, x=0.01, ha="left")
    figstyle.save(fig, a.out, arrays={"gates": np.array(gates),
                                      "metrics": np.array([[M[g][k] for k in ("SR", "T", "ST", "WT")] for g in gates])})
    print(f"{'policy':16s}" + "".join(f"{k:>8s}" for k, *_ in panels))
    for g in gates:
        m = M[g]; print(f"{LABEL.get(g, g):16s}" + "".join(f"{m[k]:8.1f}" for k, *_ in panels))
    print(f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()