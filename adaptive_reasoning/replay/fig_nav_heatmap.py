"""
fig_nav_heatmap.py — navigation performance heatmap: methods × (category × metric).

Metrics (standard, cited; no custom composite):
  SR   success rate = correct final decision (replay: decision == annotated)
  SCT  success weighted by completion time (Yokoyama et al. 2021):
       S_i * t*_i / (t*_i + stranded_i), with t*_i = approach driving time and the
       stated assumption that the robot stops during reasoning, so completion time
       = driving time + stranded time
  ST   stranded time per approach [s] = calls * measured latency (lower is better)
  WT   wrong-commit rate: the policy COMMITTED to a wrong direction (lower is better).
       Distinct from "no decision reached" (a stall), which is the other failure mode;
       on single-junction approaches 1 - SR = WT + no-decision.
  SPL  needs real path lengths -> not computable from replay; shown as n/a

Rows are INVOCATION POLICIES re-implemented on our stack (same reasoner, same VLA,
same data) — labelled as such.  SignNav is a learned policy and cannot be
re-implemented in replay: its row is n/a until end-to-end runs exist.

Categories come from the annotation csv: overall; ambiguous = temporary_present;
numeric = numeric goal; compound = named/multi-line goal (proxy, stated);
revisited = requires route bags (n/a until then).

  python -m adaptive_reasoning.replay.fig_nav_heatmap --per-approach $SCRATCH/ar_eval/gates32/per_approach.csv \\
      --annotations adaptive_reasoning/annotations/ar_extra-sep-4.csv --out ~/SignWay/figs/ar_nav_heatmap
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from . import figstyle
from .figstyle import PAGE_W

ROWS = ["signnav", "signscene", "iros", "always", "ours"]
LABEL = {"signnav": "SignNav‡", "signscene": "SignScene-style†", "iros": "IROS-style†",
         "always": "Always-invoke", "ours": "SignWay (ours)"}
GATE = {"signscene": "signscene", "iros": "iros", "always": "always", "ours": "ours"}
CATS = ["overall", "ambiguous", "numeric", "compound", "revisited"]
CAT_LABEL = {"overall": "Overall", "ambiguous": "Ambiguous\n(temp./permanent)", "numeric": "Numeric\nrange",
             "compound": "Compound\n(multi-text + icon)", "revisited": "Revisited\nsigns"}
METRICS = ["SR", "SCT", "SPL", "ST", "WT"]
LOWER_BETTER = {"ST", "WT"}


def category_of(a: dict) -> set:
    cats = {"overall"}
    if a.get("temp_conflicting", a.get("temporary_present", "0")) == "1":
        cats.add("ambiguous")
    cats.add("compound" if a.get("semantic_only") == "1" else "numeric")
    return cats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-approach", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", default="figs/ar_nav_heatmap")
    a = ap.parse_args()

    ann = {r["bag"]: r for r in csv.DictReader(open(a.annotations))}
    rows = [r for r in csv.DictReader(open(a.per_approach)) if r["bag"] in ann]
    by = defaultdict(list)
    for r in rows:
        by[r["gate"]].append(r)

    # value[row][cat][metric] -> (display, colour_score in 0..100) or None
    val = {m: {c: {k: None for k in METRICS} for c in CATS} for m in ROWS}
    for m in ROWS:
        g = GATE.get(m)
        if not g or g not in by:
            continue
        for c in CATS:
            R = [r for r in by[g] if c in category_of(ann[r["bag"]])]
            if not R:
                continue
            S = np.array([int(r["correct"]) for r in R], float)
            st = np.array([float(r["overhead_s"]) for r in R])
            dur = np.array([float(r.get("duration_s", 0) or 0) for r in R])
            wt = np.array([int(r.get("decided", "1")) * (1 - int(r["correct"])) for r in R], float)
            sr = 100 * S.mean()
            sct = 100 * float(np.mean(S * dur / np.maximum(dur + st, 1e-6))) if dur.any() else None
            val[m][c]["SR"] = (f"{sr:.0f}", sr)
            if sct is not None:
                val[m][c]["SCT"] = (f"{sct:.0f}", sct)
            val[m][c]["ST"] = (f"{st.mean():.1f}s", None)
            val[m][c]["WT"] = (f"{100*wt.mean():.0f}", None)
            val[m][c]["_n"] = len(R)
    # colour scores for lower-is-better metrics: relative to the worst row in that column
    for c in CATS:
        for k in ("ST", "WT"):
            xs = [float(val[m][c][k][0].rstrip("s")) for m in ROWS if val[m][c][k]]
            if k == "WT":                      # percentage: colour on the absolute scale, inverted
                for m in ROWS:
                    if val[m][c][k]:
                        val[m][c][k] = (val[m][c][k][0], 100 - float(val[m][c][k][0]))
                continue
            if not xs:
                continue
            hi = max(xs) or 1.0
            for m in ROWS:
                if val[m][c][k]:
                    x = float(val[m][c][k][0].rstrip("s"))
                    val[m][c][k] = (val[m][c][k][0], 100 * (1 - x / hi))

    figstyle.use()
    ncol = len(CATS) * len(METRICS)
    fig, ax = plt.subplots(figsize=(PAGE_W, 0.42 * len(ROWS) + 1.3))
    M = np.full((len(ROWS), ncol), np.nan)
    for i, m in enumerate(ROWS):
        for jc, c in enumerate(CATS):
            for jk, k in enumerate(METRICS):
                v = val[m][c][k]
                if v and v[1] is not None:
                    M[i, jc * len(METRICS) + jk] = v[1]
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#eeeeee")
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i, m in enumerate(ROWS):
        for jc, c in enumerate(CATS):
            for jk, k in enumerate(METRICS):
                j = jc * len(METRICS) + jk
                v = val[m][c][k]
                if v is None:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#eeeeee", hatch="////", ec="#aaaaaa", lw=0))
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=4.6, color="#666")
                else:
                    ax.text(j, i, v[0], ha="center", va="center", fontsize=5.6,
                            color="white" if (v[1] is not None and v[1] < 55) else "black")
    for jc in range(1, len(CATS)):
        ax.axvline(jc * len(METRICS) - 0.5, color="white", lw=2)
    ax.set_xticks(range(ncol)); ax.set_xticklabels(METRICS * len(CATS), fontsize=5.6)
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([LABEL[m] for m in ROWS], fontsize=6.5)
    for jc, c in enumerate(CATS):
        n = next((val[m][c].get("_n") for m in ROWS if val[m][c].get("_n")), None)
        ax.text(jc * len(METRICS) + (len(METRICS) - 1) / 2, -0.85,
                CAT_LABEL[c] + (f"  (n={n})" if n else ""), ha="center", va="bottom", fontsize=6.2)
    ax.set_xlabel("SR success rate · SCT success weighted by completion time · SPL success weighted by path length · "
                  "ST stranded time [s] · WT wrong-commit rate [%]", fontsize=5.4)
    ax.tick_params(length=0)
    figstyle.save(fig, a.out)
    print(f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()