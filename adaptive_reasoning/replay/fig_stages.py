"""
fig_stages.py — navigation performance by stage, per situation, per method.

Three stages, each a plain percentage (no summed score in the figure):
  Triggered   the policy fires when it should / stays quiet when it shouldn't
              (trigger mode: fig_heatmap --mode trigger -> trigger_results.csv)
  Understood  the reasoner's answer is correct given a call
              (reading mode: fig_heatmap -> results.csv)
  Action      the final navigation decision is correct
              (eval_gates -> per_approach.csv, tagged by situation via the annotation csv)
A stage a method does not have (by design) is drawn hatched "n/a", never 0.

Also prints Ajay's staged score: mean over applicable stages, equal weights,
stated as such — a single number for the table, with the stages visible in the
figure so nobody has to defend the weights.

  python -m adaptive_reasoning.replay.fig_stages \\
      --trigger $SCRATCH/ar_eval/trigger/trigger_results.csv \\
      --reading $SCRATCH/ar_eval/heatmap_final/results.csv \\
      --approach $SCRATCH/ar_eval/gates32/per_approach.csv \\
      --annotations adaptive_reasoning/annotations/ar_extra-sep-4.csv \\
      --out ~/SignWay/figs/ar_stages
Any of --trigger / --reading / --approach may be omitted; that stage is then n/a for all.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from . import figstyle
from .figstyle import C, PAGE_W

# canonical method names and how each source names them
METHODS = ["always", "iros", "signscene", "relevance", "signway"]
LABEL = {"always": "Always-invoke", "iros": "IROS-style†", "signscene": "SignScene-style†",
         "relevance": "Relevance-only", "signway": "SignWay (ours)"}
TRIG_MAP = {"always": "always", "iros_style": "iros", "signscene_style": "signscene",
            "relevance": "relevance", "signway": "signway"}
READ_MAP = {"iros_style": "iros", "signscene_style": "signscene", "signway": "signway",
            "crop_notext": None, "signway_tight": None}
APPR_MAP = {"always": "always", "iros": "iros", "signscene": "signscene",
            "relevance": "relevance", "ours": "signway"}
# stages that do not exist for a method by design (hatched)
NA = {"always": {"understood"} if False else set()}     # extend if a method lacks a stage
SITUATIONS = ["numeric_sign", "named_goal", "goal_irrelevant", "unclear_sign", "temporary_sign"]
SIT_LABEL = {"numeric_sign": "numeric range", "named_goal": "named goal", "goal_irrelevant": "goal-irrelevant",
             "unclear_sign": "unclear", "temporary_sign": "temporary sign"}
STAGES = ["triggered", "understood", "action"]
STAGE_LABEL = {"triggered": "triggered correctly", "understood": "understood correctly", "action": "correct action"}
STAGE_COLOR = {"triggered": C["baseline2"], "understood": C["accent"], "action": C["ours"]}


def load_stage(path, mapping, key_method, key_cond, key_ok, cond_map=None):
    """-> {method: {situation: [0/1,...]}}"""
    out = defaultdict(lambda: defaultdict(list))
    if not path:
        return out
    for r in csv.DictReader(open(path)):
        m = mapping.get(r[key_method])
        if m is None:
            continue
        c = cond_map(r) if cond_map else r[key_cond]
        if c is None:
            continue
        out[m][c].append(int(r[key_ok]) if r[key_ok] not in ("True", "False") else int(r[key_ok] == "True"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", default=None); ap.add_argument("--reading", default=None)
    ap.add_argument("--approach", default=None); ap.add_argument("--annotations", default=None)
    ap.add_argument("--out", default="figs/ar_stages")
    ap.add_argument("--situations", nargs="+", default=SITUATIONS)
    a = ap.parse_args()

    trig = load_stage(a.trigger, TRIG_MAP, "policy", "condition", "correct")
    read = load_stage(a.reading, READ_MAP, "strategy", "condition", "correct")
    ann = {}
    if a.annotations:
        ann = {r["bag"]: r for r in csv.DictReader(open(a.annotations))}
    def appr_cond(r):
        x = ann.get(r["bag"])
        if not x:
            return None
        if x.get("temp_conflicting", x.get("temporary_present", "0")) == "1":
            return "temporary_sign"
        return "named_goal" if x.get("semantic_only") == "1" else "numeric_sign"
    appr = load_stage(a.approach, APPR_MAP, "gate", None, "correct", appr_cond)
    data = {"triggered": trig, "understood": read, "action": appr}

    sits = [s for s in a.situations if any(s in data[st][m] for st in STAGES for m in METHODS)]
    figstyle.use()
    fig, axes = plt.subplots(1, len(sits), figsize=(PAGE_W, 2.0), sharey=True, squeeze=False)
    w = 0.26; x = np.arange(len(METHODS))
    for ax, sit in zip(axes[0], sits):
        for k, st in enumerate(STAGES):
            for i, m in enumerate(METHODS):
                vals = data[st][m].get(sit)
                xi = x[i] + (k - 1) * w
                if st in NA.get(m, set()) or not vals:
                    ax.bar(xi, 100, w, color="none", edgecolor="#bbbbbb", hatch="////", lw=0.4)
                    ax.text(xi, 50, "n/a", ha="center", va="center", fontsize=4.5, color="#777", rotation=90)
                    continue
                v = 100 * np.mean(vals)
                ax.bar(xi, v, w, color=STAGE_COLOR[st], edgecolor="none")
                ax.text(xi, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=4.6)
                ax.text(xi, 3, f"n={len(vals)}", ha="center", va="bottom", fontsize=3.8, color="white", rotation=90)
        ax.set_title(SIT_LABEL.get(sit, sit), fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in METHODS], rotation=45, ha="right", fontsize=5.4)
        ax.set_ylim(0, 112); ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("correct [%]")
    handles = [Patch(color=STAGE_COLOR[s], label=STAGE_LABEL[s]) for s in STAGES] + \
              [Patch(facecolor="none", edgecolor="#bbbbbb", hatch="////", label="stage n/a for this method")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.42))
    figstyle.save(fig, a.out)

    # staged score (Ajay's number): equal weights over the stages a method HAS, per situation
    print("\nStaged score = mean over applicable stages (equal weights); per-stage values in the figure.\n")
    print(f"{'method':16s}" + "".join(f"{SIT_LABEL.get(s, s):>18s}" for s in sits) + f"{'overall':>10s}")
    for m in METHODS:
        cells, allv = [], []
        for sit in sits:
            vs = [100 * np.mean(data[st][m][sit]) for st in STAGES
                  if st not in NA.get(m, set()) and data[st][m].get(sit)]
            cells.append(f"{np.mean(vs):>17.0f}%" if vs else f"{'—':>18s}")
            allv += vs
        print(f"{LABEL[m]:16s}" + "".join(cells) + (f"{np.mean(allv):>9.0f}%" if allv else f"{'—':>10s}"))
    print(f"\nwrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()