"""
fig_memory.py — cumulative VLM calls vs. goals served, with and without the
semantic progress memory, on per-building delivery sequences COMPOSED from
separately recorded approaches (stated in the caption).

Plate-level simulation over the replay dumps:
  Without memory   each approach costs its measured calls (per_approach.csv, ours).
  With memory      every legible plate's OCR text is stored as it is passed; at a
                   new approach, if a STORED plate's text already resolves the new
                   goal via the structural parser (fast path on remembered text),
                   the decision is free and the approach costs 0 calls; plates the
                   robot re-encounters are matched by the deployed fuzzy signature.
This is exactly the "store and reuse interpreted sign knowledge" mechanism of
Fig. 1, evaluated at the plate level.

  python -m adaptive_reasoning.replay.fig_memory --dumps $SCRATCH/ar_replay \\
      --per-approach $SCRATCH/ar_eval/gates32/per_approach.csv \\
      --csv adaptive_reasoning/annotations/ar_extra-sep-4.csv \\
      --tau 0.55 --out ~/SignWay/figs/ar_memory
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import figstyle
from .figstyle import C, COL_W
from ..evidence.relevance import Goal, relevance_lines
from ..evidence.resolve import Plate, PlateLine, resolve
from ..memory import Memory, MemoryUpdate


def building(bag: str) -> str:
    parts = bag.split("-")
    return parts[1] if len(parts) > 2 else bag


def resolves(text: str, goal: str) -> bool:
    lines = [t.strip() for t in text.split("|") if t.strip()]
    goal_ = Goal.parse(goal)
    rel = relevance_lines(lines, goal_)
    if rel.score < 0.5:
        return False
    res = resolve(Plate([PlateLine(t, []) for t in lines]), goal_, rel=rel)
    return bool(res.resolved)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--per-approach", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tau", type=float, default=0.55)
    ap.add_argument("--min-seq", type=int, default=4, help="only plot buildings with >= this many approaches")
    ap.add_argument("--out", default="figs/ar_memory")
    a = ap.parse_args()

    calls = {r["bag"]: float(r["calls"]) for r in csv.DictReader(open(a.per_approach)) if r["gate"] == "ours"}
    goals = {r["bag"]: r["goal"] for r in csv.DictReader(open(a.csv))}
    seqs = defaultdict(list)
    for bag in sorted(goals):
        if bag in calls and (Path(a.dumps) / f"{bag}.jsonl").exists():
            seqs[building(bag)].append(bag)
    seqs = {b: bags for b, bags in seqs.items() if len(bags) >= a.min_seq}

    figstyle.use()
    fig, axes = plt.subplots(1, len(seqs), figsize=(COL_W * 0.62 * len(seqs) + 0.6, 1.9),
                             sharey=False, squeeze=False)
    summary = {}
    for ax, (b, bags) in zip(axes[0], sorted(seqs.items())):
        # memory OFF: measured calls per approach
        off = np.cumsum([calls[bag] for bag in bags])
        # memory ON: plate-level simulation with stored text re-resolved per goal
        store: dict[str, str] = {}                       # plate_id -> best text
        mem = Memory("cross-goal", fuzzy=True)
        on = []
        tot = 0.0
        for bag in bags:
            goal = goals[bag]
            reused = any((not mem.is_novel(pid)) and resolves(txt, goal) for pid, txt in store.items())
            tot += 0.0 if reused else calls[bag]
            on.append(tot)
            for line in open(Path(a.dumps) / f"{bag}.jsonl"):
                r = json.loads(line)
                if r.get("ell", 0) >= a.tau and len(r.get("text", "")) > 3:
                    pid, txt = r["plate_id"], r["text"]
                    if pid not in store or len(txt) > len(store[pid]):
                        store[pid] = txt
                    mem.apply(MemoryUpdate(pid, "vlm", None, 0))
        x = np.arange(1, len(bags) + 1)
        ax.plot(x, off, "-o", ms=2.5, lw=1.1, color=C["baseline"], label="memory off")
        ax.plot(x, on, "-s", ms=2.5, lw=1.1, color=C["ours"], label="memory on")
        ax.set_title(b, fontsize=6.6)
        ax.set_xlabel("goals served", fontsize=6)
        ax.set_xticks(x)
        ax.grid(alpha=0.25, lw=0.5)
        summary[b] = {"bags": bags, "off": off.tolist(), "on": on}
    axes[0][0].set_ylabel("cumulative VLM calls", fontsize=6.5)
    axes[0][-1].legend(fontsize=5.6, frameon=False, loc="upper left")
    figstyle.save(fig, a.out)
    for b, s in summary.items():
        print(f"{b:8s} off={s['off'][-1]:.1f}  on={s['on'][-1]:.1f}  saved={s['off'][-1]-s['on'][-1]:.1f} calls over {len(s['bags'])} goals")
    print(f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
