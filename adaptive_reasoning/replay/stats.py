"""
stats.py — the numbers the tex quotes, with uncertainty.

Per gate: mean and 95% bootstrap CI (10k resamples over approaches) for
calls, overhead_s, accuracy, in-time accuracy.  Paired comparison of each
baseline against ours on the SAME approaches: Wilcoxon signed-rank (scipy) if
available, else an exact two-sided sign test — no independent-samples tests,
these are paired by construction.

  python -m adaptive_reasoning.replay.stats --per-approach $SCRATCH/ar_eval/gates/per_approach.csv \\
      --out $SCRATCH/ar_eval/gates/stats
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import comb

import numpy as np

from .eval_gates import GATES, LABEL


def boot(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_p(a, b) -> tuple[str, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    try:
        from scipy.stats import wilcoxon  # type: ignore
        if np.all(d == 0):
            return "wilcoxon", 1.0
        return "wilcoxon", float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:  # noqa: BLE001
        nz = d[d != 0]
        if len(nz) == 0:
            return "sign", 1.0
        k = int((nz > 0).sum()); n = len(nz)
        p = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2
        return "sign", float(min(p, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-approach", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ours", default="ours")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.per_approach)))
    by = defaultdict(dict)
    for r in rows:
        by[r["gate"]][r["bag"]] = r
    gates = [g for g in GATES if g in by]
    common = sorted(set.intersection(*(set(by[g]) for g in gates)))
    out = {"n": len(common), "gates": {}, "paired_vs_ours": {}}
    md = [f"n = {len(common)} approaches (paired)\n",
          "| Gate | Calls [CI] | Overhead s [CI] | Acc % [CI] | In-time % [CI] |", "|---|---|---|---|---|"]
    metrics = {"calls": lambda r: float(r["calls"]), "overhead_s": lambda r: float(r["overhead_s"]),
               "acc": lambda r: 100 * int(r["correct"]), "in_time": lambda r: 100 * int(r["in_time"])}
    vals = {g: {m: [f(by[g][b]) for b in common] for m, f in metrics.items()} for g in gates}
    for g in gates:
        out["gates"][g] = {m: dict(zip(("mean", "lo", "hi"), boot(vals[g][m]))) for m in metrics}
        s = out["gates"][g]
        md.append(f"| {LABEL[g]} | {s['calls']['mean']:.2f} [{s['calls']['lo']:.2f},{s['calls']['hi']:.2f}] "
                  f"| {s['overhead_s']['mean']:.1f} [{s['overhead_s']['lo']:.1f},{s['overhead_s']['hi']:.1f}] "
                  f"| {s['acc']['mean']:.0f} [{s['acc']['lo']:.0f},{s['acc']['hi']:.0f}] "
                  f"| {s['in_time']['mean']:.0f} [{s['in_time']['lo']:.0f},{s['in_time']['hi']:.0f}] |")
    md += ["", "Paired vs ours (same approaches):", "| Gate | Δcalls | p | Δoverhead s | p | Δacc pts | p |", "|---|---|---|---|---|---|---|"]
    for g in gates:
        if g == a.ours:
            continue
        rec = {}
        cells = []
        for m in ("calls", "overhead_s", "acc"):
            test, p = paired_p(vals[g][m], vals[a.ours][m])
            delta = float(np.mean(vals[g][m]) - np.mean(vals[a.ours][m]))
            rec[m] = {"delta": delta, "p": p, "test": test}
            cells += [f"{delta:+.2f}", f"{p:.3g}"]
        out["paired_vs_ours"][g] = rec
        md.append(f"| {LABEL[g]} | " + " | ".join(cells) + " |")
    from pathlib import Path
    Path(a.out + ".json").write_text(json.dumps(out, indent=2))
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
