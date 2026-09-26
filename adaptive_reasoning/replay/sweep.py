"""
sweep.py — E5.  Every gate over every replayed approach → Table I + Fig tradeoff.

Reads the E1 npz dumps + ar_extra.csv, runs each gate variant frame-by-frame
(same evidence, same FSM, only the trigger differs), and scores each fired
call.  Two labeling modes:

  --oracle       dry-run scoring, no VLM: a call is CORRECT iff the fired
                 plate's R_struct at fire time is 1 (it literally addresses
                 the goal) and the annotated decision is what a correct answer
                 would be; it is WASTED otherwise.  Good for pipeline bring-up
                 and for choosing the τ grid before spending money.
  (default)      real scoring via reasoner cache: the fired (plate, frame)
                 crop set is answered by the cached VLM response (cache miss →
                 live call).  This is the paper mode; after E2's labeling pass
                 the sweep is free.  [Requires the jsonl+crop replay hooks —
                 wired when E2 lands.]

Metrics per approach: calls, wasted calls, correct decision (first applicable
answer == annotation), lead time λ = (junction − t_decided)/fps, false fires
on `sign_situation ∈ {none, irrelevant_only}` controls.

Outputs: <out>/table_main.md + .tex, <out>/tradeoff.(pdf|png|npz)

Usage:
  python -m adaptive_reasoning.replay.sweep --dumps $SCRATCH/ar_replay \
      --csv annotations/ar_extra.csv --out figs/sweep --oracle \
      --taus 0.1 0.2 ... --tau-star 0.55
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..gate import EvidenceGate, PlateEvidence, VLMResponse, new_goal
from ..memory import Memory
from . import figstyle
from .figstyle import C, COL_W
from .baselines import make_gate

GATES = ["always", "periodic", "reactive", "sufficiency", "relevance", "ours"]
GATE_LABEL = {"always": "Always-invoke", "periodic": "Periodic",
              "reactive": "Reactive (necessity-only)", "sufficiency": "Sufficiency-only",
              "relevance": "Relevance-only", "ours": "Ours (Eq. gate)"}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

@dataclass
class Approach:
    bag: str
    frames: np.ndarray
    plates: list[dict]              # id,label,R,R_struct,ell,resolvable per track
    fps: float
    junction: int | None            # grid index
    decision: str
    goal: str
    control: bool                   # none / irrelevant_only row → false-fire control


def load_approaches(dumps: Path, csv_path: str) -> list[Approach]:
    out = []
    for row in csv.DictReader(open(csv_path)):
        f = dumps / f"{row['bag']}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        plates, i = [], 0
        while f"plate{i}_id" in d:
            plates.append({k: d[f"plate{i}_{k}"] for k in
                           ("id", "label", "R", "R_struct", "ell", "resolvable")})
            i += 1
        out.append(Approach(
            bag=row["bag"], frames=d["frame_idx"], plates=plates,
            fps=float(d["fps"]) if "fps" in d else int(row["fps"]),
            junction=int(d["junction_frame"]) if "junction_frame" in d else None,
            decision=row["decision"], goal=row["goal"],
            control=row["sign_situation"] in ("none", "irrelevant_only")))
    return out


# ----------------------------------------------------------------------------
# Scoring one approach under one gate
# ----------------------------------------------------------------------------

@dataclass
class Score:
    calls: int = 0
    wasted: int = 0
    correct: bool = False
    lead_s: float | None = None
    false_fires: int = 0


def oracle_answer(ap: Approach, plate_idx: int, t: int) -> VLMResponse:
    """Dry-run stand-in for the VLM: applicable iff the fired plate literally
    addresses the goal (R_struct==1) at fire time; answer = annotation."""
    p = ap.plates[plate_idx]
    if p["R_struct"][t] >= 1.0 and ap.decision in ("turn_left", "turn_right", "straight", "stop"):
        return VLMResponse(str(p["id"]), True, ap.decision)
    return VLMResponse(str(p["id"]), False)


def run_one(gate: EvidenceGate, ap: Approach, latency_frames: int) -> Score:
    mem, s, sc = Memory(ap.goal), new_goal(), Score()
    T = len(ap.frames)
    pending: tuple[int, VLMResponse] | None = None
    idx_of = {str(p["id"]): k for k, p in enumerate(ap.plates)}
    for t in range(T):
        if pending and t >= pending[0]:
            s, upds = gate.on_response(s, pending[1], t)
            for u in upds:
                mem.apply(u)
            pending = None
        evid = [PlateEvidence(str(p["id"]), float(p["R"][t]), float(p["ell"][t]))
                for p in ap.plates if p["ell"][t] > 0 or p["R"][t] > 0]
        r = gate.tick(s, evid, mem, t)
        for u in r.memory_updates:
            mem.apply(u)
        s = r.state
        if r.fire:
            sc.calls += 1
            if ap.control:
                sc.false_fires += 1
            resp = oracle_answer(ap, idx_of.get(r.best.plate_id, 0), t) \
                if r.best and r.best.plate_id in idx_of else VLMResponse("scene", False)
            if not resp.applicable:
                sc.wasted += 1
            pending = (t + latency_frames, resp)
    if s.decision is not None:
        sc.correct = (s.decision == ap.decision)
        if ap.junction is not None and s.decided_at is not None:
            sc.lead_s = (ap.junction - s.decided_at) / ap.fps
    return sc


def evaluate(gate_name: str, tau: float, aps: list[Approach], latency_s: float,
             period: int, detail_out: list | None = None) -> dict:
    rows = []
    for ap in aps:
        gate = make_gate(gate_name, tau, period=period)   # fresh state per approach
        r = run_one(gate, ap, latency_frames=max(int(latency_s * ap.fps), 1))
        rows.append(r)
        if detail_out is not None:
            detail_out.append({"gate": gate_name, "bag": ap.bag, "goal": ap.goal,
                               "calls": r.calls, "wasted": r.wasted,
                               "correct": int(r.correct),
                               "lead_s": round(r.lead_s, 2) if r.lead_s is not None else ""})
    n_dec = [r for r, ap in zip(rows, aps) if not ap.control]
    leads = [r.lead_s for r in n_dec if r.lead_s is not None]
    return {
        "gate": gate_name, "tau": tau,
        "calls": float(np.mean([r.calls for r in rows])),
        "wasted_pct": 100 * float(np.sum([r.wasted for r in rows])
                                  / max(np.sum([r.calls for r in rows]), 1)),
        "acc_pct": 100 * float(np.mean([r.correct for r in n_dec])) if n_dec else 0.0,
        "lead_s": float(np.mean(leads)) if leads else float("nan"),
        "false_fires": int(np.sum([r.false_fires for r in rows])),
    }


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------

def write_table(rows: list[dict], out: Path) -> None:
    hdr = f"| Gate | Calls/appr. | Wasted % | Acc % | λ (s) | False fires |\n|---|---|---|---|---|---|\n"
    md = hdr + "\n".join(
        f"| {GATE_LABEL[r['gate']]}{f' (τ={r_tau})' if (r_tau := r['tau']) and r['gate'] in ('ours','sufficiency','relevance') else ''} "
        f"| {r['calls']:.2f} | {r['wasted_pct']:.0f} | {r['acc_pct']:.0f} | "
        f"{r['lead_s']:.1f} | {r['false_fires']} |" for r in rows)
    (out / "table_main.md").write_text(md + "\n")
    tex = "\n".join(
        f"{GATE_LABEL[r['gate']]} & {r['calls']:.2f} & {r['wasted_pct']:.0f} & "
        f"{r['acc_pct']:.0f} & {r['lead_s']:.1f} & {r['false_fires']} \\\\" for r in rows)
    (out / "table_main.tex").write_text(tex + "\n")
    print(md)


def fig_tradeoff(sweep_rows: list[dict], baseline_rows: list[dict], out_stem: Path) -> None:
    figstyle.use()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.7), sharex=True,
                                 gridspec_kw={"wspace": 0.45})
    xs = [r["calls"] for r in sweep_rows]
    a1.plot(xs, [r["acc_pct"] for r in sweep_rows], "-o", ms=2.5, color=C["ours"],
            label="ours (τ sweep)")
    a2.plot(xs, [r["lead_s"] for r in sweep_rows], "-o", ms=2.5, color=C["ours"])
    marks = {"always": "s", "periodic": "D", "reactive": "^",
             "sufficiency": "v", "relevance": "P"}
    for r in baseline_rows:
        a1.plot(r["calls"], r["acc_pct"], marks[r["gate"]], ms=3.5,
                color=C["baseline"], label=GATE_LABEL[r["gate"]])
        a2.plot(r["calls"], r["lead_s"], marks[r["gate"]], ms=3.5, color=C["baseline"])
    a1.set_xlabel("VLM calls / approach"); a2.set_xlabel("VLM calls / approach")
    a1.set_ylabel("decision accuracy [%]"); a2.set_ylabel("lead time λ [s]")
    a1.legend(loc="lower right", fontsize=5.2, handlelength=1.0)
    figstyle.save(fig, out_stem, arrays={
        "sweep": np.array([(r["tau"], r["calls"], r["acc_pct"], r["lead_s"]) for r in sweep_rows]),
        "baselines": np.array([(r["calls"], r["acc_pct"], r["lead_s"]) for r in baseline_rows]),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="figs/sweep")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--taus", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--tau-star", type=float, required=True,
                    help="operating point for Table I (choose on train split)")
    ap.add_argument("--latency", type=float, default=3.0, help="VLM latency L in s (E3)")
    ap.add_argument("--period", type=int, default=20, help="Periodic baseline, frames")
    a = ap.parse_args()
    if not a.oracle:
        raise SystemExit("cached-VLM scoring lands with E2; run with --oracle for now")

    aps = load_approaches(Path(a.dumps), a.csv)
    if not aps:
        raise SystemExit("no npz dumps matched the csv — run dump_features first")
    print(f"{len(aps)} approaches ({sum(ap_.control for ap_ in aps)} controls)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sweep = [evaluate("ours", t, aps, a.latency, a.period) for t in a.taus]
    base = [evaluate(g, a.tau_star, aps, a.latency, a.period)
            for g in GATES if g != "ours"]
    detail: list = []
    table = base + [evaluate("ours", a.tau_star, aps, a.latency, a.period, detail_out=detail)]
    import csv as _csv
    with open(out / "per_approach.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["gate", "bag", "goal", "calls", "wasted", "correct", "lead_s"])
        w.writeheader(); w.writerows(detail)
    write_table(table, out)
    fig_tradeoff(sweep, base, out / "tradeoff")
    print(f"wrote {out}/table_main.md, {out}/tradeoff.pdf")


if __name__ == "__main__":
    main()
