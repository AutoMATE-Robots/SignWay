"""
eval_gates.py — every trigger policy on the held-out approaches, scored with
REAL cached VLM answers.  Replaces sweep.py's oracle mode for the paper.

Answers come from the crop-pass ledger (label_crops.py): for a fire on plate p
at grid frame t, the answer is the crop_pass row for (bag, p) whose frame is
nearest to t (stored every 5th sighting → ≤0.5 s away; stated approximation).
No live calls.  A fire on a plate that has no stored answer counts as
not-applicable and is tallied in `answer_misses`.

Per approach × gate:  calls (incl. SignScene parse calls), wasted calls
(answered not-applicable or wrong plate), correct (final decision == annotated),
in_time (correct AND decided before the junction), lead_s, overhead_s
(successful + wasted calls × L), cheap_s (frames × cheap-layer seconds, only
for gates that run OCR).  L = median measured image-call latency from the
ledger unless --latency overrides.

θ_KFC (IROS-style): --fit-kfc <dumps-of-held-in-bags> picks the threshold so
that the scene-change test fires on 46% of text-visible frames (their reported
System-Two share), never touching the held-out set.

Usage:
  python -m adaptive_reasoning.replay.eval_gates --dumps $SCRATCH/ar_replay \\
      --csv adaptive_reasoning/annotations/ar_extra-sep-4.csv \\
      --ledger $SCRATCH/ar_eval/ledger.jsonl --out $SCRATCH/ar_eval/gates \\
      --tau 0.55 --fit-kfc rosbag2_2026_08_20-17_12_19 rosbag2_2026_08_20-17_13_22 ...
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..evidence.resolve import Resolution
from ..gate import EvidenceGate, PlateEvidence, VLMResponse, new_goal
from ..memory import Memory
from .baselines import kfc_distance, make_gate

GATES = ["always", "always_uncond", "continuous05", "periodic5s", "reactive", "iros", "signscene",
         "relevance", "ours", "signnav"]
LABEL = {"always": "Always-invoke", "continuous05": "Continuous 0.5 Hz", "periodic5s": "Periodic 5 s",
         "reactive": "Reactive", "iros": "IROS-style†", "signscene": "SignScene-style†",
         "relevance": "Relevance-only (abl.)", "ours": "SignWay (ours)",
         "signnav": "SignNav-style‡", "always_uncond": "Always-invoke (unconditional)"}
USES_CHEAP = {"reactive", "iros", "signscene", "relevance", "ours"}
# Memory semantics per policy.  OURS consumes a sign after a not-applicable
# answer (never re-ask the same sign for the same goal — part of the method).
# The published baselines DO re-query when their trigger fires again (IROS on
# the next scene change; schedule/reactive gates on the next tick), so giving
# them our memory rule would misrepresent them.  Relevance-only is our ablation
# and keeps our rule; SignScene parses each sign once and grounds once.
CONSUME_NOT_APPLICABLE = {"ours", "relevance", "signscene"}


@dataclass
class Approach:
    bag: str
    building: str
    goal: str
    decision: str
    fps: float
    frames: np.ndarray
    junction: int | None
    plates: list[dict]
    scene: np.ndarray | None
    answers: dict[str, list[tuple[int, bool, str | None]]]   # plate_id → [(grid_t, applicable, dir)]
    fast_dir: dict[tuple[str, int], str]                      # (plate_id, grid_t) → parser's own direction
    arrows: dict[int, list[str]]                              # grid_t → arrow directions visible (all plates)
    labels: dict[str, str]                                    # plate_id → longest OCR text seen


def load_ledger_index(ledger: Path, model_filter: str = ""
                      ) -> dict[tuple[str, str], list[tuple[int, bool, str | None, str]]]:
    """(bag, plate_id) → [(raw_frame, applicable, direction, goal)] from crop_pass rows
    (deduped by key).  `model_filter` restricts to rows whose model tag contains it,
    so a ledger holding several models' passes scores each model on its own answers."""
    rows: dict[str, dict] = {}
    for line in open(ledger):
        r = json.loads(line)
        if r.get("eval") == "crop_pass" and (not model_filter or model_filter in r.get("model", "")):
            rows[r["key"]] = r                 # last write wins (cached re-logs are identical)
    idx: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows.values():
        idx[(r["bag"], r["plate_id"])].append((int(r["frame"]), bool(r["applicable"]),
                                               r.get("direction"), r["goal"]))
    return idx


def load_approaches(dumps: Path, csv_path: str, ledger_idx) -> list[Approach]:
    out = []
    for row in csv.DictReader(open(csv_path)):
        f = dumps / f"{row['bag']}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        frames = d["frame_idx"]; grid = {int(x): j for j, x in enumerate(frames)}
        plates, i = [], 0
        while f"plate{i}_id" in d:
            plates.append({k: d[f"plate{i}_{k}"] for k in ("id", "R", "R_struct", "ell", "resolvable")})
            i += 1
        answers: dict[str, list] = {}
        for p in plates:
            pid = str(p["id"])
            rows_ = [(grid[fr], ap, di) for fr, ap, di, g in ledger_idx.get((row["bag"], pid), [])
                     if g == row["goal"] and fr in grid]
            if rows_:
                # key on the frame alone — `direction` is None for not-applicable
                # answers and tuple comparison would fall through to None < str
                answers[pid] = sorted(rows_, key=lambda t: t[0])
        # what the PARSER itself resolved, per frame — the only thing the fast path may be credited with
        fast_dir: dict[tuple[str, int], str] = {}
        arrows: dict[int, list[str]] = {}
        labels: dict[str, str] = {}
        jl = dumps / f"{row['bag']}.jsonl"
        if jl.exists():
            for line in open(jl):
                r = json.loads(line)
                if r.get("resolvable") and r.get("direction") and int(r["frame"]) in grid:
                    fast_dir[(r["plate_id"], grid[int(r["frame"])])] = r["direction"]
                if r.get("arrows") and int(r["frame"]) in grid:
                    arrows.setdefault(grid[int(r["frame"])], []).extend(r["arrows"])
                if len(r.get("text", "")) > len(labels.get(r["plate_id"], "")):
                    labels[r["plate_id"]] = r["text"]
        out.append(Approach(
            bag=row["bag"], building=row.get("building", ""), goal=row["goal"], decision=row["decision"],
            fps=float(d["fps"]) if "fps" in d else 10.0, frames=frames,
            junction=int(d["junction_frame"]) if "junction_frame" in d else None,
            plates=plates, scene=d["scene_grid"] if "scene_grid" in d else None, answers=answers,
            fast_dir=fast_dir, labels=labels, arrows=arrows))
    return out


@dataclass
class Score:
    calls: int = 0
    parse_calls: int = 0
    wasted: int = 0
    correct: bool = False
    in_time: bool = False
    decided_after_end: bool = False
    lead_s: float | None = None
    misses: int = 0
    decided_at: int | None = None


def lookup(ap: Approach, plate_id: str, t: int) -> tuple[bool, str | None, bool]:
    """Nearest stored answer for the plate; (applicable, direction, found)."""
    rows = ap.answers.get(plate_id)
    if not rows:
        return False, None, False
    j = min(range(len(rows)), key=lambda k: abs(rows[k][0] - t))
    return rows[j][1], rows[j][2], True


ARROW_TO_DECISION = {"left": "turn_left", "right": "turn_right", "up": "straight", "straight": "straight"}


def run_always_unconditional(ap: Approach, latency_frames: int) -> Score:
    """Always-invoke in the literal sense: issue a call whenever none is in flight,
    whether or not a sign is visible.  Steps with no legible goal-relevant plate
    still cost a call (the robot cannot know in advance that the view is useless);
    such calls return nothing and are counted as wasted.  This is the honest upper
    bound on the cost of unconditional reasoning; the variant that only fires when
    a sign is in view is reported separately as Reactive."""
    sc, pending, decided = Score(), None, None
    for t in range(len(ap.frames)):
        if pending and t >= pending[0]:
            applicable, direction = pending[1], pending[2]
            if applicable and direction is not None and decided is None:
                decided = (direction, t)
            pending = None
        if pending is None:
            sc.calls += 1
            best, best_R = None, 0.0
            for p in ap.plates:                       # ask about the most relevant plate in view
                R = float(p["R"][t])
                if R > best_R:
                    best, best_R = str(p["id"]), R
            if best is None or best_R < 0.5:
                sc.wasted += 1                        # nothing worth asking about: call is spent
                pending = (t + latency_frames, False, None)
            else:
                applicable, direction, found = lookup(ap, best, t)
                if not found:
                    sc.misses += 1
                if not applicable or direction is None:
                    sc.wasted += 1
                pending = (t + latency_frames, applicable and direction is not None, direction)
    if decided is not None:
        sc.decided_at = decided[1]
        sc.correct = (decided[0] == ap.decision)
        if ap.junction is not None:
            sc.lead_s = (ap.junction - sc.decided_at) / ap.fps
            sc.in_time = sc.correct and sc.decided_at <= ap.junction
        else:
            sc.in_time = sc.correct
    return sc


def run_signnav(ap: Approach) -> Score:
    """SignNav-style: follows detected directional arrows with NO text grounding and
    no VLM (their goal→arrow association is delegated to an external parser, which we
    do not grant).  Commits when exactly one distinct arrow direction is visible for
    3 consecutive grid frames; frames with several distinct directions are ambiguous
    for an arrow-only policy and yield no decision.  Zero calls, zero stranded time."""
    sc = Score()
    streak_dir, streak = None, 0
    for t in range(len(ap.frames)):
        dirs = {ARROW_TO_DECISION.get(a) for a in ap.arrows.get(t, []) if ARROW_TO_DECISION.get(a)}
        if len(dirs) == 1:
            d = next(iter(dirs))
            streak = streak + 1 if d == streak_dir else 1
            streak_dir = d
            if streak >= 3 and sc.decided_at is None:
                sc.decided_at = t
                sc.correct = (d == ap.decision)
                sc.in_time = ap.junction is None or t <= ap.junction
        else:
            streak_dir, streak = None, 0
    return sc


def run_one(gate: EvidenceGate, ap: Approach, latency_frames: int, binarize_R: bool = True,
            consume_na: bool = True, fuzzy_memory: bool = True, extra_frames: int = 0) -> Score:
    """extra_frames > 0 removes the approach deadline: after the recorded frames end,
    the policy may keep reasoning on the final view until it decides (or the budget
    runs out).  Calls and stranded time still accrue, so cost is unchanged; only the
    truncation at the junction is lifted.  Used to report a baseline's ceiling when
    its call structure does not fit inside the approach."""
    mem, s, sc = Memory(ap.goal, fuzzy=fuzzy_memory), new_goal(), Score()
    T = len(ap.frames)
    pending = None
    for t in range(T + extra_frames):
        ti = min(t, T - 1)                      # past the end: keep the last view
        if pending and t >= pending[0]:
            s, upds = gate.on_response(s, pending[1], t)
            for u in upds:
                if u.outcome == "not_applicable" and not consume_na:
                    continue                      # baseline may re-query this sign
                mem.apply(u)
            pending = None
        if s.decision is not None:
            break                                # decided: stop the clock
        evid = []
        for p in ap.plates:
            if p["ell"][ti] <= 0 and p["R"][ti] <= 0:
                continue
            R = float(p["R"][ti])
            if binarize_R:
                R = 1.0 if R >= 0.5 else 0.0
            res = None
            pid = str(p["id"])
            fd = ap.fast_dir.get((pid, ti))
            if bool(p["resolvable"][ti]) and fd:
                # credit the fast path ONLY with the direction the parser produced
                res = Resolution("resolved", vla_prompt=fd, source="line", channel="struct")
            evid.append(PlateEvidence(pid, R, float(p["ell"][ti]), res, text=ap.labels.get(pid, "")))
        r = gate.tick(s, evid, mem, t)
        for u in r.memory_updates:
            mem.apply(u)
        s = r.state
        if r.fire:
            pid = r.best.plate_id
            if pid.endswith("#parse"):
                sc.parse_calls += 1; sc.calls += 1
                pending = (t + latency_frames, VLMResponse(pid, False))
                continue
            sc.calls += 1
            applicable, direction, found = lookup(ap, pid, ti)
            if not found:
                sc.misses += 1
            if not applicable or direction is None:
                sc.wasted += 1
            pending = (t + latency_frames, VLMResponse(pid, applicable and direction is not None, direction))
    if pending:      # response after the bag ended
        s, _ = gate.on_response(s, pending[1], pending[0])
    if s.decision is not None:
        sc.correct = (s.decision == ap.decision)
        sc.decided_at = s.decided_at
        sc.decided_after_end = bool(s.decided_at is not None and s.decided_at >= T)
        if ap.junction is not None and s.decided_at is not None:
            sc.lead_s = (ap.junction - s.decided_at) / ap.fps
            sc.in_time = sc.correct and s.decided_at <= ap.junction
        else:
            sc.in_time = sc.correct
    return sc


def fit_theta_kfc(dumps: Path, bags: list[str], share: float = 0.464, lag: int = 10) -> float:
    """θ so that the scene-change test fires on `share` of text-visible frames
    (distance to the frame `lag` steps earlier as the trigger-reference proxy)."""
    dists = []
    for b in bags:
        f = dumps / f"{b}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        if "scene_grid" not in d:
            continue
        g = d["scene_grid"]; i = 0; vis = np.zeros(len(g), bool)
        while f"plate{i}_id" in d:
            vis |= d[f"plate{i}_ell"] > 0; i += 1
        for t in range(lag, len(g)):
            if vis[t]:
                dists.append(kfc_distance(g[t], g[t - lag]))
    if not dists:
        raise SystemExit("no scene grids in the fit bags — re-dump them with the current code")
    return float(np.quantile(dists, 1.0 - share))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, required=True, help="legibility threshold for ours (set on held-in data)")
    ap.add_argument("--gates", nargs="+", default=GATES)
    ap.add_argument("--latency", type=float, default=None, help="L in s; default = ledger median (image calls)")
    ap.add_argument("--cheap-per-frame", type=float, default=0.25, help="cheap-layer seconds per processed frame (dump f/s)")
    ap.add_argument("--theta-kfc", type=float, default=None)
    ap.add_argument("--fit-kfc", nargs="*", default=[], help="held-in bag names with scene grids")
    ap.add_argument("--theta-parse", type=float, default=0.5)
    ap.add_argument("--no-fast-path", action="store_true")
    ap.add_argument("--model", default="", help="score only ledger rows from this model (substring, e.g. 32B)")
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument("--drop-uncovered", action="store_true",
                    help="evaluate only the bags whose answers are >= min-coverage, and print which "
                         "bags were dropped (n is then the covered subset — state it in the caption)")
    ap.add_argument("--no-deadline", action="store_true",
                    help="let policies keep reasoning on the final view after the approach ends; "
                         "reports the ceiling of a method whose call structure does not fit in time")
    ap.add_argument("--no-memory", action="store_true",
                    help="ablation: OUR gate may re-ask a sign after a not-applicable answer (memory off)")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ledger_idx = load_ledger_index(Path(a.ledger), a.model)
    if not ledger_idx:
        raise SystemExit(f"no crop_pass rows in the ledger matching model filter {a.model!r}")
    # coverage guard: every stored plate should have an answer; scoring a missing
    # answer as not-applicable silently turns an incomplete cache into a "result"
    have = set(ledger_idx)
    want, per_bag = set(), {}
    for row in csv.DictReader(open(a.csv)):
        jl = Path(a.dumps) / f"{row['bag']}.jsonl"
        if not jl.exists():
            continue
        pids = {json.loads(l)["plate_id"] for l in open(jl) if '"crop": "' in l}
        want |= {(row["bag"], pid) for pid in pids}
        per_bag[row["bag"]] = len([pid for pid in pids if (row["bag"], pid) in have]) / max(len(pids), 1)
    cov = len(want & have) / max(len(want), 1)
    print(f"answer coverage: {100*cov:.0f}% of stored plates ({len(want & have)}/{len(want)})")
    low = {b: f"{100*c:.0f}%" for b, c in per_bag.items() if c < 0.9}
    if low:
        print(f"  bags below 90%: {low}")
    drop = set()
    if a.drop_uncovered:
        drop = {b for b, c in per_bag.items() if c < a.min_coverage}
        if drop:
            print(f"  DROPPING {len(drop)} uncovered bags: {sorted(drop)}")
            kept = {b: c for b, c in per_bag.items() if b not in drop}
            print(f"  evaluating {len(kept)} bags (state this n in the caption)")
    elif cov < a.min_coverage:
        raise SystemExit(f"coverage {100*cov:.0f}% < {100*a.min_coverage:.0f}% — the crop pass is incomplete; "
                         f"finish it (or pass --min-coverage 0 to override, results will be artifacts)")
    aps = load_approaches(Path(a.dumps), a.csv, ledger_idx)
    if drop:
        aps = [x for x in aps if x.bag not in drop]
    if not aps:
        raise SystemExit("no approaches")
    # latency
    if a.latency is None:
        lats = [json.loads(l)["latency_s"] for l in open(a.ledger)
                if '"cached": false' in l and '"crop_pass"' in l and (a.model in l)]
        L = float(st.median(lats)) if lats else 3.0
    else:
        L = a.latency
    theta = a.theta_kfc
    if theta is None and a.fit_kfc:
        theta = fit_theta_kfc(Path(a.dumps), a.fit_kfc)
    if theta is None and "iros" in a.gates:
        raise SystemExit("IROS-style needs --theta-kfc or --fit-kfc <held-in bags>")
    print(f"{len(aps)} approaches | model filter={a.model!r} | L={L:.2f}s | theta_kfc={theta} | tau={a.tau}")

    rows = []
    for g in a.gates:
        for ap_ in aps:
            if g == "iros" and ap_.scene is None:
                continue
            if g == "signnav":
                gate = None
                sc = run_signnav(ap_)
            elif g == "always_uncond":
                gate = None
                sc = run_always_unconditional(ap_, latency_frames=max(int(L * ap_.fps), 1))
            else:
                gate = make_gate(g, a.tau, scene_grid=ap_.scene, theta_kfc=theta,
                                 theta_parse=a.theta_parse, fast_path=not a.no_fast_path)
                lf = max(int(L * ap_.fps), 1)
                sc = run_one(gate, ap_, latency_frames=lf,
                             consume_na=(g in CONSUME_NOT_APPLICABLE) and not (a.no_memory and g in ("ours", "relevance")),
                             fuzzy_memory=(g in ("ours", "relevance")),
                             extra_frames=(12 * lf if a.no_deadline else 0))
            succ = sc.calls - sc.wasted - sc.parse_calls
            rows.append({"gate": g, "bag": ap_.bag, "building": ap_.building, "goal": ap_.goal,
                         "calls": sc.calls, "parse_calls": sc.parse_calls, "wasted": sc.wasted,
                         "answer_misses": sc.misses, "correct": int(sc.correct), "in_time": int(sc.in_time),
                         "decided": int(sc.decided_at is not None),
                         "decided_after_end": int(getattr(sc, "decided_after_end", False)),
                         "lead_s": round(sc.lead_s, 2) if sc.lead_s is not None else "",
                         "vlm_s": round(succ * L, 2), "wasted_s": round((sc.wasted + sc.parse_calls) * L, 2),
                         "overhead_s": round(sc.calls * L, 2),
                         "duration_s": round(len(ap_.frames) / ap_.fps, 2),
                         "cheap_s": round(len(ap_.frames) * a.cheap_per_frame, 2) if g in USES_CHEAP else 0.0})
    with open(out / "per_approach.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # table
    lines = ["| Gate | Calls/ep. | Wasted % | Acc % | In-time % | λ (s) | Overhead (s) | Cheap (s) |",
             "|---|---|---|---|---|---|---|---|"]
    tex = []
    summary = {}
    for g in a.gates:
        R = [r for r in rows if r["gate"] == g]
        if not R:
            continue
        n = len(R); calls = np.mean([r["calls"] for r in R])
        wasted = 100 * sum(r["wasted"] + r["parse_calls"] for r in R) / max(sum(r["calls"] for r in R), 1)
        acc = 100 * np.mean([r["correct"] for r in R]); intime = 100 * np.mean([r["in_time"] for r in R])
        leads = [r["lead_s"] for r in R if r["lead_s"] != ""]
        lam = float(np.mean(leads)) if leads else float("nan")
        ovh = np.mean([r["overhead_s"] for r in R]); cheap = np.mean([r["cheap_s"] for r in R])
        summary[g] = dict(n=n, calls=calls, wasted_pct=wasted, acc=acc, in_time=intime, lead_s=lam,
                          overhead_s=ovh, cheap_s=cheap, misses=int(sum(r["answer_misses"] for r in R)))
        lines.append(f"| {LABEL[g]} | {calls:.2f} | {wasted:.0f} | {acc:.0f} | {intime:.0f} | {lam:.1f} | {ovh:.1f} | {cheap:.1f} |")
        tex.append(f"{LABEL[g]} & {calls:.2f} & {wasted:.0f} & {acc:.0f} & {intime:.0f} & {lam:.1f} & {ovh:.1f} \\\\")
    (out / "table_main.md").write_text("\n".join(lines) + "\n")
    (out / "table_main.tex").write_text("\n".join(tex) + "\n")
    (out / "summary.json").write_text(json.dumps({"L": L, "theta_kfc": theta, "tau": a.tau, "model": a.model,
                                                  "n_approaches": len(aps), "gates": summary}, indent=2))
    print("\n".join(lines))
    print(f"\nanswer misses (fires on plates with no stored answer): "
          f"{ {g: summary[g]['misses'] for g in summary} }")
    print(f"wrote {out}/per_approach.csv, table_main.md/.tex, summary.json")


if __name__ == "__main__":
    main()