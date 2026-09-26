"""
fig_heatmap.py — "robustness across sign-semantic challenges" heatmap.

Columns = curated condition folders; rows = reasoning STRATEGIES, all run with
the same model so only the payload/prompt design differs (Ajay's convention:
we copy each paper's approach; lineage is cited in the caption).  Every cell
prints accuracy AND n.  Every call is cached and ledgered like everything else.

Folder layout (one CSV per condition):
  <root>/<condition>/goals.csv   columns: image, goal, expected
      expected ∈ {turn_left, turn_right, straight, stop, not_applicable}
      one image may appear on several rows with different goals.
  images referenced relative to the condition folder.

Strategies (rows):
  fast_path      structural parse + OCR arrows, no VLM (answers only when a
                 unique arrow resolves; otherwise counted as no-answer)
  iros_style     FULL FRAME + OCR text + coarse spatial descriptor in the prompt
                 (their System-Two payload; spatial descriptor here is the
                 plate's position in view — their SegFormer zoning is not
                 reproduced, stated in the caption)
  signscene_style aligned SIGN CROP + symbol-dictionary prompt (arrow legend)
  crop_notext    sign crop, OCR text withheld (payload ablation)
  signway        ours: fast path if the plate resolves, else crop + OCR text

Usage (ar env, GPU node, server up):
  python -m adaptive_reasoning.replay.fig_heatmap --root figs/heatmap \\
      --out $SCRATCH/ar_eval/heatmap --model Qwen/Qwen2.5-VL-7B-Instruct \\
      --base-url http://localhost:8000/v1 --reasoning none --rpm 0
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ..evidence.detect import DetectConfig, detect, payload_crop, payload_crops
from ..evidence.relevance import Goal, relevance_lines
from ..evidence.resolve import Plate, PlateLine, resolve
from ..reasoner import PROMPT, FakeVLM, OpenAICompatVLM, Reasoner, structural_hint
from . import figstyle
from .figstyle import C, COL_W

# ----------------------------------------------------------------------------
# Prompt variants (the only thing that differs between VLM rows besides payload)
# ----------------------------------------------------------------------------

PROMPT_SIGNSCENE = PROMPT.replace(
    "From the attached image crops (sign close-ups) and scene frame, decide whether",
    "Symbol dictionary: ← = turn left, → = turn right, ↑ = go straight ahead, "
    "↗/↖ = ahead then right/left, ↓/↩ = behind you.\n"
    "From the attached sign close-up, decide whether")

PROMPT_IROS = PROMPT.replace(
    "Recognized text on the candidate sign(s) (may contain OCR errors):",
    "Scene layout: {spatial}\nRecognized text in the scene (may contain OCR errors):"
).replace(
    "From the attached image crops (sign close-ups) and scene frame, decide whether",
    "From the attached camera frame, decide whether")

PROMPT_NOTEXT = PROMPT.replace(
    "Recognized text on the candidate sign(s) (may contain OCR errors):\n{plate_texts}\n", "")

STRATEGIES = ["iros_style", "signscene_style", "crop_notext", "signway", "signway_tight", "fast_path"]
ROW_LABEL = {"fast_path": "Fast path only (no VLM)", "iros_style": "IROS-style†",
             "signscene_style": "SignScene-style†", "crop_notext": "Crop, no text grounding",
             "signway": "SignWay (ours)", "signway_tight": "SignWay, tight crop only",
             "signnav": "SignNav (analytic)‡"}   # ‡ learned policy, not runnable: scope only


def spatial_descriptor(box, W: int, H: int) -> str:
    if box is None:
        return "no sign localized"
    cx = (box[0] + box[2]) / 2 / W
    cy = (box[1] + box[3]) / 2 / H
    horiz = "left" if cx < 0.4 else "right" if cx > 0.6 else "centre"
    vert = "high on the wall" if cy < 0.4 else "at eye level"
    return f"a sign {vert}, {horiz} of view"


def run_case(img, goal_text: str, plates, strategy: str, reasoners: dict, cfg, meta: dict) -> dict:
    goal = Goal.parse(goal_text)
    scored = []
    for p in plates:
        rel = relevance_lines([l.text for l in p.lines], goal)
        res = resolve(Plate([PlateLine(l.text, list(l.arrows)) for l in p.lines]), goal, rel=rel)
        scored.append((p, rel, res))
    if not scored:
        return {"answer": None, "source": "no plates"}
    # Plate choice.  Relevance first; but when the parser finds NO match (OCR
    # garbled the range, e.g. "2-1011 to2-140") every plate ties at R=0 and the
    # first one — an EXIT sign, a building name — would be cropped.  Break ties
    # by directory-likeness: room-range tokens, line count, arrows, area.
    def rank(k):
        q, r_, _ = scored[k]
        txt = q.text
        ranges = len(re.findall(r"\d[\d-]*\s*(?:to|-|–)\s*\d[\d-]*", txt, flags=re.I))
        area = ((q.box[2] - q.box[0]) * (q.box[3] - q.box[1])) if q.box else 0.0
        return (round(r_.score, 2), ranges, len(q.lines), sum(len(l.arrows) for l in q.lines), area)
    i = max(range(len(scored)), key=rank)
    p, rel, res = scored[i]
    boxes = [q.box for q, _, _ in scored if q.box]
    H, W = img.shape[:2]
    m = {**meta, "plate": p.text, "R": rel.score}

    if strategy == "fast_path":
        return {"answer": res.vla_prompt if (res.resolved and res.vla_prompt) else None,
                "source": "fast path" if res.resolved else "unresolved"}
    if strategy == "signway" and res.resolved and res.vla_prompt:
        return {"answer": res.vla_prompt, "source": "fast path"}

    crop = payload_crop(img, p.box, boxes, cfg) if p.box else img
    r = reasoners[strategy]
    if strategy == "iros_style":
        r.prompt_template = PROMPT_IROS.replace("{spatial}", spatial_descriptor(p.box, W, H))
        ans = r.ask(goal_text, [q.text for q, _, _ in scored], [img], meta=m)   # full frame, all text
    elif strategy == "crop_notext":
        ans = r.ask(goal_text, [], [crop], meta=m)
    elif strategy == "signway_tight":    # ablation: tight crop only + OCR text + parser grounding
        ans = r.ask(goal_text, [p.text], [crop], meta=m, hint=structural_hint(p.text, goal_text))
    elif strategy == "signway":          # ours: tight crop + 2.5× context crop + OCR text + parser grounding
        pair = payload_crops(img, p.box, boxes, cfg) if p.box else [img]
        ans = r.ask(goal_text, [p.text], pair, meta=m, hint=structural_hint(p.text, goal_text))
    else:  # signscene_style: crop + symbol dictionary, no parser hint
        ans = r.ask(goal_text, [p.text], [crop], meta=m)
    return {"answer": ans.direction if ans.applicable else "not_applicable",
            "source": "vlm", "confidence": ans.confidence, "summary": ans.summary,
            "plate_obj": p,
            "plate": p.text[:80], "plate_R": round(rel.score, 2), "plate_lines": len(p.lines),
            "n_plates": len(scored),
            "other_plates": " || ".join(q.text[:30] for q, r_, _ in scored if q is not p and r_.score >= 0.5)[:120]}


HAS_MEMORY = {"signway", "signway_tight", "signscene_style"}   # SignScene: per-instance fusion


def run_repeat(r, img2, plates2, strategy, reasoners, cfg, det_cache, engine, Image) -> dict:
    """Repeated-sign case: first encounter (first_image) then a second view of the
    same sign (image).  Correct = the first view was decided correctly AND the
    second view produced NO new call.  Memory is keyed on the plate's digit
    signature, so this measures whether the same physical sign is recognised
    across viewpoints and OCR noise — a real test, not a by-design cell."""
    from ..evidence.buffer import stable_plate_id
    from ..evidence.relevance import Goal, relevance_lines
    first = r.get("first_path") or r.get("first_image")
    if not first:
        return {"answer": None, "source": "no first_image column"}
    if first not in det_cache:
        im1 = np.asarray(Image.open(first).convert("RGB"))
        det_cache[first] = (im1, detect(im1, engine, cfg) if engine else [])
    img1, plates1 = det_cache[first]
    r1 = run_case(img1, r["goal"], plates1, strategy, reasoners, cfg,
                  {"condition": "repeated_sign", "image": Path(first).name, "goal": r["goal"],
                   "expected": r["expected"], "strategy": strategy, "encounter": 1})
    first_ok = (r1.get("answer") == r["expected"])
    consumed = stable_plate_id(r1["plate_obj"].text) if r1.get("plate_obj") is not None else None
    # second view: which plate would this strategy pick?
    goal = Goal.parse(r["goal"])
    best = None
    if plates2:
        best = max(plates2, key=lambda q: relevance_lines([l.text for l in q.lines], goal).score)
    if strategy in HAS_MEMORY and best is not None and consumed is not None \
            and stable_plate_id(best.text) == consumed:
        got = "no_call"
    else:
        r2 = run_case(img2, r["goal"], plates2, strategy, reasoners, cfg,
                      {"condition": "repeated_sign", "image": r["image"], "goal": r["goal"],
                       "expected": r["expected"], "strategy": strategy, "encounter": 2})
        got = "re-ask:" + str(r2.get("answer"))
    answer = "no_call" if (first_ok and got == "no_call") else got
    return {"answer": answer, "source": "memory" if got == "no_call" else "vlm",
            "first_answer": r1.get("answer"), "first_correct": first_ok,
            "plate": r1.get("plate", ""), "second_plate": best.text[:80] if best is not None else ""}


def load_cases(root: Path, csv_dir: Path | None = None, images_root: Path | None = None,
               folder_map: dict[str, str] | None = None) -> dict[str, list[dict]]:
    """Two layouts:
      A) <root>/<condition>/goals.csv, images beside the csv
      B) --csv-dir <dir>/<condition>.csv + --images-root <dir>, image column is a
         path relative to images-root (e.g. "Numeric sign/IMG_0012.jpg")."""
    cond = {}
    if csv_dir is not None:
        for f in sorted(csv_dir.glob("*.csv")):
            rows = [r for r in csv.DictReader(open(f)) if r.get("image", "").strip()]
            sub = (folder_map or {}).get(f.stem, "")
            for r in rows:
                img = r["image"].strip()
                base = images_root or csv_dir
                cand = [base / sub / img, base / f.stem / img, base / img]
                r["path"] = str(next((c for c in cand if c.exists()), cand[0]))
                if r.get("first_image", "").strip():
                    fi = r["first_image"].strip()
                    c2 = [base / sub / fi, base / f.stem / fi, base / fi]
                    r["first_path"] = str(next((c for c in c2 if c.exists()), c2[0]))
                r["goal"] = r["goal"].strip(); r["expected"] = r["expected"].strip()
            cond[f.stem] = rows
        return cond
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "goals.csv").exists()):
        rows = list(csv.DictReader(open(d / "goals.csv")))
        for r in rows:
            r["path"] = str(d / r["image"])
        cond[d.name] = rows
    return cond


def draw(acc: dict, n: dict, conds: list[str], out_stem: str, strategies: list[str] | None = None,
         abstain_ok: tuple = (), scope: dict | None = None) -> None:
    figstyle.use()
    scope = scope or {}
    STR = [s for s in (strategies or STRATEGIES) if s in acc or s in scope]
    for s_ in STR:                                   # analytic-only rows have no measurements
        acc.setdefault(s_, {}); n.setdefault(s_, {})
        for c in conds + ["Overall"]:
            acc[s_].setdefault(c, np.nan); n[s_].setdefault(c, 0)
    M = np.array([[acc[s][c] for c in conds + ["Overall"]] for s in STR], dtype=float)
    Mplot = np.ma.masked_invalid(M)
    fig, ax = plt.subplots(figsize=(COL_W * 2.05, 0.42 * len(STR) + 0.9))
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#eeeeee")
    im = ax.imshow(Mplot * 100, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(conds) + 1))
    ax.set_xticklabels([c.replace("_", " ") + ("\n(abstain = correct)" if c in abstain_ok else "")
                        + ("\n(no re-ask = correct)" if c == "repeated_sign" else "")
                        for c in conds] + ["Overall"], rotation=20, ha="right")
    ax.set_yticks(range(len(STR)))
    ax.set_yticklabels([ROW_LABEL[s] for s in STR])
    for i, s in enumerate(STR):
        for j, c in enumerate(conds + ["Overall"]):
            if c in scope.get(s, {}):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, fc="#eeeeee",
                                       hatch="////", ec="#999999", lw=0))
                ax.text(j, i, "not supported\nby design", ha="center", va="center", fontsize=5.2, color="#444")
                continue
            v = M[i, j] * 100
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#444"); continue
            ax.text(j, i, f"{v:.0f}%\n(n={n[s][c]})", ha="center", va="center", fontsize=6,
                    color="white" if v < 60 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("correct [%]")
    figstyle.save(fig, out_stem, arrays={"acc": M, "conds": np.array(conds + ["Overall"]),
                                         "rows": np.array(STR)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="layout A: <root>/<condition>/goals.csv")
    ap.add_argument("--csv-dir", default=None, help="layout B: one <condition>.csv per condition")
    ap.add_argument("--images-root", default=None, help="layout B: folder the csv image paths are relative to")
    ap.add_argument("--folder-map", nargs="*", default=[],
                    help="csvstem=subfolder pairs, e.g. numeric_sign=numeric unclear_sign=unclear")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--rpm", type=float, default=0.0)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES[:4])
    ap.add_argument("--mode", choices=["reading", "trigger"], default="reading",
                    help="reading: does the reasoner answer correctly given a call (needs a VLM). "
                         "trigger: does the POLICY fire on this frame when it should and not when it "
                         "shouldn't (no VLM; measures the invocation decision itself)")
    ap.add_argument("--tau", type=float, default=0.55, help="trigger mode: our legibility threshold")
    ap.add_argument("--from-ledger", default=None,
                    help="build the figure from an existing ledger: no OCR, no reasoner, no calls")
    ap.add_argument("--ledger-map", nargs="*", default=["signway_ctx=signway", "signway=signway_tight"],
                    help="ledger strategy tag → row name (the context-crop run was logged as signway_ctx)")
    ap.add_argument("--scope", default=None,
                    help="JSON {strategy: {condition: 'reason'}} — situations a method does NOT support by "
                         "design (from its paper); drawn as hatched cells, never scored as a percentage")
    ap.add_argument("--abstain-ok", nargs="*", default=["unclear_sign"],
                    help="conditions where declining (not_applicable / no plate) is the CORRECT behaviour: "
                         "the sign is unreadable, so committing to a direction is the failure")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.from_ledger:
        return from_ledger(a)
    if a.mode == "trigger":
        return trigger_mode(a)
    if a.csv_dir:
        fmap = dict(kv.split("=", 1) for kv in a.folder_map)
        conds = load_cases(Path("."), Path(a.csv_dir), Path(a.images_root) if a.images_root else None, fmap)
    elif a.root:
        conds = load_cases(Path(a.root))
    else:
        raise SystemExit("need --root or --csv-dir/--images-root")
    if not conds:
        raise SystemExit("no condition csvs found")
    missing = [r["path"] for rows in conds.values() for r in rows if not Path(r["path"]).exists()]
    missing += [r["first_path"] for rows in conds.values() for r in rows
                if r.get("first_path") and not Path(r["first_path"]).exists()]
    if "repeated_sign" in conds and any(not r.get("first_path") for r in conds["repeated_sign"]):
        raise SystemExit("repeated_sign.csv needs a first_image column (first encounter of the same sign)")
    if missing:
        raise SystemExit(f"{len(missing)} image paths in the csvs don't exist, e.g. {missing[:3]}")
    bad = [r for rows in conds.values() for r in rows
           if r["expected"] not in ("turn_left", "turn_right", "straight", "stop", "not_applicable")]
    if bad:
        raise SystemExit(f"{len(bad)} rows have an invalid 'expected' value, e.g. {bad[0]}")
    print("conditions:", {k: len(v) for k, v in conds.items()})
    from PIL import Image
    if a.fake:
        engine, client = None, FakeVLM({})
    else:
        from ..evidence.detect import DocTREngine
        engine = DocTREngine()
        reasoning = None if a.reasoning.lower() in ("none", "off", "") else a.reasoning
        client = OpenAICompatVLM(model=a.model, base_url=a.base_url, reasoning_effort=reasoning, rpm=a.rpm)
    templates = {"iros_style": PROMPT_IROS, "signscene_style": PROMPT_SIGNSCENE,
                 "crop_notext": PROMPT_NOTEXT, "signway": PROMPT, "signway_tight": PROMPT}
    reasoners = {s: Reasoner(client, model_tag=f"{a.model}:{a.reasoning}:{s}", cache_dir=out / "vlm_cache",
                             payload_dir=out / "payloads" / s, ledger_path=out / "ledger.jsonl",
                             eval_tag=f"heatmap_{s}", prompt_template=t)
                 for s, t in templates.items()}
    cfg = DetectConfig()
    det_cache: dict[str, list] = {}

    scope_ = json.loads(Path(a.scope).read_text()) if a.scope else {}
    acc = {s: {} for s in a.strategies}; n = {s: {} for s in a.strategies}
    results = []
    for cname, rows in conds.items():
        for s in a.strategies:
            if cname in scope_.get(s, {}) or s == "signnav":
                acc[s][cname] = float("nan"); n[s][cname] = 0
                continue
            ok = tot = 0
            for r in rows:
                if r["path"] not in det_cache:
                    img = np.asarray(Image.open(r["path"]).convert("RGB"))
                    det_cache[r["path"]] = (img, detect(img, engine, cfg) if engine else [])
                img, plates = det_cache[r["path"]]
                if cname == "repeated_sign":
                    res = run_repeat(r, img, plates, s, reasoners, cfg, det_cache, engine, Image)
                else:
                    res = run_case(img, r["goal"], plates, s, reasoners, cfg,
                                   {"condition": cname, "image": r["image"], "goal": r["goal"],
                                    "expected": r["expected"], "strategy": s})
                res.pop("plate_obj", None)
                got = res["answer"] if res["answer"] is not None else "no_answer"
                if cname == "repeated_sign":
                    correct = (got == "no_call")
                else:
                    correct = (got == r["expected"]) or \
                              (cname in a.abstain_ok and got in ("not_applicable", "no_answer"))
                ok += correct; tot += 1
                results.append({"condition": cname, "strategy": s, "image": r["image"],
                                "goal": r["goal"], "expected": r["expected"], "got": got,
                                "correct": correct, **{k: v for k, v in res.items() if k != "answer"}})
            acc[s][cname] = ok / tot if tot else 0.0; n[s][cname] = tot
            print(f"{cname:18s} {s:16s} {ok:3d}/{tot:<3d} = {100*ok/max(tot,1):5.1f}%")
    for s in a.strategies:
        cs = [c for c in conds if c not in scope_.get(s, {}) and n[s].get(c, 0) > 0]
        tot = sum(n[s][c] for c in cs); ok = sum(acc[s][c] * n[s][c] for c in cs)
        acc[s]["Overall"] = ok / tot if tot else float("nan"); n[s]["Overall"] = tot
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in results for k in r}))
        w.writeheader(); w.writerows(results)
    (out / "summary.json").write_text(json.dumps({"acc": acc, "n": n}, indent=2, default=str))
    draw(acc, n, list(conds), str(out / "ar_heatmap"), a.strategies, tuple(a.abstain_ok), scope_)
    print(f"wrote {out}/ar_heatmap.pdf/.png, results.csv, summary.json, ledger.jsonl")




def from_ledger(a) -> None:
    """Score every (strategy, condition, image, goal) from the ledger's last row;
    cases with no ledger row never reached the reasoner (no plate detected) and
    count as 'no_answer'.  Draws the same figure.  Zero model calls."""
    fmap = dict(kv.split("=", 1) for kv in a.folder_map)
    conds = load_cases(Path("."), Path(a.csv_dir), Path(a.images_root) if a.images_root else None, fmap) \
        if a.csv_dir else load_cases(Path(a.root))
    lmap = dict(kv.split("=", 1) for kv in a.ledger_map)
    ans: dict = {}
    for line in open(a.from_ledger):
        r = json.loads(line)
        ev = r.get("eval", "")
        if not ev.startswith("heatmap_") or "condition" not in r:
            continue
        tag = ev[len("heatmap_"):]
        row = lmap.get(tag, tag)
        got = r["direction"] if r.get("applicable") else "not_applicable"
        ans[(row, r["condition"], r["image"], r["goal"])] = got     # last row wins
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scope_ = json.loads(Path(a.scope).read_text()) if a.scope else {}
    acc = {s: {} for s in a.strategies}; n = {s: {} for s in a.strategies}
    results = []
    for cname, rows in conds.items():
        for s in a.strategies:
            if cname in scope_.get(s, {}) or s == "signnav":
                acc[s][cname] = float("nan"); n[s][cname] = 0
                continue
            ok = tot = 0
            for r in rows:
                got = ans.get((s, cname, r["image"].strip(), r["goal"]), "no_answer")
                correct = (got == r["expected"]) or (cname in a.abstain_ok and got in ("not_applicable", "no_answer"))
                ok += correct; tot += 1
                results.append({"condition": cname, "strategy": s, "image": r["image"], "goal": r["goal"],
                                "expected": r["expected"], "got": got, "correct": correct})
            acc[s][cname] = ok / tot if tot else 0.0; n[s][cname] = tot
            print(f"{cname:18s} {s:16s} {ok:3d}/{tot:<3d} = {100*ok/max(tot,1):5.1f}%")
    for s in a.strategies:
        cs = [c for c in conds if c not in scope_.get(s, {}) and n[s].get(c, 0) > 0]
        tot = sum(n[s][c] for c in cs); ok = sum(acc[s][c] * n[s][c] for c in cs)
        acc[s]["Overall"] = ok / tot if tot else float("nan"); n[s]["Overall"] = tot
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    (out / "summary.json").write_text(json.dumps({"acc": acc, "n": n, "source": a.from_ledger}, indent=2, default=str))
    draw(acc, n, list(conds), str(out / "ar_heatmap"), a.strategies, tuple(a.abstain_ok), scope_)
    print(f"wrote {out}/ar_heatmap.pdf/.png (from ledger, no model calls)")


# ----------------------------------------------------------------------------
# Trigger mode: the invocation decision per frame, no VLM
# ----------------------------------------------------------------------------

TRIGGER_ROWS = ["always", "reactive", "iros_style", "signscene_style", "relevance", "signway"]
TRIGGER_LABEL = {"always": "Always-invoke", "reactive": "Reactive (any text)", "iros_style": "IROS-style†",
                 "signscene_style": "SignScene-style†", "relevance": "Relevance-only (abl.)",
                 "signway": "SignWay (ours)"}
SHOULD_FIRE = {"numeric_sign": True, "named_goal": True, "goal_irrelevant": False, "unclear_sign": False}
FAST_PATH_POLICIES = {"iros_style", "relevance", "signway"}   # may decide without a call


def resolved_without_call(plates, goal_text: str) -> bool:
    from ..evidence.resolve import Plate, PlateLine, resolve
    goal = Goal.parse(goal_text)
    for p in plates:
        rel = relevance_lines([l.text for l in p.lines], goal)
        res = resolve(Plate([PlateLine(l.text, list(l.arrows)) for l in p.lines]), goal, rel=rel)
        if res.resolved and rel.score >= 0.5:
            return True
    return False


def fires(policy: str, plates, goal_text: str, leg, tau: float) -> bool:
    """Single-frame approximation of each policy's trigger on this frame's evidence.
    always: every frame.  reactive: any text.  iros: text present and no plate
    resolves via the fast path (their ambiguity test; KFC assumed passed, which is
    what the fitted threshold did).  signscene: any sign-like plate legible enough
    to parse (their viewpoint gate ~ legibility >= 0.5).  relevance: a goal-relevant
    plate.  signway: a goal-relevant plate that is also legible (>= tau)."""
    from ..evidence.features import phi
    from ..evidence.resolve import Plate, PlateLine, resolve
    from .baselines import sign_like
    if policy == "always":
        return True
    if not plates:
        return False
    if policy == "reactive":
        return True
    goal = Goal.parse(goal_text)
    scored = []
    for p in plates:
        rel = relevance_lines([l.text for l in p.lines], goal)
        res = resolve(Plate([PlateLine(l.text, list(l.arrows)) for l in p.lines]), goal, rel=rel)
        scored.append((p, rel.score, leg(phi(p, 1.0)), res.resolved))
    if policy == "iros_style":
        return not any(r for _, _, _, r in scored)
    if policy == "signscene_style":
        return any(sign_like(p.text) and ell >= 0.5 for p, _, ell, _ in scored)
    if policy == "relevance":
        return any(R >= 0.5 for _, R, _, _ in scored)
    if policy == "signway":
        return any(R >= 0.5 and ell >= tau for _, R, ell, _ in scored)
    raise ValueError(policy)


def trigger_mode(a) -> None:
    from ..evidence.legibility import Legibility
    from ..evidence.detect import DocTREngine
    from PIL import Image
    fmap = dict(kv.split("=", 1) for kv in a.folder_map)
    conds = load_cases(Path("."), Path(a.csv_dir), Path(a.images_root) if a.images_root else None, fmap) \
        if a.csv_dir else load_cases(Path(a.root))
    engine, cfg, leg = DocTREngine(), DetectConfig(), Legibility.load()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    det = {}
    rows_out, acc, n = [], {r: {} for r in TRIGGER_ROWS}, {r: {} for r in TRIGGER_ROWS}
    for cname, rows in conds.items():
        if cname not in SHOULD_FIRE:
            print(f"skip {cname}: no fire/no-fire rule for this category"); continue
        want = SHOULD_FIRE[cname]
        tallies = {r: [0, 0] for r in TRIGGER_ROWS}
        for r in rows:
            if r["path"] not in det:
                img = np.asarray(Image.open(r["path"]).convert("RGB"))
                det[r["path"]] = detect(img, engine, cfg)
            plates = det[r["path"]]
            fast = resolved_without_call(plates, r["goal"])
            for pol in TRIGGER_ROWS:
                f = fires(pol, plates, r["goal"], leg, a.tau)
                # a should-fire frame is also correct if the policy's fast path decided it without a call
                ok = (f == want) or (want and fast and pol in FAST_PATH_POLICIES)
                tallies[pol][0] += ok; tallies[pol][1] += 1
                rows_out.append({"condition": cname, "policy": pol, "image": r["image"], "goal": r["goal"],
                                 "should_fire": int(want), "fired": int(f), "fast_path": int(fast),
                                 "correct": int(ok), "n_plates": len(plates)})
        for pol in TRIGGER_ROWS:
            ok, tot = tallies[pol]
            acc[pol][cname] = ok / tot if tot else float("nan"); n[pol][cname] = tot
            print(f"{cname:18s} {pol:16s} {ok:3d}/{tot:<3d} = {100*ok/max(tot,1):5.1f}%   (should_fire={want})")
    conds_used = [c for c in conds if c in SHOULD_FIRE]
    for pol in TRIGGER_ROWS:
        tot = sum(n[pol][c] for c in conds_used); ok = sum(acc[pol][c] * n[pol][c] for c in conds_used)
        acc[pol]["Overall"] = ok / tot if tot else float("nan"); n[pol]["Overall"] = tot
    with open(out / "trigger_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    (out / "trigger_summary.json").write_text(json.dumps({"acc": acc, "n": n, "tau": a.tau}, indent=2, default=str))
    # draw with the trigger labels
    global ROW_LABEL
    saved = dict(ROW_LABEL); ROW_LABEL.update(TRIGGER_LABEL)
    try:
        draw(acc, n, conds_used, str(out / "ar_trigger_heatmap"), TRIGGER_ROWS, ())
    finally:
        ROW_LABEL.clear(); ROW_LABEL.update(saved)
    print(f"wrote {out}/ar_trigger_heatmap.pdf/.png, trigger_results.csv, trigger_summary.json  (no model calls)")


if __name__ == "__main__":
    main()
