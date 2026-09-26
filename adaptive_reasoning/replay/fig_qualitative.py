"""
fig_qualitative.py — "different decisions under different circumstances" figure.

Each panel = one real frame + one goal.  The pipeline runs FOR REAL on the
frame: docTR → plates → relevance vs goal → the winning plate → payload crop
→ reasoner (fast path if the plate resolves, else the VLM) → action.  The
panel shows the frame, the chosen plate boxed (green), other detected plates
in grey, and a caption "goal → action (source)".  Two goals on the same scene
(stacked, SignNav-Fig.4 style) make the causal point: same sign, different
answer, only the goal changed.

Spec file (JSON list), e.g.
  [{"image": "figs/src/rapson-12.png", "goals": ["49", "20"],
    "title": "(a) Room-range directory", "expect": ["turn_left", "turn_right"]},
   ...]
`expect` is optional and only used to mark ✓/✗ in the JSON sidecar (never on
the figure).  Everything that was sent/received is written to
<out>_panels.json for the caption and for auditing.

Usage (ar env; OCR needs docTR; reasoning needs a server or an API key):
  python -m adaptive_reasoning.replay.fig_qualitative --spec figs/qual_spec.json \
      --out figs/ar_qualitative --model Qwen/Qwen2.5-VL-7B-Instruct \
      --base-url http://localhost:8000/v1 --reasoning none --rpm 0
  # or Gemini while the local server is down:
  ... --model gemini-3.1-flash-lite --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --reasoning low
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ..evidence.detect import DetectConfig, detect, payload_box, payload_crop
from ..evidence.legibility import Legibility
from ..evidence.relevance import Goal, relevance_lines
from ..evidence.resolve import Plate, PlateLine, resolve
from ..reasoner import OpenAICompatVLM, Reasoner, FakeVLM
from . import figstyle
from .figstyle import C, PAGE_W

ARROW = {"turn_left": "←", "turn_right": "→", "straight": "↑", "stop": "■", None: "?"}


def run_panel(img: np.ndarray, goal_text: str, engine, reasoner, leg, det_cfg,
              use_fast_path: bool = True) -> dict:
    plates = detect(img, engine, det_cfg)
    goal = Goal.parse(goal_text)
    scored = []
    for p in plates:
        rel = relevance_lines([l.text for l in p.lines], goal)
        res = resolve(Plate([PlateLine(l.text, list(l.arrows)) for l in p.lines]), goal, rel=rel)
        scored.append((p, rel, res))
    if not scored:
        return {"goal": goal_text, "action": None, "source": "no plates", "plates": []}
    best_i = max(range(len(scored)), key=lambda i: scored[i][1].score)
    p, rel, res = scored[best_i]
    ell = leg(__import__("adaptive_reasoning.evidence.features", fromlist=["phi"]).phi(p, 1.0))
    out = {"goal": goal_text, "best_plate_text": p.text, "R": rel.score, "ell": ell,
           "plates": [{"text": q.text, "R": r.score, "box": list(q.box) if q.box else None}
                      for q, r, _ in scored],
           "best_box": list(p.box) if p.box else None}
    if use_fast_path and res.resolved and res.vla_prompt:
        out.update({"action": res.vla_prompt, "source": "fast path", "vlm": None})
        return out
    boxes = [q.box for q, _, _ in scored if q.box is not None]
    crop = payload_crop(img, p.box, boxes, det_cfg) if p.box is not None else img
    ans = reasoner.ask(goal_text, [p.text], [crop], scene=None,
                       meta={"panel_goal": goal_text, "plate": p.text})
    out.update({"action": ans.direction if ans.applicable else None,
                "source": "VLM" + (" (cached)" if ans.cached else ""),
                "vlm": {"applicable": ans.applicable, "direction": ans.direction,
                        "confidence": ans.confidence, "summary": ans.summary,
                        "latency_s": ans.latency_s}})
    return out


def _center_crop(img: np.ndarray, aspect: float = 1.5) -> tuple[np.ndarray, int, int]:
    """Crop to a common aspect ratio so every panel is the same shape; returns
    (crop, x_offset, y_offset) so boxes can be shifted."""
    H, W = img.shape[:2]
    if W / H > aspect:
        w = int(H * aspect); x0 = (W - w) // 2
        return img[:, x0:x0 + w], x0, 0
    h = int(W / aspect); y0 = (H - h) // 2
    return img[y0:y0 + h], 0, y0


def draw(panels: list[dict], images: list[np.ndarray], out_stem: str, ncols: int) -> None:
    """One column per image; rows = goals of that image (SignNav-Fig.4 style)."""
    figstyle.use()
    n_img = len(images)
    max_goals = max(len(p["results"]) for p in panels)
    col_w = PAGE_W / n_img
    fig, axes = plt.subplots(max_goals, n_img,
                             figsize=(PAGE_W, max_goals * (col_w / 1.5 + 0.36) + 0.05),
                             squeeze=False, gridspec_kw={"wspace": 0.04, "hspace": 0.42})
    for j, (panel, img) in enumerate(zip(panels, images)):
        crop, ox, oy = _center_crop(img)
        for i in range(max_goals):
            ax = axes[i][j]
            ax.axis("off")
            if i >= len(panel["results"]):
                continue
            r = panel["results"][i]
            ax.imshow(crop)
            for pl in r["plates"]:
                if pl["box"] and pl["box"] != r["best_box"]:
                    x0, y0, x1, y1 = pl["box"]
                    ax.add_patch(Rectangle((x0 - ox, y0 - oy), x1 - x0, y1 - y0, fill=False,
                                           ec=C["baseline2"], lw=0.5, alpha=0.7))
            if r["best_box"]:
                x0, y0, x1, y1 = r["best_box"]
                ax.add_patch(Rectangle((x0 - ox, y0 - oy), x1 - x0, y1 - y0, fill=False,
                                       ec=C["DECIDED"], lw=1.4))
            act = r.get("action")
            src = r.get("source", "").replace(" (cached)", "")
            goal = r["goal"] if len(r["goal"]) <= 18 else r["goal"][:17] + "…"
            ax.text(0.0, -0.04, f"goal: {goal}", transform=ax.transAxes, fontsize=6, va="top", ha="left")
            ax.text(0.0, -0.20, f"{ARROW.get(act, '?')} {(act or 'not applicable').replace('_', ' ')}  ·  {src}",
                    transform=ax.transAxes, fontsize=6, va="top", ha="left",
                    color=C["ours"] if act else C["bad"])
            if i == 0:
                label = panel.get("title", "").split(")")[0] + ")" if ")" in panel.get("title", "") else ""
                ax.text(0.02, 0.96, label, transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
                        color="white", bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.7))
    figstyle.save(fig, out_stem)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", default="figs/ar_qualitative")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--rpm", type=float, default=0.0)
    ap.add_argument("--cache-dir", default="vlm_cache_qual")
    ap.add_argument("--no-fast-path", action="store_true", help="always ask the VLM")
    ap.add_argument("--fake", action="store_true", help="fake OCR+VLM (layout test only)")
    a = ap.parse_args()

    spec = json.loads(Path(a.spec).read_text())
    from PIL import Image
    images = [np.asarray(Image.open(s["image"]).convert("RGB")) for s in spec]

    if a.fake:
        engine = _FakeOcr(spec)
        client = FakeVLM({})
    else:
        from ..evidence.detect import DocTREngine
        engine = DocTREngine()
        reasoning = None if a.reasoning.lower() in ("none", "off", "") else a.reasoning
        client = OpenAICompatVLM(model=a.model, base_url=a.base_url,
                                 reasoning_effort=reasoning, rpm=a.rpm)
    reasoner = Reasoner(client, model_tag=f"{a.model}:{a.reasoning}", cache_dir=Path(a.cache_dir),
                        payload_dir=Path(a.out + "_payloads"), ledger_path=Path(a.out + "_ledger.jsonl"),
                        eval_tag="qualitative")
    leg = Legibility.load()
    det_cfg = DetectConfig()

    panels = []
    for s, img in zip(spec, images):
        results = [run_panel(img, g, engine, reasoner, leg, det_cfg, not a.no_fast_path)
                   for g in s["goals"]]
        for r, exp in zip(results, s.get("expect", [])):
            r["expected"] = exp
            r["correct"] = (r.get("action") == exp)
        panels.append({"image": s["image"], "title": s.get("title", ""), "results": results})
        for r in results:
            mark = "" if "correct" not in r else (" ✓" if r["correct"] else " ✗")
            print(f"{Path(s['image']).name:14s} goal={r['goal']!r:36s} → {r.get('action')}"
                  f" [{r.get('source')}]{mark}   plate='{str(r.get('best_plate_text'))[:50]}'")
    Path(a.out + "_panels.json").write_text(json.dumps(panels, indent=2))
    draw(panels, images, a.out, ncols=len(spec))
    print(f"wrote {a.out}.pdf/.png, {a.out}_panels.json")


class _FakeOcr:
    """Layout-test OCR: one plate per image from an optional 'fake_box' in the spec,
    keyed by image content (detect() is called once per goal, not once per image)."""

    def __init__(self, spec):
        import hashlib
        from PIL import Image
        self.by_key = {}
        for s in spec:
            arr = np.asarray(Image.open(s["image"]).convert("RGB"))
            self.by_key[hashlib.md5(arr[::50, ::50].tobytes()).hexdigest()] = s

    def __call__(self, img):
        import hashlib
        from ..evidence.detect import DetectedLine
        s = self.by_key[hashlib.md5(img[::50, ::50].tobytes()).hexdigest()]
        x0, y0, x1, y1 = s.get("fake_box", [100, 100, 400, 200])
        lines = s.get("fake_lines", ["5-201 to 5-217 >"])
        h = (y1 - y0) / max(len(lines), 1)
        return [DetectedLine(t, 0.9, (x0, y0 + k * h, x1, y0 + (k + 1) * h)) for k, t in enumerate(lines)]


if __name__ == "__main__":
    main()
