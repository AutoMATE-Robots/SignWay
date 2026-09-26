"""
run_images.py — the pipeline on standalone frames, raw assets only.

For each (image, goal): docTR → plates → relevance → chosen plate → payload
crop → fast path or reasoner.  Writes, per image and goal, into <out>/:

  <name>__<goal>_replay.png  the frame drawn EXACTLY as the replay video draws
                             it (same function): every plate boxed, coloured by
                             relevance, "R=.. l=.." above, OCR text below.
  <name>__<goal>_boxes.png   clean version: chosen plate green, others thin
                             grey, no text (--chosen-only to drop the others).
  <name>__<goal>_crop.jpg    the exact crop the reasoner received
  <name>__<goal>.json        OCR lines + boxes, R per plate, fast-path result,
                             VLM answer (direction / confidence / summary / latency)
  results.json               everything, one file

Usage (ar env; server up or Gemini key):
  python -m adaptive_reasoning.replay.run_images --out figs/panels \\
      figs/src/rapson-12.png:49 figs/src/rapson-12.png:20 \\
      figs/src/hsec-28.png:"Diehl Hall" ...
Each positional arg is  path:goal .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..evidence.detect import DetectConfig, detect, payload_crop
from ..evidence.features import phi
from ..evidence.buffer import stable_plate_id
from ..evidence.legibility import Legibility
from ..evidence.relevance import Goal, relevance_lines
from ..evidence.resolve import Plate, PlateLine, resolve
from ..reasoner import FakeVLM, OpenAICompatVLM, Reasoner, structural_hint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+", help="path:goal")
    ap.add_argument("--out", default="figs/panels")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--rpm", type=float, default=0.0)
    ap.add_argument("--always-vlm", action="store_true", help="ask the VLM even if the fast path resolves")
    ap.add_argument("--chosen-only", action="store_true", help="draw only the chosen plate")
    ap.add_argument("--line", type=int, default=0, help="box line width px (0 = auto)")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.fake:
        engine, client = None, FakeVLM({})
    else:
        from ..evidence.detect import DocTREngine
        engine = DocTREngine()
        reasoning = None if a.reasoning.lower() in ("none", "off", "") else a.reasoning
        client = OpenAICompatVLM(model=a.model, base_url=a.base_url,
                                 reasoning_effort=reasoning, rpm=a.rpm)
    reasoner = Reasoner(client, model_tag=f"{a.model}:{a.reasoning}", cache_dir=out / "vlm_cache",
                        payload_dir=out / "payloads", ledger_path=out / "ledger.jsonl",
                        eval_tag="run_images")
    leg = Legibility.load()
    cfg = DetectConfig()
    results = []
    cache: dict[str, tuple] = {}

    for item in a.items:
        path, goal_text = item.rsplit(":", 1)
        img = np.asarray(Image.open(path).convert("RGB"))
        if path not in cache:                       # OCR once per image
            cache[path] = detect(img, engine, cfg) if engine else []
        plates = cache[path]
        goal = Goal.parse(goal_text)
        scored = []
        for p in plates:
            rel = relevance_lines([l.text for l in p.lines], goal)
            res = resolve(Plate([PlateLine(l.text, list(l.arrows)) for l in p.lines]), goal, rel=rel)
            scored.append((p, rel, res))
        rec = {"image": path, "goal": goal_text,
               "plates": [{"text": p.text, "box": list(p.box) if p.box else None, "R": rel.score,
                           "arrows": [l.arrows for l in p.lines]} for p, rel, _ in scored]}
        if not scored:
            rec["action"] = None; rec["source"] = "no plates"
            results.append(rec); print(f"{Path(path).name:14s} {goal_text!r:34s} no plates"); continue
        i = max(range(len(scored)), key=lambda k: scored[k][1].score)
        p, rel, res = scored[i]
        rec.update({"chosen": {"text": p.text, "box": list(p.box) if p.box else None, "R": rel.score,
                               "ell_placeholder": leg(phi(p, 1.0))}})
        stem = f"{Path(path).stem}__{goal_text.replace(' ', '_').replace('/', '-')}"

        # replay-style overlay: identical drawing code to render_video
        import cv2
        from .render_video import draw_plates
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        recs_draw = [{"box": list(q.box) if q.box else None, "R": r_.score,
                      "ell": leg(phi(q, 1.0)), "text": q.text,
                      "plate_id": stable_plate_id(q.text)} for q, r_, _ in scored]
        draw_plates(frame_bgr, recs_draw, 1.0)
        cv2.imwrite(str(out / f"{stem}_replay.png"), frame_bgr)

        # clean boxes overlay: no text
        im = Image.open(path).convert("RGB")
        d = ImageDraw.Draw(im)
        lw = a.line or max(2, im.width // 400)
        if not a.chosen_only:
            for q, _, _ in scored:
                if q.box and q is not p:
                    d.rectangle(q.box, outline=(160, 160, 160), width=max(1, lw // 2))
        if p.box:
            d.rectangle(p.box, outline=(74, 222, 128), width=lw)
        im.save(out / f"{stem}_boxes.png")

        # payload crop
        boxes = [q.box for q, _, _ in scored if q.box]
        crop = payload_crop(img, p.box, boxes, cfg) if p.box else img
        Image.fromarray(crop).save(out / f"{stem}_crop.jpg", quality=92)

        if res.resolved and res.vla_prompt and not a.always_vlm:
            rec.update({"action": res.vla_prompt, "source": "fast path", "vlm": None})
        else:
            ans = reasoner.ask(goal_text, [p.text], [crop], scene=None,
                               meta={"image": path, "goal": goal_text},
                               hint=structural_hint(p.text, goal_text))
            rec.update({"action": ans.direction if ans.applicable else None,
                        "source": "vlm",
                        "vlm": {"applicable": ans.applicable, "direction": ans.direction,
                                "confidence": ans.confidence, "summary": ans.summary,
                                "latency_s": ans.latency_s, "cached": ans.cached}})
        (out / f"{stem}.json").write_text(json.dumps(rec, indent=2))
        results.append(rec)
        print(f"{Path(path).name:14s} {goal_text!r:34s} -> {rec['action']} [{rec['source']}]  "
              f"plate='{p.text[:60]}'")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out}/  (*_boxes.png, *_crop.jpg, *.json, results.json)")


if __name__ == "__main__":
    main()
