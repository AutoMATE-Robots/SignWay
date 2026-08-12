"""Run Grounding DINO over robot corridor videos and write an annotated video of what it sees.

    python tools/dino_video.py --video 'videos/*.mp4' --every 3 --out dino_videos

Real model, real weights, no stubs: every processed frame goes through Grounding DINO. This
is DETECTION ONLY — no legibility gate, no text extraction, no reading. The question this tool
answers is just: what does DINO find in real corridor footage, where, and how confidently. For
each video it writes <name>_dino.mp4 (boxes drawn) and <name>_dino.jsonl (every box with score
and phrase, every processed frame) — the raw material for detector metrics and prompt tuning.

Colours in the output video, by category:
  gold   sign
  red    hazard (cone, tape, warning/wet-floor sign)
  gray   obstacle (person, cart, box)

The header shows per-frame detector latency, which is the number that decides how often the
robot can afford to run DINO (the every-frame vs every-N ablation).

Reading and writing video needs imageio + imageio-ffmpeg (`pip install imageio[ffmpeg]`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tools.viz import ascii_safe, draw_labeled_boxes, merge_for_drawing

COLOURS = {"sign": (240, 200, 40),
           "hazard": (220, 60, 50),
           "obstacle": (140, 140, 140)}


def ascii_safe(s: str) -> str:
    return (s.replace("\u2014", "-").replace("\u2013", "-")
             .replace("\u2018", "'").replace("\u2019", "'")
             .replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u2192", "->").replace("\u2190", "<-")
             .encode("ascii", "replace").decode())


def annotate_frame(arr: np.ndarray, boxes: list, header: str, font_size: int = 0):
    if not font_size:
        font_size = max(14, int(round(arr.shape[0] * 0.030)))
    """Shared drawer: readable font, and labels nudged apart when boxes overlap."""

    items = [{"bbox": b["bbox"], "colour": COLOURS[b["colour_key"]], "text": b["tag"],
              "score": b.get("score", 0.0)} for b in boxes]
    return draw_labeled_boxes(arr, items, title=header, font_size=font_size)


def tag_boxes(objects):
    """Detection-only: one draw record + one jsonl entry per box, coloured by category."""
    draw, entries = [], []
    for ob in objects:
        entries.append({"bbox": [round(v, 1) for v in ob.bbox], "score": round(ob.score, 3),
                        "phrase": ob.phrase, "category": ob.category})
        draw.append({"bbox": ob.bbox, "colour_key": ob.category,
                     "tag": f"{ob.phrase} {ob.score:.2f}", "score": ob.score})
    return draw, entries


def process_video(path: str, backend, out_dir: str, every: int, fps_out: float,
                  limit: int = 0, font_size: int = 18) -> dict:
    import imageio.v2 as imageio
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0]
    out_mp4 = os.path.join(out_dir, f"{name}_dino.mp4")
    out_jsonl = os.path.join(out_dir, f"{name}_dino.jsonl")

    reader = imageio.get_reader(path)
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", 30.0))
    writer = imageio.get_writer(out_mp4, fps=fps_out or max(1.0, src_fps / every))
    log = open(out_jsonl, "w")

    n_frames = n_boxes = 0
    t_total = 0.0
    counts = {"sign": 0, "hazard": 0, "obstacle": 0}
    for i, frame in enumerate(reader):
        if i % every:
            continue
        arr = np.asarray(frame)[..., :3]          # drop alpha if present
        t0 = time.time()
        objects = backend.detect_objects(arr)
        dt = time.time() - t0
        t_total += dt
        n_frames += 1
        n_boxes += len(objects)
        for ob in objects:
            counts[ob.category] = counts.get(ob.category, 0) + 1

        draw, entries = tag_boxes(objects, merge_iou, top_k)
        header = (f"{name}  frame {i}  detect {dt * 1000:.0f}ms  "
                  f"signs {counts['sign']}  hazards {counts['hazard']}  "
                  f"obstacles {counts['obstacle']}")
        writer.append_data(annotate_frame(arr, draw, header, font_size))
        log.write(json.dumps({"frame": i, "detect_s": round(dt, 3),
                              "boxes": entries}) + "\n")
        if n_frames % 25 == 0:
            print(f"  [{name}] {n_frames} frames, {n_boxes} boxes, "
                  f"{t_total / n_frames * 1000:.0f}ms/frame avg", flush=True)
        if limit and n_frames >= limit:
            break

    writer.close()
    log.close()
    summary = {"video": name, "frames": n_frames, "boxes": n_boxes,
               "ms_per_frame": round(t_total / max(1, n_frames) * 1000, 1),
               "counts": counts, "out": out_mp4}
    print(f"[{name}] done: {n_frames} frames, {n_boxes} boxes, "
          f"{summary['ms_per_frame']}ms/frame -> {out_mp4}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="mp4 path or glob, e.g. 'videos/*.mp4'")
    ap.add_argument("--out", default="dino_videos")
    ap.add_argument("--every", type=int, default=3,
                    help="detect every Nth source frame (output plays at src_fps/N)")
    ap.add_argument("--fps-out", type=float, default=0.0,
                    help="output fps; 0 = source fps divided by --every")
    ap.add_argument("--limit", type=int, default=0, help="stop after N processed frames")
    ap.add_argument("--backend", default="dino", choices=["dino", "omdet"],
                    help="which open-vocabulary detector to run")
    ap.add_argument("--model", default=None,
                    help="model id; defaults to the backend's standard checkpoint")
    ap.add_argument("--prompts", default=None,
                    help="override the phrase vocabulary, e.g. 'sign. chair. pallet.' or a "
                         "comma list. Categories are inferred: anything containing 'sign' "
                         "routes as sign unless it is a known hazard phrase. Default: "
                         "DEFAULT_PROMPTS in c3_reasoning/detector_dino.py")
    ap.add_argument("--font-size", type=int, default=18,
                    help="label text size in pixels")
    ap.add_argument("--box-threshold", type=float, default=0.25,
                    help="confidence floor. OmDet scores on a different scale to DINO and "
                         "floods at low settings — try 0.35-0.45 for --backend omdet")
    ap.add_argument("--merge-iou", type=float, default=0.7,
                    help="same-category boxes overlapping this much draw as one labelled "
                         "box. Affects DRAWING only; the jsonl keeps every raw detection")
    ap.add_argument("--top-k", type=int, default=0,
                    help="draw only the N highest-scoring boxes per frame (0 = all)")
    ap.add_argument("--font-size", type=int, default=0,
                    help="label text size in px; 0 scales with frame height")
    ap.add_argument("--text-threshold", type=float, default=0.25)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.video))
    if not paths:
        raise SystemExit(f"no videos match {args.video}")
    os.makedirs(args.out, exist_ok=True)

    from c3_reasoning.detector_dino import DEFAULT_PROMPTS, split_prompts
    from c3_reasoning.backends import DEFAULT_MODELS, build_backend
    prompts = split_prompts(args.prompts) if args.prompts else DEFAULT_PROMPTS
    model_id = args.model or DEFAULT_MODELS[args.backend]
    print(f"backend: {args.backend} ({model_id})", flush=True)
    print("vocabulary: " + "; ".join(f"{c}: {', '.join(p)}" for c, p in prompts.items()),
          flush=True)
    backend = build_backend(args.backend, model_id=model_id, prompts=prompts,
                            box_threshold=args.box_threshold,
                            text_threshold=args.text_threshold)

    summaries = [process_video(p, backend, args.out, args.every, args.fps_out,
                               limit=args.limit, font_size=args.font_size)
                 for p in paths]
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n{len(paths)} videos -> {args.out}/ (annotated mp4 + jsonl each, "
          f"+ summary.json)", flush=True)


if __name__ == "__main__":
    main()