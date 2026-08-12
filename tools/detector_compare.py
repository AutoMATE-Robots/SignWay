"""Run two detectors over the same footage and show them side by side.

    python tools/detector_compare.py --video 'videos/*.mp4' --every 5 --out compare

For every processed frame both backends see the identical image with the identical vocabulary,
and their boxes are drawn on two copies placed next to each other — left vs right, same
moment. That is the only honest way to judge "which is better for our use case": counts and
mAP on some other dataset say nothing about whether a model finds YOUR corridor's signs.

Per video it writes:
  <name>_compare.mp4   side-by-side annotated video, header showing each model's latency
  <name>_compare.jsonl one record per frame with both models' boxes
and across all videos:
  summary.json         latency median/p95 per model, detections per category, agreement rate

AGREEMENT is worth explaining: two boxes agree when they overlap (IoU >= --iou) and land in
the same category. High agreement with different latency means take the faster one. Low
agreement means they are finding genuinely different things and the annotated video is the
only way to tell which is right — so watch it before trusting any number here.

Colours are per category (gold sign, red hazard, gray obstacle), identical on both sides, so
a difference in the picture is a difference in detection rather than in drawing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from c3_reasoning.backends import BACKENDS, DEFAULT_MODELS, build_backend
from tools.viz import (ascii_safe, box_iou as iou, draw_labeled_boxes,
                       merge_for_drawing)

COLOURS = {"sign": (240, 200, 40), "hazard": (220, 60, 50), "obstacle": (140, 140, 140)}


def ascii_safe(s: str) -> str:
    return (s.replace("\u2014", "-").replace("\u2013", "-")
             .replace("\u2192", "->").replace("\u2190", "<-")
             .encode("ascii", "replace").decode())




def count_agreements(left, right, thr: float) -> int:
    """Greedy one-to-one match: a pair agrees on overlap AND category."""
    used = set()
    hits = 0
    for lb in left:
        best_j, best_iou = -1, thr
        for j, rb in enumerate(right):
            if j in used or rb.category != lb.category:
                continue
            v = iou(lb.bbox, rb.bbox)
            if v >= best_iou:
                best_j, best_iou = j, v
        if best_j >= 0:
            used.add(best_j)
            hits += 1
    return hits


def draw(arr: np.ndarray, boxes, title: str, font_size: int = 0,
         merge_iou: float = 0.7, top_k: int = 0):
    """Adapt DinoBoxes to the shared drawer.

    Duplicates are merged first so one object carries one caption; font_size=0 scales the
    text with frame height, which matters because two panes side by side are viewed at half
    width each.
    """
    if not font_size:
        font_size = max(14, int(round(arr.shape[0] * 0.030)))
    groups = merge_for_drawing(boxes, merge_iou=merge_iou)
    if top_k:
        groups = sorted(groups, key=lambda g: -g["score"])[:top_k]
    items = [{"bbox": g["bbox"], "colour": COLOURS.get(g["category"], (200, 200, 200)),
              "text": g["label"], "score": g["score"]} for g in groups]
    return draw_labeled_boxes(arr, items, title=title, font_size=font_size)


def stitch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    gap = np.zeros((left.shape[0], 6, 3), np.uint8)
    return np.concatenate([left, gap, right], axis=1)


def process(path: str, backends: dict, out_dir: str, every: int, limit: int,
            iou_thr: float, font_size: int = 0, merge_iou: float = 0.7,
            top_k: int = 0) -> dict:
    import imageio.v2 as imageio
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0]
    reader = imageio.get_reader(path)
    src_fps = float(reader.get_meta_data().get("fps", 30.0))
    writer = imageio.get_writer(os.path.join(out_dir, f"{name}_compare.mp4"),
                                fps=max(1.0, src_fps / every), macro_block_size=None)
    log = open(os.path.join(out_dir, f"{name}_compare.jsonl"), "w")

    names = list(backends)
    lat = {n: [] for n in names}
    counts = {n: {"sign": 0, "hazard": 0, "obstacle": 0} for n in names}
    agree_total = agree_hits = 0
    n_frames = 0

    for i, frame in enumerate(reader):
        if i % every:
            continue
        arr = np.asarray(frame)[..., :3]
        dets, took = {}, {}
        for n in names:
            t0 = time.time()
            dets[n] = backends[n].detect_objects(arr)
            took[n] = time.time() - t0
            lat[n].append(took[n])
            for b in dets[n]:
                counts[n][b.category] = counts[n].get(b.category, 0) + 1

        a, b = names[0], names[1]
        hits = count_agreements(dets[a], dets[b], iou_thr)
        agree_hits += hits
        agree_total += max(len(dets[a]), len(dets[b]))

        panes = [draw(arr.copy(), dets[n],
                      f"{n}  {len(dets[n])} boxes  {took[n] * 1000:.0f}ms",
                      font_size, merge_iou, top_k)
                 for n in names]
        writer.append_data(stitch(*panes))
        log.write(json.dumps({
            "frame": i,
            **{n: {"ms": round(took[n] * 1000, 1),
                   "boxes": [{"bbox": [round(v, 1) for v in d.bbox],
                              "score": round(d.score, 3), "phrase": d.phrase,
                              "category": d.category} for d in dets[n]]}
               for n in names},
            "agreements": hits,
        }) + "\n")

        n_frames += 1
        if n_frames % 20 == 0:
            msg = "  ".join(f"{n} {statistics.median(lat[n]) * 1000:.0f}ms" for n in names)
            print(f"  [{name}] {n_frames} frames | {msg}", flush=True)
        if limit and n_frames >= limit:
            break

    writer.close()
    log.close()

    def stats(v):
        if not v:
            return {}
        s = sorted(v)
        return {"median_ms": round(statistics.median(s) * 1000, 1),
                "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))] * 1000, 1),
                "fps": round(1.0 / statistics.median(s), 1)}

    summary = {"video": name, "frames": n_frames,
               "models": {n: {**stats(lat[n]), "counts": counts[n],
                              "total_boxes": sum(counts[n].values())} for n in names},
               "agreement_rate": round(agree_hits / agree_total, 3) if agree_total else None}
    print(f"[{name}] {n_frames} frames  " +
          "  ".join(f"{n}: {summary['models'][n].get('median_ms')}ms/"
                    f"{summary['models'][n]['total_boxes']} boxes" for n in names) +
          f"  agreement {summary['agreement_rate']}", flush=True)
    return summary


def per_model_threshold(name: str, args) -> float:
    """Each detector gets its own confidence floor, falling back to the shared one."""
    override = getattr(args, f"{name}_threshold", None)
    return args.box_threshold if override is None else override


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="mp4 path or glob")
    ap.add_argument("--out", default="compare")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="stop after N processed frames")
    ap.add_argument("--backends", default="dino,omdet",
                    help=f"two of: {', '.join(BACKENDS)}")
    ap.add_argument("--dino-model", default=DEFAULT_MODELS["dino"])
    ap.add_argument("--omdet-model", default=DEFAULT_MODELS["omdet"])
    ap.add_argument("--prompts", default=None,
                    help="shared vocabulary override, e.g. 'sign. chair. traffic cone.' "
                         "Both models get exactly the same phrases")
    ap.add_argument("--box-threshold", type=float, default=0.25,
                    help="shared fallback confidence threshold")
    ap.add_argument("--dino-threshold", type=float, default=None,
                    help="Grounding DINO confidence; overrides --box-threshold")
    ap.add_argument("--omdet-threshold", type=float, default=None,
                    help="OmDet-Turbo confidence; overrides --box-threshold. OmDet scores on "
                         "a different scale and floods at DINO's setting, so it usually wants "
                         "a higher value — 0.35-0.45 is a sane starting range")
    ap.add_argument("--text-threshold", type=float, default=0.25, help="Grounding DINO only")
    ap.add_argument("--nms-threshold", type=float, default=0.3, help="OmDet-Turbo only")
    ap.add_argument("--merge-iou", type=float, default=0.7,
                    help="boxes of one category overlapping this much are drawn as a single "
                         "labelled box. Affects DRAWING only; counts stay raw")
    ap.add_argument("--top-k", type=int, default=0,
                    help="draw only the N highest-scoring boxes per frame (0 = all)")
    ap.add_argument("--font-size", type=int, default=0,
                    help="label text size in pixels; raise it for high-resolution video")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="overlap needed to call two boxes the same detection")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.video))
    if not paths:
        raise SystemExit(f"no videos match {args.video}")
    names = [n.strip().lower() for n in args.backends.split(",") if n.strip()]
    if len(names) != 2:
        raise SystemExit("--backends needs exactly two, e.g. dino,omdet")

    from c3_reasoning.detector_dino import DEFAULT_PROMPTS, split_prompts
    prompts = split_prompts(args.prompts) if args.prompts else DEFAULT_PROMPTS
    print("shared vocabulary: " +
          "; ".join(f"{c}: {', '.join(p)}" for c, p in prompts.items()) + "\n", flush=True)

    models = {"dino": args.dino_model, "omdet": args.omdet_model}

    backends = {}
    for n in names:
        print(f"loading {n} ({models.get(n)}) at threshold "
              f"{per_model_threshold(n, args)} ...", flush=True)
        backends[n] = build_backend(n, model_id=models.get(n), prompts=prompts,
                                    box_threshold=per_model_threshold(n, args),
                                    text_threshold=args.text_threshold,
                                    nms_threshold=args.nms_threshold)

    os.makedirs(args.out, exist_ok=True)
    summaries = [process(p, backends, args.out, args.every, args.limit, args.iou,
                         args.font_size, args.merge_iou, args.top_k)
                 for p in paths]
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    # roll-up across every video
    print("\n" + "=" * 62)
    for n in names:
        med = [s["models"][n]["median_ms"] for s in summaries if s["models"][n]]
        boxes = sum(s["models"][n]["total_boxes"] for s in summaries)
        cats = {c: sum(s["models"][n]["counts"].get(c, 0) for s in summaries)
                for c in ("sign", "hazard", "obstacle")}
        if med:
            print(f"{n:>6}: {statistics.median(med):6.0f}ms median  "
                  f"{1000 / statistics.median(med):5.1f} fps  {boxes:5d} boxes  "
                  f"signs {cats['sign']}, hazards {cats['hazard']}, "
                  f"obstacles {cats['obstacle']}", flush=True)
    rates = [s["agreement_rate"] for s in summaries if s["agreement_rate"] is not None]
    if rates:
        print(f"agreement between the two: {statistics.median(rates):.1%} of boxes")
    print("=" * 62)
    print(f"\nside-by-side videos -> {args.out}/  (watch before trusting the counts)",
          flush=True)


if __name__ == "__main__":
    main()