"""Tune Grounding DINO prompts and thresholds against a handful of frames, in seconds.

A full video pass costs minutes per vocabulary guess, which makes tuning miserable. This pulls
the specific frames you care about and sweeps prompt sets x thresholds against them, printing
what each combination finds and writing one annotated image per combination.

    # grab the frames where something interesting is happening
    python tools/dino_probe.py --video videos/test3.mp4 --frame 708 --frame 300 --extract-only

    # then sweep: does lowering the threshold find the EXIT sign?
    python tools/dino_probe.py --images probe/*.png \
        --prompts "sign. exit sign." --threshold 0.35 --threshold 0.2 --threshold 0.1

    # or compare vocabularies head to head
    python tools/dino_probe.py --images probe/*.png \
        --prompts "chair. pallet. box." --prompts "office chair. wooden pallet. cardboard box."

The model loads ONCE and every variant reuses it, so a sweep costs one inference per
(frame x prompt-set x threshold) and nothing else.

Prompt sets accept either "a. b. c." or "a, b, c" — both normalise to the period-separated,
lowercase, trailing-period form Grounding DINO expects.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

PALETTE = [(240, 200, 40), (220, 60, 50), (70, 130, 220), (120, 200, 120), (200, 120, 220)]


def normalise_query(raw: str) -> str:
    """'a, b, c' or 'a. b. c.' -> 'a. b. c.' — lowercase, period-separated, trailing period.
    A malformed query silently detects nothing, which looks exactly like a bad model."""
    parts = [p.strip() for chunk in raw.split(".") for p in chunk.split(",")]
    parts = [p.lower() for p in parts if p]
    return ". ".join(parts) + "." if parts else ""


class Probe:
    """Model loaded once, queried many times."""

    def __init__(self, model_id: str, device: str | None = None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"loading {model_id} on {self.device} ...", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to(self.device).eval()
        import inspect
        params = inspect.signature(
            self.processor.post_process_grounded_object_detection).parameters
        self._kw = "box_threshold" if "box_threshold" in params else "threshold"
        print(f"ready (post-processor uses '{self._kw}')", flush=True)

    def detect(self, arr: np.ndarray, query: str, box_thr: float, text_thr: float):
        import torch
        from PIL import Image as PILImage
        pil = PILImage.fromarray(arr.astype(np.uint8))
        inputs = self.processor(images=pil, text=query, return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            **{self._kw: box_thr}, text_threshold=text_thr,
            target_sizes=[pil.size[::-1]])[0]
        dt = time.time() - t0
        labels = res["text_labels"] if "text_labels" in res else res["labels"]
        out = []
        for box, score, phrase in zip(res["boxes"], res["scores"], labels):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            out.append(((x0, y0, x1, y1), float(score), str(phrase)))
        return out, dt


def extract_frames(video: str, wanted: list, out_dir: str) -> list:
    import imageio.v2 as imageio
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video))[0]
    reader = imageio.get_reader(video)
    want = set(wanted)
    paths = []
    from PIL import Image as PILImage
    for i, frame in enumerate(reader):
        if i in want:
            p = os.path.join(out_dir, f"{name}_f{i:05d}.png")
            PILImage.fromarray(np.asarray(frame)[..., :3]).save(p)
            paths.append(p)
            want.discard(i)
            if not want:
                break
    reader.close()
    if want:
        print(f"  (frames not reached: {sorted(want)})", flush=True)
    return paths


def annotate(arr: np.ndarray, dets, header: str, phrase_colour: dict):
    from PIL import Image, ImageDraw
    pil = Image.fromarray(arr.astype(np.uint8))
    dr = ImageDraw.Draw(pil)
    for (x0, y0, x1, y1), score, phrase in dets:
        col = phrase_colour.setdefault(phrase, PALETTE[len(phrase_colour) % len(PALETTE)])
        dr.rectangle([x0, y0, x1, y1], outline=col, width=3)
        ty = y0 - 12 if y0 > 14 else y1 + 2
        dr.text((x0 + 2, ty), f"{phrase} {score:.2f}"[:40], fill=col)
    dr.rectangle([0, 0, pil.width, 18], fill=(0, 0, 0))
    dr.text((4, 3), header[:150], fill=(255, 255, 255))
    return pil


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_argument_group("input (use --video+--frame to extract, or --images)")
    src.add_argument("--video", help="video to pull frames from")
    src.add_argument("--frame", type=int, action="append", default=[],
                     help="frame index to extract; repeatable")
    src.add_argument("--images", help="glob of already-extracted frames")
    ap.add_argument("--extract-only", action="store_true",
                    help="just write the frames and stop (no model load)")

    ap.add_argument("--prompts", action="append", default=[],
                    help="a prompt set, e.g. 'sign. exit sign.' — repeatable for a sweep. "
                         "Defaults to DEFAULT_PROMPTS from detector_dino.")
    ap.add_argument("--threshold", type=float, action="append", default=[],
                    help="box threshold; repeatable. Default 0.35 0.25 0.15")
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--out", default="probe")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- gather frames ----
    if args.video:
        if not args.frame:
            ap.error("--video needs at least one --frame N")
        print(f"extracting {len(args.frame)} frames from {os.path.basename(args.video)}",
              flush=True)
        paths = extract_frames(args.video, args.frame, args.out)
        print(f"  wrote {len(paths)} frames to {args.out}/", flush=True)
        if args.extract_only:
            return
    elif args.images:
        paths = sorted(glob.glob(args.images))
    else:
        ap.error("need --video with --frame, or --images")
    if not paths:
        raise SystemExit("no frames to probe")

    # ---- variants ----
    if args.prompts:
        queries = [normalise_query(p) for p in args.prompts]
    else:
        from c3_reasoning.detector_dino import DEFAULT_PROMPTS
        queries = [normalise_query(". ".join(p for ps in DEFAULT_PROMPTS.values() for p in ps))]
    thresholds = args.threshold or [0.35, 0.25, 0.15]

    probe = Probe(args.model)
    from PIL import Image as PILImage

    print(f"\n{len(paths)} frames x {len(queries)} prompt sets x {len(thresholds)} thresholds "
          f"= {len(paths) * len(queries) * len(thresholds)} runs\n", flush=True)

    for path in paths:
        arr = np.asarray(PILImage.open(path).convert("RGB"))
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"=== {stem} ===", flush=True)
        for qi, query in enumerate(queries):
            print(f"  prompts[{qi}]: {query}", flush=True)
            for thr in thresholds:
                dets, dt = probe.detect(arr, query, thr, args.text_threshold)
                found = {}
                for _, score, phrase in dets:
                    found.setdefault(phrase, []).append(score)
                summary = ", ".join(
                    f"{p} x{len(s)} (max {max(s):.2f})" for p, s in sorted(found.items())
                ) or "nothing"
                print(f"    thr={thr:<5} {len(dets):>2} boxes  {dt * 1000:>5.0f}ms  {summary}",
                      flush=True)
                out_png = os.path.join(args.out, f"{stem}_q{qi}_t{thr}.png")
                annotate(arr, dets, f"{stem}  q{qi} thr={thr}  {len(dets)} boxes", {}
                         ).save(out_png)
        print("", flush=True)

    print(f"annotated variants -> {args.out}/", flush=True)


if __name__ == "__main__":
    main()