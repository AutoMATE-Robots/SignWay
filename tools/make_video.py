#!/usr/bin/env python3
"""make_video.py — stitch a run's frames into a video you can actually watch.

Run OUTSIDE the Isaac container (any env with pillow), after a run_sim run:

    python tools/make_video.py --frames isaac_out --out isaac_out/run.mp4
    python tools/make_video.py --frames isaac_out --out isaac_out/run.gif --fps 5

mp4 needs imageio-ffmpeg (`pip install imageio[ffmpeg]`); gif needs nothing but pillow, so if
mp4 fails it falls back to gif automatically.

With --log it overlays the step number, pose and goal on each frame, so the video shows what the
robot knew at that moment — much more useful than raw frames for showing people what happened.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load_log(path):
    """frame filename -> record, so overlays match frames by name not by index."""
    if not path or not os.path.exists(path):
        return {}
    recs = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "meta" or "frame" not in d:
            continue
        recs[d["frame"]] = d
    return recs


_SUBS = {"\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
         "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2192": "->",
         "\u2190": "<-", "\u2191": "^", "\u2193": "v", "\u00b0": "deg"}


def _ascii(text):
    """PIL's built-in bitmap font is latin-1 only and raises on anything else.

    This is not paranoia about my own strings: the VLM's rationale is free text from a language
    model and will contain curly quotes, em-dashes and arrows sooner or later. One of those
    crashes the whole video render at frame 400, so everything drawn goes through here.
    """
    t = str(text)
    for k, v in _SUBS.items():
        t = t.replace(k, v)
    return t.encode("ascii", "replace").decode("ascii")


def _wrap(text, n=46):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def annotate(img, rec):
    """Burn the story onto the frame: where we are, what the gate saw, what the VLM said.

    The gate line is the point of the whole project — it shows the sign being seen and
    deliberately NOT read, frame after frame, until it is big enough to be worth 4 seconds.
    """
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    x, y, yaw = rec.get("pose", [0, 0, 0])
    gf, gl = (rec.get("goal_robot", [0, 0, 0]))[:2]
    state = rec.get("state", "")
    lines = [(f"step {rec.get('i','?')}  {state}   pose ({x:.2f}, {y:.2f}, {yaw:+.2f})",
              (255, 255, 255))]
    lines.append((f"subgoal  fwd {gf:+.2f}  left {gl:+.2f}", (200, 200, 200)))

    g = rec.get("gate")
    if g:
        if g.get("legible"):
            lines.append((f"sign: {g['h_px']:.0f}px  LEGIBLE -> worth reading", (120, 255, 120)))
        else:
            lines.append((f"sign: {g['h_px']:.0f}px  {g.get('reason','')} -> skip, drive on",
                          (255, 190, 90)))
    if rec.get("sign"):
        lines.append(("READING SIGN - asking the VLM...", (255, 255, 120)))

    v = rec.get("vlm")
    if v:
        fired = rec.get("vlm_fired")
        head = (f"VLM call #{v.get('call_index', 1)}  {v['latency_s']}s  ->  "
                f"{v['decision'].upper()}  (conf {v['confidence']})")
        lines.append((head, (255, 240, 120) if fired else (140, 200, 255)))
        lines.append((f'mission: "{v.get("mission","")}"', (150, 170, 200)))
        for w in _wrap(v.get("rationale", ""), 52)[:4]:
            lines.append(("  " + w, (170, 190, 215)))
    lines.append((f"VLM calls so far: {rec.get('vlm_calls', 0)}", (160, 160, 160)))

    pad, lh = 5, 13
    d.rectangle([0, 0, 470, lh * len(lines) + pad * 2], fill=(0, 0, 0))
    for i, (t, col) in enumerate(lines):
        d.text((pad, pad + i * lh), _ascii(t), fill=col)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder of frame_*.png from run_sim")
    ap.add_argument("--out", default=None, help="output .mp4 or .gif")
    ap.add_argument("--log", default=None, help="run.jsonl — adds pose/goal overlay")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--pattern", default="frame_*.png")
    ap.add_argument("--map", default=None,
                    help="folder holding map_XXXX.png (from tools/make_map.py --animate). "
                         "Puts the top-down decision view beside the camera view.")
    args = ap.parse_args()

    from PIL import Image

    files = sorted(glob.glob(os.path.join(args.frames, args.pattern)))
    if not files:
        raise SystemExit(f"no frames matching {args.pattern} in {args.frames}")
    recs = load_log(args.log)
    print(f"[video] {len(files)} frames" + (f", {len(recs)} log records" if recs else ""))

    # Frames rendered BETWEEN decisions (run_sim --sub-steps) have no log record of their own.
    # Carry the last decision forward so the overlay stays put instead of flickering off.
    imgs, last = [], None
    for f in files:
        im = Image.open(f).convert("RGB")
        rec = recs.get(os.path.basename(f))
        if rec:
            last = rec
        if last:
            im = annotate(im, last)
        if args.map and last is not None:
            # camera on the left (what it saw), decision map on the right (what it did)
            mp = os.path.join(args.map, f"map_{last['i']:04d}.png")
            if os.path.exists(mp):
                m = Image.open(mp).convert("RGB")
                h = im.height
                m = m.resize((int(m.width * h / m.height), h), Image.LANCZOS)
                combo = Image.new("RGB", (im.width + m.width, h), (255, 255, 255))
                combo.paste(im, (0, 0))
                combo.paste(m, (im.width, 0))
                im = combo
        imgs.append(im)

    out = args.out or os.path.join(args.frames, "run.mp4")
    if out.lower().endswith(".mp4"):
        try:
            import imageio.v2 as imageio
            import numpy as np
            with imageio.get_writer(out, fps=args.fps, macro_block_size=None) as w:
                for im in imgs:
                    w.append_data(np.asarray(im))
            print(f"[video] wrote {out}")
            return
        except Exception as e:
            out = out[:-4] + ".gif"
            print(f"[video] mp4 unavailable ({e.__class__.__name__}) — falling back to {out}")

    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(args.fps, 1)), loop=0)
    print(f"[video] wrote {out}")


if __name__ == "__main__":
    main()