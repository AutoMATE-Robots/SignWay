"""
render_submission.py — ICRA video-grade replay of the adaptive-reasoning gate.

Same gate, same numbers as render_video.py (EvidenceGate over the dump_features
jsonl; real VLM via the OpenAI-compatible endpoint at fire time, cached) — only
the presentation layer is new:

  * 1920x1080 canvas: camera panel (left), reasoning panel (right), timeline (bottom)
  * reasoning panel follows the gate state:
      NO_SIGN  -> "scanning for signs"
      ARMED    -> tracked plate crop + legibility meter vs tau ("waiting for a readable view")
      PENDING  -> the crop that was SENT + in-flight ring over the call latency
      DECIDED  -> decision chip with arrow + one-line rationale
  * timeline: state bands fill in live, call/answer markers, annotated junction
    drawn from frame 0 (so the decision lead is visible), moving cursor
  * freeze-holds at fire (--hold-fire) and at answer landing (--hold-answer)
  * captions from a JSON file: [{"from": raw_idx, "to": raw_idx, "text": "..."}]
  * NO model name, NO reactive counter, NO sim/real tag on screen

Usage (vLLM serving Qwen locally on the compute node):
  python -m adaptive_reasoning.replay.render_submission \
      --bag ~/SignWay/ros2_bags/<bag> --jsonl $SCRATCH/ar_replay/<bag>.jsonl \
      --tau 0.45 --goal "room 6-210" --junction 530 \
      --vlm real --model <served-model-name> --base-url http://localhost:8000/v1 \
      --rpm 0 --reasoning none --latency 3.0 \
      --captions clipA_captions.json --h264 --out figs/clipA.mp4

Optional: --odom-csv t,x,y,yaw (exported separately) adds a distance-to-junction
readout; without it the timeline is frame-based, which is fine.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- style ------
W, H = 1920, 1080
CAM_W, CAM_H = 1240, 800          # camera panel (letterboxed inside)


def set_layout(timeline: bool) -> None:
    """Without the timeline strip the camera and panel grow to fill the canvas."""
    global CAM_H
    CAM_H = 800 if timeline else 1000
PANEL_X, PANEL_W = 1300, 580      # reasoning panel
TL_Y, TL_H = 880, 160             # timeline strip
MARGIN = 40

BG = (16, 18, 22)
PANEL_BG = (26, 29, 35)
TEXT = (236, 238, 241)
MUTED = (150, 156, 166)
STATE_RGB = {"NO_SIGN": (150, 156, 166), "ARMED": (250, 204, 21),
             "PENDING": (251, 146, 60), "DECIDED": (74, 222, 128)}
# scheduler modes (paper vocabulary) for each internal gate state
MODE = {"NO_SIGN": "execute", "ARMED": "acquire evidence", "PENDING": "reason",
        "DECIDED": "execute \u00b7 decided", "RECALL": "recall \u00b7 known junction"}
STATE_RGB["RECALL"] = (52, 211, 153)
MEMORY = (52, 211, 153)
CALL = (251, 146, 60)             # orange = VLM call (paper Fig. 3 palette)
PROMPT = (253, 224, 71)           # yellow = standing prompt
RELEVANT = (34, 211, 238)         # cyan box for the relevant plate
IRRELEVANT = (110, 115, 125)
ARROW = {"turn_left": "\u2190", "left": "\u2190", "turn_right": "\u2192",
         "right": "\u2192", "straight": "\u2191", "stop": "\u25A0"}

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
_FONT_BOLD = [p.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf").replace("Arial.ttf", "Arial Bold.ttf")
              for p in _FONT_CANDIDATES]


def _font(size: int, bold: bool = False, path: str | None = None) -> ImageFont.FreeTypeFont:
    cands = ([path] if path else []) + (_FONT_BOLD if bold else _FONT_CANDIDATES)
    for p in cands:
        try:
            return ImageFont.truetype(p, size)
        except (OSError, TypeError):
            continue
    return ImageFont.load_default()


class Fonts:
    def __init__(self, path: str | None = None):
        self.h1 = _font(34, True, path)
        self.h2 = _font(26, True, path)
        self.body = _font(22, False, path)
        self.small = _font(18, False, path)
        self.tiny = _font(15, False, path)
        self.arrow = _font(96, True, path)
        self.caption = _font(30, False, path)


# ---------------------------------------------------------------- data -------
def load_jsonl(path: str) -> dict[int, list[dict]]:
    per_frame: dict[int, list[dict]] = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        per_frame[r["frame"]].append(r)
    return per_frame


def load_captions(path: str | None) -> list[dict]:
    if not path:
        return []
    caps = json.load(open(path))
    return sorted(caps, key=lambda c: c["from"])


def load_odom(path: str | None):
    """Optional 't,x,y,yaw' csv -> arrays; returns None if not given."""
    if not path:
        return None
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr


# ---------------------------------------------------------------- drawing ----
def _fit(img: Image.Image, box_w: int, box_h: int) -> tuple[Image.Image, float]:
    s = min(box_w / img.width, box_h / img.height)
    return img.resize((int(img.width * s), int(img.height * s)), Image.BILINEAR), s


def _rounded(d: ImageDraw.ImageDraw, xy, fill, r=14, outline=None, width=2):
    d.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def _text_w(d, text, font):
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    return r - l


def draw_camera(canvas: Image.Image, d: ImageDraw.ImageDraw, frame_rgb: np.ndarray,
                recs: list[dict], best_id: str | None, F: Fonts,
                hide_irrelevant: bool) -> tuple[int, int, float]:
    img = Image.fromarray(frame_rgb)
    fitted, s = _fit(img, CAM_W, CAM_H)
    ox = MARGIN + (CAM_W - fitted.width) // 2
    oy = MARGIN + (CAM_H - fitted.height) // 2
    _rounded(d, (MARGIN - 6, MARGIN - 6, MARGIN + CAM_W + 6, MARGIN + CAM_H + 6), PANEL_BG, r=18)
    canvas.paste(fitted, (ox, oy))
    for r in recs:
        if not r.get("box"):
            continue
        rel = r["R"] > 0.5
        if hide_irrelevant and not rel:
            continue
        x0, y0, x1, y1 = [int(v * s) for v in r["box"]]
        x0, x1 = x0 + ox, x1 + ox
        y0, y1 = y0 + oy, y1 + oy
        col = RELEVANT if rel else IRRELEVANT
        d.rounded_rectangle((x0, y0, x1, y1), radius=4, outline=col, width=3 if rel else 1)
    return ox, oy, s


def draw_panel(canvas: Image.Image, d: ImageDraw.ImageDraw, F: Fonts, state: str,
               best_rec: dict | None, best_crop: Image.Image | None, sent_crop: Image.Image | None,
               tau: float, in_flight_frac: float | None, decision: str | None,
               rationale: str | None, n_calls: int, goal: str,
               source: str | None = None, matched_text: str | None = None):
    x0, y0, x1, y1 = PANEL_X, MARGIN, PANEL_X + PANEL_W, MARGIN + CAM_H
    _rounded(d, (x0, y0, x1, y1), PANEL_BG, r=18)
    d.text((x0 + 24, y0 + 20), "Adaptive reasoning", font=F.h1, fill=TEXT)
    d.text((x0 + 24, y0 + 64), f"goal: {goal}", font=F.body, fill=MUTED)
    # state pill
    col = STATE_RGB[state]
    label = MODE[state]
    _rounded(d, (x0 + 24, y0 + 104, x0 + 24 + _text_w(d, label, F.h2) + 28, y0 + 146), col, r=21)
    d.text((x0 + 38, y0 + 111), label, font=F.h2, fill=(0, 0, 0))
    d.text((x1 - 24 - _text_w(d, f"VLM calls  {n_calls}", F.body), y0 + 114),
           f"VLM calls  {n_calls}", font=F.body, fill=TEXT)

    cy = y0 + 170
    if state == "NO_SIGN":
        d.text((x0 + 24, cy + 10), "no goal-relevant sign in view", font=F.body, fill=MUTED)

    elif state == "ARMED" and best_rec is not None:
        d.text((x0 + 24, cy), "Relevant sign tracked \u2014 not yet readable", font=F.h2, fill=TEXT)
        if best_crop is not None:
            th, s = _fit(best_crop, PANEL_W - 48, 260)
            canvas.paste(th, (x0 + 24, cy + 70))
        ell = float(best_rec["ell"])
        my = cy + 350
        d.text((x0 + 24, my), "legibility", font=F.small, fill=MUTED)
        bx0, bx1 = x0 + 24, x1 - 24
        d.rounded_rectangle((bx0, my + 28, bx1, my + 46), radius=9, fill=(60, 64, 72))
        d.rounded_rectangle((bx0, my + 28, bx0 + int((bx1 - bx0) * min(ell, 1)), my + 46),
                            radius=9, fill=STATE_RGB["ARMED"])
        tx = bx0 + int((bx1 - bx0) * tau)
        d.line((tx, my + 20, tx, my + 54), fill=TEXT, width=3)
        d.text((bx1 - 60, my + 56), f"{ell:.2f}", font=F.small, fill=TEXT)

    elif state == "PENDING":
        d.text((x0 + 24, cy), "Sign readable \u2014 reasoning", font=F.h2, fill=TEXT)
        if sent_crop is not None:
            th, s = _fit(sent_crop, PANEL_W - 48, 300)
            canvas.paste(th, (x0 + 24, cy + 70))
        # in-flight ring
        cx, cyy, rr = x0 + PANEL_W // 2, cy + 470, 44
        d.ellipse((cx - rr, cyy - rr, cx + rr, cyy + rr), outline=(60, 64, 72), width=8)
        if in_flight_frac is not None:
            d.arc((cx - rr, cyy - rr, cx + rr, cyy + rr), start=-90,
                  end=-90 + 360 * min(in_flight_frac, 1.0), fill=CALL, width=8)
        d.text((cx - _text_w(d, "VLM call in flight", F.small) // 2, cyy + rr + 10),
               "VLM call in flight", font=F.small, fill=CALL)

    elif state == "RECALL":
        d.text((x0 + 24, cy), "Known junction \u2014 decision recalled", font=F.h2, fill=TEXT)
        glyph = ARROW.get(decision or "", "?")
        _rounded(d, (x0 + 24, cy + 44, x1 - 24, cy + 184), (36, 40, 48), r=16, outline=MEMORY, width=3)
        d.text((x0 + 44, cy + 62), glyph, font=F.arrow, fill=MEMORY)
        d.text((x0 + 170, cy + 92), (decision or "").replace("_", " "), font=F.h1, fill=TEXT)
        d.text((x0 + 24, cy + 204), "no VLM call", font=F.body, fill=MUTED)

    elif state == "DECIDED":
        hdr = "Decision from sign text" if source == "ocr" else "Standing decision"
        d.text((x0 + 24, cy), hdr, font=F.h2, fill=TEXT)
        glyph = ARROW.get(decision or "", "?")
        _rounded(d, (x0 + 24, cy + 44, x1 - 24, cy + 184), (36, 40, 48), r=16, outline=PROMPT, width=3)
        d.text((x0 + 44, cy + 62), glyph, font=F.arrow, fill=PROMPT)
        d.text((x0 + 170, cy + 92), (decision or "").replace("_", " "), font=F.h1, fill=TEXT)



def _wrap(d, text, xy, width, font, fill, max_lines=4):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if _text_w(d, t, font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    x, y = xy
    for i, ln in enumerate(lines[:max_lines]):
        if i == max_lines - 1 and len(lines) > max_lines:
            ln = ln.rstrip(".,") + " \u2026"
        d.text((x, y + i * (font.size + 8)), ln, font=font, fill=fill)


def draw_minimap(d: ImageDraw.ImageDraw, F: Fonts, mm: dict, box):
    """Route trace + junction nodes only. mm: trace [(x,y)], nodes [(x,y,known)],
    pose (x,y) | None, bounds (xmin,xmax,ymin,ymax)."""
    bx0, by0, bx1, by1 = box
    _rounded(d, box, (20, 23, 28), r=12)
    xmin, xmax, ymin, ymax = mm["bounds"]
    pad = 18
    sw, sh = (bx1 - bx0 - 2 * pad), (by1 - by0 - 2 * pad)
    sc = min(sw / max(xmax - xmin, 1e-6), sh / max(ymax - ymin, 1e-6))
    offx = bx0 + pad + (sw - (xmax - xmin) * sc) / 2
    offy = by0 + pad + (sh - (ymax - ymin) * sc) / 2
    P = lambda x, y: (offx + (x - xmin) * sc, by1 - pad - (sh - (ymax - ymin) * sc) / 2 - (y - ymin) * sc)
    tr = mm.get("trace") or []
    if len(tr) > 1:
        d.line([P(x, y) for x, y in tr], fill=(120, 126, 136), width=3)
    for (x, y, known) in mm.get("nodes") or []:
        cx, cy = P(x, y)
        r = 9
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=MEMORY if known else (60, 64, 72),
                  outline=MEMORY, width=2)
    if mm.get("pose"):
        cx, cy = P(*mm["pose"])
        d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=TEXT)
    d.text((bx0 + 12, by0 + 8), "route", font=F.tiny, fill=MUTED)


def draw_timeline(d: ImageDraw.ImageDraw, F: Fonts, pos: int, n: int, bands: list[str],
                  fires: list[int], lands: list[int], junction_pos: int | None,
                  decision: str | None, dist_to_junction: float | None,
                  recalls: list[int] | None = None):
    x0, x1 = MARGIN, W - MARGIN
    _rounded(d, (x0 - 6, TL_Y, x1 + 6, TL_Y + TL_H), PANEL_BG, r=18)
    d.text((x0 + 18, TL_Y + 14), "scheduler mode over the approach", font=F.small, fill=MUTED)
    bx0, bx1, by0, by1 = x0 + 18, x1 - 18, TL_Y + 50, TL_Y + 86
    d.rounded_rectangle((bx0, by0, bx1, by1), radius=8, fill=(40, 44, 52))
    px = lambda p: bx0 + int((bx1 - bx0) * p / max(n - 1, 1))
    # bands (runs of identical state up to current pos)
    start = 0
    for i in range(1, pos + 2):
        if i > pos or bands[i] != bands[start]:
            d.rectangle((px(start), by0, px(min(i, pos)), by1), fill=STATE_RGB[bands[start]])
            start = i
    for f in fires:
        fx = px(f)
        d.polygon([(fx - 8, by0 - 14), (fx + 8, by0 - 14), (fx, by0 - 2)], fill=CALL)
    for f in lands:
        fx = px(f)
        d.polygon([(fx - 8, by1 + 14), (fx + 8, by1 + 14), (fx, by1 + 2)], fill=STATE_RGB["DECIDED"])
    for f in (recalls or []):
        fx = px(f)
        d.polygon([(fx, by0 - 16), (fx + 8, by0 - 8), (fx, by0 - 1), (fx - 8, by0 - 8)], fill=MEMORY)
    cx = px(pos)
    d.line((cx, by0 - 18, cx, by1 + 18), fill=TEXT, width=3)
    # legend + readouts
    ly = TL_Y + 118
    lx = bx0
    for name, col in (("EXECUTE", STATE_RGB["NO_SIGN"]), ("ACQUIRE", STATE_RGB["ARMED"]),
                      ("REASON", STATE_RGB["PENDING"]), ("EXECUTE \u00b7 decided", STATE_RGB["DECIDED"])):
        d.rounded_rectangle((lx, ly + 4, lx + 18, ly + 22), radius=4, fill=col)
        d.text((lx + 26, ly), name, font=F.small, fill=MUTED)
        lx += 40 + _text_w(d, name, F.small) + 30
    d.polygon([(lx, ly + 20), (lx + 16, ly + 20), (lx + 8, ly + 6)], fill=CALL)
    d.text((lx + 26, ly), "VLM call", font=F.small, fill=MUTED)
    if recalls:
        lx += 40 + _text_w(d, "VLM call", F.small) + 30
        d.polygon([(lx + 8, ly + 4), (lx + 16, ly + 12), (lx + 8, ly + 20), (lx, ly + 12)], fill=MEMORY)
        d.text((lx + 26, ly), "memory recall", font=F.small, fill=MUTED)
    # readouts live on the title row (top-right) so the junction label below the bar is never covered
    right = f"standing prompt: {(decision or 'straight').replace('_', ' ')}"
    d.text((bx1 - _text_w(d, right, F.body), TL_Y + 10), right, font=F.body, fill=PROMPT)


def draw_caption(d: ImageDraw.ImageDraw, F: Fonts, text: str):
    tw = _text_w(d, text, F.caption)
    cx = MARGIN + CAM_W // 2
    y = MARGIN + CAM_H - 70
    _rounded(d, (cx - tw // 2 - 22, y - 10, cx + tw // 2 + 22, y + 46), (0, 0, 0), r=12)
    d.text((cx - tw // 2, y), text, font=F.caption, fill=TEXT)


def draw_callout(d: ImageDraw.ImageDraw, F: Fonts, text: str, color):
    tw = _text_w(d, text, F.h2)
    cx = MARGIN + CAM_W // 2
    y = MARGIN + 30
    _rounded(d, (cx - tw // 2 - 26, y - 12, cx + tw // 2 + 26, y + 44), (0, 0, 0), r=12,
             outline=color, width=3)
    d.text((cx - tw // 2, y), text, font=F.h2, fill=color)


def compose(frame_rgb: np.ndarray, recs: list[dict], st: dict, F: Fonts) -> np.ndarray:
    """Pure function: one composed 1920x1080 frame from a state dict.  Keys of st:
    state, best_id, best_rec, best_crop, sent_crop, tau, in_flight_frac, decision,
    rationale, n_calls, goal, pos, n, bands, fires, lands, junction_pos, caption,
    callout, callout_color, hide_irrelevant, dist."""
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    draw_camera(canvas, d, frame_rgb, recs, st["best_id"], F, st.get("hide_irrelevant", False))
    draw_panel(canvas, d, F, st["state"], st.get("best_rec"), st.get("best_crop"),
               st.get("sent_crop"), st["tau"], st.get("in_flight_frac"), st.get("decision"),
               st.get("rationale"), st["n_calls"], st["goal"],
               st.get("source"), st.get("matched_text"))
    if st.get("minimap"):
        draw_minimap(d, F, st["minimap"], (PANEL_X + 24, MARGIN + CAM_H - 24 - 230,
                                           PANEL_X + PANEL_W - 24, MARGIN + CAM_H - 24))
    if not st.get("no_timeline"):
        draw_timeline(d, F, st["pos"], st["n"], st["bands"], st["fires"], st["lands"],
                      st.get("junction_pos"), st.get("decision"), st.get("dist"),
                      st.get("recalls", []))
    if st.get("caption"):
        draw_caption(d, F, st["caption"])
    if st.get("callout"):
        draw_callout(d, F, st["callout"], st.get("callout_color", CALL))
    return np.asarray(canvas)


# ---------------------------------------------------------------- main -------
def main() -> None:
    import cv2
    from ..gate import EvidenceGate, GateConfig, PlateEvidence, VLMResponse, new_goal
    from ..memory import Memory
    from .dump_features import RosbagSource

    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--topic", default="/c1/image_raw")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--decision", default="turn_right", help="oracle answer if --vlm sim")
    ap.add_argument("--vlm", choices=["sim", "real"], default="real")
    ap.add_argument("--model", default="")
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--rpm", type=float, default=0.0)
    ap.add_argument("--cache-dir", default="vlm_cache")
    ap.add_argument("--latency", type=float, default=3.0,
                    help="seconds shown for cached answers (use the measured median)")
    ap.add_argument("--fixed-latency", action="store_true",
                    help="always display --latency for the call, regardless of measured API latency "
                         "(video presentation only; paper latency must be measured)")
    ap.add_argument("--junction", type=int, default=None, help="annotated junction_frame (raw idx)")
    ap.add_argument("--captions", default=None, help="json: [{from,to,text}] in raw frame idx")
    ap.add_argument("--odom-csv", default=None, help="optional t,x,y,yaw csv for distance readout")
    ap.add_argument("--hold-fire", type=float, default=1.0, help="freeze seconds at fire")
    ap.add_argument("--hold-answer", type=float, default=1.5, help="freeze seconds at answer")
    ap.add_argument("--hide-irrelevant", action="store_true", help="draw only R>0.5 plates")
    ap.add_argument("--direct", action="store_true",
                    help="at fire time, try the direct read (text + arrow icon); if confident, "
                         "decide without a VLM call, else fall through to the VLM")
    ap.add_argument("--direct-min-conf", type=float, default=None,
                    help="override DIRECT_MIN_CONF from evidence/direct.py (paper value must come from the tune split)")
    ap.add_argument("--min-plate-h", type=float, default=60.0,
                    help="do not call the VLM while the plate is shorter than this (px in the source "
                         "frame); keep acquiring instead. 0 disables. Stand-in for calibrated legibility.")
    ap.add_argument("--fast-path", action="store_true",
                    help="let the gate decide from OCR text alone when the sign is unambiguous (no VLM)")
    ap.add_argument("--fps-out", type=float, default=30.0)
    ap.add_argument("--font", default=None, help="path to a .ttf (default: DejaVu/Arial search)")
    ap.add_argument("--h264", action="store_true", help="re-encode with ffmpeg libx264 (yuv420p)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    F = Fonts(a.font)
    per_frame = load_jsonl(a.jsonl)
    frames_wanted = sorted(per_frame.keys())
    n = len(frames_wanted)
    pos_of = {f: i for i, f in enumerate(frames_wanted)}
    stride = int(min(np.diff(frames_wanted))) if n > 1 else 1
    bag_fps_eff = 10.0                          # dataset rate after stride (matches project)
    repeat = max(int(round(a.fps_out / bag_fps_eff)), 1)
    latency_frames = max(int(a.latency * bag_fps_eff), 1)
    captions = load_captions(a.captions)
    odom = load_odom(a.odom_csv)
    junction_pos = pos_of.get(a.junction) if a.junction is not None else None

    reasoner = None
    if a.vlm == "real":
        from ..reasoner import OpenAICompatVLM, Reasoner
        client = OpenAICompatVLM(model=a.model, base_url=a.base_url, rpm=a.rpm,
                                 reasoning_effort=None if a.reasoning.lower() in ("", "none", "off")
                                 else a.reasoning)
        run_dir = Path(a.out).parent / "payloads" / Path(a.bag).name
        reasoner = Reasoner(client, model_tag=f"{a.model}:{a.reasoning}",
                            cache_dir=Path(a.cache_dir), payload_dir=run_dir)

    gate = EvidenceGate(GateConfig(tau=a.tau, fast_path=a.fast_path))
    mem, state = Memory(a.goal), new_goal()
    pending = None            # (land_raw_idx, VLMResponse)
    pending_start = None
    last_answer = None
    sent_crop = None
    fires, lands, bands = [], [], []
    prev_decision, source, ocr_decided_now = None, None, False
    n_direct, n_veto = 0, 0
    seen_relevant, last_best_rec, last_best_crop = False, None, None
    src = RosbagSource(a.bag, a.topic, stride=stride)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp_out = a.out if not a.h264 else str(Path(a.out).with_suffix(".raw.mp4"))
    writer = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps_out, (W, H))

    def emit(frame, k):
        for _ in range(k):
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    for raw_idx, img in src:
        recs = per_frame.get(raw_idx, [])
        pos = pos_of.get(raw_idx, len(bands))
        landed_now = False
        if pending and raw_idx >= pending[0]:
            state, upds = gate.on_response(state, pending[1], raw_idx)
            for u in upds:
                mem.apply(u)
            pending, pending_start, landed_now = None, None, True
            lands.append(pos)
        evid = [PlateEvidence(r["plate_id"], float(r["R"]), float(r["ell"])) for r in recs]
        res = gate.tick(state, evid, mem, raw_idx)
        for u in res.memory_updates:
            mem.apply(u)
        fired_now = False
        acquiring = False           # fire vetoed: sign too small to read yet -> keep acquiring
        if res.fire:
            rec = next((r for r in recs if r["plate_id"] == res.best.plate_id), None)
            crop_np, plate_h = None, 0.0
            if rec and rec.get("box"):
                from ..evidence.detect import payload_crop
                others = [r["box"] for r in recs if r.get("box")]
                crop_np = payload_crop(img, rec["box"], others)
                plate_h = float(rec["box"][3] - rec["box"][1])
            # 1) direct read: text already matched the goal; can the arrow be read too?
            direct_dir = None
            if a.direct and crop_np is not None:
                from ..evidence.direct import direct_for_crop, DIRECT_MIN_CONF
                dr = direct_for_crop(crop_np, goal_matched=(rec is not None and float(rec["R"]) > 0.5))
                min_conf = a.direct_min_conf if a.direct_min_conf is not None else DIRECT_MIN_CONF
                if dr["direction"] and dr["conf"] >= min_conf:
                    direct_dir = {"left": "turn_left", "right": "turn_right",
                                  "up": "straight"}.get(dr["direction"])
                att = Path(a.out).parent / "direct_attempts"; att.mkdir(exist_ok=True)
                Image.fromarray(crop_np).save(att / f"f{raw_idx}_{dr['direction']}_{dr['conf']:.2f}.jpg")
                print(f"direct read at f{raw_idx}: {dr['direction']} conf={dr['conf']} "
                      f"({dr['reason']})  plate_h={plate_h:.0f}px")
            if direct_dir is not None:
                # decided from the sign itself: no VLM call
                state, upds = gate.on_response(res.state, VLMResponse(res.best.plate_id, True, direct_dir),
                                               raw_idx)
                for u in upds:
                    mem.apply(u)
                n_direct += 1
                sent_crop = Image.fromarray(crop_np)
                Image.fromarray(crop_np).save(Path(a.out).parent / f"direct_f{raw_idx}.jpg")
            # 2) sign still too small to be readable -> keep acquiring, retry next frame
            elif plate_h < a.min_plate_h:                 # applies with or without --direct
                acquiring = True                  # keep the pre-tick state (ARMED); no call
                n_veto += 1
            # 3) readable, direct read could not resolve it -> the VLM earns its call
            else:
                state = res.state
                fired_now = True
                fires.append(pos)
                sent_crop = Image.fromarray(crop_np) if crop_np is not None else None
                if reasoner is not None:
                    ans = reasoner.ask(a.goal, [rec["text"] if rec else ""],
                                       [crop_np if crop_np is not None else img],
                                       scene=None, memory_summary=f"{len(mem)} signs consumed")
                    last_answer = ans
                    lat_s = a.latency if (ans.cached or a.fixed_latency) else max(ans.latency_s, 0.1)
                    pending = (raw_idx + max(int(lat_s * bag_fps_eff), 1) * stride,
                               VLMResponse(res.best.plate_id, ans.applicable, ans.direction))
                else:
                    applicable = any(r["plate_id"] == res.best.plate_id and r["R_struct"] >= 1.0
                                     for r in recs)
                    pending = (raw_idx + latency_frames * stride,
                               VLMResponse(res.best.plate_id, applicable,
                                           a.decision if applicable else None))
                pending_start = raw_idx
        else:
            state = res.state
        gs = state.state.value
        if acquiring or gs == "ARMED":
            seen_relevant = True
        if gs in ("PENDING", "DECIDED"):
            seen_relevant = False              # acquire phase is over
        # sticky ACQUIRE: once a relevant sign has been seen, stay in acquire until reason/decision
        disp_state = "ARMED" if (acquiring or (seen_relevant and gs == "NO_SIGN")) else gs
        bands.append(disp_state)
        ocr_decided_now = False
        if state.decision != prev_decision:
            if state.decision:
                source = "vlm" if landed_now else "ocr"
                ocr_decided_now = source == "ocr"
            prev_decision = state.decision

        # ---- presentation state --------------------------------------------
        best_rec = None
        best_crop = None
        if res.best is not None:
            best_rec = next((r for r in recs if r["plate_id"] == res.best.plate_id), None)
            if best_rec and best_rec.get("box") and disp_state == "ARMED":
                x0, y0, x1, y1 = [int(v) for v in best_rec["box"]]
                pad = int(0.25 * (x1 - x0))
                best_crop = Image.fromarray(img[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad])
        if best_rec is not None and best_crop is not None:
            last_best_rec, last_best_crop = best_rec, best_crop
        if disp_state == "ARMED" and best_rec is None:
            best_rec, best_crop = last_best_rec, last_best_crop
        in_flight = None
        if pending and pending_start is not None:
            in_flight = (raw_idx - pending_start) / max(pending[0] - pending_start, 1)
        caption = next((c["text"] for c in captions if c["from"] <= raw_idx <= c["to"]), None)
        dist = None
        if odom is not None and a.junction is not None:
            # nearest odom rows by frame fraction; good enough for a readout
            i_now = min(int(len(odom) * pos / max(n, 1)), len(odom) - 1)
            i_j = min(int(len(odom) * junction_pos / max(n, 1)), len(odom) - 1) if junction_pos else i_now
            dist = float(np.hypot(*(odom[i_j, 1:3] - odom[i_now, 1:3])))
        st = dict(state=disp_state, best_id=res.best.plate_id if res.best else None,
                  best_rec=best_rec, best_crop=best_crop, sent_crop=sent_crop, tau=a.tau,
                  in_flight_frac=in_flight, decision=state.decision,
                  rationale=(last_answer.summary if last_answer is not None else None),
                  n_calls=len(fires), goal=a.goal, pos=pos, n=n, bands=bands,
                  fires=fires, lands=lands, junction_pos=junction_pos, caption=caption,
                  hide_irrelevant=a.hide_irrelevant, dist=dist, source=source,
                  matched_text=(best_rec["text"] if best_rec else None))

        frame = compose(img, recs, st, F)
        emit(frame, repeat)

        if fired_now and best_rec is not None:
            st2 = dict(st, callout="sign readable \u2192 reason", callout_color=CALL)
            emit(compose(img, recs, st2, F), int(a.hold_fire * a.fps_out))
        if landed_now and state.decision:
            st3 = dict(st, callout=f"decision:  {state.decision.replace('_', ' ')}",
                       callout_color=STATE_RGB["DECIDED"])
            emit(compose(img, recs, st3, F), int(a.hold_answer * a.fps_out))
        if ocr_decided_now:
            st4 = dict(st, callout=f"decided from sign text — no VLM call:  "
                                   f"{state.decision.replace('_', ' ')}",
                       callout_color=STATE_RGB["DECIDED"])
            emit(compose(img, recs, st4, F), int(a.hold_answer * a.fps_out))

    if pending:
        state, upds = gate.on_response(state, pending[1], pending[0])
        for u in upds:
            mem.apply(u)
    writer.release()

    if a.h264:
        encs = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                              capture_output=True, text=True).stdout
        vcodec = (["-c:v", "libx264", "-crf", "18"] if "libx264" in encs
                  else ["-c:v", "libopenh264", "-b:v", "12M"])
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_out, *vcodec,
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True)
        Path(tmp_out).unlink()
    print(f"wrote {a.out}  vlm_calls={len(fires)} direct_decisions={n_direct} "
          f"acquire_frames={sum(b == 'ARMED' for b in bands)} (size_vetoes={n_veto}) "
          f"final={state.state.value} prompt={state.decision}")
    if last_answer is not None:
        print(f"answer: {last_answer.direction} conf={last_answer.confidence:.2f} "
              f"latency={last_answer.latency_s:.2f}s cached={last_answer.cached}")


if __name__ == "__main__":
    main()