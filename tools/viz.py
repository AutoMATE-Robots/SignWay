"""Drawing detections so a human can actually read them.

Two problems make annotated frames useless in practice, and both are solved here:

  TINY TEXT — PIL's built-in bitmap font is about 11px, which is unreadable on a 1280-wide
  video played at any sensible size. A real TrueType face is loaded at a configurable size,
  falling back through several standard paths so this works on a cluster, a Mac or Windows.

  STACKED LABELS — when detections overlap (two cones side by side, a sign inside a doorway),
  their labels land on top of each other and you cannot tell which text belongs to which box.
  Labels are placed one at a time, highest-confidence first, and each one is nudged to the
  first free slot: above the box, below it, inside it, then stepped down until it fits. A
  placed label reserves its rectangle so nothing later overlaps it.

Each label is drawn as a filled chip in the box's colour with black or white text chosen by
the chip's luminance, so it stays legible against a bright corridor or a dark doorway.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]

_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
)

_font_cache: dict = {}


def load_font(size: int):
    """A readable TrueType face at `size`, or the bitmap default if none is installed."""
    if size in _font_cache:
        return _font_cache[size]
    from PIL import ImageFont
    font = None
    for path in _FONT_PATHS:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)      # Pillow >= 10.1
        except TypeError:
            font = ImageFont.load_default()               # older: fixed tiny size
    _font_cache[size] = font
    return font


def ascii_safe(s: str) -> str:
    """Bitmap fallback fonts are latin-1; model phrases and VLM text are not."""
    return (str(s).replace("\u2014", "-").replace("\u2013", "-")
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2192", "->").replace("\u2190", "<-")
            .encode("ascii", "replace").decode())


def _text_colour(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Black on light chips, white on dark ones (Rec. 709 luma)."""
    lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def _overlaps(a, b, pad: int = 2) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0]
                or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def box_iou(a, b) -> float:
    """Intersection over union for two (x0, y0, x1, y1) boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_for_drawing(boxes, merge_iou: float = 0.7, max_phrases: int = 2):
    """Collapse near-duplicate same-category boxes into one drawable entry.

    Open-vocabulary detectors routinely return one physical object several times under
    different phrases — a single door plate comes back as "sign", "exit sign" and "door sign".
    Avoiding label OVERLAP does not help there: three boxes on one object legitimately need
    three labels, and the frame becomes unreadable. So they are merged: the highest-scoring
    box keeps its outline, and the phrases fold into one caption ("sign / exit sign 0.44").

    Cosmetic only. Callers keep logging and counting the raw detections, because how many
    boxes a model emits is a real property of that model and the renderer must not quietly
    edit it.

    Returns dicts with bbox, score, category, phrases, label, merged.
    """
    groups = []
    for b in sorted(boxes, key=lambda d: -float(d.score)):
        for g in groups:
            if g["category"] == b.category and box_iou(g["bbox"], b.bbox) >= merge_iou:
                if b.phrase not in g["phrases"]:
                    g["phrases"].append(b.phrase)
                g["merged"] += 1
                break
        else:
            groups.append({"bbox": tuple(b.bbox), "score": float(b.score),
                           "category": b.category, "phrases": [b.phrase], "merged": 1})
    for g in groups:
        shown = g["phrases"][:max_phrases]
        extra = len(g["phrases"]) - len(shown)
        name = " / ".join(shown) + (f" +{extra}" if extra else "")
        g["label"] = f"{name} {g['score']:.2f}"
    return groups


def draw_labeled_boxes(arr: np.ndarray,
                       items: Sequence[dict],
                       title: Optional[str] = None,
                       font_size: int = 18,
                       box_width: int = 3,
                       title_size: Optional[int] = None) -> np.ndarray:
    """Draw boxes with non-overlapping labels.

    items: dicts with 'bbox' (x0,y0,x1,y1), 'colour' (r,g,b), 'text', optional 'score'
           (used only to decide who gets the best label position).
    """
    from PIL import Image, ImageDraw
    pil = Image.fromarray(arr.astype(np.uint8)) if isinstance(arr, np.ndarray) else arr
    dr = ImageDraw.Draw(pil)
    font = load_font(font_size)
    W, H = pil.size

    title_h = 0
    if title:
        tfont = load_font(title_size or max(font_size, 20))
        tb = dr.textbbox((0, 0), ascii_safe(title), font=tfont)
        title_h = tb[3] - tb[1] + 10
        dr.rectangle([0, 0, W, title_h], fill=(0, 0, 0))
        dr.text((6, 4), ascii_safe(title), fill=(255, 255, 255), font=tfont)

    # confident detections claim the best label slots first
    ordered = sorted(items, key=lambda d: -float(d.get("score", 0.0)))
    taken: list = []

    for it in ordered:
        x0, y0, x1, y1 = [float(v) for v in it["bbox"]]
        col = tuple(it.get("colour", (240, 200, 40)))
        dr.rectangle([x0, y0, x1, y1], outline=col, width=box_width)

        text = ascii_safe(it.get("text", ""))
        if not text:
            continue
        tb = dr.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        cw, ch = tw + 8, th + 6                       # chip size with padding

        # candidate anchors, best first: above the box, below it, inside its top,
        # then stepped downward until something is free
        cands = [(x0, y0 - ch - 2), (x0, y1 + 2), (x0 + 2, y0 + 2)]
        cands += [(x0, y0 - ch - 2 + k * (ch + 3)) for k in range(1, 9)]

        chip = None
        for cx, cy in cands:
            cx = min(max(0.0, cx), max(0.0, W - cw))
            cy = min(max(float(title_h), cy), max(float(title_h), H - ch))
            rect = (cx, cy, cx + cw, cy + ch)
            if not any(_overlaps(rect, t) for t in taken):
                chip = rect
                break
        if chip is None:                              # frame is crowded; accept a collision
            cx = min(max(0.0, x0), max(0.0, W - cw))
            cy = min(max(float(title_h), y0 - ch - 2), max(float(title_h), H - ch))
            chip = (cx, cy, cx + cw, cy + ch)

        dr.rectangle(list(chip), fill=col)
        # a thin tie-line when the label had to move away from its box
        if abs(chip[1] - (y0 - ch - 2)) > ch + 4 or abs(chip[0] - x0) > 4:
            dr.line([chip[0] + 3, chip[3], x0 + 3, y0], fill=col, width=1)
        dr.text((chip[0] + 4, chip[1] + 3), text, fill=_text_colour(col), font=font)
        taken.append(chip)

    return np.asarray(pil)
