"""
adaptive_reasoning/evidence/direct.py — the DIRECT-READ path.

When a sign's relationship to the goal is unambiguous from the evidence alone
(the goal's text line is present AND that line carries exactly one clear arrow
icon), the scheduler can decide WITHOUT a VLM call.  This module provides:

  detect_arrows(img)        -> arrow icons in a plate crop (direction + confidence)
  resolve_direct(...)       -> (direction | None, confidence, reason)   conservative
  CLI: evaluate on crops + annotations, report coverage/precision per bag

It is deliberately CONSERVATIVE: anything short of "one goal line, one clear
arrow, all arrows on the plate agree or are row-separable" returns None and the
gate falls through to the VLM.  A wrong direct read costs a wrong turn, so the
precision of this path must be measured (see `evaluate`) before the paper
claims it.  docTR reads text, not icons; arrows on institutional signage are
almost always icons, which is exactly why this module exists.

Integration (three touch points, see NOTES at the bottom of the file):
  1. dump_features: per plate record, add  "direct": {"direction", "conf", "reason"}
  2. gate.tick:     if fast_path and ell >= tau and rec.direct.direction and
                    rec.direct.conf >= DIRECT_MIN_CONF -> DECIDED, source="direct", no fire
  3. render_submission already shows source "ocr"/"direct" decisions with 0 calls
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

DIRECTIONS = ("left", "right", "up", "down")
DIRECT_MIN_CONF = 0.70          # gate trusts direct reads at/above this; RE-TUNE on the tune split via `evaluate`


@dataclass
class Arrow:
    cx: float
    cy: float
    w: float
    h: float
    direction: str
    conf: float


# ----------------------------------------------------------------- detector --
def _components(binary: np.ndarray):
    import cv2
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        yield i, x, y, w, h, area, labels


def _arrow_from_mask(mask: np.ndarray) -> tuple[str | None, float]:
    """Classify one connected component as an arrow: returns (direction, conf)."""
    import cv2
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 30:
        return None, 0.0
    hull = cv2.convexHull(c)
    solidity = area / max(cv2.contourArea(hull), 1e-6)
    # arrows are concave (head + shaft): solidity typically 0.45..0.85
    if not (0.40 <= solidity <= 0.88):
        return None, 0.0
    pts = c.reshape(-1, 2).astype(np.float64)
    mu = pts.mean(0)
    cov = np.cov((pts - mu).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]          # principal axis (unit)
    perp = np.array([-axis[1], axis[0]])
    proj = (pts - mu) @ axis
    spread_pos = np.std((pts[proj > 0] - mu) @ perp) if (proj > 0).sum() > 5 else 0
    spread_neg = np.std((pts[proj < 0] - mu) @ perp) if (proj < 0).sum() > 5 else 0
    if spread_pos == 0 or spread_neg == 0:
        return None, 0.0
    # the pointed (head) end has LARGER perpendicular spread (the barbs) than the shaft end
    ratio = max(spread_pos, spread_neg) / min(spread_pos, spread_neg)
    if ratio < 1.35:
        return None, 0.0
    tip = axis if spread_pos > spread_neg else -axis
    ang = math.degrees(math.atan2(tip[1], tip[0]))   # image coords: +y is down
    if -45 <= ang < 45:
        d = "right"
    elif 45 <= ang < 135:
        d = "down"
    elif -135 <= ang < -45:
        d = "up"
    else:
        d = "left"
    # confidence: asymmetry strength + closeness to a clean arrow's solidity (~0.70)
    conf = min(1.0, 0.5 * min((ratio - 1.35) / 1.0, 1.0) + 0.5 * (1 - abs(solidity - 0.70) / 0.25))
    return d, float(max(conf, 0.0))


def detect_arrows(img: np.ndarray, text_boxes: list | None = None) -> list[Arrow]:
    """Find arrow icons in a plate crop (RGB or BGR uint8).  Tries both polarities
    (dark-on-light and light-on-dark plates).  Rejects components that sit in a
    row of similar-height neighbours (those are letters, not icons)."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    g = cv2.GaussianBlur(g, (3, 3), 0)
    Hh, Ww = g.shape
    out: list[Arrow] = []
    for pol in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, b = cv2.threshold(g, 0, 255, pol + cv2.THRESH_OTSU)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        comps = list(_components(b))
        for i, x, y, w, h, area, labels in comps:
            if area < 0.0004 * Hh * Ww or area > 0.08 * Hh * Ww:
                continue
            if not (0.35 <= w / max(h, 1) <= 2.8):
                continue
            # letters live in rows: count similar-height components in the same y-band nearby
            row_mates = sum(1 for j, x2, y2, w2, h2, a2, _ in comps
                            if j != i and abs((y2 + h2 / 2) - (y + h / 2)) < 0.5 * h
                            and 0.6 <= h2 / max(h, 1) <= 1.6 and abs(x2 - x) < 6 * h)
            if row_mates >= 3:
                continue
            if text_boxes and any(_inside((x + w / 2, y + h / 2), tb) for tb in text_boxes):
                continue
            mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
            d, conf = _arrow_from_mask(mask)
            if d is not None and conf > 0.3:
                out.append(Arrow(float(x + w / 2), float(y + h / 2), float(w), float(h), d, float(conf)))
    # icon-box pass: a light, square-ish icon (solid) that contains a dark arrow.
    # Global thresholds merge that arrow with a dark plate; threshold INSIDE the icon instead.
    _, lb = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for i, x, y, w, h, area, labels in _components(lb):
        if area < 0.0005 * Hh * Ww or area > 0.12 * Hh * Ww or not (0.5 <= w / max(h, 1) <= 2.0):
            continue
        fill = area / float(w * h)
        if fill < 0.35:                      # a thick arrow cutout lowers fill; letters are far lower
            continue
        roi = g[y:y + h, x:x + w]
        _, ib = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ib[[0, -1], :] = 0; ib[:, [0, -1]] = 0          # cut anything touching the icon border
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(ib, connectivity=8)
        if n2 < 2:
            continue
        j = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
        if st2[j, cv2.CC_STAT_AREA] < 0.08 * w * h:      # too small to be the icon's glyph
            continue
        mask = (lab2 == j).astype(np.uint8) * 255
        d, conf = _arrow_from_mask(mask)
        if d is not None:
            out.append(Arrow(float(x + w / 2), float(y + h / 2), float(w), float(h), d,
                             float(min(1.0, conf + 0.2))))   # icon context is strong evidence
    # de-duplicate across polarities (same icon found twice): keep higher conf
    out.sort(key=lambda a: -a.conf)
    kept: list[Arrow] = []
    for a in out:
        if all(abs(a.cx - k.cx) > 0.5 * max(a.w, k.w) or abs(a.cy - k.cy) > 0.5 * max(a.h, k.h)
               for k in kept):
            kept.append(a)
    return kept


def _inside(p, box) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


# ----------------------------------------------------------------- resolver --
def resolve_direct(goal_line_box: tuple | None, arrows: list[Arrow],
                   n_lines: int, goal_matched: bool) -> tuple[str | None, float, str]:
    """Decide from evidence alone, or refuse.
    goal_line_box: box of the OCR line that structurally matches the goal (None if
                   only plate-level text is available), n_lines: text lines on the plate."""
    if not goal_matched:
        return None, 0.0, "goal not on plate"
    if not arrows:
        return None, 0.0, "no arrow icon found"
    # single-arrow plate, or all arrows agree -> plate-level direction
    dirs = Counter(a.direction for a in arrows)
    if len(dirs) == 1:
        conf = min(1.0, np.mean([a.conf for a in arrows]) * (1.0 if len(arrows) == 1 else 0.95))
        return arrows[0].direction, float(conf), "single arrow direction on plate"
    # mixed arrows: need the goal line's row to pick the right one
    if goal_line_box is None:
        return None, 0.0, "mixed arrows, no line association"
    y0, y1 = goal_line_box[1], goal_line_box[3]
    hh = max(y1 - y0, 1)
    row = [a for a in arrows if (y0 - 0.6 * hh) <= a.cy <= (y1 + 0.6 * hh)]
    rdirs = Counter(a.direction for a in row)
    if len(rdirs) == 1 and n_lines >= 1:
        return row[0].direction, float(min(1.0, np.mean([a.conf for a in row]) * 0.9)), \
            "arrow in the goal line's row"
    return None, 0.0, "ambiguous arrows in goal row"


UPSCALE_TO = 900   # small (far) crops are upscaled before detection; tiny icons lose their shape otherwise


def direct_for_crop(crop: np.ndarray, goal_matched: bool, goal_line_box=None,
                    n_lines: int = 1, text_boxes=None) -> dict:
    import cv2
    s = UPSCALE_TO / max(crop.shape[:2])
    if s > 1.0:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        if goal_line_box is not None:
            goal_line_box = tuple(v * s for v in goal_line_box)
        if text_boxes:
            text_boxes = [tuple(v * s for v in b) for b in text_boxes]
    arrows = detect_arrows(crop, text_boxes)
    d, conf, why = resolve_direct(goal_line_box, arrows, n_lines, goal_matched)
    return {"direction": d, "conf": round(conf, 3), "reason": why,
            "arrows": [asdict(a) for a in arrows]}


# ----------------------------------------------------------------- evaluate --
def _annot_dir(transcript: str, decision_gt: str) -> str | None:
    """Ground-truth direction: prefer decision_gt; fall back to the transcript's arrow glyphs."""
    m = {"turn_left": "left", "left": "left", "turn_right": "right", "right": "right",
         "straight": "up"}
    if decision_gt in m:
        return m[decision_gt]
    t = transcript or ""
    if "<" in t or "\u2190" in t:
        return "left"
    if ">" in t or "\u2192" in t:
        return "right"
    if "^" in t or "\u2191" in t:
        return "up"
    return None


def evaluate(crops_root: str, csv_path: str, out_json: str | None = None) -> None:
    """Run the detector over <crops_root>/<bag>_crops/*.jpg and compare the majority
    direction per bag against ar_extra.csv.  Reports coverage (how often the direct
    path would decide) and precision (how often it's right)."""
    import cv2
    rows = list(csv.reader(open(csv_path)))
    # columns per Akul's csv sample: bag,?,building,floor,room,sign_type,?,transcript,?,legible,junction,done,decision,...
    gt = {}
    for r in rows:
        if not r or r[0].startswith("#") or r[0] == "bag":
            continue
        try:
            gt[r[0]] = (_annot_dir(r[7], r[12]), r[7])
        except IndexError:
            continue
    results, decided, correct = {}, 0, 0
    for d in sorted(glob.glob(os.path.join(crops_root, "*_crops"))):
        bag = os.path.basename(d).replace("_crops", "")
        votes = Counter()
        for f in sorted(glob.glob(os.path.join(d, "*.jpg")))[-8:]:   # the last (closest) sightings
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            r = direct_for_crop(img, goal_matched=True)
            if r["direction"] and r["conf"] >= DIRECT_MIN_CONF:
                votes[r["direction"]] += 1
        pred = votes.most_common(1)[0][0] if votes else None
        truth = gt.get(bag, (None, ""))[0]
        results[bag] = {"pred": pred, "truth": truth, "votes": dict(votes)}
        if pred is not None and truth is not None:
            decided += 1
            correct += int(pred == truth)
        print(f"{bag:28s} direct={str(pred):6s} truth={str(truth):6s} votes={dict(votes)}")
    n = sum(1 for b in results if results[b]["truth"] is not None)
    print(f"\ncoverage {decided}/{n} bags decided directly; precision "
          f"{correct}/{decided if decided else 1} = {correct / max(decided, 1):.2f}")
    if out_json:
        json.dump(results, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate", help="coverage/precision of the direct path on annotated bags")
    e.add_argument("--crops-root", required=True, help="e.g. $SCRATCH/ar_replay")
    e.add_argument("--csv", required=True, help="ar_extra.csv")
    e.add_argument("--out", default=None)
    o = sub.add_parser("one", help="run on a single crop image")
    o.add_argument("image")
    a = ap.parse_args()
    if a.cmd == "evaluate":
        evaluate(a.crops_root, a.csv, a.out)
    else:
        import cv2
        img = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
        print(json.dumps(direct_for_crop(img, goal_matched=True), indent=1))

# NOTES — wiring (needs gate.py / dump_features.py / detect.py, which I haven't seen):
#  * dump_features: where each plate record is built, call
#        rec["direct"] = direct_for_crop(plate_crop, goal_matched=(R_struct >= 1.0),
#                                        goal_line_box=<box of matched line if detect.py keeps lines>,
#                                        n_lines=<line count>, text_boxes=<line boxes>)
#    so the jsonl carries the direct read and the gate/table/video all use the same numbers.
#  * gate.tick (fast_path): when the top plate is sufficient (ell >= tau) and
#        rec.direct.direction and rec.direct.conf >= DIRECT_MIN_CONF
#    transition ARMED -> DECIDED with decision = rec.direct.direction, source = "direct",
#    WITHOUT firing.  Everything else falls through to the existing fire rule.
#  * Report in the paper: coverage and precision from `evaluate` on the eval split.