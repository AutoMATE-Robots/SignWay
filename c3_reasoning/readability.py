"""Is this sign legible YET? — the gate that decides whether reasoning is worth its cost.

This is the project's central claim in one file. The expensive VLM costs 1-3 seconds; deciding
whether to spend that must cost ~nothing. So the gate uses three cheap measurements on the
detected text region and never invokes a model.

WHY NOT ASK THE VLM: the earlier approach queried Qwen three times and checked whether the
answers agreed. That spends 9-15s to decide whether to spend 3s, and it measures the wrong
thing — agreement is not correctness. Three confident identical misreads look exactly like
three correct reads. Self-consistency is known to be overconfident, and the OCR-abstention
literature notes semantic-agreement methods are unsuitable for OCR specifically.

WHY THESE THREE FEATURES:
  1. text height (px) — the hard physical floor. Recognition fails below a cap-height of
     roughly 20-30px regardless of how good the model is: industry guidance (Cognex) says
     >=30px tall for reliable OCR; Tesseract/Nuance guidance lands at 20-40px cap-height.
     This is also the feature that makes the gate PREDICTIVE: height grows as the robot
     approaches, so the gate fires exactly once, at the right moment, with no tuning per sign.
  2. blur (variance of Laplacian) — a moving robot motion-blurs text that is nominally large
     enough. Cheap, single-pass, standard (Pech-Pacheco et al. 2000). Threshold is
     camera-specific and must be calibrated, which is why it is a Params field, not a constant.
  3. detector confidence — the text detector already produces it for free.

Deliberately NOT generic image-quality metrics (NIQE/BRISQUE): the document-IQA literature is
explicit that perceptual quality does not predict OCR success — an image can look poor and OCR
at 95%. Only OCR-goal-oriented features belong here.

The gate is intentionally interpretable and training-free: every competitor gates on a learned
anomaly score or encoder confidence. Grounding the trigger in a measurable physical condition
means it needs no training data and degrades gracefully — as the robot approaches a sign,
readability rises monotonically, so the trigger is a threshold crossing rather than a guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Readability:
    """Why the gate said yes or no. Logged for every frame — this is the ablation data."""
    legible: bool
    text_height_px: float
    blur_var: float
    det_conf: float
    reason: str            # which feature blocked it (empty when legible)

    def as_dict(self) -> dict:
        return {"legible": self.legible, "h_px": round(self.text_height_px, 1),
                "blur": round(self.blur_var, 1), "conf": round(self.det_conf, 3),
                "reason": self.reason}


def laplacian_variance(gray: np.ndarray) -> float:
    """Blur score: high = sharp, low = blurry. Variance of the 3x3 Laplacian response.

    Implemented directly rather than via cv2 so the gate has no OpenCV dependency and runs
    anywhere (including inside the Isaac container, which ships a minimal Python).
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim == 3:
        g = g.mean(axis=2)
    if g.size == 0 or min(g.shape) < 3:
        return 0.0
    # 4-neighbour Laplacian on the interior; edges are dropped rather than padded, since
    # padding invents gradients that inflate the variance of a small crop.
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def text_height_from_bbox(bbox: Tuple[float, float, float, float]) -> float:
    """Cap-height proxy: the height of the detected text region in pixels."""
    _x0, y0, _x1, y1 = bbox
    return float(abs(y1 - y0))


def expected_height_px(char_height_m: float, distance_m: float, fy_px: float) -> float:
    """Pinhole prediction of text height in pixels — lets the gate ANTICIPATE.

    h_px = f_y * (character height in metres) / (distance in metres)

    Useful for two things a reactive gate cannot do: (a) telling the robot how much closer it
    must get before a sign becomes readable, and (b) generating benchmark scenarios at a known
    readability, rather than discovering it by trial.
    """
    if distance_m <= 1e-6:
        return float("inf")
    return float(fy_px * char_height_m / distance_m)


def assess(crop_gray: Optional[np.ndarray],
           bbox: Optional[Tuple[float, float, float, float]],
           det_conf: float,
           min_height_px: float = 22.0,
           min_blur_var: float = 60.0,
           min_det_conf: float = 0.5) -> Readability:
    """The gate. Returns legible + the evidence, checked cheapest-first.

    Order matters: bbox height is arithmetic on four numbers, blur touches the pixels. Check
    the free one first so most frames cost almost nothing.

    Defaults: min_height_px=22 sits at the low end of the 20-30px OCR floor (recall over
    precision — a marginal read that fails is cheaper than sailing past a sign). min_blur_var
    and min_det_conf are CAMERA- AND DETECTOR-SPECIFIC and must be calibrated per deployment;
    they are arguments, not constants, for exactly that reason.
    """
    if bbox is None:
        return Readability(False, 0.0, 0.0, float(det_conf), "no_text_detected")

    h = text_height_from_bbox(bbox)
    if h < min_height_px:
        return Readability(False, h, 0.0, float(det_conf), "too_small")

    if det_conf < min_det_conf:
        return Readability(False, h, 0.0, float(det_conf), "low_detector_confidence")

    blur = laplacian_variance(crop_gray) if crop_gray is not None else float("inf")
    if blur < min_blur_var:
        return Readability(False, h, blur, float(det_conf), "too_blurry")

    return Readability(True, h, blur, float(det_conf), "")
