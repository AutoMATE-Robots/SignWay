"""Component 3, cheap half: find text, measure whether it is legible yet. Never read it.

The division of labour is the whole design. This runs on EVERY frame and must be nearly free,
so it answers only three questions: is there text, where is it, and is it big/sharp/confident
enough to be worth reading? Reading and reasoning belong to the VLM, which runs rarely.

That is why this uses text DETECTION only (bounding boxes) and not recognition. Detection is
the cheap half of an OCR stack — DBNet reports ~62 FPS with a ResNet-18 backbone — while
recognition is the expensive half and would duplicate what the VLM is about to do anyway.

WHAT REPLACED THE MOCK: MockDetector faked a detection by comparing the robot's pose to sign
coordinates passed on the command line. It had perfect knowledge — no misses, no false
positives, no range error — and never looked at a pixel. This looks at pixels.

DISTANCE comes from the depth map when one is available (sampling the median inside the box,
which is robust to a few bad pixels), because a monocular box cannot tell a small near sign
from a large far one. BEARING comes from the box centre and the camera intrinsics, which is
pure geometry and always available. Both are needed by the orchestrator: bearing aims the
subgoal, and distance anchors the debounce key to the sign's world position rather than the
robot's.
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from common.interfaces import Detector
from common.types import Detection

from c3_reasoning.readability import Readability, assess

BBox = Tuple[float, float, float, float]      # x0, y0, x1, y1 in pixels


class TextBackend(Protocol):
    """Anything that turns an image into candidate text boxes with confidences."""

    def detect_text(self, image) -> Sequence[Tuple[BBox, float]]:
        ...


class PaddleTextBackend:
    """Real scene-text detection via PaddleOCR, detection-only (rec=False).

    Detection-only is deliberate: recognition is the expensive half and the VLM re-reads the
    crop anyway. Imported lazily so the rest of the pipeline — and the tests — never need
    PaddleOCR installed.
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False, **kw):
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(det=True, rec=False, cls=False, lang=lang,
                              use_gpu=use_gpu, show_log=False, **kw)

    def detect_text(self, image) -> Sequence[Tuple[BBox, float]]:
        out = self._ocr.ocr(np.asarray(image), det=True, rec=False, cls=False)
        if not out or out[0] is None:
            return []
        boxes = []
        for poly in out[0]:
            pts = np.asarray(poly, float).reshape(-1, 2)
            boxes.append(((float(pts[:, 0].min()), float(pts[:, 1].min()),
                           float(pts[:, 0].max()), float(pts[:, 1].max())), 1.0))
        return boxes


class StubTextBackend:
    """Fixed boxes, for tests and for exercising the gate without a model."""

    def __init__(self, boxes: Sequence[Tuple[BBox, float]] = ()):
        self.boxes = list(boxes)

    def detect_text(self, image) -> Sequence[Tuple[BBox, float]]:
        return list(self.boxes)


def _to_gray(image) -> np.ndarray:
    a = np.asarray(image)
    return a.mean(axis=2) if a.ndim == 3 else a


def merge_boxes(boxes: Sequence[BBox], gap_px: float = 24.0) -> List[BBox]:
    """Group text lines that belong to one physical sign.

    A directory sign is many text lines on one board; treating each line as its own sign would
    call the VLM once per line and destroy the whole point. Lines that overlap horizontally and
    sit within `gap_px` vertically are merged into one region. This is deliberately crude — the
    VLM sorts out which line belongs to which arrow, because that grouping IS the sign's meaning
    and no box-merging heuristic can recover it.
    """
    if not boxes:
        return []
    remaining = [tuple(map(float, b)) for b in boxes]
    out: List[BBox] = []
    while remaining:
        cur = list(remaining.pop(0))
        changed = True
        while changed:
            changed = False
            for other in list(remaining):
                x_overlap = min(cur[2], other[2]) - max(cur[0], other[0])
                v_gap = max(cur[1] - other[3], other[1] - cur[3])   # negative if overlapping
                if x_overlap > -gap_px and v_gap < gap_px:
                    cur = [min(cur[0], other[0]), min(cur[1], other[1]),
                           max(cur[2], other[2]), max(cur[3], other[3])]
                    remaining.remove(other)
                    changed = True
        out.append((cur[0], cur[1], cur[2], cur[3]))
    return out


class TextSignDetector(Detector):
    """Cheap always-on detector: text boxes in, readability-gated Detections out."""

    def __init__(self, backend: Optional[TextBackend] = None, K: Optional[np.ndarray] = None,
                 min_height_px: float = 22.0, min_blur_var: float = 60.0,
                 min_det_conf: float = 0.5, merge_gap_px: float = 24.0):
        self.backend = backend or StubTextBackend()
        self.K = None if K is None else np.asarray(K, float)
        self.min_height_px = min_height_px
        self.min_blur_var = min_blur_var
        self.min_det_conf = min_det_conf
        self.merge_gap_px = merge_gap_px
        self.last: List[Tuple[Detection, Readability]] = []   # per-frame evidence, for logging
        self._depth: Optional[np.ndarray] = None
        self._n = 0

    def set_pose(self, pose) -> None:
        """Forwarded to the backend if it cares. The oracle backend needs the robot pose to
        project signs; a real image-only detector (DBNet) ignores this entirely."""
        if hasattr(self.backend, "set_pose"):
            self.backend.set_pose(pose)

    def set_depth(self, depth) -> None:
        """Optional. Without it, distance is unknown (inf) — bearing still works."""
        self._depth = None if depth is None else np.asarray(depth, float)

    # ---- geometry ----
    def _bearing(self, bbox: BBox, width: int) -> float:
        """Left of image centre is a positive (left) bearing, matching the robot's convention."""
        cx = (bbox[0] + bbox[2]) / 2.0
        if self.K is not None:
            fx, ppx = float(self.K[0, 0]), float(self.K[0, 2])
        else:
            fx, ppx = float(width), width / 2.0        # ~53 deg horizontal fov fallback
        return float(np.arctan2(ppx - cx, fx))

    def _distance(self, bbox: BBox) -> float:
        if self._depth is None or not self._depth.size:
            return float("inf")
        h, w = self._depth.shape[:2]
        x0, y0, x1, y1 = (int(max(0, bbox[0])), int(max(0, bbox[1])),
                          int(min(w, bbox[2])), int(min(h, bbox[3])))
        if x1 <= x0 or y1 <= y0:
            return float("inf")
        patch = self._depth[y0:y1, x0:x1]
        finite = patch[np.isfinite(patch) & (patch > 0)]
        # median, not mean: a few pixels bleeding onto the wall behind the sign should not
        # drag the estimate, and the debounce depends on this landing in the right 2m bin.
        return float(np.median(finite)) if finite.size else float("inf")

    # ---- Detector API ----
    def detect(self, image) -> List[Detection]:
        gray = _to_gray(image)
        h_img, w_img = gray.shape[:2]
        raw = list(self.backend.detect_text(image))
        if not raw:
            self.last = []
            return []

        conf_of = {}
        for bbox, conf in raw:
            conf_of[tuple(map(float, bbox))] = float(conf)
        merged = merge_boxes([b for b, _ in raw], self.merge_gap_px)

        dets: List[Detection] = []
        self.last = []
        for bbox in merged:
            # a merged region inherits the weakest confidence it contains — the sign is only as
            # trustworthy as its shakiest line
            parts = [c for b, c in conf_of.items()
                     if b[0] >= bbox[0] - 1 and b[2] <= bbox[2] + 1
                     and b[1] >= bbox[1] - 1 and b[3] <= bbox[3] + 1]
            conf = min(parts) if parts else 1.0

            x0, y0, x1, y1 = (int(max(0, bbox[0])), int(max(0, bbox[1])),
                              int(min(w_img, bbox[2])), int(min(h_img, bbox[3])))
            crop = gray[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else None

            r = assess(crop, bbox, conf, self.min_height_px, self.min_blur_var, self.min_det_conf)
            det = Detection(
                id=f"text_{self._n}",
                label="sign",                       # NOT read yet — that is the VLM's job
                conf=conf,
                bbox=bbox,
                est_distance_m=self._distance(bbox),
                est_bearing_rad=self._bearing(bbox, w_img),
                hazard=False,
            )
            self._n += 1
            self.last.append((det, r))
            if r.legible:                           # THE GATE: only legible text is worth reasoning about
                dets.append(det)
        return dets

    def crop(self, image, det: Detection, margin: float = 0.15):
        """The pixels handed to the VLM. Cropping is also the biggest latency win available —
        fewer visual tokens to prefill — so the gate pays for itself twice."""
        a = np.asarray(image)
        h_img, w_img = a.shape[:2]
        x0, y0, x1, y1 = det.bbox
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        return a[int(max(0, y0 - my)):int(min(h_img, y1 + my)),
                 int(max(0, x0 - mx)):int(min(w_img, x1 + mx))]