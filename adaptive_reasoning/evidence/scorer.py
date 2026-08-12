"""Evidence scoring: sign candidates -> scored crops -> best-K buffer.

score = ocr_conf * sqrt(area_frac_norm) * sharpness_norm

Every component is kept on the EvidenceItem so the combination stays ablatable
and the v1 sufficiency predictor can train on (ocr_conf, area, sharpness) ->
P(VLM correct) from the offline curve.

OCR backends are pluggable and lazily imported:
  * "doctr"  : docTR det+reco (what IROS used) -- preferred.
  * "paddle" : PaddleOCR fallback.
  * "fake"   : deterministic synthetic backend for tests / dry runs.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np


@dataclasses.dataclass
class EvidenceItem:
    frame_idx: int
    bbox: tuple                 # (x0, y0, x1, y1) pixels
    ocr_conf: float
    text: str
    area_frac: float
    sharpness: float            # normalized 0..1
    score: float
    crop: Optional[np.ndarray] = None   # HxWx3 uint8 (kept only for buffered items)

    def light(self) -> "EvidenceItem":
        """Copy without the crop (for logs/ring buffer)."""
        return dataclasses.replace(self, crop=None)


# --------------------------------------------------------------------------- #
# components
# --------------------------------------------------------------------------- #
def sharpness_norm(gray: np.ndarray, lo: float = 20.0, hi: float = 800.0) -> float:
    """Variance of Laplacian, squashed to 0..1 (lo/hi from indoor-camera practice)."""
    import cv2

    v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def compute_score(ocr_conf: float, area_frac: float, sharp: float,
                  area_norm: float = 0.02) -> float:
    """area_norm: bbox area fraction at which the area term saturates to 1
    (~a readable sign fills ~2% of the frame)."""
    a = min(1.0, area_frac / area_norm) ** 0.5
    return float(ocr_conf * a * sharp)


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class FakeBackend:
    """Synthetic backend: one detection whose confidence/area grow with frame_idx.

    Lets the gate/buffer/replay be tested end-to-end with zero OCR dependencies.
    """

    def __init__(self, start_frame: int = 0, ramp: int = 100):
        self.start, self.ramp = start_frame, ramp

    def detect(self, frame_rgb: np.ndarray, frame_idx: int) -> List[dict]:
        if frame_idx < self.start:
            return []
        p = min(1.0, (frame_idx - self.start) / self.ramp)
        h, w = frame_rgb.shape[:2]
        s = int(20 + 80 * p)
        x0, y0 = w // 2 - s // 2, h // 3 - s // 4
        return [dict(bbox=(x0, y0, x0 + s, y0 + s // 2),
                     conf=0.2 + 0.75 * p, text="ROOM 301-320 ->")]


class DoctrBackend:
    def __init__(self):
        from doctr.models import ocr_predictor  # lazy

        self.model = ocr_predictor(pretrained=True)

    def detect(self, frame_rgb: np.ndarray, frame_idx: int) -> List[dict]:
        res = self.model([frame_rgb])
        h, w = frame_rgb.shape[:2]
        out = []
        for page in res.pages:
            for block in page.blocks:
                for line in block.lines:
                    (x0, y0), (x1, y1) = line.geometry
                    words = [wd.value for wd in line.words]
                    confs = [wd.confidence for wd in line.words] or [0.0]
                    out.append(dict(
                        bbox=(int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)),
                        conf=float(np.mean(confs)), text=" ".join(words)))
        return out


class PaddleBackend:
    def __init__(self):
        from paddleocr import PaddleOCR  # lazy

        self.model = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)

    def detect(self, frame_rgb: np.ndarray, frame_idx: int) -> List[dict]:
        res = self.model.ocr(frame_rgb, cls=False)
        out = []
        for line in (res[0] or []):
            poly, (text, conf) = line
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            out.append(dict(bbox=(int(min(xs)), int(min(ys)),
                                  int(max(xs)), int(max(ys))),
                            conf=float(conf), text=text))
        return out


def make_backend(name: str, **kw):
    return {"doctr": DoctrBackend, "paddle": PaddleBackend,
            "fake": FakeBackend}[name](**kw) if name == "fake" else \
        {"doctr": DoctrBackend, "paddle": PaddleBackend}[name]()


# --------------------------------------------------------------------------- #
# scorer + buffer
# --------------------------------------------------------------------------- #
class EvidenceScorer:
    def __init__(self, backend, pad: int = 6):
        self.backend = backend
        self.pad = pad

    def score_frame(self, frame_rgb: np.ndarray, frame_idx: int) -> List[EvidenceItem]:
        import cv2

        h, w = frame_rgb.shape[:2]
        items = []
        for det in self.backend.detect(frame_rgb, frame_idx):
            x0, y0, x1, y1 = det["bbox"]
            x0, y0 = max(0, x0 - self.pad), max(0, y0 - self.pad)
            x1, y1 = min(w, x1 + self.pad), min(h, y1 + self.pad)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = frame_rgb[y0:y1, x0:x1]
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            sharp = sharpness_norm(gray)
            area = (x1 - x0) * (y1 - y0) / float(h * w)
            items.append(EvidenceItem(
                frame_idx=frame_idx, bbox=(x0, y0, x1, y1),
                ocr_conf=float(det["conf"]), text=det.get("text", ""),
                area_frac=area, sharpness=sharp,
                score=compute_score(det["conf"], area, sharp), crop=crop))
        return items


class BestKBuffer:
    """Top-K evidence by score with a frame-diversity constraint: buffered items
    must be >= min_frame_gap frames apart, so the VLM sees distinct
    distances/angles instead of three near-duplicates."""

    def __init__(self, k: int = 3, min_frame_gap: int = 5):
        self.k, self.gap = k, min_frame_gap
        self.items: List[EvidenceItem] = []

    def add(self, item: EvidenceItem) -> None:
        near = [i for i in self.items if abs(i.frame_idx - item.frame_idx) < self.gap]
        if near:
            worst = min(near, key=lambda i: i.score)
            if item.score > worst.score:
                self.items.remove(worst)
                self.items.append(item)
        else:
            self.items.append(item)
        self.items.sort(key=lambda i: -i.score)
        del self.items[self.k:]

    @property
    def best_score(self) -> float:
        return self.items[0].score if self.items else 0.0

    def crops(self):
        return [i.crop for i in self.items if i.crop is not None]

    def clear(self):
        self.items = []
