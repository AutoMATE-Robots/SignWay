"""Open-vocabulary detection backend: Grounding DINO finds sign-like OBJECTS, not text.

WHY THIS EXISTS: the oracle backend is given the sign's location; nothing is detected. This
detects. Grounding DINO is prompted with plain phrases ("wall sign", "warning cone") and
returns boxes for whatever matches — so one model covers all three jobs the project needs:

  1. signs             -> routed through the readability gate, then the VLM reads them
  2. temporary signage -> textual ones ("staff only" plates) gate like signs; SYMBOLIC ones
                          (wet-floor cones, barrier tape) are recognisable at any distance and
                          bypass the gate entirely — a hazard flag must not wait for legibility
  3. obstacles         -> semantics only ("person", "cart"). Geometry already comes from the
                          depth occupancy grid; DINO adds WHAT the obstacle is, never WHERE.

Why an object detector and not a text detector for signs: a sign is findable long before its
text is readable. Detecting the sign OBJECT early lets the gate watch its height grow frame by
frame and fire the VLM at exactly the moment the text becomes legible — which is the paper's
whole trigger. A text detector only sees the sign once the text is nearly readable anyway.

The class implements the same TextBackend protocol as the oracle and Paddle backends
(detect_text), so TextSignDetector and the whole downstream pipeline are unchanged. The richer
per-category view (detect_objects) is for the hazard path and the offline harness.

transformers/torch are imported lazily: the rest of the pipeline and the tests never need them.
Model weights come from the HuggingFace hub — download once on a login node with internet, set
HF_HOME to scratch so the cache survives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]

# One detector pass, three routings. Format follows published Grounding DINO usage: lowercase
# single-concept category names, period-separated, trailing period. The model encodes each
# phrase independently (sub-sentence representation), so SHORT names detect well and long
# descriptive phrases ("box on the floor") bleed attention and detect worse — category names
# only, no referring expressions.
#
# SCOPE: these name what an INDOOR CORRIDOR contains — hospital and office hallways, which is
# what the robot footage actually shows. An earlier warehouse-flavoured list (forklift, no
# chair, no pallet) missed a chair and a pallet sitting in plain view, because open-vocabulary
# detection finds only what you name.
#
# TRADE-OFF worth knowing: every phrase added to a query dilutes attention across the text
# encoder, so a long list can detect each individual thing slightly worse than a short one.
# The alternative is several passes with short lists, which multiplies latency (already
# ~250ms/frame for base). Use tools/dino_probe.py to measure rather than guess.
DEFAULT_PROMPTS: Dict[str, List[str]] = {
    "sign": [
        "sign", "exit sign", "door sign", "wall sign",
    ],
    # temporary/warning signage and hazards. Routing differs from signs downstream (no
    # legibility gating for symbolic hazards) but detection is the same single pass.
    "hazard": [
        "warning sign", "wet floor sign", "traffic cone", "caution tape",
        "barrel", "barricade",
    ],
    "obstacle": [
        "person", "chair", "cart", "box", "pallet", "ladder", "trash can", "table",
    ],
}


# words that mark a phrase as a hazard rather than a plain sign or obstacle, used when the
# vocabulary arrives from the command line with no category structure
_HAZARD_WORDS = ("cone", "tape", "barrel", "barricade", "barrier", "caution",
                 "warning", "wet floor", "hazard")


def split_prompts(raw: str) -> Dict[str, List[str]]:
    """Turn a flat CLI vocabulary ('sign. chair. traffic cone.') into the category dict.

    Routing rule: a hazard word anywhere in the phrase makes it a hazard; otherwise a phrase
    containing 'sign' is a sign; everything else is an obstacle. This exists so prompt
    experiments do not require editing source — the categories only decide DOWNSTREAM routing,
    never what the detector looks for.
    """
    out: Dict[str, List[str]] = {"sign": [], "hazard": [], "obstacle": []}
    parts = [p.strip().lower() for chunk in raw.split(".") for p in chunk.split(",")]
    for p in (p for p in parts if p):
        if any(w in p for w in _HAZARD_WORDS):
            out["hazard"].append(p)
        elif "sign" in p:
            out["sign"].append(p)
        else:
            out["obstacle"].append(p)
    return {k: v for k, v in out.items() if v}


@dataclass
class DinoBox:
    """One detection with everything the three routings need."""
    bbox: BBox            # x0, y0, x1, y1 pixels
    score: float
    phrase: str           # the matched prompt text, e.g. "wall sign"
    category: str         # "sign" | "hazard" | "obstacle"


class GroundingDinoBackend:
    """Grounding DINO behind the TextBackend protocol.

    detect_text()    -> sign-category boxes only, as (bbox, conf) — what TextSignDetector eats.
    detect_objects() -> every box with phrase + category — for hazards and the offline harness.
    """

    def __init__(self,
                 model_id: str = "IDEA-Research/grounding-dino-tiny",
                 device: Optional[str] = None,
                 box_threshold: float = 0.35,
                 text_threshold: float = 0.25,
                 prompts: Optional[Dict[str, List[str]]] = None,
                 categories: Sequence[str] = ("sign", "hazard", "obstacle")):
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.prompts = {k: list(v) for k, v in (prompts or DEFAULT_PROMPTS).items()
                        if k in categories}
        self._device = device
        self._model = None            # lazy — built on first detect
        self._processor = None
        self._post_kw = None          # resolved on first call: box_threshold vs threshold
        # phrase -> category lookup for routing the model's matched text back to a category
        self._phrase_cat = {p: c for c, ps in self.prompts.items() for p in ps}
        # HF grounding-dino wants "phrase. phrase. phrase." — lowercase, period-separated
        self._text_query = ". ".join(p for ps in self.prompts.values() for p in ps) + "."
        self.last_objects: List[DinoBox] = []     # per-frame evidence, mirrors detector.last

    # ---- lazy model ----
    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id).to(dev).eval()
        self._device = dev

    def _category_of(self, phrase: str) -> str:
        """The model returns the matched span, which may be a sub-phrase ("sign") or a join of
        prompts. Exact match first, then longest prompt contained in the span, else 'sign' —
        failing toward the gated path is the safe failure (a gated box that turns out to be a
        cone costs one wasted look; an ungated box that was a sign skips the gate)."""
        p = phrase.strip().lower()
        if p in self._phrase_cat:
            return self._phrase_cat[p]
        best, best_len = "sign", 0
        for known, cat in self._phrase_cat.items():
            if known in p and len(known) > best_len:
                best, best_len = cat, len(known)
        return best

    # ---- the full three-category view ----
    def detect_objects(self, image) -> List[DinoBox]:
        self._ensure_model()
        import torch
        from PIL import Image as PILImage
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        pil = PILImage.fromarray(arr.astype(np.uint8))

        inputs = self._processor(images=pil, text=self._text_query,
                                 return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        # transformers renamed box_threshold -> threshold in this post-processor. Pick the
        # name the installed version actually accepts rather than pinning a version.
        if self._post_kw is None:
            import inspect
            params = inspect.signature(
                self._processor.post_process_grounded_object_detection).parameters
            self._post_kw = "box_threshold" if "box_threshold" in params else "threshold"
        res = self._processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            **{self._post_kw: self.box_threshold},
            text_threshold=self.text_threshold,
            target_sizes=[pil.size[::-1]])[0]

        out: List[DinoBox] = []
        for box, score, phrase in zip(res["boxes"], res["scores"], res["text_labels"]
                                      if "text_labels" in res else res["labels"]):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            out.append(DinoBox(bbox=(x0, y0, x1, y1), score=float(score),
                               phrase=str(phrase), category=self._category_of(str(phrase))))
        self.last_objects = out
        return out

    # ---- TextBackend protocol: sign boxes only ----
    def detect_text(self, image) -> Sequence[Tuple[BBox, float]]:
        return [(b.bbox, b.score) for b in self.detect_objects(image)
                if b.category == "sign"]


class StubDinoBackend:
    """Fixed DinoBoxes for tests and for exercising the routing without torch installed."""

    def __init__(self, objects: Sequence[DinoBox] = ()):
        self.objects = list(objects)
        self.last_objects: List[DinoBox] = []

    def detect_objects(self, image) -> List[DinoBox]:
        self.last_objects = list(self.objects)
        return self.last_objects

    def detect_text(self, image) -> Sequence[Tuple[BBox, float]]:
        return [(b.bbox, b.score) for b in self.objects if b.category == "sign"]