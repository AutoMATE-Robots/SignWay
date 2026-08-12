"""OmDet-Turbo detection backend — the real-time alternative to Grounding DINO.

Same protocol as GroundingDinoBackend (detect_objects / detect_text), so it drops into the
same tools and the same downstream routing. The two differ in one way that matters beyond
speed:

  Grounding DINO takes ONE string containing every phrase ("sign. chair. traffic cone.") and
  encodes it as a whole, so phrases share attention — a long vocabulary can detect each
  individual thing slightly worse than a short one.

  OmDet-Turbo takes a LIST of class names encoded separately, decoupled from the task
  embedding. That decoupling is exactly why it is fast (no per-class decoding at inference),
  and it means adding vocabulary behaves differently than it does for DINO.

That difference is worth measuring, not assuming: if signs go undetected under a long DINO
query but appear under OmDet with the same vocabulary, the cause was query dilution rather
than the model being unable to see signs.

OmDet-Turbo also applies NMS in post-processing (nms_threshold), which Grounding DINO does
not — so duplicate/overlapping boxes are suppressed for you.

Reference: Zhao et al., "Real-time Transformer-based Open-Vocabulary Detection with Efficient
Fusion Head"; the base model is reported at up to 100.2 FPS with 53.4 AP zero-shot on COCO,
though that figure is with optimised serving rather than plain PyTorch.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from c3_reasoning.detector_dino import DEFAULT_PROMPTS, DinoBox

BBox = Tuple[float, float, float, float]


class OmDetTurboBackend:
    """OmDet-Turbo behind the same backend protocol as GroundingDinoBackend.

    detect_text()    -> sign-category boxes only, as (bbox, conf)
    detect_objects() -> every box with phrase + category
    """

    def __init__(self,
                 model_id: str = "omlab/omdet-turbo-swin-tiny-hf",
                 device: Optional[str] = None,
                 box_threshold: float = 0.25,
                 nms_threshold: float = 0.3,
                 prompts: Optional[Dict[str, List[str]]] = None,
                 categories: Sequence[str] = ("sign", "hazard", "obstacle")):
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.nms_threshold = nms_threshold
        self.prompts = {k: list(v) for k, v in (prompts or DEFAULT_PROMPTS).items()
                        if k in categories}
        self._device = device
        self._model = None
        self._processor = None
        self._post_sig = None         # resolved on first call; the API was renamed once
        self._phrase_cat = {p: c for c, ps in self.prompts.items() for p in ps}
        # a flat LIST of class names — not a joined string, unlike Grounding DINO
        self._classes = [p for ps in self.prompts.values() for p in ps]
        self.last_objects: List[DinoBox] = []

    # ---- lazy model ----
    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, OmDetTurboForObjectDetection
        dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = OmDetTurboForObjectDetection.from_pretrained(
            self.model_id).to(dev).eval()
        self._device = dev

    def _post_kwargs(self, target_size):
        """transformers renamed this post-processor's arguments once:
            old:  classes=[...],     score_threshold=...
            new:  text_labels=[...], threshold=...
        Resolve against the installed signature rather than pinning a version."""
        if self._post_sig is None:
            import inspect
            self._post_sig = set(inspect.signature(
                self._processor.post_process_grounded_object_detection).parameters)
        kw = {"target_sizes": [target_size], "nms_threshold": self.nms_threshold}
        kw["text_labels" if "text_labels" in self._post_sig else "classes"] = self._classes
        kw["threshold" if "threshold" in self._post_sig else "score_threshold"] = \
            self.box_threshold
        return kw

    def _category_of(self, phrase: str) -> str:
        """OmDet returns one of the class names we supplied, so exact match nearly always
        hits. The containment fallback and the 'sign' default mirror the DINO backend:
        failing toward the gated path is the safe failure."""
        p = str(phrase).strip().lower()
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
        pil = PILImage.fromarray(arr[..., :3].astype(np.uint8))

        inputs = self._processor(pil, text=self._classes,
                                 return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        res = self._processor.post_process_grounded_object_detection(
            outputs, **self._post_kwargs((pil.height, pil.width)))[0]

        labels = res.get("text_labels", res.get("classes", res.get("labels")))
        out: List[DinoBox] = []
        for box, score, phrase in zip(res["boxes"], res["scores"], labels):
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            out.append(DinoBox(bbox=(x0, y0, x1, y1), score=float(score),
                               phrase=str(phrase), category=self._category_of(phrase)))
        self.last_objects = out
        return out

    # ---- TextBackend protocol: sign boxes only ----
    def detect_text(self, image) -> Sequence[Tuple[BBox, float]]:
        return [(b.bbox, b.score) for b in self.detect_objects(image)
                if b.category == "sign"]
