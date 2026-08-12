"""One place that knows which open-vocabulary detectors exist, so tools take a --backend name
instead of importing a specific model.

Both backends satisfy the same protocol (detect_objects -> [DinoBox], detect_text -> [(bbox,
conf)]) and consume the same category->phrases vocabulary, which is what makes a like-for-like
comparison meaningful: identical prompts, identical routing, identical downstream code, only
the detector swapped.
"""
from __future__ import annotations

from typing import Dict, List, Optional

BACKENDS = ("dino", "omdet")

DEFAULT_MODELS = {
    "dino": "IDEA-Research/grounding-dino-base",
    "omdet": "omlab/omdet-turbo-swin-tiny-hf",
}


def build_backend(name: str,
                  model_id: Optional[str] = None,
                  prompts: Optional[Dict[str, List[str]]] = None,
                  box_threshold: float = 0.25,
                  text_threshold: float = 0.25,
                  nms_threshold: float = 0.3,
                  device: Optional[str] = None):
    """Construct a detector backend by name. Extra thresholds that a backend does not use are
    ignored rather than rejected, so callers can pass one flat set of options."""
    name = name.lower()
    if name == "dino":
        from c3_reasoning.detector_dino import GroundingDinoBackend
        return GroundingDinoBackend(model_id=model_id or DEFAULT_MODELS["dino"],
                                    device=device,
                                    box_threshold=box_threshold,
                                    text_threshold=text_threshold,
                                    prompts=prompts)
    if name == "omdet":
        from c3_reasoning.detector_omdet import OmDetTurboBackend
        return OmDetTurboBackend(model_id=model_id or DEFAULT_MODELS["omdet"],
                                 device=device,
                                 box_threshold=box_threshold,
                                 nms_threshold=nms_threshold,
                                 prompts=prompts)
    raise ValueError(f"unknown backend '{name}' — choose from {', '.join(BACKENDS)}")