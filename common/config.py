"""Shared configuration + repo-path bootstrap for the adaptive_reasoning package.

This package lives at <repo>/adaptive_reasoning/ and reuses bag I/O from
<repo>/tools/bag_to_episode.py (read_bag, interp_odom). `bootstrap()` makes those
importable whether a script is run as `python -m adaptive_reasoning...` from the
repo root or executed directly by path.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent          # .../adaptive_reasoning
REPO_ROOT = PKG_ROOT.parent                          # .../SignWay
TOOLS_DIR = REPO_ROOT / "tools"


def bootstrap() -> None:
    """Make repo root + tools/ importable (idempotent)."""
    for p in (REPO_ROOT, TOOLS_DIR, PKG_ROOT.parent):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


@dataclass
class GateConfig:
    """All gate thresholds in one place (logged with every run for reproducibility)."""

    # evidence buffer
    buffer_k: int = 6                 # best-K crops sent to the VLM.
                                      # K=6 (was 3) after a live run sent Gemini three
                                      # crops of a department header and none of the
                                      # room-range sign beside it -> wrong decision from
                                      # correct reasoning. The evidence score ranks by
                                      # legibility, which favours big bold headers over
                                      # smaller directional signs; sending more crops
                                      # lets the VLM do the relevance selection it is
                                      # already good at (it returns not_applicable
                                      # correctly). Revisit with a goal-affinity term.
    min_frame_gap: int = 2            # diversity: buffered crops >= this many CALLS apart
                                      # (calls, not dataset frames: at 2 Hz this is ~1 s)
    min_arm_frames: int = 6           # sufficiency cannot fire until this many calls
                                      # after arming (~3 s at 2 Hz).
                                      # WHY: measured on the robot, a legible sign scores
                                      # ~0.999 on the FIRST readable frame, so a
                                      # sufficiency-only gate fires instantly with a
                                      # single crop -- no evidence accumulation at all.
                                      # The deadline may still force an earlier fire.
    score_floor: float = 0.05         # ignore candidates below this evidence score
    min_ocr_conf: float = 0.15        # a crop needs at least this OCR conf to buffer

    # scene context for the VLM: prepend a downscaled full frame to the crops so
    # arrows/layout are always visible even if plate clustering misses something
    include_scene_frame: bool = True
    scene_max_dim: int = 1280

    # sufficiency (v0 = threshold on best buffered score; v1 = learned predictor)
    tau_sufficiency: float = 0.55

    # deadline: t* = d_q10 / v - L - margin
    latency_p90_init_s: float = 3.0   # prior for VLM latency before measurements exist
    latency_window: int = 20          # running window for the latency p90
    margin_s: float = 0.75            # set >= deadline-estimator q10 violation error
    min_speed_for_deadline: float = 0.05  # below this v, time-to-junction is "infinite"

    # speed-for-evidence: scale commanded max_speed when evidence-poor near deadline
    speed_for_evidence: bool = True
    slow_factor: float = 0.5          # multiply max_speed by this while buying time
    slow_trigger_s: float = 1.5       # engage when time-to-fire-deadline < this AND insufficient

    # memory
    tau_memory: float = 0.72          # combined sign*context score to accept a hit
    sign_sim_floor: float = 0.80      # min sign-embedding cosine to even consider
    context_partial_agreement: float = 0.5  # weighted fraction of context that must match


@dataclass
class MemoryConfig:
    store_dir: Path = field(default_factory=lambda: Path.home() / "SignWay" / "memory")
    building: str = "keller"
    ring_capacity: int = 200
    ttl_permanent_days: float = 365.0
    ttl_temporary_days: float = 1.0
    # class-permanence prior: weight of a context object in the agreement score
    permanence: dict = field(default_factory=lambda: {
        "trash can": 1.0, "fire extinguisher": 1.0, "door": 0.9, "exit sign": 0.9,
        "poster": 0.6, "plant": 0.5, "chair": 0.4, "cup": 0.2, "person": 0.0,
    })
    default_permanence: float = 0.5