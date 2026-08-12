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
    buffer_k: int = 3                 # best-K crops sent to the VLM
    min_frame_gap: int = 5            # diversity: buffered crops >= this many frames apart
    score_floor: float = 0.05         # ignore candidates below this evidence score
    min_ocr_conf: float = 0.15        # a crop needs at least this OCR conf to buffer

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
