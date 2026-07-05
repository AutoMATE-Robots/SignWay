"""Tunable parameters (orchestration spec §12). Load from YAML or use the defaults below.
All thresholds live here in one place so tuning never means hunting through the code."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Params:
    tau_conf: float = 0.5        # min sign detection confidence to act (Phase 2)
    d_read: float = 3.5          # max distance (m) a sign is considered readable (Phase 2)
    d_arrive: float = 0.5        # subgoal-reached distance (m)
    d_look: float = 2.5          # subgoal look-ahead distance (m)
    t_stall: float = 3.0         # stall window (s)
    eps_progress: float = 0.2    # min displacement (m) over t_stall before 'stalled'
    leg_max_dist: float = 8.0    # max travel (m) before a leg auto-closes (Phase 2)
    reason_budget: int = 30      # max VLM calls per mission (Phase 2)
    occ_staleness_s: float = 0.5  # max occupancy age before 'hold'
    det_staleness_s: float = 0.3  # max detection age before 'hold'
    omni_range_m: float = 30.0   # clip relative goal to this (matches OmniVLA training)

    @staticmethod
    def load(path: str) -> "Params":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid = Params().__dict__
        return Params(**{k: v for k, v in data.items() if k in valid})
