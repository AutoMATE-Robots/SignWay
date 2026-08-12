"""The adaptive-reasoning gate: NO_SIGN -> ARMED -> DECIDED.

Pure logic, no I/O, no model imports -- everything heavy (scorer, deadline
estimator, VLM, memory) is injected. This is the module the paper describes;
keeping it dependency-free is what makes it unit-testable and auditable.

Fire condition while ARMED (evidence path, L3):
    sufficiency: best buffered evidence score >= tau_sufficiency
    OR deadline: time_to_junction <= L_p90 + margin
where time_to_junction = d_q10 / v  (PESSIMISTIC quantile: if the junction is
believed 10-15 m away, plan against 10).

Memory preemption (L1/L2): a memory decision while ARMED resolves the gate with
zero VLM calls and cancels any pending fire. Memory can only preempt the
reasoning path, never delay it.

Speed-for-evidence: when evidence is insufficient and the fire-deadline is
< slow_trigger_s away, emit slow_factor so the controller scales max_speed --
d/v grows, the deadline recedes, evidence keeps accumulating. The graceful,
continuous alternative to stopping to think.
"""
from __future__ import annotations

import dataclasses
from collections import deque
from typing import List, Optional

import numpy as np

from adaptive_reasoning.config import GateConfig
from adaptive_reasoning.evidence.scorer import BestKBuffer, EvidenceItem

NO_SIGN, ARMED, DECIDED = "NO_SIGN", "ARMED", "DECIDED"


@dataclasses.dataclass
class GateOutput:
    state: str
    fire: bool = False                 # caller should invoke the VLM now (L3)
    fire_reason: Optional[str] = None  # "sufficiency" | "deadline"
    decision: Optional[str] = None     # set when resolved via memory (L1/L2)
    decision_source: Optional[str] = None  # "memory" | "vlm"
    slow_factor: float = 1.0           # multiply commanded max_speed by this
    time_to_junction_s: Optional[float] = None
    fire_deadline_s: Optional[float] = None   # time left before deadline forces
    best_score: float = 0.0
    buffer_size: int = 0
    d_q10: Optional[float] = None


class LatencyTracker:
    """Running p90 of measured VLM latencies; seeds from the config prior."""

    def __init__(self, init_s: float, window: int):
        self.samples = deque(maxlen=window)
        self.init = init_s

    def update(self, latency_s: float) -> None:
        self.samples.append(float(latency_s))

    @property
    def p90(self) -> float:
        if not self.samples:
            return self.init
        return float(np.percentile(list(self.samples), 90))


class Gate:
    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        self.state = NO_SIGN
        self.buffer = BestKBuffer(cfg.buffer_k, cfg.min_frame_gap)
        self.latency = LatencyTracker(cfg.latency_p90_init_s, cfg.latency_window)
        self._fired = False

    # ---- lifecycle -----------------------------------------------------------
    def rearm(self) -> None:
        """Call after turn completion (the node's existing turn-done event)."""
        self.state = NO_SIGN
        self.buffer.clear()
        self._fired = False

    def report_vlm_latency(self, latency_s: float) -> None:
        self.latency.update(latency_s)

    def resolve(self, decision: str, source: str) -> None:
        """External resolution (VLM response arrived, or manual override)."""
        self.state = DECIDED
        self._fired = False
        self._decision, self._source = decision, source

    # ---- the per-frame step --------------------------------------------------
    def step(
        self,
        frame_idx: int,
        items: List[EvidenceItem],
        d_q10: Optional[float],
        p_junction: float,
        speed_mps: float,
        memory_decision: Optional[str] = None,
    ) -> GateOutput:
        cfg = self.cfg

        if self.state == DECIDED:
            return GateOutput(state=DECIDED)

        # arm on first credible sign evidence
        credible = [i for i in items
                    if i.score >= cfg.score_floor and i.ocr_conf >= cfg.min_ocr_conf]
        if self.state == NO_SIGN:
            if not credible:
                return GateOutput(state=NO_SIGN)
            self.state = ARMED

        for i in credible:
            self.buffer.add(i)

        # memory preemption: cheapest path wins, cancels any pending fire
        if memory_decision is not None:
            self.resolve(memory_decision, "memory")
            return GateOutput(state=DECIDED, decision=memory_decision,
                              decision_source="memory",
                              best_score=self.buffer.best_score,
                              buffer_size=len(self.buffer.items))

        # deadline math (pessimistic d_q10)
        ttj = None
        fire_deadline = None
        if (d_q10 is not None and p_junction >= 0.5
                and speed_mps > cfg.min_speed_for_deadline):
            ttj = d_q10 / speed_mps
            fire_deadline = ttj - (self.latency.p90 + cfg.margin_s)

        sufficient = self.buffer.best_score >= cfg.tau_sufficiency
        deadline_forces = fire_deadline is not None and fire_deadline <= 0.0

        out = GateOutput(state=ARMED, best_score=self.buffer.best_score,
                         buffer_size=len(self.buffer.items),
                         time_to_junction_s=ttj, fire_deadline_s=fire_deadline,
                         d_q10=d_q10)

        if not self._fired and self.buffer.items and (sufficient or deadline_forces):
            self._fired = True
            out.fire = True
            out.fire_reason = "sufficiency" if sufficient else "deadline"
            return out

        # speed-for-evidence: insufficient AND deadline close -> buy time
        if (cfg.speed_for_evidence and not sufficient and not self._fired
                and fire_deadline is not None
                and 0.0 < fire_deadline < cfg.slow_trigger_s):
            out.slow_factor = cfg.slow_factor
        return out
