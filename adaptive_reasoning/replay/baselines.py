"""
baselines.py — the competitor gates for Table I / Fig tradeoff.

Every baseline SHARES the EvidenceGate FSM (arm/pending/decided, memory,
single-pending-call) and overrides only the trigger — so the comparison in the
paper is between triggers, and nothing else.  Lineages (see
AR_BASELINES_AND_CITATIONS.md):

  AlwaysInvoke        fixed schedule: call whenever any novel text is visible.
  Periodic(k)         fixed schedule: at most one call per k frames, when text
                      is visible (deployed fast/slow systems; AdaNav's
                      "reasoning at fixed steps" strawman-that-isn't).
  ReactiveNecessity   perception-change: fire on the FIRST frame with any
                      readable novel text — IROS-style necessity-only; the
                      τ→0 limit of Eq. (gate).
  SufficiencyOnly(τ)  ablation: ℓ ≥ τ, relevance ignored (fires on notices).
  RelevanceOnly(τ)    ablation: R ≥ τ, legibility ignored (fires on blur).
  UncertaintyGate(τ)  internal-state lineage (AdaNav/KnowNo): fires when a
                      per-frame policy-uncertainty series crosses its
                      calibrated quantile.  The series is supplied externally
                      (VLA TTA variance, computed offline in its own replay).

Ours is EvidenceGate itself.  fast_path is disabled for every baseline AND for
ours inside the sweep unless explicitly ablated, so calls are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from ..gate import EvidenceGate, GateConfig, GateState, PlateEvidence, State, TickResult
from ..memory import Memory


def _cfg(tau: float) -> GateConfig:
    return GateConfig(tau=tau, fast_path=False)


@dataclass(frozen=True)
class AlwaysInvoke(EvidenceGate):
    """Call on every frame with any novel evidence (τ plays no role)."""

    def __init__(self):
        super().__init__(_cfg(tau=0.0))

    def _candidates(self, plates, memory):
        return [p for p in plates if memory.is_novel(p.plate_id)]

    def _score(self, p):
        return 1.0


@dataclass(frozen=True)
class ReactiveNecessity(EvidenceGate):
    """IROS-style: first frame the cheap layer sees readable novel text, call.
    Identical to Eq. (gate) with τ→0 over presence rather than quality."""

    def __init__(self):
        super().__init__(_cfg(tau=0.0))

    def _candidates(self, plates, memory):
        return [p for p in plates if memory.is_novel(p.plate_id)]

    def _score(self, p):
        return 1.0 if (p.legibility > 0.0 or p.relevance > 0.0) else 0.0


@dataclass(frozen=True)
class SufficiencyOnly(EvidenceGate):
    def _candidates(self, plates, memory):
        return [p for p in plates if memory.is_novel(p.plate_id)]

    def _score(self, p):
        return p.legibility


@dataclass(frozen=True)
class RelevanceOnly(EvidenceGate):
    def _candidates(self, plates, memory):
        return [p for p in plates if memory.is_novel(p.plate_id)]

    def _score(self, p):
        return p.relevance


class Periodic(EvidenceGate):
    """At most one call per `period` frames while any text is visible."""

    def __init__(self, period: int):
        super().__init__(_cfg(tau=0.0))
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "_last_fire", {"t": -10**9})

    def _candidates(self, plates, memory):
        return [p for p in plates if memory.is_novel(p.plate_id)]

    def _score(self, p):
        return 1.0

    def tick(self, prev: GateState, plates: Sequence[PlateEvidence],
             memory: Memory, t: int) -> TickResult:
        r = super().tick(prev, plates, memory, t)
        if r.fire and t - self._last_fire["t"] < self.period:
            # too soon: stay ARMED instead of firing
            return TickResult(replace(r.state, state=State.ARMED, pending_plate=None,
                                      fired_at=prev.fired_at),
                              False, r.necessity, r.q, r.best, r.memory_updates,
                              reason=f"periodic hold ({t - self._last_fire['t']} < {self.period})")
        if r.fire:
            self._last_fire["t"] = t
        return r


class UncertaintyGate(EvidenceGate):
    """Internal-uncertainty lineage (AdaNav action entropy / KnowNo set size):
    fires when `series[t]` ≥ τ_u, regardless of what's in view.  `series` is a
    per-frame uncertainty trace computed offline (e.g. VLA waypoint variance
    under test-time augmentation); τ_u is a quantile fit on training approaches."""

    def __init__(self, series: np.ndarray, tau_u: float):
        super().__init__(_cfg(tau=0.5))
        object.__setattr__(self, "series", np.asarray(series, float))
        object.__setattr__(self, "tau_u", float(tau_u))

    def tick(self, prev: GateState, plates: Sequence[PlateEvidence],
             memory: Memory, t: int) -> TickResult:
        novel = self._candidates(plates, memory)
        best = max(novel, key=lambda p: p.q) if novel else None
        u = self.series[t] if t < len(self.series) else 0.0
        if prev.necessity == 0:
            return TickResult(prev, False, 0, u, best, reason="necessity 0")
        if u < self.tau_u:
            st = State.ARMED if best is not None else State.NO_SIGN
            return TickResult(replace(prev, state=st), False, 1, u, best,
                              reason=f"u={u:.3f} < tau_u={self.tau_u:.3f}")
        if best is None:
            # uncertain but nothing to ask about: burn the call on the scene
            best = PlateEvidence("scene", 0.0, 0.0)
        new = replace(prev, state=State.PENDING, pending_plate=best.plate_id, fired_at=t)
        return TickResult(new, True, 1, u, best, reason=f"u={u:.3f} ≥ tau_u")

    def _candidates(self, plates, memory):
        return [p for p in plates if memory.is_novel(p.plate_id)]


def make_gate(name: str, tau: float, **kw) -> EvidenceGate:
    """Factory used by sweep.py / eval_gates.py."""
    return {
        "ours": lambda: EvidenceGate(GateConfig(tau=tau, fast_path=kw.get("fast_path", True))),
        "always": lambda: AlwaysInvoke(),
        "continuous05": lambda: Periodic(20),          # 0.5 Hz at the 10 Hz grid
        "periodic5s": lambda: Periodic(50),
        "periodic": lambda: Periodic(kw.get("period", 20)),
        "reactive": lambda: ReactiveNecessity(),
        "sufficiency": lambda: SufficiencyOnly(_cfg(tau)),
        "relevance": lambda: RelevanceOnly(_cfg(tau)),
        "iros": lambda: IROSStyle(kw["scene_grid"], kw.get("theta_kfc", 0.1)),
        "signscene": lambda: SignSceneStyle(kw.get("theta_parse", 0.5)),
    }[name]()


# ----------------------------------------------------------------------------
# Paper-derived invocation policies (see AR_COMPARISON_PROTOCOL.md §1)
# ----------------------------------------------------------------------------

def kfc_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Patch-level scene change: mean over the 4x4 grid cells of (1 - cosine)."""
    a = a.astype(np.float32); b = b.astype(np.float32)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return float(np.mean(1.0 - num / den))


class IROSStyle(EvidenceGate):
    """IROS [2601.21506] System-One/Two trigger, re-implemented:
    Key-Frame-Compare — the current frame's patch-grid descriptor is compared
    with the LAST TRIGGERING frame; if the change exceeds theta_kfc AND the
    cheap layer cannot resolve a unique action (no plate resolves via the fast
    path), escalate to the VLM.  If a plate DOES resolve, System One acts.
    theta_kfc is set on held-in approaches (protocol §1); scene_grid comes from
    the dump (SigLIP patch grid or the flagged proxy)."""

    def __init__(self, scene_grid: np.ndarray, theta_kfc: float):
        super().__init__(GateConfig(tau=0.0, fast_path=True))
        object.__setattr__(self, "grid", scene_grid)
        object.__setattr__(self, "theta", float(theta_kfc))
        object.__setattr__(self, "_last", {"t": None})

    def _candidates(self, plates, memory):
        return [p for p in plates if memory.is_novel(p.plate_id)]

    def _score(self, p):
        return 1.0

    def tick(self, prev: GateState, plates: Sequence[PlateEvidence],
             memory: Memory, t: int) -> TickResult:
        novel = self._candidates(plates, memory)
        if prev.necessity == 0:
            return TickResult(prev, False, 0, 0.0, None, reason="necessity 0")
        if not novel:
            return TickResult(replace(prev, state=State.NO_SIGN), False, 1, 0.0, None, reason="no text")
        # System One: a uniquely resolvable plate → act without the VLM
        res = [p for p in novel if p.resolvable]
        if len(res) == 1:
            p = res[0]; d = p.resolution.vla_prompt
            new = replace(prev, state=State.DECIDED, decision=d, decision_source="fast",
                          decision_plate=p.plate_id, decided_at=t, pending_plate=None)
            return TickResult(new, False, 1, 1.0, p, (MemoryUpdate(p.plate_id, "fast", d, t),),
                              reason="system one resolved")
        # KFC: scene changed enough since the last trigger?
        last = self._last["t"]
        ti = min(t, len(self.grid) - 1)                       # past the approach end: hold the last view
        li = min(last, len(self.grid) - 1) if last is not None else None
        dist = 1.0 if li is None else kfc_distance(self.grid[ti], self.grid[li])
        if dist < self.theta:
            return TickResult(replace(prev, state=State.ARMED), False, 1, dist, novel[0],
                              reason=f"kfc {dist:.3f} < {self.theta:.3f}")
        self._last["t"] = t
        best = max(novel, key=lambda p: p.relevance)
        new = replace(prev, state=State.PENDING, pending_plate=best.plate_id, fired_at=t)
        return TickResult(new, True, 1, dist, best, reason=f"kfc {dist:.3f} ≥ theta & ambiguous → VLM")


def sign_like(text: str) -> bool:
    """Proxy for SignScene's sign DETECTOR (GroundingDINO prompted with 'signs'):
    a directory-like plate — several text lines, or a room code — rather than
    every text fragment docTR finds (posters, door plates, labels)."""
    if not text:
        return False
    lines = [t for t in text.split("|") if t.strip()]
    has_code = any(ch.isdigit() for ch in text) and any(ch.isalpha() for ch in text)
    multiword = len(text.split()) >= 2          # a place-name plaque ("Lind Hall")
    return len(lines) >= 2 or has_code or multiword


class SignSceneStyle(EvidenceGate):
    """SignScene [2602.12686] call structure, re-implemented: one PARSE call per
    newly legible sign (viewpoint gate approximated by ℓ ≥ theta_parse — their
    τ_dist/τ_angle are unreported), then one GROUNDING call once a parsed sign
    matches the goal.  Parse calls are tagged '#parse' so the shared memory does
    not consume the real plate before grounding.  Their servo-to-align and
    top-view-map steps are not reproducible in replay (stated in the caption)."""

    def __init__(self, theta_parse: float = 0.5):
        super().__init__(GateConfig(tau=0.0, fast_path=False))
        object.__setattr__(self, "theta_parse", float(theta_parse))
        object.__setattr__(self, "_parsed", set())

    def tick(self, prev: GateState, plates: Sequence[PlateEvidence],
             memory: Memory, t: int) -> TickResult:
        if prev.necessity == 0:
            return TickResult(prev, False, 0, 0.0, None, reason="necessity 0")
        novel = [p for p in plates if memory.is_novel(p.plate_id)]
        if not novel:
            return TickResult(replace(prev, state=State.NO_SIGN), False, 1, 0.0, None, reason="no text")
        # grounding: a parsed, goal-relevant plate → the real call
        ground = [p for p in novel if p.plate_id in self._parsed and p.relevance >= 0.5]
        if ground:
            best = max(ground, key=lambda p: p.legibility)
            new = replace(prev, state=State.PENDING, pending_plate=best.plate_id, fired_at=t)
            return TickResult(new, True, 1, best.legibility, best, reason="grounding call")
        # parse: first legible view of a new sign
        fresh = [p for p in novel if p.plate_id not in self._parsed and p.legibility >= self.theta_parse
                 and sign_like(p.text)]
        if fresh:
            best = max(fresh, key=lambda p: p.legibility)
            self._parsed.add(best.plate_id)
            parse_plate = PlateEvidence(best.plate_id + "#parse", best.relevance, best.legibility)
            new = replace(prev, state=State.PENDING, pending_plate=parse_plate.plate_id, fired_at=t)
            return TickResult(new, True, 1, best.legibility, parse_plate, reason="parse call")
        return TickResult(replace(prev, state=State.ARMED), False, 1, 0.0, novel[0], reason="waiting")


from ..gate import MemoryUpdate  # noqa: E402  (used above)
