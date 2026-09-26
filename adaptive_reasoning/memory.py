"""
memory.py — M_g: which sign plates have already been *consumed* for the current goal.

"Consumed" means one of:
  * the VLM was asked about it and answered (decision or not-applicable),
  * the fast path read a direction off it,
  * a human/operator told us to ignore it.
A consumed plate can never trigger another call for the same goal — this is
the novelty term E_t \\ M_g in the gate equation, and it replaces the old
"ignore for 5 seconds" cooldown with a goal-scoped fact.

Scope: ONE goal.  When the goal changes, start a fresh Memory (the richer
cross-goal memory — the reasoning-event graph for Section C — will wrap this,
not replace it).

Purity: the gate never mutates memory.  It returns MemoryUpdate objects; the
caller (replay loop or server) applies them with memory.apply(update).  That
keeps gate.tick a pure function and makes replays reproducible.

Keys: plate_id is whatever detect.py / buffer.py decide is the stable identity
of a plate (normalized text for now).  Merging OCR variants of the same sign
is the buffer's job, not memory's — memory does exact lookups only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .evidence.relevance import normalize


def plate_key(text: str) -> str:
    """Default plate identity: normalized text with lines joined.  Exact, no fuzz."""
    return normalize(text.replace("|", " "))


@dataclass(frozen=True)
class MemoryUpdate:
    plate_id: str
    outcome: str                  # 'fast' | 'vlm' | 'not_applicable' | 'ignored'
    decision: Optional[str] = None
    t: Optional[int] = None


@dataclass
class Memory:
    goal: str
    consumed: dict[str, MemoryUpdate] = field(default_factory=dict)

    # a consumed plate is recognised again if its digit signature overlaps a consumed
    # one by >= this (the evidence buffer's association rule); exact-id equality
    # broke on a single OCR dropout ("43-58" vs "43 58") and re-asked the same sign.
    # fuzzy=False -> exact-id consumption only.  Baselines keep exact identity:
    # their papers dedup per tracked instance, and OUR overlap rule must never
    # decide when a BASELINE's sign counts as already-seen (a false match would
    # suppress a call they needed — tuning against them).
    match_overlap: float = 0.5
    fuzzy: bool = True

    @staticmethod
    def _sig(plate_id: str) -> frozenset:
        if plate_id.startswith("SIG:"):
            return frozenset(plate_id[4:].split())
        return frozenset(t for t in plate_id.split() if len(t) >= 3)   # text key: word tokens

    def matches(self, plate_id: str):
        """The consumed plate this id refers to, or None."""
        if plate_id in self.consumed:
            return plate_id
        if not self.fuzzy:
            return None
        a = self._sig(plate_id)
        if len(a) < 2:
            return None
        best, best_j = None, 0.0
        for k in self.consumed:
            b = self._sig(k)
            if len(b) < 2 or (k.startswith("SIG:") != plate_id.startswith("SIG:")):
                continue
            j = len(a & b) / min(len(a), len(b))
            if j > best_j:
                best, best_j = k, j
        return best if best_j >= self.match_overlap else None

    def is_novel(self, plate_id: str) -> bool:
        return self.matches(plate_id) is None

    def apply(self, update: MemoryUpdate) -> None:
        self.consumed[update.plate_id] = update

    def outcome(self, plate_id: str) -> Optional[str]:
        k = self.matches(plate_id)
        u = self.consumed.get(k) if k else None
        return u.outcome if u else None

    def __len__(self) -> int:
        return len(self.consumed)