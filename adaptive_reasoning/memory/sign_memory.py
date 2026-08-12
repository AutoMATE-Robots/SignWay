"""Sign memory: cache the READING, not the decision.

Long-term store: one deduped record per physically distinct sign face ever read
at L3. `content` (the parsed reading) is GOAL-AGNOSTIC -- the expensive VLM read
amortizes across every future goal; goal matching against cached content is
nearly free (L1 local / L2 text-only).

Two-stage retrieval (the aliasing defense):
  1. candidate retrieval by sign-embedding cosine (floor cfg.sign_sim_floor)
  2. context verification: permanence-weighted partial agreement over co-visible
     objects (label/emb match + coarse same-side position). sign_sim HIGH but
     context FAIL => aliased-sign detection (logged; the demo dataset).

Staleness: the VLM's own `sign_type` drives TTL (temporary -> ~1 day,
permanent -> ~1 year). Embedding functions are injected so tests run with
synthetic vectors and production can use SigLIP/CLIP.
"""
from __future__ import annotations

import dataclasses
import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np

from adaptive_reasoning.config import GateConfig, MemoryConfig


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclasses.dataclass
class ContextObj:
    label: str
    emb: np.ndarray                    # crop embedding
    side: str                          # "left" | "right" | "above" | "below" (of the sign)


@dataclasses.dataclass
class SignRecord:
    rec_id: str
    sign_emb: np.ndarray
    context: List[ContextObj]
    heading_rad: float                 # topological cue at sighting
    content: dict                      # parsed reading: sign_type, arrows/targets, raw_text, read_conf
    created: float
    last_seen: float
    hits: int = 0

    @property
    def sign_type(self) -> str:
        return self.content.get("sign_type", "permanent")


@dataclasses.dataclass
class MatchResult:
    record: Optional[SignRecord]
    aliased: bool                      # high sign-sim but context verification failed
    sign_sim: float
    context_agreement: float
    combined: float


class SignMemory:
    def __init__(self, mem_cfg: MemoryConfig, gate_cfg: GateConfig):
        self.mc, self.gc = mem_cfg, gate_cfg
        self.records: List[SignRecord] = []
        self.ring = deque(maxlen=mem_cfg.ring_capacity)   # short-term sightings
        self.alias_log: List[dict] = []
        self.dir = Path(mem_cfg.store_dir) / mem_cfg.building
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------ matching ---------------------------------
    def _context_agreement(self, stored: List[ContextObj],
                           current: List[ContextObj]) -> float:
        """Permanence-weighted fraction of stored context found in the current view."""
        if not stored:
            return 0.5   # no context recorded: neutral (sign_sim carries the match)
        wsum = matched = 0.0
        for s in stored:
            w = self.mc.permanence.get(s.label, self.mc.default_permanence)
            if w <= 0.0:
                continue
            wsum += w
            for c in current:
                lbl_ok = (c.label == s.label) or cos(c.emb, s.emb) > 0.75
                if lbl_ok and c.side == s.side:
                    matched += w
                    break
        return matched / wsum if wsum > 0 else 0.5

    def _fresh(self, r: SignRecord) -> bool:
        ttl_days = (self.mc.ttl_temporary_days if r.sign_type == "temporary"
                    else self.mc.ttl_permanent_days)
        return (time.time() - r.last_seen) < ttl_days * 86400.0

    def query(self, sign_emb: np.ndarray, context: List[ContextObj],
              heading_rad: float) -> MatchResult:
        best = MatchResult(None, False, 0.0, 0.0, 0.0)
        aliased_any = False
        for r in self.records:
            ss = cos(sign_emb, r.sign_emb)
            if ss < self.gc.sign_sim_floor:
                continue
            # topological sanity: heading within +-45 deg
            dh = abs((heading_rad - r.heading_rad + np.pi) % (2 * np.pi) - np.pi)
            if dh > np.pi / 4:
                aliased_any = True     # same template, incompatible approach heading
                continue
            ca = self._context_agreement(r.context, context)
            if ca < self.gc.context_partial_agreement:
                aliased_any = True
                self.alias_log.append(dict(t=time.time(), rec_id=r.rec_id,
                                           sign_sim=ss, context_agreement=ca))
                continue
            comb = ss * (0.5 + 0.5 * ca)
            if comb > best.combined:
                best = MatchResult(r, False, ss, ca, comb)
        if best.record is not None and best.combined >= self.gc.tau_memory \
                and self._fresh(best.record):
            best.record.hits += 1
            best.record.last_seen = time.time()
            return best
        return MatchResult(None, aliased_any, best.sign_sim,
                           best.context_agreement, best.combined)

    # ------------------------------ writing ----------------------------------
    def add_or_merge(self, sign_emb, context, heading_rad, content) -> SignRecord:
        """Called once per L3 fire. Dedup by the same two-stage match; merge keeps
        the higher-confidence reading and unions context."""
        m = self.query(sign_emb, context, heading_rad)
        now = time.time()
        if m.record is not None:
            r = m.record
            if content.get("read_conf", 0) > r.content.get("read_conf", 0):
                r.content = content
            known = {(c.label, c.side) for c in r.context}
            r.context += [c for c in context if (c.label, c.side) not in known]
            r.last_seen = now
            return r
        r = SignRecord(rec_id=uuid.uuid4().hex[:10], sign_emb=np.asarray(sign_emb),
                       context=list(context), heading_rad=float(heading_rad),
                       content=dict(content), created=now, last_seen=now, hits=0)
        self.records.append(r)
        return r

    # ------------------------------ short-term -------------------------------
    def note_sighting(self, sign_emb, heading_rad, odom_progress_m, t=None) -> None:
        self.ring.append(dict(emb=np.asarray(sign_emb), heading=float(heading_rad),
                              s=float(odom_progress_m), t=t or time.time()))

    def loop_check(self, sign_emb, heading_rad, min_gap_s: float = 60.0,
                   sim_thr: float = 0.9) -> bool:
        """Loop = same signature + consistent heading seen > min_gap_s ago."""
        now = time.time()
        for e in self.ring:
            if now - e["t"] < min_gap_s:
                continue
            dh = abs((heading_rad - e["heading"] + np.pi) % (2 * np.pi) - np.pi)
            if cos(sign_emb, e["emb"]) > sim_thr and dh < np.pi / 6:
                return True
        return False

    # ------------------------------ L1/L2 ------------------------------------
    @staticmethod
    def resolve_goal(content: dict, goal: str) -> Optional[str]:
        """L1: local goal match against cached content. Returns a decision or None
        (None => caller escalates to L2 text-only LLM with the same content)."""
        goal_l = goal.lower().strip()
        for arrow in content.get("arrows", []):
            for tgt in arrow.get("targets", []):
                t = str(tgt).lower()
                if goal_l == t:
                    return arrow.get("direction")
                # room-range match: "301-320" covers goal "room 305"
                if "-" in t:
                    try:
                        lo, hi = (int(x) for x in t.replace("room", "").split("-"))
                        digits = "".join(ch for ch in goal_l if ch.isdigit())
                        if digits and lo <= int(digits) <= hi:
                            return arrow.get("direction")
                    except ValueError:
                        pass
        return None

    # ------------------------------ persistence ------------------------------
    def save(self) -> None:
        meta = []
        embs, ctx_embs = [], []
        for r in self.records:
            meta.append(dict(rec_id=r.rec_id, heading=r.heading_rad,
                             content=r.content, created=r.created,
                             last_seen=r.last_seen, hits=r.hits,
                             context=[dict(label=c.label, side=c.side)
                                      for c in r.context]))
            embs.append(r.sign_emb)
            ctx_embs.append(np.stack([c.emb for c in r.context])
                            if r.context else np.zeros((0, len(r.sign_emb))))
        (self.dir / "signs.jsonl").write_text(
            "\n".join(json.dumps(m) for m in meta))
        np.savez_compressed(self.dir / "embs.npz",
                            sign=np.stack(embs) if embs else np.zeros((0, 1)),
                            **{f"ctx_{i}": c for i, c in enumerate(ctx_embs)})

    def load(self) -> int:
        jp, ep = self.dir / "signs.jsonl", self.dir / "embs.npz"
        if not jp.exists() or not ep.exists():
            return 0
        metas = [json.loads(l) for l in jp.read_text().splitlines() if l.strip()]
        z = np.load(ep)
        self.records = []
        for i, m in enumerate(metas):
            ctx_arr = z.get(f"ctx_{i}", np.zeros((0, z["sign"].shape[1])))
            ctx = [ContextObj(c["label"], ctx_arr[j], c["side"])
                   for j, c in enumerate(m["context"]) if j < len(ctx_arr)]
            self.records.append(SignRecord(
                rec_id=m["rec_id"], sign_emb=z["sign"][i], context=ctx,
                heading_rad=m["heading"], content=m["content"],
                created=m["created"], last_seen=m["last_seen"], hits=m["hits"]))
        return len(self.records)
