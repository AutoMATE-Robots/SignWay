"""Tests for gate logic, sign memory, and deadline-label geometry.

All tests run with numpy only (no torch / OCR / ROS): heavy deps stay lazy in
the modules under test. Run from the repo root:  pytest adaptive_reasoning/tests
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaptive_reasoning.config import GateConfig, MemoryConfig
from adaptive_reasoning.deadline.make_deadline_labels import (
    detect_turn_onset, label_frames)
from adaptive_reasoning.evidence.scorer import BestKBuffer, EvidenceItem
from adaptive_reasoning.gate.gate import ARMED, DECIDED, NO_SIGN, Gate
from adaptive_reasoning.memory.sign_memory import ContextObj, SignMemory


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def item(frame, score, conf=0.9):
    return EvidenceItem(frame_idx=frame, bbox=(0, 0, 10, 10), ocr_conf=conf,
                        text="x", area_frac=0.01, sharpness=0.8, score=score,
                        crop=np.zeros((4, 4, 3), np.uint8))


def unit(seed, dim=32):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #
class TestGate:
    def test_stays_no_sign_without_evidence(self):
        g = Gate(GateConfig())
        o = g.step(0, [], d_q10=None, p_junction=0.0, speed_mps=0.5)
        assert o.state == NO_SIGN and not o.fire

    def test_arms_and_fires_on_sufficiency(self):
        cfg = GateConfig(tau_sufficiency=0.5)
        g = Gate(cfg)
        o = g.step(10, [item(10, 0.2)], None, 0.0, 0.5)
        assert o.state == ARMED and not o.fire
        o = g.step(20, [item(20, 0.7)], None, 0.0, 0.5)
        assert o.fire and o.fire_reason == "sufficiency"
        # does not double-fire
        o = g.step(30, [item(30, 0.9)], None, 0.0, 0.5)
        assert not o.fire

    def test_deadline_forces_with_weak_evidence(self):
        cfg = GateConfig(tau_sufficiency=0.9, latency_p90_init_s=2.0, margin_s=0.5)
        g = Gate(cfg)
        # far away: no fire (ttj = 10/0.5 = 20s >> 2.5s)
        o = g.step(0, [item(0, 0.3)], d_q10=10.0, p_junction=1.0, speed_mps=0.5)
        assert o.state == ARMED and not o.fire and o.fire_deadline_s > 0
        # close: 1.0/0.5 = 2.0s <= 2.5s -> deadline fire despite weak evidence
        o = g.step(50, [item(50, 0.3)], d_q10=1.0, p_junction=1.0, speed_mps=0.5)
        assert o.fire and o.fire_reason == "deadline"

    def test_pessimistic_quantile_is_what_matters(self):
        """q10=1.5m must force even if the median belief is comfortable."""
        cfg = GateConfig(tau_sufficiency=0.9, latency_p90_init_s=2.0, margin_s=0.5)
        g = Gate(cfg)
        o = g.step(0, [item(0, 0.2)], d_q10=1.2, p_junction=1.0, speed_mps=0.5)
        assert o.fire and o.fire_reason == "deadline"

    def test_memory_preempts_and_cancels_fire(self):
        cfg = GateConfig(tau_sufficiency=0.5)
        g = Gate(cfg)
        o = g.step(10, [item(10, 0.8)], None, 0.0, 0.5,
                   memory_decision="turn_left")
        assert o.state == DECIDED and o.decision == "turn_left"
        assert o.decision_source == "memory" and not o.fire
        # subsequent frames: resolved, nothing fires
        o = g.step(20, [item(20, 0.9)], d_q10=0.5, p_junction=1.0, speed_mps=0.5)
        assert o.state == DECIDED and not o.fire

    def test_speed_for_evidence_engages_then_not_after_fire(self):
        cfg = GateConfig(tau_sufficiency=0.9, latency_p90_init_s=2.0,
                         margin_s=0.5, slow_trigger_s=1.5, slow_factor=0.5)
        g = Gate(cfg)
        # fire_deadline = 2.0/0.5 - 2.5 = 1.5 -> not <1.5; nudge closer:
        o = g.step(0, [item(0, 0.2)], d_q10=1.9, p_junction=1.0, speed_mps=0.5)
        assert 0 < o.fire_deadline_s < 1.5 and o.slow_factor == 0.5 and not o.fire

    def test_rearm_resets(self):
        g = Gate(GateConfig(tau_sufficiency=0.5))
        g.step(10, [item(10, 0.8)], None, 0.0, 0.5)
        g.resolve("turn_left", "vlm")
        g.rearm()
        assert g.state == NO_SIGN and len(g.buffer.items) == 0

    def test_latency_tracker_p90(self):
        g = Gate(GateConfig(latency_p90_init_s=3.0))
        assert g.latency.p90 == 3.0
        for L in (1.0, 1.2, 5.0, 1.1):
            g.report_vlm_latency(L)
        assert 1.2 <= g.latency.p90 <= 5.0


class TestBuffer:
    def test_diversity_gap(self):
        b = BestKBuffer(k=3, min_frame_gap=5)
        b.add(item(10, 0.5))
        b.add(item(12, 0.9))       # within gap of 10: replaces (higher score)
        assert len(b.items) == 1 and b.items[0].frame_idx == 12
        b.add(item(11, 0.4))       # within gap, lower score: rejected
        assert len(b.items) == 1
        b.add(item(20, 0.6))
        b.add(item(30, 0.7))
        assert len(b.items) == 3
        b.add(item(40, 0.95))      # k full: evicts the weakest (0.6 @20)
        assert len(b.items) == 3 and b.best_score == 0.95
        assert all(i.frame_idx != 20 for i in b.items)


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #
def make_mem(tmp_path):
    return SignMemory(MemoryConfig(store_dir=tmp_path, building="test"),
                      GateConfig())


class TestMemory:
    def test_hit_after_write(self, tmp_path):
        mem = make_mem(tmp_path)
        s = unit(1)
        ctx = [ContextObj("trash can", unit(2), "left"),
               ContextObj("door", unit(3), "right")]
        mem.add_or_merge(s, ctx, 0.0, dict(sign_type="permanent", read_conf=0.9,
                                           arrows=[dict(direction="turn_left",
                                                        targets=["room 301-320"])]))
        m = mem.query(s + 0.01 * unit(9), ctx, 0.05)
        assert m.record is not None and not m.aliased
        assert SignMemory.resolve_goal(m.record.content, "room 305") == "turn_left"

    def test_aliased_same_sign_different_context(self, tmp_path):
        mem = make_mem(tmp_path)
        s = unit(1)
        mem.add_or_merge(s, [ContextObj("trash can", unit(2), "left")], 0.0,
                         dict(sign_type="permanent", read_conf=0.9))
        # identical sign template, entirely different surroundings
        m = mem.query(s, [ContextObj("plant", unit(7), "right")], 0.0)
        assert m.record is None and m.aliased
        assert len(mem.alias_log) >= 1

    def test_moved_object_partial_agreement_still_matches(self, tmp_path):
        mem = make_mem(tmp_path)
        s = unit(1)
        ctx = [ContextObj("trash can", unit(2), "left"),
               ContextObj("fire extinguisher", unit(3), "right"),
               ContextObj("cup", unit(4), "left")]
        mem.add_or_merge(s, ctx, 0.0, dict(sign_type="permanent", read_conf=0.9))
        # the cup walked away; heavy fixtures remain -> still a hit
        m = mem.query(s, ctx[:2], 0.0)
        assert m.record is not None

    def test_heading_mismatch_rejects(self, tmp_path):
        mem = make_mem(tmp_path)
        s = unit(1)
        ctx = [ContextObj("door", unit(3), "right")]
        mem.add_or_merge(s, ctx, 0.0, dict(sign_type="permanent", read_conf=0.9))
        m = mem.query(s, ctx, math.pi)     # approaching from the opposite direction
        assert m.record is None

    def test_merge_dedup_keeps_best_reading(self, tmp_path):
        mem = make_mem(tmp_path)
        s = unit(1)
        ctx = [ContextObj("door", unit(3), "right")]
        mem.add_or_merge(s, ctx, 0.0, dict(sign_type="permanent", read_conf=0.5,
                                           raw_text="blurry"))
        mem.add_or_merge(s, ctx, 0.0, dict(sign_type="permanent", read_conf=0.95,
                                           raw_text="ROOM 301-320"))
        assert len(mem.records) == 1
        assert mem.records[0].content["raw_text"] == "ROOM 301-320"

    def test_temporary_ttl_expires(self, tmp_path):
        mem = make_mem(tmp_path)
        s = unit(1)
        ctx = [ContextObj("door", unit(3), "right")]
        r = mem.add_or_merge(s, ctx, 0.0, dict(sign_type="temporary", read_conf=0.9))
        r.last_seen = time.time() - 2 * 86400          # 2 days old, TTL 1 day
        assert mem.query(s, ctx, 0.0).record is None

    def test_persistence_roundtrip(self, tmp_path):
        mem = make_mem(tmp_path)
        s = unit(1)
        ctx = [ContextObj("door", unit(3), "right")]
        mem.add_or_merge(s, ctx, 0.2, dict(sign_type="permanent", read_conf=0.9,
                                           raw_text="ROOM 301"))
        mem.save()
        mem2 = make_mem(tmp_path)
        assert mem2.load() == 1
        assert mem2.query(s, ctx, 0.2).record is not None

    def test_loop_detection(self, tmp_path):
        mem = make_mem(tmp_path)
        s = unit(5)
        mem.note_sighting(s, 0.1, 3.0, t=time.time() - 120)
        assert mem.loop_check(s, 0.12)
        assert not mem.loop_check(unit(6), 0.12)


# --------------------------------------------------------------------------- #
# deadline label geometry (synthetic odom)
# --------------------------------------------------------------------------- #
def synth_turn_bag(v=0.5, hz=20.0, straight_s=20.0, turn_s=3.0, after_s=5.0,
                   decision="turn_right"):
    """Straight at v, then a 90-deg turn, then straight. Returns frame_times, odom."""
    dt = 1.0 / hz
    ts, xs, ys, yaws = [], [], [], []
    t = x = y = yaw = 0.0
    sgn = -1.0 if decision == "turn_right" else 1.0
    n1, n2, n3 = int(straight_s * hz), int(turn_s * hz), int(after_s * hz)
    for i in range(n1 + n2 + n3):
        ts.append(t); xs.append(x); ys.append(y); yaws.append(yaw)
        if n1 <= i < n1 + n2:
            yaw += sgn * (math.pi / 2) / n2
        x += v * dt * math.cos(yaw)
        y += v * dt * math.sin(yaw)
        t += dt
    frame_times = np.array(ts)                # camera at odom rate for simplicity
    odom = (np.array(ts), np.array(xs), np.array(ys), np.array(yaws))
    return frame_times, odom, n1              # n1 = true onset index


class TestDeadlineLabels:
    def test_onset_detected_at_true_turn(self):
        ft, odom, true_onset = synth_turn_bag()
        t_on, method = detect_turn_onset(odom[0], odom[3], t_lo=ft[true_onset - 100],
                                         t_hi=ft[-1], decision="turn_right")
        assert t_on is not None
        assert abs(t_on - ft[true_onset]) < 0.5          # within 0.5 s

    def test_distance_labels_decrease_linearly(self):
        ft, odom, true_onset = synth_turn_bag(v=0.5, straight_s=40.0)  # 20 m approach
        rows, meta = label_frames(ft, odom, flip_frame=true_onset - 150,
                                  decision="turn_right",
                                  turn_done_frame=true_onset + 80, d_max=15.0)
        assert rows is not None
        ahead = [r for r in rows if r["junction_ahead"]]
        assert ahead, "no junction_ahead frames"
        # d should shrink ~ v * dt between consecutive ahead frames
        ds = [r["d_m"] for r in ahead]
        diffs = -np.diff(ds)
        assert np.all(diffs > 0)
        assert abs(np.median(diffs) - 0.5 / 20.0) < 0.01
        # last ahead frame is essentially at the junction
        assert ds[-1] < 0.2
        # frames beyond d_max are labeled 0
        far = [r for r in rows if r["frame_index"] < 30]
        assert all(r["junction_ahead"] == 0 for r in far)

    def test_straight_bag_all_negative(self):
        ft, odom, _ = synth_turn_bag(turn_s=0.0, after_s=0.0)
        rows, meta = label_frames(ft, odom, flip_frame=0, decision="straight",
                                  turn_done_frame=None, d_max=15.0)
        assert all(r["junction_ahead"] == 0 and r["past_junction"] == 0
                   for r in rows)

    def test_wrong_direction_no_onset(self):
        ft, odom, true_onset = synth_turn_bag(decision="turn_right")
        t_on, method = detect_turn_onset(odom[0], odom[3], ft[0], ft[-1],
                                         decision="turn_left")   # wrong sign
        assert t_on is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
