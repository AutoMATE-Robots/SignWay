"""Tests for detect.py (clustering), buffer.py (tracking), reasoner.py
(parse + cache), baselines, and an end-to-end pipeline smoke test that runs
FakeOcr frames → replay_bag → save_npz → timeline.load_tracks → gate sweep."""
import json

import numpy as np
import pytest

from adaptive_reasoning.evidence.buffer import BufferConfig, EvidenceBuffer, iou
from adaptive_reasoning.evidence.detect import (
    DetectConfig, DetectedLine, cluster_lines, detect, split_arrows,
)
from adaptive_reasoning.evidence.features import OcrLine, PlateObservation
from adaptive_reasoning.evidence.legibility import Legibility
from adaptive_reasoning.evidence.relevance import Goal
from adaptive_reasoning.gate import PlateEvidence
from adaptive_reasoning.reasoner import FakeVLM, Reasoner, VLMAnswer
from adaptive_reasoning.replay.baselines import (
    AlwaysInvoke, ReactiveNecessity, RelevanceOnly, SufficiencyOnly, make_gate,
)
from adaptive_reasoning.replay.dump_features import replay_bag, save_npz
from adaptive_reasoning.memory import Memory
from adaptive_reasoning.gate import new_goal


# ---------------- detect: arrows & clustering ----------------

def test_split_arrows_tokens_and_glyphs():
    assert split_arrows("6-201 to 6-250 \u2192") == ("6-201 to 6-250", ["right"])
    assert split_arrows("Main Elevators <") == ("Main Elevators", ["left"])
    assert split_arrows("Elevators") == ("Elevators", [])   # 'v' inside a word survives


def line(text, x0, y0, x1, y1, conf=0.9):
    return DetectedLine(text, conf, (x0, y0, x1, y1))


def test_cluster_two_plates_apart():
    lines = [line("6-201 to 6-250 >", 100, 100, 400, 130),
             line("Main Elevators <", 100, 140, 380, 170),      # 10px gap, ~30px lines
             line("COMPOSTO", 900, 600, 1100, 640)]             # far away
    groups = cluster_lines(lines, DetectConfig())
    assert sorted(map(len, groups)) == [1, 2]


def test_cluster_respects_vertical_gap():
    lines = [line("A", 100, 100, 300, 130),
             line("B", 100, 400, 300, 430)]                     # 270px gap >> 1.5*30
    assert len(cluster_lines(lines, DetectConfig())) == 2


class FakeOcr:
    """Scripted OCR: frame index → detected lines (set per test)."""

    def __init__(self, script):
        self.script = script
        self._t = None

    def set_frame(self, t):
        self._t = t

    def __call__(self, image):
        return self.script.get(self._t, [])


def test_detect_builds_observations_with_crops():
    img = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    ocr = FakeOcr({0: [line("6-201 to 6-250 >", 100, 100, 400, 130),
                       line("Main Elevators <", 100, 140, 380, 170)]})
    ocr.set_frame(0)
    plates = detect(img, ocr)
    assert len(plates) == 1
    p = plates[0]
    assert [l.text for l in p.lines] == ["6-201 to 6-250", "Main Elevators"]
    assert p.lines[0].arrows == ["right"] and p.lines[1].arrows == ["left"]
    assert p.crop is not None and p.crop.shape[0] > 50 and p.box is not None


# ---------------- buffer ----------------

def obs(text, box=(100, 100, 400, 170), conf=0.9):
    return PlateObservation([OcrLine(t.strip(), conf, 30.0) for t in text.split("|")], box=box)


def test_buffer_same_text_same_track_and_agreement_rises():
    buf = EvidenceBuffer(BufferConfig(k=4))
    t0 = buf.update([obs("6-201 to 6-250")], 0)[0]
    t1 = buf.update([obs("6-201 to 6-250")], 1)[0]
    assert t0.track_id == t1.track_id and t1.agreement_frac == pytest.approx(2 / 4)


def test_buffer_fuzzy_ocr_flicker_keeps_track():
    buf = EvidenceBuffer()
    a = buf.update([obs("6-201 to 6-250")], 0)[0]
    b = buf.update([obs("6-2O1 to 6-25O")], 1)[0]        # O/0 flicker
    assert a.track_id == b.track_id
    assert b.plate_id == a.plate_id                       # id from best-conf read


def test_buffer_iou_rescues_bad_read_and_two_plates_stay_separate():
    buf = EvidenceBuffer()
    a = buf.update([obs("6-201 to 6-250", box=(100, 100, 400, 170))], 0)[0]
    b = buf.update([obs("###", box=(110, 105, 390, 165), conf=0.1)], 1)[0]
    assert a.track_id == b.track_id                       # IoU match despite garbage text
    c = buf.update([obs("COMPOSTO", box=(900, 600, 1100, 660))], 2)[0]
    assert c.track_id != a.track_id


def test_buffer_ages_out():
    buf = EvidenceBuffer(BufferConfig(miss_tolerance=2))
    a = buf.update([obs("A")], 0)[0]
    buf.update([], 1); buf.update([], 2); buf.update([], 3)
    d = buf.update([obs("A")], 4)[0]
    assert d.track_id != a.track_id


def test_iou():
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


# ---------------- reasoner ----------------

def test_vlm_answer_parse_variants():
    good = VLMAnswer.parse('{"applicable": true, "direction": "turn_left", "confidence": 0.9, "summary": "s"}')
    assert good.applicable and good.direction == "turn_left"
    fenced = VLMAnswer.parse('```json\n{"applicable": false, "direction": null, "confidence": 0.3, "summary": "no"}\n```')
    assert not fenced.applicable and fenced.direction is None
    bad_dir = VLMAnswer.parse('{"applicable": true, "direction": "left", "confidence": 1}')
    assert not bad_dir.applicable                          # unknown direction → not applicable
    garbage = VLMAnswer.parse("I think you should turn left")
    assert not garbage.applicable and garbage.summary == "unparseable"


def test_reasoner_cache_prevents_second_call(tmp_path):
    fake = FakeVLM({"6-217": {"applicable": True, "direction": "turn_right",
                              "confidence": 0.9, "summary": "range sign"}})
    r = Reasoner(fake, model_tag="fake-1", cache_dir=tmp_path)
    crop = np.zeros((50, 100, 3), dtype=np.uint8)
    a1 = r.ask("6-217", ["6-201 to 6-250"], [crop])
    a2 = r.ask("6-217", ["6-201 to 6-250"], [crop])
    assert a1.direction == "turn_right" and not a1.cached
    assert a2.direction == "turn_right" and a2.cached and fake.calls == 1
    a3 = r.ask("6-217", ["6-201 to 6-250"], [np.full((50, 100, 3), 7, np.uint8)])
    assert fake.calls == 2 and not a3.cached               # different crop → new key


# ---------------- baselines ----------------

def _ev(pid="p", R=1.0, L=1.0):
    return PlateEvidence(pid, R, L)


def test_baseline_triggers_differ_on_the_telling_cases():
    mem = Memory("g"); s = new_goal()
    blurry_relevant = [_ev("dir", R=1.0, L=0.05)]
    sharp_notice = [_ev("note", R=0.0, L=0.95)]
    assert AlwaysInvoke().tick(s, blurry_relevant, mem, 0).fire
    assert ReactiveNecessity().tick(s, blurry_relevant, mem, 0).fire
    assert not SufficiencyOnly(make_gate("sufficiency", 0.5).cfg).tick(s, blurry_relevant, mem, 0).fire
    assert SufficiencyOnly(make_gate("sufficiency", 0.5).cfg).tick(s, sharp_notice, mem, 0).fire   # the notice bug
    assert RelevanceOnly(make_gate("relevance", 0.5).cfg).tick(s, blurry_relevant, mem, 0).fire    # the blur bug
    assert not make_gate("ours", 0.5).tick(s, blurry_relevant, mem, 0).fire
    assert not make_gate("ours", 0.5).tick(s, sharp_notice, mem, 0).fire


def test_periodic_holds_between_fires():
    g = make_gate("periodic", 0.0, period=5)
    mem = Memory("g"); s = new_goal(); fires = []
    from adaptive_reasoning.gate import VLMResponse
    for t in range(12):
        r = g.tick(s, [_ev(f"p{t}", R=0.5, L=0.5)], mem, t)   # new plate每 frame
        s = r.state
        if r.fire:
            fires.append(t)
            s, upds = g.on_response(s, VLMResponse(r.best.plate_id, False), t)  # not-applicable → keep going
            for u in upds:
                mem.apply(u)
    assert fires == [0, 5, 10]


# ---------------- end-to-end pipeline smoke ----------------

def test_pipeline_fakeocr_to_npz_to_timeline_and_gate(tmp_path):
    """20 frames: directory sign appears at frame 6, grows; notice always there.
    replay_bag → save_npz → timeline.load_tracks → gate fires once."""
    rng = np.random.default_rng(1)

    script = {}
    for t in range(20):
        lines = [DetectedLine("COMPOSTO", 0.9, (900, 600, 1050, 630))]
        if t >= 6:
            h = 14 + 3.0 * (t - 6)                       # grows as we approach
            conf = min(0.35 + 0.05 * (t - 6), 0.95)
            lines.append(DetectedLine("6-201 to 6-250 >", conf, (100, 100, 100 + 12 * h, 100 + h)))
        script[t] = lines

    class SeqOcr:
        def __init__(self): self.t = -1
        def __call__(self, image):
            self.t += 1
            return script.get(self.t, [])

    class Frames:
        def __iter__(self):
            for t in range(20):
                yield t, rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    frames, logs = replay_bag(Frames(), Goal.parse("6-217"), SeqOcr(),
                              Legibility.placeholder(),
                              jsonl_path=tmp_path / "t.jsonl",
                              crops_dir=tmp_path / "crops", crop_every=3)
    assert len(frames) == 20 and len(logs) == 2
    crop_files = list((tmp_path / "crops").glob("*.jpg"))
    assert crop_files, "crops must be saved for E2"
    from adaptive_reasoning.replay.dump_features import contact_sheet
    contact_sheet(tmp_path / "crops", tmp_path / "sheet.jpg")
    assert (tmp_path / "sheet.jpg").exists()
    save_npz(tmp_path / "t.npz", frames, logs,
             ann={"flip_frame": "10", "junction_frame": "18", "fps": "10",
                  "_stride": 1, "decision": "turn_right", "goal": "6-217"})

    d = np.load(tmp_path / "t.npz", allow_pickle=True)
    dir_i = 0 if "6-201" in str(d["plate0_label"]) else 1
    R, ell = d[f"plate{dir_i}_R"], d[f"plate{dir_i}_ell"]
    assert R.max() == 1.0 and R[:6].max() == 0.0
    assert ell[19] > ell[7]                                # legibility rises
    assert d[f"plate{1-dir_i}_R"].max() == 0.0             # notice irrelevant
    assert bool(d[f"plate{dir_i}_resolvable"][12])         # arrow read → fast path possible
    assert d[f"plate{dir_i}_phi"].shape == (20, 6)
    # jsonl written and parseable
    rec = json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])
    assert "phi" in rec and "R" in rec

    # timeline can load it
    from adaptive_reasoning.replay.timeline import load_tracks
    tracks, junction, _ = load_tracks(str(tmp_path / "t.npz"))
    assert len(tracks) == 2 and junction == 18

    # and the real gate over these arrays fires exactly once (fast path off)
    from adaptive_reasoning.replay.sweep import Approach, run_one
    ap = Approach(bag="t", frames=d["frame_idx"],
                  plates=[{k: d[f"plate{i}_{k}"] for k in
                           ("id", "label", "R", "R_struct", "ell", "resolvable")}
                          for i in range(2)],
                  fps=10.0, junction=18, decision="turn_right", goal="6-217",
                  control=False)
    sc = run_one(make_gate("ours", 0.5), ap, latency_frames=2)
    assert sc.calls == 1 and sc.correct and sc.wasted == 0 and sc.lead_s > 0


# ---------------- buffer v2: the real-t1 fragmentation patterns ----------------

from adaptive_reasoning.evidence.buffer import digit_signature, merge_observations, sig_overlap, stable_plate_id
from adaptive_reasoning.memory import plate_key


def test_digit_signature_stable_under_real_ocr_mutations():
    reads = ["E | 5-231 to5-250 | 5-201 to5-217 | 5-117 to5-196",
             "€ | 5-231105-250 | N | 5-201105-217 | 5-1171o5-196",
             "E | 5-2311 t05-250 | - | 5-2011 105-217 | 5-117105-196"]
    sigs = [digit_signature(r) for r in reads]
    # mutations glue codes ("5-201105-217") but 3-digit windows keep them recoverable
    assert sig_overlap(sigs[0], sigs[1]) >= 0.8
    assert sig_overlap(sigs[0], sigs[2]) >= 0.8
    assert sig_overlap(sigs[0], digit_signature("COMPOSTO")) == 0.0
    # a lone door plate never sig-matches the directory listing its room
    assert sig_overlap(digit_signature("5-201"), sigs[0]) == 0.0


def test_half_whole_split_stays_one_track_and_merges():
    """The f184→f186 pattern: full sign, then top+bottom halves in one frame."""
    buf = EvidenceBuffer()
    full = obs("E | 5-231 to5-250 | 5-201 to5-217 | 5-117 to5-196", box=(100, 100, 400, 300))
    a = buf.update([full], 0)[0]
    top = obs("€ | 5-231 to5-250", box=(100, 100, 400, 160))
    bot = obs("5-201 105-217 | 5-117105-196", box=(100, 180, 400, 300))
    seen = buf.update([top, bot], 1)
    assert len(seen) == 1 and seen[0].track_id == a.track_id          # merged, same track
    assert "5-231" in seen[0].obs.text and "5-117" in seen[0].obs.text  # both halves present
    assert seen[0].obs.box == (100, 100, 400, 300)
    back = buf.update([obs("E | 5-2311 105-250 | 5-201105-217 | 5-117 to5-196",
                           box=(95, 95, 405, 305))], 2)[0]
    assert back.track_id == a.track_id                                 # whole again, same track


def test_plate_id_stable_across_mutations_and_splits():
    buf = EvidenceBuffer()
    a = buf.update([obs("5-231 to 5-250 | 5-201 to 5-217", box=(0, 0, 100, 60))], 0)[0]
    pid0 = a.plate_id
    b = buf.update([obs("5-2311 105-250 | 5-201105-217", box=(2, 2, 102, 62))], 1)[0]
    assert b.plate_id.startswith("SIG:")
    # identity is sticky, so memory keys survive OCR churn
    assert sig_overlap(digit_signature(a.best_text), digit_signature(b.obs.text)) >= 0.8
    assert b.plate_id == pid0


def test_two_phase_association_no_first_come_corruption():
    """Second half must match the PRE-frame state even after the first half
    was assigned (the bug where the first detection rewrote the track)."""
    buf = EvidenceBuffer()
    buf.update([obs("5-231 to5-250 | 5-201 to5-217 | 5-117 to5-196", box=(100, 100, 400, 300))], 0)
    # halves arrive in top-then-bottom order; with one-pass update the bottom
    # would have compared against the top-half box and text and spawned a track
    seen = buf.update([obs("5-231 to5-250", box=(100, 100, 400, 160)),
                       obs("5-117 105-196", box=(100, 230, 400, 300))], 1)
    assert len(seen) == 1


def test_unrelated_plates_still_separate():
    buf = EvidenceBuffer()
    a = buf.update([obs("5-231 to 5-250", box=(100, 100, 400, 160))], 0)[0]
    c = buf.update([obs("COMPOSTO", box=(900, 600, 1100, 660)),
                    obs("5-231 to 5-250", box=(100, 100, 400, 160))], 1)
    assert len({tr.track_id for tr in c}) == 2
    assert any(tr.track_id == a.track_id for tr in c)


def test_merge_observations_order_and_crop():
    top = obs("TOP", box=(0, 0, 100, 40))
    bot = obs("BOT", box=(0, 60, 100, 200))
    m = merge_observations([bot, top])
    assert [l.text for l in m.lines] == ["TOP", "BOT"] and m.box == (0, 0, 100, 200)


def test_stable_plate_id_fallback_for_textual_plates():
    assert stable_plate_id("Main Elevators") == plate_key("Main Elevators")
    assert stable_plate_id("5-231 to 5-250").startswith("SIG:")
