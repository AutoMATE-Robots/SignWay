"""The reasoner must survive everything a network and a language model can do to it.

No API key needed: parse_decision is pure, and the transport is stubbed.
"""
import numpy as np

from c3_reasoning.reasoner_vlm import GeminiReasoner, parse_decision
from common.types import DecisionType


def test_clean_json_becomes_a_decision():
    d = parse_decision('{"reasoning":"arrow points left","text_read":"CAFETERIA",'
                       '"decision":"turn_left","confidence":0.92}')
    assert d.type == DecisionType.TURN_LEFT and d.confidence == 0.92
    assert "arrow" in d.rationale
    assert d.target["text_read"] == "CAFETERIA"


def test_markdown_fence_is_tolerated():
    d = parse_decision('Sure!\n```json\n{"decision":"turn_right","confidence":0.8}\n```')
    assert d.type == DecisionType.TURN_RIGHT


def test_prose_instead_of_json_stops_safely():
    d = parse_decision("I think the robot should probably go left?")
    assert d.type == DecisionType.STOP and d.confidence == 0.0


def test_invented_decision_name_stops_safely():
    d = parse_decision('{"decision":"swerve_gracefully","confidence":0.99}')
    assert d.type == DecisionType.STOP and d.confidence == 0.0


def test_broken_json_stops_safely():
    d = parse_decision('{"decision":"turn_left", "confidence":}')
    assert d.type == DecisionType.STOP


def test_empty_reply_stops_safely():
    assert parse_decision("").type == DecisionType.STOP


def test_confidence_is_clamped_not_trusted():
    assert parse_decision('{"decision":"continue","confidence":5}').confidence == 1.0
    assert parse_decision('{"decision":"continue","confidence":-2}').confidence == 0.0
    assert parse_decision('{"decision":"continue","confidence":"high"}').confidence == 0.0


def test_network_failure_stops_the_robot_rather_than_crashing_it():
    """An API call mid-run WILL fail. It must never raise into the control loop."""
    r = GeminiReasoner(api_key="fake")
    r._post = lambda body: {"_error": "HTTP 429: rate limited"}
    d = r.reason({"sign_image": np.zeros((40, 120, 3), np.uint8), "mission_goal": "cafeteria"})
    assert d.type == DecisionType.STOP and d.confidence == 0.0
    assert "429" in d.rationale
    assert r.calls == 1                       # a failed call still costs a call — count it


def test_missing_image_is_not_an_exception():
    r = GeminiReasoner(api_key="fake")
    assert r.reason({"mission_goal": "cafeteria"}).type == DecisionType.STOP


def test_a_normal_call_is_timed_and_logged():
    r = GeminiReasoner(api_key="fake")
    r._post = lambda body: {"candidates": [{"content": {"parts": [
        {"text": '{"reasoning":"cafeteria is left","decision":"turn_left","confidence":0.9}'}]}}]}
    d = r.reason({"sign_image": np.zeros((40, 120, 3), np.uint8), "mission_goal": "cafeteria"})
    assert d.type == DecisionType.TURN_LEFT
    assert r.stats()["calls"] == 1
    assert r.log[0]["decision"] == "turn_left" and "latency_s" in r.log[0]


def test_thinking_budget_reaches_the_request():
    """thinkingBudget is the latency dial and the ablation axis — it must actually be sent."""
    sent = {}
    r = GeminiReasoner(api_key="fake", thinking_budget=0)
    r._post = lambda body: sent.update(body) or {"_error": "stub"}
    r.reason({"sign_image": np.zeros((8, 8, 3), np.uint8)})
    assert sent["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0


def test_temperature_is_zero_because_a_sign_has_one_meaning():
    sent = {}
    r = GeminiReasoner(api_key="fake")
    r._post = lambda body: sent.update(body) or {"_error": "stub"}
    r.reason({"sign_image": np.zeros((8, 8, 3), np.uint8)})
    assert sent["generationConfig"]["temperature"] == 0.0
