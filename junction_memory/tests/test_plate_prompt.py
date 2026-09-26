#!/usr/bin/env python3
"""
test_plate_prompt.py — the directory extractor must survive real model output.

These are the shapes Gemini actually produces: fenced JSON, the canonical list
of {label, direction}, a {label: direction} object, arrow glyphs instead of
words, rows keyed 'text' instead of 'label', and outright garbage. A malformed
directory must degrade to "no directory" (which the run reports as
DIRECTORY_PARTIAL), never raise mid-bag.
"""
import json

from plate_prompt import PLATE_PROMPT, extract_directory
from .memory_runtime import directory_from_vlm


def test_canonical_fenced_response():
    raw = """```json
{"applicable": true, "direction": "turn_left", "confidence": 0.91,
 "summary": "sign lists 6-201 to 6-250 to the left",
 "directory": [{"label": "6-201 to 6-250", "direction": "turn_left"},
               {"label": "Cafeteria", "direction": "straight"},
               {"label": "Exit", "direction": "stop"}]}
```"""
    d = extract_directory(raw)
    assert len(d) == 3
    assert d[0] == {"label": "6-201 to 6-250", "direction": "turn_left"}
    # stop is not a bearing -> dropped on the way into memory
    assert directory_from_vlm(d) == {"6-201 to 6-250": "left",
                                     "cafeteria": "straight"}


def test_object_form_and_arrow_glyphs():
    raw = json.dumps({"applicable": True, "direction": "turn_right",
                      "confidence": 0.8, "summary": "",
                      "directory": {"Rooms 340-360": "→",
                                    "Library": "↑",
                                    "Loading dock": "left"}})
    d = extract_directory(raw)
    got = {e["label"]: e["direction"] for e in d}
    assert got == {"Rooms 340-360": "turn_right", "Library": "straight",
                   "Loading dock": "turn_left"}


def test_alternate_row_keys_and_junk_rows():
    raw = json.dumps({"directory": [
        {"text": "Auditorium", "arrow": "<-"},        # text/arrow keys
        {"label": "", "direction": "straight"},        # empty label
        {"label": "Nowhere", "direction": "sideways"},  # unknown direction
        ["Gym", "turn_right"],                          # pair form
        "garbage",                                      # not a row at all
    ]})
    d = extract_directory(raw)
    got = {e["label"]: e["direction"] for e in d}
    assert got == {"Auditorium": "turn_left", "Gym": "turn_right"}


def test_missing_or_malformed_degrades_quietly():
    # the CURRENT (unpatched) reasoner: valid answer, no directory field
    assert extract_directory(json.dumps(
        {"applicable": True, "direction": "straight", "confidence": 0.7,
         "summary": "x"})) == []
    assert extract_directory("not json at all") == []
    assert extract_directory("") == []
    assert extract_directory(json.dumps({"directory": "left"})) == []
    assert directory_from_vlm([]) == {}
    assert directory_from_vlm(None) == {}


def test_prompt_keeps_the_reasoner_contract():
    # must still format with exactly the three fields reasoner.py supplies
    s = PLATE_PROMPT.format(goal="6-352", plate_texts="- 6-201 to 6-250",
                            memory_summary="none")
    assert "6-352" in s and "6-201 to 6-250" in s
    assert "JSON ONLY" in s and '"directory"' in s
    # and the single-goal contract the gate depends on is untouched
    for field in ('"applicable"', '"direction"', '"confidence"', '"summary"'):
        assert field in s


def test_install_swaps_and_returns_previous():
    import types
    fake = types.SimpleNamespace(PROMPT="OLD PROMPT")
    from plate_prompt import install
    prev = install(fake)
    assert prev == "OLD PROMPT"
    assert fake.PROMPT is PLATE_PROMPT


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
