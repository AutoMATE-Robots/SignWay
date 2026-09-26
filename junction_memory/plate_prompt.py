#!/usr/bin/env python3
"""
plate_prompt.py — the §2.1 change: ask the VLM for the WHOLE plate.

The reusable asset from a paid call is not "turn left", it is the mapping from
every destination on the sign to its branch. Without it, node A stores one
entry, the next goal misses, and the reduction never materialises.

Two things live here:

  PLATE_PROMPT       reasoner.PROMPT plus a `directory` field. Same JSON-only
                     contract, same grounding in the OCR text, so the existing
                     VLMAnswer.parse keeps working unchanged — `directory` just
                     rides along in `raw`.
  extract_directory  pulls the directory out of the raw model text, tolerantly
                     (code fences, missing field, dict-or-list form, arrow
                     glyphs, "turn left" with a space). Returns [] when absent.

Because VLMAnswer.raw already holds the full response, nothing in reasoner.py's
dataclass needs to change: the only permanent edit is swapping PROMPT. Until
you make that edit, render_video_memory can install this prompt at runtime
(--plate-prompt), which is why the model_tag carries a ':dirv1' suffix — the
cache key hashes model_tag but NOT the prompt, so without that bump you would
be served pre-directory answers from cache and see no directory at all.
"""
from __future__ import annotations

import json
import re

PLATE_PROMPT = """You are the sign-reading module of an indoor robot.
Goal: "{goal}"
Recognized text on the candidate sign(s) (may contain OCR errors):
{plate_texts}
Previously used signs/decisions for this goal: {memory_summary}

From the attached image crops (sign close-ups) and scene frame, do BOTH:
 1. decide whether any sign tells the robot which way to go NOW for the goal;
 2. transcribe the ENTIRE sign as a directory: every destination label on the
    plate paired with the direction its arrow points, not just the goal's.
    One entry per label. An arrow may sit above, beside, or below the labels it
    governs, and a single arrow may govern several labels -- group each label
    with the arrow that applies to it, using the sign's layout.
Answer with JSON ONLY, no prose, exactly:
{{"applicable": true/false, "direction": "turn_left"|"turn_right"|"straight"|"stop"|null,
  "confidence": 0.0-1.0, "summary": "<one short sentence>",
  "directory": [{{"label": "<text as printed>", "direction": "turn_left"|"turn_right"|"straight"|"stop"}}]}}
If no sign addresses the goal, applicable=false and direction=null, but STILL
fill directory with every label you can read. If no sign is legible at all,
directory=[]."""

_ARROWS = {"←": "turn_left", "<-": "turn_left", "<--": "turn_left",
           "→": "turn_right", "->": "turn_right", "-->": "turn_right",
           "↑": "straight", "^": "straight", "straight ahead": "straight",
           "↓": "stop"}


def _norm_direction(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in _ARROWS:
        return _ARROWS[s]
    s = s.replace("-", "_").replace(" ", "_")
    aliases = {"left": "turn_left", "turn_left": "turn_left",
               "right": "turn_right", "turn_right": "turn_right",
               "straight": "straight", "forward": "straight",
               "ahead": "straight", "up": "straight",
               "stop": "stop", "back": "back", "behind": "back",
               "u_turn": "back", "uturn": "back"}
    return aliases.get(s)


def extract_directory(raw: str) -> list[dict]:
    """Best-effort pull of the plate directory from a raw model response.

    Accepts the canonical list-of-dicts form and the two shapes models drift
    into: a {label: direction} object, and rows keyed 'text'/'destination'
    instead of 'label'. Unparseable rows are dropped, never raised on -- a
    malformed directory must degrade to "no directory", not crash a run.
    """
    if not raw:
        return []
    t = str(raw).strip()
    if "```" in t:
        parts = t.split("```")
        if len(parts) > 1:
            t = parts[1]
            t = t[4:] if t.lstrip().startswith("json") else t
    try:
        d = json.loads(t[t.index("{"):t.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return []
    raw_dir = d.get("directory")
    out = []
    if isinstance(raw_dir, dict):
        items = [{"label": k, "direction": v} for k, v in raw_dir.items()]
    elif isinstance(raw_dir, list):
        items = raw_dir
    else:
        return []
    for it in items:
        if isinstance(it, dict):
            label = it.get("label") or it.get("text") or it.get("destination")
            direction = _norm_direction(it.get("direction") or it.get("arrow"))
        elif isinstance(it, (list, tuple)) and len(it) == 2:
            label, direction = it[0], _norm_direction(it[1])
        else:
            continue
        if not label or not direction:
            continue
        out.append({"label": str(label).strip(), "direction": direction})
    return out


def install(reasoner_module) -> str:
    """Swap the module-level PROMPT for the directory-aware one, at runtime.

    Returns the previous prompt so a caller can restore it. Use this to try the
    change without touching the repo; make it permanent by editing
    reasoner.PROMPT (and keep a version suffix in Reasoner.model_tag).
    """
    if not hasattr(reasoner_module, "PROMPT"):
        raise RuntimeError(
            f"{reasoner_module.__name__} has no module-level PROMPT to swap; "
            "set reasoner.PROMPT = PLATE_PROMPT by hand (and add a version "
            "suffix to Reasoner.model_tag so the cache does not serve "
            "pre-directory answers).")
    prev = reasoner_module.PROMPT
    reasoner_module.PROMPT = PLATE_PROMPT
    return prev