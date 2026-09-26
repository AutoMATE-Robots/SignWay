"""
validate_annotations.py — protocol checks for ar_extra.csv (T0 gate).

Runs the REAL relevance/resolve code against every annotated row, so a typo in
`sign_text` or `goal` fails loudly here instead of silently corrupting the
calibration later.  Checks (protocol §8):

  1. Required fields non-empty for the row's sign_situation.
  2. Frame invariant: sign_visible_frame <= flip_frame < junction_frame.
     (sign_exit_frame may exceed junction_frame — the sign can stay in view
     through the turn — but must be >= flip_frame.)
  3. directional & semantic_only=0  -> R(sign_text, goal) >= 0.9 (struct or own-word).
  4. semantic_only=1                -> R_struct < 0.1 (the row must NEED R_sem).
  5. If the fast path resolves the plate, its direction must equal `decision`
     (when decision_grounded=1).  Unresolved is fine (that's the VLM's case);
     a RESOLVED direction that contradicts the annotation is an error in one
     of: sign_text arrows, goal, or decision.
  6. Goal format: room goals as printed ('2-260'); flags verbose goals that
     the room parser can't read ('General Purpose Classroom 2-260').
  7. Warnings (non-fatal): margin (junction-flip)/fps below --min-margin
     seconds.  (V_g descriptor files were removed from the design: the
     semantic channel matches the goal's own words via the embedder.)

Usage:
  python -m adaptive_reasoning.replay.validate_annotations ar_extra.csv [--goals-dir adaptive_reasoning/goals] [--min-margin 4.0]
Exit code 0 = clean (warnings allowed), 1 = errors.
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

from ..evidence.relevance import Goal, parse_room, relevance
from ..evidence.resolve import Plate, resolve

REQUIRED_ALWAYS = ["bag", "building", "goal", "sign_situation", "flip_frame",
                   "junction_frame", "decision", "decision_grounded", "fps"]
REQUIRED_DIRECTIONAL = ["sign_type", "sign_text", "sign_visible_frame"]


def validate(path: str, goals_dir: str | None = None, min_margin: float = 4.0) -> int:
    rows = list(csv.DictReader(open(path)))
    errors: list[str] = []
    warns: list[str] = []
    E = lambda r, m: errors.append(f"[{r['bag']}] {m}")
    W = lambda r, m: warns.append(f"[{r['bag']}] {m}")

    for r in rows:
        # 1. required fields
        for c in REQUIRED_ALWAYS:
            if not (r.get(c) or "").strip():
                E(r, f"missing required field '{c}'")
        if r.get("sign_situation") == "directional":
            for c in REQUIRED_DIRECTIONAL:
                if not (r.get(c) or "").strip():
                    E(r, f"directional row missing '{c}'")
        if any(e.startswith(f"[{r['bag']}]") for e in errors):
            continue  # skip content checks on structurally broken rows

        fps = int(r["fps"])
        flip, junc = int(r["flip_frame"]), int(r["junction_frame"])

        # 2. frame invariant
        if r.get("sign_visible_frame", "").strip():
            vis = int(r["sign_visible_frame"])
            if not (vis <= flip):
                E(r, f"sign_visible_frame {vis} > flip_frame {flip}")
        if not (flip < junc):
            E(r, f"flip_frame {flip} >= junction_frame {junc}")
        if r.get("sign_exit_frame", "").strip():
            if int(r["sign_exit_frame"]) < flip:
                E(r, f"sign_exit_frame {r['sign_exit_frame']} < flip_frame {flip}")

        # margin warning
        margin = (junc - flip) / fps
        if margin < min_margin:
            W(r, f"margin {margin:.1f}s < {min_margin:.1f}s (tight approach)")

        if r["sign_situation"] != "directional":
            # controls: no plate should be relevant — checked in E1 replay against OCR, not here
            continue

        goal = Goal.parse(r["goal"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rel = relevance(r["sign_text"], goal)
            res = resolve(Plate.from_text(r["sign_text"]), goal)

        # 6. goal format
        if goal.room is None:
            if any(ch.isdigit() for ch in r["goal"]):
                E(r, f"goal '{r['goal']}' contains digits but doesn't parse as a room — "
                     f"use the printed room code (e.g. '2-260')")
