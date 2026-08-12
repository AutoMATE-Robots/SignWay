#!/usr/bin/env python3
# one-off: one-shot patch adding the SIGNWAY_FLIP_ZERO prompt option to tfds_builder.py, applied 2026-08-09, kept for provenance
r"""
patch_flip_zero.py -- add the SIGNWAY_FLIP_ZERO option to tfds_builder.py.

WHAT IT DOES
    With SIGNWAY_FLIP_ZERO=1, every TRAIN bag whose decision is a turn gets
    flip_frame forced to 0. No images change, no actions change, no new data --
    only the prompt string on the frames before the original flip.

WHY
    Those pre-flip frames currently carry prompt="straight" with a straight action,
    which teaches nothing about patience. Relabelled, they become
        "you have a standing instruction to turn right, and the correct action is
         still to go straight"
    which is exactly the decision-vs-timing signal the policy needs. Measured gain
    on the v7 annotation set: +1859 steps under a turn prompt, a 23% increase in
    turn-prompt frames, all of them patience examples.

IS IT LEGITIMATE?
    flip_frame models when the *VLM* could read the sign -- a system-level concern.
    For *policy* training the standing instruction is true from frame 0 regardless
    of sign legibility, so this is a truthful relabel, not a fudge.

    val (e1/e2/e3) is deliberately NOT touched: it isn't used in training, and the
    eval passes --flip-frame on the command line anyway, so the frozen anchor stays
    byte-comparable across v4..v8.

RUN
    python patch_flip_zero.py
    # v7 dataset (as annotated):
    python -c "from tfds_builder import SignwayDataset; SignwayDataset(data_dir='$SCRATCH/rlds_v7').download_and_prepare()"
    # v8 dataset (relabelled):
    SIGNWAY_FLIP_ZERO=1 python -c "from tfds_builder import SignwayDataset; SignwayDataset(data_dir='$SCRATCH/rlds_v8').download_and_prepare()"

Idempotent.
"""
import ast
import sys
from pathlib import Path

P = Path.home() / "SignWay" / "signway_dataset" / "tfds_builder.py"
s = P.read_text()

if "SIGNWAY_FLIP_ZERO" in s:
    print("already patched")
    sys.exit(0)

anchor = '''_ANN = _os.environ.get("SIGNWAY_ANNOTATIONS", "")
if _ANN and Path(_ANN).exists():
    BAGS = _bags_from_csv(_ANN)'''
assert anchor in s, "anchor not found -- is this the cache-backed builder?"

addition = anchor + '''


# ---------------------------------------------------------------------------
# SIGNWAY_FLIP_ZERO: relabel TRAIN turn bags with flip_frame=0 so the frames
# before the original flip carry the turn prompt instead of "straight". Same
# images, same actions -- only the prompt changes. Those frames then teach
# "standing turn command, correct action is still straight", which is the
# patience signal the policy is short on. val is left alone.
# ---------------------------------------------------------------------------
if _os.environ.get("SIGNWAY_FLIP_ZERO", "").lower() in ("1", "true", "yes"):
    _before = sum(f for sp, nm, f, dec, td, st in BAGS
                  if sp == "train" and dec != "straight")
    BAGS = [(sp, nm, (0 if (sp == "train" and dec != "straight") else f), dec, td, st)
            for sp, nm, f, dec, td, st in BAGS]
    _n = sum(1 for sp, nm, f, dec, td, st in BAGS if sp == "train" and dec != "straight")
    print(f"[builder] SIGNWAY_FLIP_ZERO: {_n} train turn bags relabelled to flip_frame=0 "
          f"({_before} raw frames moved into the turn-prompt regime)")'''

s = s.replace(anchor, addition, 1)
ast.parse(s)
P.write_text(s)
print("patched OK -- set SIGNWAY_FLIP_ZERO=1 to build the relabelled variant")
