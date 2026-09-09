"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.

------------------------------------------------------------------------------------------------------
RECONSTRUCTED 2026-08-25 from prismatic/vla/__pycache__/constants.cpython-310.pyc after the original
was lost in a /scratch.global purge. Every value below was read out of the compiled bytecode
(`dis.dis`), not from memory. The SIGNWAY block is additionally corroborated by arithmetic: with
ACTION_DIM = 16, L1RegressionActionHead builds MLPResNet(input_dim=4096*16=65536, hidden_dim=4096,
num_blocks=2, output_dim=16) = 302,223,376 parameters, which matches the count printed in
oft_train_v11.log exactly.

KEEP THIS FILE IN VERSION CONTROL. It is part of the training recipe and it lives on purgeable
scratch; losing it means no training or evaluation can start (action_heads.py imports ACTION_DIM at
module load).
------------------------------------------------------------------------------------------------------
"""

import sys
from enum import Enum

# Defines supported normalization schemes for action and proprioceptive state.
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2


class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [q01, q99] -> [-1, 1]
    # fmt: on


# LIBERO simulation benchmark constants
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# ALOHA real-world robot constants
ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

# Bridge V2 real-world robot constants
BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# SignWay: single-step prediction of an 8-waypoint trajectory.
#   NUM_ACTIONS_CHUNK = 1   one action per step (no chunking) -> "Next Actions L1 Loss = nan" is
#                           expected in wandb, since there is no second chunk to score.
#   ACTION_DIM = 16         [x1,y1,...,x8,y8]: 8 cumulative waypoints, robot frame, metres.
#   PROPRIO_DIM = 0         no proprioceptive input -> forward speed is unobservable from a single
#                           frame, which is why forward error dominates and is reported separately.
SIGNWAY_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 1,
    "ACTION_DIM": 16,
    "PROPRIO_DIM": 0,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()
    if "signway" in cmd_args:
        return "SIGNWAY"
    elif "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    else:
        # Default to LIBERO
        return "LIBERO"


# Detect the robot platform from the launch command.
# NOTE: detection is a substring match on sys.argv, so any script path containing "signway" selects
# the SIGNWAY block automatically. A bare `python -` heredoc has no such path and must do
# `sys.argv.append("--signway")` BEFORE importing anything from prismatic.
ROBOT_PLATFORM = detect_robot_platform()

if ROBOT_PLATFORM == "SIGNWAY":
    constants = SIGNWAY_CONSTANTS
elif ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS

NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

print("Using", ROBOT_PLATFORM, "constants:")
print("  NUM_ACTIONS_CHUNK =", NUM_ACTIONS_CHUNK)
print("  ACTION_DIM =", ACTION_DIM)
print("  PROPRIO_DIM =", PROPRIO_DIM)
print("  ACTION_PROPRIO_NORMALIZATION_TYPE =", ACTION_PROPRIO_NORMALIZATION_TYPE)
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
