"""Core data types for the SignWay pipeline. Pure Python + numpy; no ROS2, no Habitat.

These are the shared contract from the orchestration spec: the things every node passes
around. Keeping them here (with no heavy imports) is what lets the FSM be unit-tested in plain
Python and reused unchanged in sim and on the real robot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

import numpy as np


class FSMState(str, Enum):
    DRIVE = "DRIVE"
    REASON = "REASON"        # Phase 2
    MANEUVER = "MANEUVER"    # Phase 2/3
    HALT = "HALT"
    ARRIVED = "ARRIVED"
    FAILED = "FAILED"


class DecisionType(str, Enum):
    CONTINUE = "continue"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    U_TURN = "u_turn"
    GOTO = "goto"
    STOP = "stop"
    ARRIVED = "arrived"    # the sign marks the destination itself — the mission is complete.
    # This is how a mapless robot knows it is done: not by reaching a coordinate it was never
    # given, but by READING a sign that names the goal as being here. Arrival is a sign-read.


class TriggerType(str, Enum):
    SAFETY = "safety"
    HAZARD = "hazard"                  # Phase 2
    SIGN_READABLE = "sign_readable"    # Phase 2
    DECISION_POINT = "decision_point"  # Phase 2
    SUBGOAL_REACHED = "subgoal_reached"
    STALLED = "stalled"


@dataclass
class Pose:
    """World SE(2) pose. Robot frame convention elsewhere: x forward, y left, yaw CCW."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.yaw)


@dataclass
class Detection:
    id: str
    label: str
    conf: float
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    est_distance_m: float = float("inf")
    est_bearing_rad: float = 0.0
    hazard: bool = False


@dataclass
class Occupancy:
    """Robot-centric occupancy. Opaque to the core; produced and consumed by the Safety backend."""
    res: float = 0.05
    x_max: float = 4.0
    y_half: float = 2.0
    cells: List[Tuple[int, int]] = field(default_factory=list)  # occupied (row, col)
    stamp: float = 0.0


@dataclass
class Decision:
    """Structured output of the VLM (Phase 2). Its only effect is a new subgoal or maneuver."""
    type: DecisionType
    target: Optional[dict] = None      # e.g. {"bearing_rad": .., "distance_m": ..}
    rationale: str = ""
    confidence: float = 1.0


@dataclass
class Subgoal:
    pose: Pose
    stamp: float = 0.0
    source: str = "mission"    # "mission" | "sign" | "continue"
    # Provenance decides what ARRIVING means. Reaching the MISSION goal ends the run. Reaching
    # a nudge a sign gave us only means that instruction is done — the robot re-arms sign
    # reading and drives on. Without this the two are indistinguishable and any completed leg
    # ends the mission.


@dataclass
class Leg:
    """A committed segment of travel under one decision — used for debounce and logging."""
    start_pose: Pose
    decision: Optional[Decision] = None
    start_stamp: float = 0.0
    sign_key: Optional[str] = None


@dataclass
class Observation:
    """One sensor bundle handed to FSM.tick() by the harness (sim or ROS2)."""
    images: List[Any] = field(default_factory=list)     # RGB frames (np arrays)
    robot_pose: Pose = field(default_factory=Pose)
    occupancy: Optional[Occupancy] = None
    detections: List[Detection] = field(default_factory=list)
    stamp: float = 0.0


@dataclass
class Command:
    """What the FSM tells the controller to do this tick."""
    kind: str = "stop"                       # "waypoints" | "stop" | "maneuver"
    waypoints: Optional[np.ndarray] = None    # (N, 2) robot frame, x fwd / y left
    maneuver: Optional[str] = None            # e.g. "u_turn"
    meta: dict = field(default_factory=dict)