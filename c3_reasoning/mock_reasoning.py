"""Fake versions of the two reasoning pieces, so the whole pipeline can be tested with no VLM,
no GPU, and no network.

MockDetector  — pretends to spot signs at fixed places in the world.
MockReasoner  — pretends to read them and returns a canned decision.

These exist so the orchestrator's REASON state can be tested for correctness *now*. Swapping in
the real GroundingDINO + VLM later changes nothing else: they implement the same two methods.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from common.interfaces import Detector, Reasoner
from common.types import Decision, DecisionType, Detection


class MockDetector(Detector):
    """Reports a sign when the robot is within `visible_from` metres of a placed sign.

    signs: list of (world_x, world_y, label, hazard). The detector needs to know where the robot
    is to fake distance, so call `set_pose()` each tick (the real detector reads the image
    instead).
    """

    def __init__(self, signs: Optional[List[Tuple[float, float, str, bool]]] = None,
                 visible_from: float = 3.0, conf: float = 0.9):
        self.signs = signs or []
        self.visible_from = visible_from
        self.conf = conf
        self._pose = (0.0, 0.0, 0.0)

    def set_pose(self, x: float, y: float, yaw: float) -> None:
        self._pose = (x, y, yaw)

    def detect(self, image) -> List[Detection]:
        x, y, yaw = self._pose
        out = []
        for i, (sx, sy, label, hazard) in enumerate(self.signs):
            dx, dy = sx - x, sy - y
            dist = float(np.hypot(dx, dy))
            if dist > self.visible_from:
                continue
            bearing = float(np.arctan2(dy, dx) - yaw)
            out.append(Detection(id=f"sign_{i}", label=label, conf=self.conf,
                                 bbox=(0, 0, 10, 10), est_distance_m=dist,
                                 est_bearing_rad=bearing, hazard=hazard))
        return out


class MockReasoner(Reasoner):
    """Maps a sign label to a decision with a lookup table. The real VLM will look at the image
    and decide; this just proves the plumbing and lets us test every branch."""

    DEFAULT: Dict[str, DecisionType] = {
        "cafeteria left": DecisionType.TURN_LEFT,
        "cafeteria right": DecisionType.TURN_RIGHT,
        "staff only": DecisionType.U_TURN,
        "elevator closed": DecisionType.U_TURN,
        "stop": DecisionType.STOP,
        "straight ahead": DecisionType.CONTINUE,
        "cafeteria": DecisionType.ARRIVED,          # the destination nameplate — you are here
        "cafeteria here": DecisionType.ARRIVED,
    }

    def __init__(self, table: Optional[Dict[str, DecisionType]] = None,
                 confidence: float = 0.9):
        self.table = dict(self.DEFAULT)
        if table:
            self.table.update(table)
        self.confidence = confidence
        self.calls = 0                     # how many times the slow model ran — the metric

    def reason(self, request: dict) -> Decision:
        self.calls += 1
        label = (request.get("label") or "").lower()
        dtype = self.table.get(label, DecisionType.CONTINUE)
        return Decision(type=dtype,
                        rationale=f"sign said '{label}' -> {dtype.value}",
                        confidence=self.confidence)