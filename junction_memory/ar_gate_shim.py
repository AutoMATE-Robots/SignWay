#!/usr/bin/env python3
"""
ar_gate_shim.py
---------------
A replayable stand-in for the SignWay Adaptive Reasoning gate, driven by a
per-bag annotation YAML instead of live OCR + Gemini. It exists so the memory
layer can be validated on the 35 GB bag without the full evidence stack; the
interface to memory is exactly the production one (README §9):

    necessity = pending AND (memory.resolves(goal) is None)

so swapping in the real gate later means replacing this file, not the runtime.

Annotation schema — windows can be keyed by IMAGE FRAME index (the usual
SignWay convention) or by bag-relative SECONDS (for the fast odom-only loop,
which never touches the image topic):

    bag: rosbag2_2026_08_22-20_29_37
    goals:
      - {from_frame: 0,    goal: goal_A}         # or from_s: 0.0
      - {from_frame: 2400, goal: goal_E}
    signs:
      - arm_frame: 380          # evidence starts accumulating (gate arms)
        decide_frame: 520       # sufficiency reached -> VLM call dispatched
        # ... or arm_s / decide_s in seconds
        decision: left          # what the VLM answered for the active goal
        conf: 0.91
        directory:              # the WHOLE plate (README §2.1)
          goal_A: left
          goal_E: straight

Gate states mirror the production FSM: NO_SIGN -> ARMED -> PENDING -> DECIDED,
plus SUPPRESSED when memory zeroes necessity before sufficiency is reached.
"""
from __future__ import annotations

from dataclasses import dataclass

import yaml

from .memory_runtime import MemoryRuntime, norm_goal


@dataclass
class SignWindow:
    decision: str
    directory: dict
    arm_frame: int = None
    decide_frame: int = None
    arm_s: float = None
    decide_s: float = None
    conf: float = None
    node_hint: str = None
    state: str = "IDLE"       # IDLE|ARMED|PENDING|DECIDED|SUPPRESSED
    counted: bool = False

    def pos(self, frame, t_rel):
        """-1 before arm, 0 inside [arm, decide), +1 at/after decide."""
        if self.arm_frame is not None:
            a, d, q = self.arm_frame, self.decide_frame, frame
        else:
            a, d, q = self.arm_s, self.decide_s, t_rel
        if q is None or q < a:
            return -1
        return 1 if q >= d else 0


class ARGateShim:
    def __init__(self, annotations_path: str, runtime: MemoryRuntime):
        with open(annotations_path) as f:
            self.ann = yaml.safe_load(f) or {}
        self.rt = runtime
        self.goals = self.ann.get("goals", [])
        self.signs = [SignWindow(**s) for s in self.ann.get("signs", [])]
        self.signs.sort(key=lambda w: (w.arm_frame if w.arm_frame is not None
                                       else w.arm_s))
        self.state = "NO_SIGN"
        self.vlm_calls = 0
        self.counterfactual_calls = 0     # what always-reason would have paid
        self.standing_prompt = "straight"
        self.prompt_source = "default"
        self.events = []

    # ------------------------------------------------------------------
    def goal_at(self, frame, t_rel):
        g = self.goals[0]["goal"] if self.goals else "goal"
        for entry in self.goals:
            if "from_frame" in entry and frame is not None:
                if frame >= entry["from_frame"]:
                    g = entry["goal"]
            elif "from_s" in entry and t_rel is not None:
                if t_rel >= entry["from_s"]:
                    g = entry["goal"]
        return g

    def _emit(self, frame, kind, msg, severity="info", **data):
        e = self.rt._emit(kind, msg, severity, **data)
        if frame is not None:
            e.frame = frame
        self.events.append(e)
        return e

    # ------------------------------------------------------------------
    def step(self, frame=None, t_rel=None):
        """Advance the gate. Call once per image frame (video replay) or per
        odom sample (fast loop with seconds-keyed annotations)."""
        goal = self.goal_at(frame, t_rel)
        prev_goal = getattr(self.rt, "current_goal", None)
        self.rt.current_goal = goal
        if prev_goal is not None and norm_goal(prev_goal) != norm_goal(goal):
            self._emit(frame, "GOAL_SWITCH",
                       f"goal switched: {prev_goal} -> {goal}", "major",
                       goal=goal)
            if self.rt.standing and \
                    norm_goal(self.rt.standing["goal"]) != norm_goal(goal):
                self.rt.standing = None
            self.standing_prompt, self.prompt_source = "straight", "default"

        # memory can hand us a standing prompt at any time (edge entry)
        hit = self.rt.resolves(goal)
        if hit and self.prompt_source != "memory":
            self.standing_prompt = hit["action"]
            self.prompt_source = "memory"

        gate_state = "NO_SIGN"
        for w in self.signs:
            p = w.pos(frame, t_rel)
            if p < 0:
                continue
            if w.state in ("DECIDED", "SUPPRESSED") and p > 0:
                continue

            if not w.counted:
                w.counted = True
                self.counterfactual_calls += 1    # always-invoke pays here

            if w.state == "IDLE":
                if hit:
                    w.state = "SUPPRESSED"
                    self._emit(frame, "GATE_SUPPRESSED",
                               f"AR gate arming point reached but necessity is "
                               f"ZERO — memory already resolved "
                               f"'{goal}' -> {hit['action'].upper()} "
                               f"{hit['lead_m']:.0f} m ago via "
                               f"{hit['source']}; no VLM call", "major",
                               goal=goal, action=hit["action"],
                               source=hit["source"])
                else:
                    w.state = "ARMED"
                    self._emit(frame, "GATE_ARMED",
                               "sign evidence accumulating — gate ARMED "
                               "(necessity: pending, memory: no hit)", "info")

            elif w.state == "ARMED":
                w.state = "PENDING"
            elif w.state == "PENDING" and hit:
                w.state = "SUPPRESSED"
                self._emit(frame, "GATE_CALL_CANCELLED",
                           f"memory resolved '{goal}' while gate was pending "
                           f"— VLM call cancelled before dispatch", "major",
                           goal=goal, action=hit["action"])

            if w.state in ("ARMED", "PENDING") and p > 0:
                w.state = "DECIDED"
                self.vlm_calls += 1
                self.standing_prompt = w.decision
                self.prompt_source = "vlm"
                self._emit(frame, "VLM_CALL",
                           f"sufficiency reached -> VLM CALL #{self.vlm_calls} "
                           f"dispatched; answer for '{goal}': "
                           f"{str(w.decision).upper()} (plate has "
                           f"{len(w.directory)} entries)", "major",
                           goal=goal, decision=w.decision,
                           n_entries=len(w.directory), conf=w.conf)
                self.rt.provide_directory(w.directory, goal=goal,
                                          decision=w.decision, conf=w.conf,
                                          frame=frame)

            gate_state = w.state
            break

        self.state = gate_state
        return gate_state

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        reuse = sum(1 for w in self.signs if w.state == "SUPPRESSED")
        return {
            "vlm_calls": self.vlm_calls,
            "counterfactual_always_invoke": self.counterfactual_calls,
            "calls_saved": self.counterfactual_calls - self.vlm_calls,
            "reuse_rate": (reuse / max(self.counterfactual_calls, 1)),
        }
