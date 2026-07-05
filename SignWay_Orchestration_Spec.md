# SignNav — Orchestration Spec (Phase 0)

**Purpose.** Define how the VLA (action) and VLM (reasoning) are synchronized during
autonomous indoor sign-guided navigation: when reasoning starts, stops, and how; how the
nodes communicate; and how every edge case is handled. This document is the contract the
whole team builds against, and its scenario list doubles as the benchmark definition.

**Scope.** Covers the runtime system (nodes, state machine, messages, triggers, edge cases)
and the sim needed to test it end-to-end. Does not cover model training or the OmniVLA
internals (those are settled elsewhere).

---

## 1. Core model: three loops sharing one state

The system is **not** "two models talking." It is three control loops at three rates, where
the fast loop gates the slow ones, all coordinated through one arbiter.

| Loop | Rate | Runs | Blocks on heavy model? |
|---|---|---|---|
| **Safety + detection** | per-frame (~10–30 Hz) | occupancy/collision veto + cheap sign detector (GroundingDINO) | never |
| **Action (VLA)** | ~0.5 Hz (OmniVLA chunk) | Omni emits 8 waypoints toward the current subgoal; controller executes | no |
| **Reasoning (VLM)** | on-trigger only, rare | reads signs / resolves decisions; **output is a new subgoal or maneuver** | this *is* the heavy model |

The single most important design rule: **the VLM never commands motors.** It only writes the
subgoal that the VLA is already chasing. That one sentence is the communication design.

Second rule: **the VLA + safety layer must run correctly with the VLM absent.** Reasoning is a
selective add-on, never a dependency in the hot path. This is both good engineering and a
restatement of the selective-reasoning thesis (Objective #3).

---

## 2. Nodes and topics (ROS2)

Five nodes. The **orchestrator owns the shared state** (the "blackboard") as its own internal
memory, updated by subscription callbacks — no separate blackboard process (keeps it simple).

| Node | Subscribes | Publishes / Serves | Notes |
|---|---|---|---|
| `perception` | `/cam/image_raw` | `/detections` | GDINO (or lighter) sign detector; always on |
| `safety` | `/cam/depth` or `/scan`, `/odom` | `/occupancy` | builds costmap; sim uses GT depth |
| `policy` (VLA) | `/subgoal`, `/cam/image_raw` | `/vla/waypoints` | OmniVLA; can run off-board as a server |
| `reasoner` (VLM) | — | **action server** `reason` | request/response, **cancelable** (matters for async + preemption) |
| `orchestrator` | `/detections`, `/occupancy`, `/odom`, `/vla/waypoints` | `/subgoal`, `/cmd` (via controller), calls `reason` | the state machine; single arbiter |

**Why the VLM is a ROS2 action, not a topic:** a reasoning call is a discrete request that
returns one result and may need to be **canceled** (preemption by a hazard, or the async gate
deciding the window passed). Actions give cancel/feedback semantics for free; topics don't.

**Controller.** A thin low-level node (or a function inside the orchestrator) turns either
(a) a VLA waypoint chunk or (b) a discrete maneuver into `/cmd` velocities. It also enforces
the safety veto as a last line (stop if `/occupancy` says the immediate cell ahead is blocked).

---

## 3. The blackboard (orchestrator internal state)

```
mission_goal      : str            # high-level, e.g. "cafeteria" (from operator)
robot_pose        : (x, y, yaw)    # latest odom, world SE(2)
subgoal_pose      : (x, y, yaw)    # what the VLA pursues; anchored to RECENT pose, re-anchored often
occupancy         : Grid + stamp   # latest costmap
detections        : [Detection]    # latest sign detections + stamp
last_decision      : Decision|None  # last committed VLM decision
committed_signs   : set[SignKey]   # sign anchors already acted on (debounce)
state             : FSMState        # current state (Section 5)
reason_budget     : int            # VLM calls remaining this mission (safety cap)
leg               : Leg|None        # current committed segment (debounce + logging)
```

**Anchoring rule (localization drift):** `subgoal_pose` is always set relative to the
**recent** `robot_pose`, never carried as one far global goal over 20 m. Re-anchor every time a
new decision commits. This is why odometry drift over a long horizon does not accumulate into
the goal.

---

## 4. Message schemas

Minimal, stable. These are the interfaces; do not add fields without updating this doc.

**Detection** (`/detections`, list per frame)
```json
{"id": "sign_0007", "label": "elevator closed", "conf": 0.82,
 "bbox": [x,y,w,h], "est_distance_m": 2.6, "est_bearing_rad": -0.15}
```
`est_distance_m` is coarse (bbox size heuristic or depth lookup); used only for the *readable*
threshold, not for control.

**Occupancy** (`/occupancy`)
```json
{"stamp": 172..., "res": 0.05, "x_max": 4.0, "y_half": 2.0,
 "cells": [[r,c], ...]}          # robot-centric; from occupancy_replan.Grid
```

**VLA waypoints** (`/vla/waypoints`)
```json
{"stamp": 172..., "wp": [[dx,dy], ... x8]}   # robot frame, x fwd / y left, meters
```

**Reason request** (action goal → `reasoner`)
```json
{"image_ref": "...", "sign_crop": [x,y,w,h]|null,
 "mission_goal": "cafeteria", "context": "at junction, last turn: left 12s ago"}
```

**Decision** (action result ← `reasoner`) — the structured output the VLM must return:
```json
{"type": "continue|turn_left|turn_right|u_turn|goto|stop",
 "target": {"bearing_rad": 1.57, "distance_m": 2.5} | null,
 "rationale": "sign says cafeteria is left",
 "confidence": 0.0}
```
`type` maps to an action (Section 7). `target` is optional guidance for `goto`. `rationale` and
`confidence` are for logging + the misread/hallucination guard.

**Subgoal** (`/subgoal`) — what the orchestrator hands the VLA:
```json
{"stamp": 172..., "pose": [x, y, yaw]}   # anchored to recent robot_pose
```

---

## 5. The state machine

Six states. Tight on purpose.

| State | What runs | Meaning |
|---|---|---|
| `DRIVE` | VLA → controller, safety veto, detector scanning | default; pursuing subgoal, no VLM |
| `REASON` | VLM action in flight | a trigger fired; deciding next subgoal/maneuver |
| `MANEUVER` | controller executes a discrete action, VLA bypassed | for U-turns / turn-in-place Omni can't do well |
| `HALT` | zero velocity | safety stop (no safe path) or hard stop sign; recoverable |
| `ARRIVED` | zero velocity | mission goal reached (terminal-ish) |
| `FAILED` | zero velocity | gave up: stuck with no options, or reason_budget exhausted (terminal) |

### 5.1 Triggers (evaluated in `DRIVE` every tick, in priority order)

Priority resolves preemption (a hazard beats a directory read).

| # | Trigger | Condition | Fires transition |
|---|---|---|---|
| T6 | `SAFETY` | replanner returns **no** collision-free path to subgoal | → `HALT` |
| T5 | `HAZARD` | detection with hazard label, `conf ≥ τ_conf` | → `REASON` (or → `HALT` if a hard "stop") |
| T1 | `SIGN_READABLE` | detection `conf ≥ τ_conf` **and** `est_distance ≤ d_read` **and** sign not in `committed_signs` | → `REASON` |
| T2 | `DECISION_POINT` | ≥2 free-space openings ahead (from occupancy) **and** current subgoal doesn't resolve which | → `REASON` |
| T3 | `SUBGOAL_REACHED` | `dist(robot, subgoal) < d_arrive` | → `REASON` for next, or → `ARRIVED` if that was the mission goal |
| T4 | `STALLED` | forward progress `< ε_progress` over `T_stall` seconds | → `REASON` (recovery) |

If none fire, stay in `DRIVE`. If `reason_budget == 0` and a reason trigger fires, go to the
**degraded default** (Section 8), not `REASON`.

### 5.2 Events (from nodes, drive transitions out of REASON/MANEUVER)

| Event | From | To | Action |
|---|---|---|---|
| `DECISION(continue)` | REASON | DRIVE | keep/refresh straight-ahead subgoal |
| `DECISION(goto/turn_*)` | REASON | DRIVE or MANEUVER | set new subgoal (Section 7); U-turn/turn-in-place → MANEUVER |
| `DECISION(stop)` | REASON | HALT | stop; wait for operator or re-trigger |
| `VLM_TIMEOUT` | REASON | per policy | degraded default (Section 8) |
| `MANEUVER_DONE` | MANEUVER | DRIVE | resume VLA on the post-maneuver subgoal |
| `PATH_CLEARED` | HALT | DRIVE | safe path reappeared (dynamic obstacle moved) |

### 5.3 Orchestrator tick (pseudocode)

```python
def tick():
    update_blackboard()                      # from latest callbacks
    if stale(occupancy) or stale(detections):
        controller.slow_or_hold()            # never act on stale safety data

    if state == DRIVE:
        for trig in [T6, T5, T1, T2, T3, T4]:   # priority order
            if trig.fires(bb):
                return handle_trigger(trig)
        wp = bb.vla_waypoints
        safe, refined = safety.refine(bb.occupancy, wp)   # occupancy veto/replan
        controller.execute(refined if not safe else wp)

    elif state == REASON:
        if MODE == "async":
            drive_last_safe_goal_within_gate()   # Section 6
        # else stop-and-reason: controller already holding
        if reason_action.done():
            apply_decision(reason_action.result())
        elif reason_action.deadline_passed():    # async only
            controller.hold()                    # reached commit point, must wait

    elif state == MANEUVER:
        if controller.maneuver_done(): set_state(DRIVE)

    elif state in (HALT,):
        if safety.path_exists(bb): set_state(DRIVE)
```

---

## 6. Stop-and-reason vs. async (the two modes)

Both modes share the state machine. The difference is what the robot does **while REASON is in
flight**. Build stop-and-reason first (it is a complete result); async is the stronger second
experiment.

**Mode A — stop-and-reason (baseline, Phase 2).** On a reason trigger, controller commands zero
velocity, enter `REASON`, wait for the decision, act. Simple and safe. Cost: full VLM latency
added to travel time, robot stationary during it. This already delivers Objective #3 (reasoning
only fires on triggers, not every frame).

**Mode B — async / reason-while-driving (the contribution, Phase 3).** On a reason trigger,
enter `REASON` but **keep executing the last committed safe subgoal**, gated by a **commit
deadline**:

- `commit_deadline` = the point (distance/time) past which a decision becomes necessary — e.g.,
  the detected junction location, or where the current safe corridor ends. Computed from the
  triggering detection's `est_distance` and current speed.
- The robot may keep driving **only while** it stays before the commit point and the motion is
  reversible (no committing to a branch). As it nears the commit point without a decision, it
  **decelerates and holds**.
- When the decision returns, apply it. If a higher-priority trigger fires mid-reason (hazard),
  **cancel** the in-flight VLM action (that's why reasoner is a cancelable action) and handle the
  new trigger.

Async never trades safety for latency: the gate guarantees the robot cannot blow through an
unread decision point. The measured win is *time saved by overlapping compute with safe travel*.

---

## 7. Decision → action mapping

How a VLM `type` becomes control. Key point: turns become **pose subgoals** the VLA pursues; the
VLA is not bypassed except for maneuvers it is bad at.

| `type` | Handling |
|---|---|
| `continue` | set subgoal straight ahead at `d_look` from recent pose; state → DRIVE |
| `turn_left` / `turn_right` | **baseline:** rotate recent pose yaw by ±90°, set subgoal `d_look` ahead → DRIVE. **refined:** pick the corresponding free-space opening from occupancy and set the subgoal into it (cleaner, needs opening detection) |
| `goto` | set subgoal from `target` (bearing+distance) relative to recent pose → DRIVE |
| `u_turn` | → MANEUVER: turn 180° in place, then DRIVE (Omni is forward-biased; do not ask it to reverse) |
| `stop` | → HALT |

`d_look` starting value ~2.5 m (within a corridor segment; re-anchored on the next decision).

**Why turns route through a pose subgoal:** it keeps the VLA doing the actual locomotion and
obstacle-hugging, so we get Omni's learned corridor-following instead of open-loop dead reckoning.
Only in-place rotation (u_turn, sharp turn-in-place) bypasses it, because that is exactly what a
forward-biased waypoint policy cannot express.

---

## 8. Debounce, legs, and graceful degradation

**Debounce (don't re-reason the same sign every frame).** When a decision commits, record a
`SignKey` = (odom anchor position, label) in `committed_signs`, and open a **leg** = the segment
traveled under that decision. While the same sign stays in view, `SIGN_READABLE` is suppressed for
it. The leg closes when the subgoal is reached, a new higher-priority trigger fires, or the robot
travels `leg_max_dist`. (This mirrors the commit/leg structure already in the reasoner player.)

**Graceful degradation (VLM slow, down, or budget exhausted).** The safety + VLA loops keep
running regardless. On `VLM_TIMEOUT` or `reason_budget == 0`, apply the **degraded default**,
which is a configurable policy:
- *conservative* (default for demos): `HALT` at an unresolved decision point; keep driving only on
  open corridor with no pending decision.
- *permissive*: continue straight through unresolved junctions on the VLA.

Either way the robot never crashes for lack of reasoning; it just navigates without
sign-informed choices. This is the thesis in code.

---

## 9. Edge cases → handling (this list *is* the benchmark)

The hard cases for "all of autonomous indoor driving" and the benchmark scenarios are the same
set. Enumerate once; use for both the FSM and the eval.

| Edge case | Trigger / state | Handling | Benchmark scenario |
|---|---|---|---|
| Sign readable only ~2 s but VLM ~3 s | T1 + async gate | detect **early** (far); slow near detected-but-unread sign; cache clearest frame and reason on it | warning / short-window |
| Multiple signs in view (directory) | T1 | VLM reasons over all detections, picks the entry matching `mission_goal` | directory |
| Conflicting signs (temp vs permanent) | T1 | VLM arbitrates on recency/authority/relevance (prompt design) | conflicting |
| Maneuver behind robot (U-turn) | DECISION(u_turn) → MANEUVER | in-place 180°, bypass VLA | detour / dead-end |
| Junction, no signage | T2 | reason (goal-directed pick) or degraded default | ambiguous |
| Dead-end / no progress | T4 STALLED | reason for recovery; if none, backtrack via MANEUVER | dead-end |
| Dynamic obstacle (person) | safety loop | occupancy veto + replan; must be temporally fresh (staleness check) | (orthogonal, always-on) |
| Overshoot decision point during latency | async commit deadline | gate holds robot before commit point until decision returns | (async correctness) |
| Misread / hallucinated sign | DECISION.confidence | confidence threshold; optional re-read; benchmark measures error rate | (all sign scenarios) |
| Localization drift over long horizon | anchoring rule (§3) | subgoals relative to recent pose; re-anchor per decision | long-horizon |
| Trigger preemption (hazard mid-reason) | priority order + action cancel | cancel in-flight VLM, handle hazard first | (safety) |
| Stop sign / hard prohibition | T5 HAZARD → HALT | immediate stop, no reasoning needed | prohibition |

---

## 10. Occupancy plan (sim-first, sidesteps calibration)

The real-robot occupancy pain is the uncalibrated camera + monocular scale. **In sim we get
ground-truth depth**, so the occupancy + replanner (already built: `occupancy_replan.py`) get
clean input immediately.

- **Now (sim):** feed Habitat GT depth → `occupancy_from_depth` → `Grid` → `refine()`. Validates
  the safety layer and the whole architecture.
- **Later (real robot):** monocular depth (Depth Anything V2) + camera calibration, or a real
  depth/LiDAR sensor. This is a separate real-robot task.
- **Honest limit:** sim occupancy validates the *architecture*, not the *real monocular pipeline*.
  That's fine — the architecture is what must be proven before the real-robot phase.

---

## 11. Phasing and milestones

Each phase de-risks one thing and always leaves a working pipeline + a number.

| Phase | Build | Milestone (de-risks) |
|---|---|---|
| **0 (this doc)** | orchestration spec: states, triggers, schemas, edge cases | team has one contract; benchmark defined |
| **1** | Habitat + VLA closed loop + sim-depth occupancy/safety; **no VLM, no signs** | robot reaches a pose goal without hitting walls, replanner engaging → sim + occupancy + closed-loop |
| **2** | perception + reasoner + FSM, **stop-and-reason** | robot drives, sees sign, stops, reasons, turns correctly, continues → orchestration + reasoning correctness |
| **3** | **async** with commit-deadline gating | async cuts latency vs. Phase 2 without hurting success; edge cases pass → the contribution |
| **4** | full pipeline across all scenarios | metrics collected → the paper |

**Signs-in-sim fork (decide before Phase 2):** either (a) texture the printed A4 benchmark onto
walls at junctions for full sign-guided eval in sim, or (b) split — sim tests pipeline mechanics +
scale + the async latency win, and a **real printed-sign hallway course** tests sign-reasoning
realism. Given the 11-page A4 printed benchmark already exists, (b) is the natural division and sim
does not need signs for its job. Recommend (b) unless sign-reasoning at sim scale is specifically
required.

---

## 12. Parameters to tune (start here, then calibrate empirically)

| Param | Meaning | Starting value |
|---|---|---|
| `τ_conf` | min sign detection confidence to act | 0.5 |
| `d_read` | max distance a sign is considered readable | 3.5 m |
| `d_arrive` | subgoal-reached distance | 0.5 m |
| `d_look` | subgoal look-ahead distance | 2.5 m |
| `T_stall` / `ε_progress` | stall detection | 3 s / 0.2 m |
| `leg_max_dist` | max travel before a leg auto-closes | 8 m |
| `reason_budget` | max VLM calls per mission (safety cap) | 30 |
| occupancy / detection staleness | max age before "hold" | 0.5 s / 0.3 s |
| VLM latency budget | expected reason time (sets commit deadline) | ~2–3 s (measure) |

---

## 13. Suggested node/file layout

```
signnav/
  orchestrator/    orchestrator_node.py     # the FSM (Section 5) + blackboard
  perception/      detector_node.py         # GDINO sign detector
  reasoner/        reasoner_node.py          # VLM action server (cancelable)
  policy/          vla_node.py               # OmniVLA (wraps omni_backend.py)
  safety/          safety_node.py            # occupancy (wraps occupancy_replan.py)
  controller/      controller_node.py        # waypoints|maneuver -> /cmd, final veto
  sim/             habitat_bridge.py         # publishes sim sensor topics; env-split for Omni
  msgs/            Detection, Occupancy, Decision, Subgoal, VlaWaypoints
```

Existing assets that drop in: `omni_backend.py` (policy), `occupancy_replan.py` (safety),
`odom_replay.py` / `signnav_trip_viewer.html` (offline visualization + logging format), the
reasoner player (leg/commit structure → orchestrator debounce), the A4 printed benchmark (Phase 2/4).

---

## 14. Open decisions (need a call before/within the phase noted)

1. **Signs-in-sim fork** (§11) — before Phase 2. *Recommend (b).*
2. **Junction/opening detection** for T2 and refined turns — Phase 1/2. Start from occupancy
   free-space branching; upgrade to a learned detector only if needed.
3. **VLM model + latency** — Phase 2. Sets the commit-deadline budget; measure real latency early.
4. **Degraded default policy** (conservative vs permissive) — Phase 2. Default conservative.
5. **Odometry source** in sim vs real, and re-anchoring cadence — Phase 1.
6. **Async gate math** (exact commit-deadline from distance/speed) — Phase 3.
```
