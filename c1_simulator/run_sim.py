"""run_sim.py — the Phase 1 closed loop.

Each tick: bridge renders an observation -> FSM ticks (policy proposes a chunk, safety vetoes/
replans against occupancy) -> execute the first few waypoints -> bridge steps -> repeat, until the
robot reaches the pose goal (ARRIVED), gives up (HALT/FAILED), or hits --max-steps. Logs to a
JSON-lines file in the trip-viewer format, so every run replays in the tools/ HTML viewer.

Runs in the `habitat` env. The policy is the OmniVLA client (server in the omnivla env) or a mock.

    # fake corridor + mock policy — tests the loop with nothing else running:
    python -m c1_simulator.run_sim --bridge fake --policy mock \
        --goal 6 0 0 --max-steps 120 --out sim_out --log sim_out/run.jsonl

    # Habitat + real Omni (server must be listening on --port):
    python -m c1_simulator.run_sim --bridge habitat --scene <scene.glb> \
        --policy omni --port 5556 --goal 6 0 0 --out sim_out --log sim_out/run.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import numpy as np

from common.config import Params
from c5_orchestrator.fsm import FSM
from c2_action.controller import DT, step_unicycle, waypoints_to_velocity
from common.geometry import world_goal_to_robot
from common.types import FSMState, Pose


def build_bridge(args):
    if args.bridge == "fake":
        from c1_simulator.fake_bridge import FakeBridge
        obstacles = [(2.0, 2.5, -0.5, 0.5)] if args.obstacle else []
        return FakeBridge(obstacles=obstacles)
    if args.bridge == "isaac":
        from c1_simulator.isaac_bridge import IsaacBridge
        return IsaacBridge(scene=args.scene, width=args.width, height=args.height,
                           sensor_height=args.sensor_height, assets_root=args.assets_root,
                           grid_range=args.grid_range)
    from c1_simulator.habitat_bridge import HabitatBridge
    return HabitatBridge(scene=args.scene, width=args.width, height=args.height,
                         hfov=args.hfov, sensor_height=args.sensor_height)


def build_policy(args):
    if args.policy == "mock":
        from c2_action.mock_policy import MockPolicy
        return MockPolicy()
    from c2_action.policy_omnivla import OmniVLAClient
    return OmniVLAClient(host=args.host, port=args.port)


# The robot's heading used to be invented here ("face the next waypoint") and then capped with
# a hand-tuned --max-turn. Both are gone: OmniVLA emits [dx, dy, hx, hy] and SignNav's own PD
# controller turns that into (v, w), with the turn limit being the model's maxw. See
# c2_action/controller.py.


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bridge", choices=["fake", "habitat", "isaac"], default="fake")
    ap.add_argument("--policy", choices=["mock", "omni"], default="mock")
    ap.add_argument("--scene", default=None, help="Habitat scene .glb (habitat bridge)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--goal", type=float, nargs=3, default=[6.0, 0.0, 0.0],
                    metavar=("X", "Y", "YAW"))
    ap.add_argument("--start", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "YAW"), help="where the robot spawns")
    ap.add_argument("--grid-range", type=float, default=6.0,
                    help="how far the occupancy grid sees (m)")
    ap.add_argument("--sign", action="append", nargs=5, default=None,
                    metavar=("LABEL", "X", "Y", "Z", "YAW_DEG"),
                    help="hang a readable sign, e.g. --sign 'cafeteria left' -4 -4 1.6 180. "
                         "Repeatable. YAW_DEG is the direction the sign FACES.")
    ap.add_argument("--sign-size", type=float, nargs=2, default=[0.8, 0.4],
                    metavar=("W", "H"), help="sign board size in metres")
    ap.add_argument("--sign-visible-from", type=float, default=4.0,
                    help="(mock detector only) how close before a sign is 'seen'")
    ap.add_argument("--detector", choices=["mock", "oracle"], default="mock",
                    help="mock = fakes detection from pose; oracle = projects the sign into the "
                         "image and MEASURES readability from real pixels")
    ap.add_argument("--reasoner", choices=["mock", "gemini"], default="mock",
                    help="mock = label lookup table; gemini = a real VLM reading the crop")
    ap.add_argument("--vlm-model", default="gemini-3.1-flash-lite")
    ap.add_argument("--min-text-px", type=float, default=22.0,
                    help="readability gate: text shorter than this cannot be read by anything")
    ap.add_argument("--min-blur-var", type=float, default=20.0,
                    help="readability gate: CAMERA-SPECIFIC, calibrate it")
    # Orchestrator knobs — the ablation axes. These were dataclass defaults, unreachable from
    # the command line, which made the thing the paper is about the one thing you could not vary.
    ap.add_argument("--d-look", type=float, default=2.5,
                    help="how far ahead/aside a decision places the subgoal (m)")
    ap.add_argument("--d-arrive", type=float, default=1.0,
                    help="radius counting as 'reached' (m); 1.0 matches the protocol the field "
                         "uses for OmniVLA, which is not trained to terminate at goals")
    ap.add_argument("--leg-max-m", type=float, default=5.0,
                    help="travel a decision is worth before sign reading re-arms (m)")
    ap.add_argument("--tau-conf", type=float, default=0.5,
                    help="minimum detection confidence to act on")
    ap.add_argument("--mission", default="cafeteria", help="where the robot is trying to go")
    ap.add_argument("--sub-steps", type=int, default=1,
                    help="render this many frames between decisions — smooth video without "
                         "changing the control rate (real robots do exactly this: think slowly, "
                         "look often)")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--obstacle", action="store_true", help="(fake bridge) add a corridor obstacle")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--sensor-height", type=float, default=0.88)
    ap.add_argument("--assets-root", default=None,
                    help="local Isaac asset packs root, e.g. .../isaacsim_assets/Assets/Isaac/5.1")
    ap.add_argument("--out", default="sim_out")
    ap.add_argument("--log", default=None)
    ap.add_argument("--save-frames", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bridge = build_bridge(args)
    policy = build_policy(args)
    from c4_safety.safety_occupancy import SafetyOccupancy
    safety = SafetyOccupancy()

    # ---- component 3: signs ----
    detector = reasoner = None
    if args.sign:
        HAZARD = ("stop", "danger", "no entry", "staff only")
        specs, oracle_specs = [], []
        for label, sx, sy, sz, syaw in args.sign:
            specs.append((float(sx), float(sy), label,
                          any(h in label.lower() for h in HAZARD)))
            oracle_specs.append((label, float(sx), float(sy), float(sz), float(syaw),
                                 args.sign_size[0], args.sign_size[1]))
            if hasattr(bridge, "add_sign"):
                bridge.add_sign(label, (float(sx), float(sy), float(sz)), float(syaw), args.out,
                                width=args.sign_size[0], height=args.sign_size[1])

        if args.detector == "oracle":
            from c1_simulator.sign_oracle import ProjectedSignBackend
            from c3_reasoning.detector import TextSignDetector
            detector = TextSignDetector(
                ProjectedSignBackend(oracle_specs, bridge.intrinsics(), args.sensor_height),
                K=bridge.intrinsics(), min_height_px=args.min_text_px,
                min_blur_var=args.min_blur_var)
        else:
            from c3_reasoning.mock_reasoning import MockDetector
            detector = MockDetector(signs=specs, visible_from=args.sign_visible_from)

        if args.reasoner == "gemini":
            from c3_reasoning.reasoner_vlm import GeminiReasoner
            reasoner = GeminiReasoner(model=args.vlm_model)
        else:
            from c3_reasoning.mock_reasoning import MockReasoner
            reasoner = MockReasoner()
        print(f"[run_sim] {len(specs)} sign(s) | detector={args.detector} "
              f"reasoner={args.reasoner} | mission='{args.mission}'", flush=True)

    cfg = Params(d_look=args.d_look, d_arrive=args.d_arrive,
                 leg_max_m=args.leg_max_m, tau_conf=args.tau_conf)
    print(f"[run_sim] d_look={cfg.d_look} d_arrive={cfg.d_arrive} "
          f"leg_max_m={cfg.leg_max_m} tau_conf={cfg.tau_conf}", flush=True)
    fsm = FSM(policy, safety, cfg, mission_goal=args.mission,
              detector=detector, reasoner=reasoner)
    goal = Pose(*args.goal)
    fsm.set_subgoal(goal)

    obs = bridge.reset(Pose(*args.start) if args.start else None)
    # Real sim time. The stall trigger asks "how far have I moved in the last t_stall SECONDS",
    # which is meaningless if every observation is stamped 0.0 (Isaac) or counts ticks (fake).
    # Each control cycle is exactly DT, so time is now honest and the trigger works.
    t_sim = [0.0]

    def stamped(o):
        return dataclasses.replace(o, stamp=t_sim[0])

    obs = stamped(obs)

    # log meta (trip-viewer format)
    logf = open(args.log, "w") if args.log else None
    if logf:
        margin = 2.0
        xs = [obs.robot_pose.x, goal.x]
        ys = [obs.robot_pose.y, goal.y]
        meta = {"type": "meta",
                "goal_world": [goal.x, goal.y, goal.yaw],
                "grid": {"res": 0.05, "x_max": 4.0, "y_half": 2.0},
                "world_bounds": [min(xs) - margin, max(xs) + margin,
                                 min(ys) - margin, max(ys) + margin],
                "path": []}
        logf.write(json.dumps(meta) + "\n")

    from PIL import Image
    frame_i = [0]

    def save_frame(o):
        """Every render gets a frame. Decisions happen every --sub-steps frames."""
        if not (args.save_frames and o.images):
            return None
        name = f"frame_{frame_i[0]:04d}.png"
        Image.fromarray(np.asarray(o.images[-1]).astype(np.uint8)).save(
            os.path.join(args.out, name))
        frame_i[0] += 1
        return name

    frame_name = save_frame(obs)
    reached = None
    last_vw = (0.0, 0.0)
    for k in range(args.max_steps):
        prev_obs = obs                 # the observation this tick's decision was made from
        if detector is not None and hasattr(detector, "set_pose"):
            try:
                detector.set_pose(obs.robot_pose)                     # oracle backend
            except TypeError:
                detector.set_pose(obs.robot_pose.x, obs.robot_pose.y, obs.robot_pose.yaw)
        if detector is not None and hasattr(detector, "set_depth") and obs.occupancy is not None:
            pass                                                       # depth wired via bridge
        cmd = fsm.tick(obs)
        if logf:
            gr = world_goal_to_robot(goal, obs.robot_pose)
            omni = cmd.meta.get("omni")
            refined = cmd.meta.get("refined")
            sg = fsm.bb.subgoal.pose if fsm.bb.subgoal else None
            dec = fsm.bb.last_decision
            # The gate's evidence for THIS frame — why we did or did not pay for the VLM.
            # This is the paper's raw data, not debug output.
            gate = None
            if detector is not None and getattr(detector, "last", None):
                det0, r0 = detector.last[0]
                gate = r0.as_dict()
            # vlm = the most recent call, re-logged every frame so the overlay can show the
            # decision the robot is currently acting on. It does NOT mean a call happened this
            # frame — vlm_calls is the running total, and vlm_fired marks the frame that paid.
            vlm = None
            vlm_calls = getattr(reasoner, "calls", 0) if reasoner is not None else 0
            if reasoner is not None and getattr(reasoner, "log", None):
                vlm = dict(reasoner.log[-1])
                vlm["call_index"] = vlm_calls
            rec = {"i": k, "ts": round(t_sim[0], 3), "frame": frame_name,
                   "gate": gate, "vlm": vlm,
                   "vlm_calls": vlm_calls,                  # running total: THE metric
                   "vlm_fired": bool(cmd.meta.get("sign")),  # did THIS frame pay for a call?
                   "pose": [obs.robot_pose.x, obs.robot_pose.y, obs.robot_pose.yaw],
                   "state": fsm.bb.state.value,
                   "subgoal": ([sg.x, sg.y, sg.yaw] if sg else None),
                   "decision": (dec.type.value if dec else None),
                   "sign": cmd.meta.get("sign"),
                   "goal_robot": [gr[0], gr[1], gr[2]],
                   "omni": (np.asarray(omni).round(3).tolist() if omni is not None else []),
                   "occ_cells": (obs.occupancy.cells if obs.occupancy else []),
                   "refined": (None if refined is None else np.asarray(refined).round(3).tolist()),
                   "used_policy": bool(cmd.meta.get("used_policy", True))}
            logf.write(json.dumps(rec) + "\n")

        # ---- terminal states ----
        if fsm.bb.state in (FSMState.ARRIVED, FSMState.FAILED):
            reached = fsm.bb.state.value
            print(f"[run_sim] step {k}: {reached}")
            break

        # ---- execute ----
        if cmd.kind == "waypoints" and cmd.waypoints is not None and len(cmd.waypoints):
            # OmniVLA's own control law: chunk -> (linear_vel, angular_vel), then integrate the
            # unicycle for one control period. No invented heading, no hand-tuned turn cap.
            v, w = waypoints_to_velocity(cmd.waypoints)
            last_vw = (v, w)
            p = obs.robot_pose
            n = max(1, args.sub_steps)
            for _ in range(n):            # same motion, just rendered n times for smooth video
                x, y, yaw = step_unicycle(p.x, p.y, p.yaw, v, w, DT / n)
                p = Pose(x, y, yaw)
                t_sim[0] += DT / n
                obs = stamped(bridge.move_to(p))
                frame_name = save_frame(obs)
        elif cmd.kind == "maneuver":
            # Pivot in place. The harness has to actually DO this — until now maneuver commands
            # fell through to "re-observe", so the robot could never turn at a junction.
            dyaw = {"turn_left": np.pi / 2, "turn_right": -np.pi / 2,
                    "u_turn": np.pi}.get(cmd.maneuver, 0.0)
            p0 = obs.robot_pose
            n = max(1, args.sub_steps * 2)      # pivots are quick; render enough to see them
            for j in range(1, n + 1):
                t_sim[0] += DT / n
                obs = stamped(bridge.move_to(Pose(p0.x, p0.y, p0.yaw + dyaw * j / n)))
                frame_name = save_frame(obs)
        else:
            t_sim[0] += DT
            obs = stamped(bridge.get_obs())   # HALT/stop: hold, re-observe (obstacle may clear)
            frame_name = save_frame(obs)

        # ---- per-step component trace: who did what, this tick ----
        _p = prev_obs.robot_pose
        _occ = len(prev_obs.occupancy.cells) if prev_obs.occupancy else 0
        _active = fsm.bb.subgoal.pose if fsm.bb.subgoal else goal
        _gr = world_goal_to_robot(_active, _p)
        _omni = cmd.meta.get("omni")
        if _omni is not None and len(_omni):
            _v, _w = waypoints_to_velocity(np.asarray(_omni))
            _act = f"v={_v:.2f} w={_w:+.2f}"
        else:
            _act = "-- none --"
        _used = cmd.meta.get("used_policy", True)
        _safe = "pass  " if _used else "REPLAN"
        if cmd.kind != "waypoints":
            _safe = "hold  "
        print(f"step {k:2d} | c1 sim pose=({_p.x:+.2f},{_p.y:+.2f},{_p.yaw:+.2f}) occ={_occ:4d} "
              f"| c5 fsm {fsm.bb.state.value:8s} | c2 {args.policy:5s} {_act} "
              f"| c4 safety {_safe} | subgoal fwd={_gr[0]:.2f} left={_gr[1]:+.2f}", flush=True)

    if logf:
        logf.close()
        print(f"[run_sim] log: {args.log}")
    if reached is None:
        print(f"[run_sim] hit max-steps ({args.max_steps}) in state {fsm.bb.state.value}")
    bridge.close()


if __name__ == "__main__":
    main()