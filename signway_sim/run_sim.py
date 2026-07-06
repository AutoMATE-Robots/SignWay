"""run_sim.py — the Phase 1 closed loop.

Each tick: bridge renders an observation -> FSM ticks (policy proposes a chunk, safety vetoes/
replans against occupancy) -> execute the first few waypoints -> bridge steps -> repeat, until the
robot reaches the pose goal (ARRIVED), gives up (HALT/FAILED), or hits --max-steps. Logs to a
JSON-lines file in the trip-viewer format, so every run replays in signway_tools' HTML viewer.

Runs in the `habitat` env. The policy is the OmniVLA client (server in the omnivla env) or a mock.

    # fake corridor + mock policy — tests the loop with nothing else running:
    python -m signway_sim.run_sim --bridge fake --policy mock \
        --goal 6 0 0 --max-steps 120 --out sim_out --log sim_out/run.jsonl

    # Habitat + real Omni (server must be listening on --port):
    python -m signway_sim.run_sim --bridge habitat --scene <scene.glb> \
        --policy omni --port 5556 --goal 6 0 0 --out sim_out --log sim_out/run.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from signway_core.config import Params
from signway_core.fsm import FSM
from signway_core.geometry import wp_robot_to_world
from signway_core.types import FSMState, Pose


def build_bridge(args):
    if args.bridge == "fake":
        from signway_sim.fake_bridge import FakeBridge
        obstacles = [(2.0, 2.5, -0.5, 0.5)] if args.obstacle else []
        return FakeBridge(obstacles=obstacles)
    from signway_sim.habitat_bridge import HabitatBridge
    return HabitatBridge(scene=args.scene, width=args.width, height=args.height,
                         hfov=args.hfov, sensor_height=args.sensor_height)


def build_policy(args):
    if args.policy == "mock":
        from signway_backends.mocks import MockPolicy
        return MockPolicy()
    from signway_backends.policy_omnivla import OmniVLAClient
    return OmniVLAClient(host=args.host, port=args.port)


def heading(a: Pose, tx: float, ty: float) -> float:
    dx, dy = tx - a.x, ty - a.y
    return float(np.arctan2(dy, dx)) if np.hypot(dx, dy) > 1e-3 else a.yaw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bridge", choices=["fake", "habitat"], default="fake")
    ap.add_argument("--policy", choices=["mock", "omni"], default="mock")
    ap.add_argument("--scene", default=None, help="Habitat scene .glb (habitat bridge)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--goal", type=float, nargs=3, default=[6.0, 0.0, 0.0],
                    metavar=("X", "Y", "YAW"))
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--exec-steps", type=int, default=4, help="waypoints to execute per tick")
    ap.add_argument("--obstacle", action="store_true", help="(fake bridge) add a corridor obstacle")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--sensor-height", type=float, default=0.88)
    ap.add_argument("--out", default="sim_out")
    ap.add_argument("--log", default=None)
    ap.add_argument("--save-frames", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bridge = build_bridge(args)
    policy = build_policy(args)
    from signway_backends.safety_occupancy import SafetyOccupancy
    safety = SafetyOccupancy()
    fsm = FSM(policy, safety, Params())
    goal = Pose(*args.goal)
    fsm.set_subgoal(goal)

    obs = bridge.reset()

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
    reached = None
    for k in range(args.max_steps):
        cmd = fsm.tick(obs)

        # ---- log this frame ----
        frame_name = f"frame_{k:04d}.png"
        if args.save_frames and obs.images:
            Image.fromarray(np.asarray(obs.images[-1]).astype(np.uint8)).save(
                os.path.join(args.out, frame_name))
        if logf:
            from signway_core.geometry import world_goal_to_robot
            gr = world_goal_to_robot(goal, obs.robot_pose)
            omni = cmd.meta.get("omni")
            refined = cmd.meta.get("refined")
            rec = {"i": k, "ts": k, "frame": frame_name,
                   "pose": [obs.robot_pose.x, obs.robot_pose.y, obs.robot_pose.yaw],
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
            sub = np.asarray(cmd.waypoints)[:args.exec_steps]
            tw = wp_robot_to_world(sub, obs.robot_pose)[-1]
            tgt = Pose(float(tw[0]), float(tw[1]), heading(obs.robot_pose, tw[0], tw[1]))
            obs = bridge.move_to(tgt)
        else:
            obs = bridge.get_obs()   # HALT/stop: hold and re-observe (obstacle may clear)

        if k % 10 == 0:
            print(f"  step {k}: state={fsm.bb.state.value} pose="
                  f"({obs.robot_pose.x:.2f},{obs.robot_pose.y:.2f},{obs.robot_pose.yaw:.2f})")

    if logf:
        logf.close()
        print(f"[run_sim] log: {args.log}")
    if reached is None:
        print(f"[run_sim] hit max-steps ({args.max_steps}) in state {fsm.bb.state.value}")
    bridge.close()


if __name__ == "__main__":
    main()
