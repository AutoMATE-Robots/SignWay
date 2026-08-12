"""End-to-end drive-loop test on the FakeBridge with a mock policy — no Habitat, no server, no
GPU. Proves the loop reaches the goal, the safety veto engages around an obstacle, and the log is
valid trip-viewer JSON-lines.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from common.config import Params
from c5_orchestrator.fsm import FSM
from common.geometry import wp_robot_to_world
from common.types import FSMState, Pose
from c2_action.mock_policy import MockPolicy
from c4_safety.safety_occupancy import SafetyOccupancy
from c1_simulator.fake_bridge import FakeBridge


def _drive(bridge, goal, max_steps=200, exec_steps=4):
    fsm = FSM(MockPolicy(), SafetyOccupancy(), Params())
    fsm.set_subgoal(goal)
    obs = bridge.reset()
    for _ in range(max_steps):
        cmd = fsm.tick(obs)
        if fsm.bb.state in (FSMState.ARRIVED, FSMState.FAILED):
            return fsm.bb.state, obs
        if cmd.kind == "waypoints" and cmd.waypoints is not None and len(cmd.waypoints):
            sub = np.asarray(cmd.waypoints)[:exec_steps]
            tw = wp_robot_to_world(sub, obs.robot_pose)[-1]
            yaw = float(np.arctan2(tw[1] - obs.robot_pose.y, tw[0] - obs.robot_pose.x))
            obs = bridge.move_to(Pose(float(tw[0]), float(tw[1]), yaw))
        else:
            obs = bridge.get_obs()
    return fsm.bb.state, obs


def test_reaches_goal_open_corridor():
    state, obs = _drive(FakeBridge(), Pose(5.0, 0.0, 0.0))
    assert state == FSMState.ARRIVED
    assert abs(obs.robot_pose.x - 5.0) < 0.6


def test_obstacle_engages_safety():
    # obstacle spanning the whole corridor -> no way past -> should HALT/FAILED, not clip through
    bridge = FakeBridge(obstacles=[(2.0, 2.4, -1.0, 1.0)])
    state, obs = _drive(bridge, Pose(5.0, 0.0, 0.0), max_steps=80)
    assert state != FSMState.ARRIVED         # cannot reach the goal through a full wall
    assert obs.robot_pose.x < 2.2            # stopped before the wall, didn't pass through


def test_run_sim_cli_writes_valid_log(tmp_path):
    out = tmp_path / "sim_out"
    log = out / "run.jsonl"
    r = subprocess.run(
        [sys.executable, "-m", "c1_simulator.run_sim", "--bridge", "fake", "--policy", "mock",
         "--goal", "5", "0", "0", "--max-steps", "120", "--out", str(out), "--log", str(log)],
        cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert lines[0]["type"] == "meta"
    assert "world_bounds" in lines[0] and "goal_world" in lines[0]
    frames = [d for d in lines if "omni" in d]
    assert len(frames) > 3
    assert "ARRIVED" in r.stdout
