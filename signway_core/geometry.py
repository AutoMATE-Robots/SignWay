"""SE(2) helpers. Robot frame: x forward, y left, yaw CCW (world uses the same convention)."""
from __future__ import annotations

import numpy as np

from .types import Pose


def world_goal_to_robot(goal: Pose, robot: Pose, max_range: float = 30.0):
    """Express a world-frame goal pose in the robot's frame as (forward, left, dtheta).
    Clipped to max_range to stay in OmniVLA's training distribution."""
    dx, dy = goal.x - robot.x, goal.y - robot.y
    c, s = np.cos(robot.yaw), np.sin(robot.yaw)
    fwd = dx * c + dy * s
    left = -dx * s + dy * c
    r = float(np.hypot(fwd, left))
    if r > max_range:
        fwd *= max_range / r
        left *= max_range / r
    dtheta = float(np.arctan2(np.sin(goal.yaw - robot.yaw), np.cos(goal.yaw - robot.yaw)))
    return (float(fwd), float(left), dtheta)


def wp_robot_to_world(wp: np.ndarray, robot: Pose) -> np.ndarray:
    """Transform a robot-frame waypoint chunk (N,2) into world coordinates."""
    c, s = np.cos(robot.yaw), np.sin(robot.yaw)
    wp = np.asarray(wp, float).reshape(-1, 2)
    wx = robot.x + wp[:, 0] * c - wp[:, 1] * s
    wy = robot.y + wp[:, 0] * s + wp[:, 1] * c
    return np.column_stack([wx, wy])


def dist(a: Pose, b: Pose) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def pose_ahead(robot: Pose, d_forward: float, dyaw: float = 0.0) -> Pose:
    """A pose d_forward metres ahead of `robot` after rotating heading by dyaw. Turns a
    continue/turn decision into a metric subgoal anchored to the recent pose (spec §7)."""
    yaw = robot.yaw + dyaw
    return Pose(robot.x + d_forward * np.cos(yaw), robot.y + d_forward * np.sin(yaw), yaw)
