"""FakeBridge — a synthetic straight corridor with optional obstacle, for testing the full drive
loop (FSM + policy + safety + logging) with no simulator, no GPU, no scene. The corridor runs
along +world_x with walls at |world_y| = wall_half; obstacles are axis-aligned world boxes.

It builds a real occupancy grid in the robot frame each step (walls + obstacles projected into the
grid), so the safety veto/replanner is genuinely exercised — just without Habitat's renderer.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from signway_core.types import Observation, Occupancy, Pose
from signway_sim.bridge import Bridge


class FakeBridge(Bridge):
    def __init__(self, wall_half: float = 1.0, obstacles: Optional[List[Tuple]] = None,
                 res: float = 0.05, x_max: float = 4.0, y_half: float = 2.0,
                 img_hw: Tuple[int, int] = (120, 160)):
        self.wall_half = wall_half
        self.obstacles = obstacles or []          # list of (x0,x1,y0,y1) world boxes
        self.res, self.x_max, self.y_half = res, x_max, y_half
        self.img_hw = img_hw
        self.pose = Pose(0.0, 0.0, 0.0)
        self.t = 0.0

    # ---- Bridge API ----
    def reset(self, start: Optional[Pose] = None) -> Observation:
        self.pose = start or Pose(0.0, 0.0, 0.0)
        self.t = 0.0
        return self.get_obs()

    def move_to(self, pose: Pose) -> Observation:
        # clamp inside the corridor walls (can't pass through), keep commanded heading
        y = float(np.clip(pose.y, -(self.wall_half - 0.1), self.wall_half - 0.1))
        self.pose = Pose(pose.x, y, pose.yaw)
        self.t += 1.0
        return self.get_obs()

    def get_obs(self) -> Observation:
        occ = self._occupancy()
        rgb = np.full((*self.img_hw, 3), 70, np.uint8)   # placeholder frame
        return Observation(images=[rgb], robot_pose=self.pose, occupancy=occ,
                           detections=[], stamp=self.t)

    def intrinsics(self) -> np.ndarray:
        h, w = self.img_hw
        fx = (w / 2) / np.tan(np.deg2rad(90) / 2)
        return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], float)

    # ---- occupancy in the robot frame ----
    def _occupancy(self) -> Occupancy:
        nx = int(round(self.x_max / self.res))
        ny = int(round(2 * self.y_half / self.res))
        cells = []
        c, s = np.cos(self.pose.yaw), np.sin(self.pose.yaw)
        for r in range(nx):
            fwd = r * self.res
            for cc in range(ny):
                left = -self.y_half + cc * self.res
                # robot-frame cell -> world
                wx = self.pose.x + fwd * c - left * s
                wy = self.pose.y + fwd * s + left * c
                blocked = abs(wy) > self.wall_half
                if not blocked:
                    for (x0, x1, y0, y1) in self.obstacles:
                        if x0 <= wx <= x1 and y0 <= wy <= y1:
                            blocked = True
                            break
                if blocked:
                    cells.append((r, cc))
        return Occupancy(res=self.res, x_max=self.x_max, y_half=self.y_half,
                         cells=cells, stamp=self.t)
