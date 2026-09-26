#!/usr/bin/env python3
"""RefinementAdapter — the ONLY thing pepper_vla_node.py needs to touch.

Owns: /local_occupancy_grid subscription, /odom ring buffer (to look up the
robot pose at the grid's capture stamp), grid->snapshot conversion (distance
transform precomputed once per grid), per-cycle JSONL logging, shadow mode.

Integration in pepper_vla_node.py — three lines:

    from refinement_adapter import RefinementAdapter
    self.refinement = RefinementAdapter(self, shadow=True,
                                        log_path="~/signway_refine_log.jsonl")
    ...
    # where the policy server's 16-dim action currently goes to the executor:
    action_to_execute = self.refinement.process(action16)

SHADOW MODE (default True): process() computes and LOGS everything the
refiner would do, but returns the RAW action unchanged. First live runs
should be shadow — zero behavior change, and the log simultaneously yields
the raw-VLA baseline row for the ablation table AND a full audit of the
refiner's would-be decisions. Flip shadow=False only after reviewing a
shadow log.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque

import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry

from trajectory_refiner import (
    RefinerConfig, TrajectoryRefiner, make_snapshot, unpack_occupancy_grid,
)


def _yaw_from_quat(q) -> float:
    # planar robot: yaw from quaternion
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class RefinementAdapter:
    def __init__(self, node, cfg: RefinerConfig = None, shadow: bool = True,
                 grid_topic: str = "/local_occupancy_grid",
                 odom_topic: str = "/odom",
                 log_path: str = "~/signway_refine_log.jsonl"):
        self.node = node
        self.cfg = cfg or RefinerConfig()
        self.refiner = TrajectoryRefiner(self.cfg)
        self.shadow = shadow

        self._lock = threading.Lock()
        self._snapshot = None
        self._odom = deque(maxlen=600)   # ~ last 5-10 s of (t, x, y, yaw)

        self._log_path = os.path.expanduser(log_path) if log_path else None
        self._log_file = open(self._log_path, "a") if self._log_path else None

        node.create_subscription(OccupancyGrid, grid_topic, self._grid_cb, 1)
        node.create_subscription(Odometry, odom_topic, self._odom_cb, 20)
        mode = ("ON — raw executed, decisions logged" if shadow
                else "OFF — refined actions EXECUTED")
        node.get_logger().info(f"refinement adapter ready (shadow={mode})")

    # ------------------------------------------------------------ callbacks
    def _odom_cb(self, msg: Odometry) -> None:
        t = _stamp_to_sec(msg.header.stamp)
        p = msg.pose.pose
        with self._lock:
            self._odom.append((t, p.position.x, p.position.y,
                               _yaw_from_quat(p.orientation)))

    def _pose_at(self, t: float):
        """Nearest odom sample to time t (None if nothing within 0.5 s)."""
        best, best_dt = None, 0.5
        for rec in self._odom:
            dt = abs(rec[0] - t)
            if dt < best_dt:
                best, best_dt = rec, dt
        return None if best is None else (best[1], best[2], best[3])

    def _grid_cb(self, msg: OccupancyGrid) -> None:
        try:
            grid = unpack_occupancy_grid(msg.data, msg.info.width,
                                         msg.info.height)
            stamp = _stamp_to_sec(msg.header.stamp)
            # sanity: geometry must match the config the refiner assumes
            if abs(msg.info.resolution - self.cfg.resolution) > 1e-6:
                self.node.get_logger().warn(
                    f"grid resolution {msg.info.resolution} != config "
                    f"{self.cfg.resolution} — check server params")
            with self._lock:
                pose = self._pose_at(stamp)
            snap = make_snapshot(grid, stamp, self.cfg.resolution, pose=pose)
            with self._lock:
                self._snapshot = snap
        except Exception as exc:
            self.node.get_logger().error(f"grid_cb failed: {exc}")

    # ------------------------------------------------------------ main call
    def process(self, action16) -> np.ndarray:
        """Call with the policy server's 16-dim action; returns the 16-dim
        action to execute (raw in shadow mode). Never raises."""
        now = time.time()
        with self._lock:
            snap = self._snapshot
            pose_now = self._odom[-1][1:] if self._odom else None
        # NOTE on clocks: snapshot stamp comes from the image header; if the
        # camera driver stamps with ROS time and this host clock differs,
        # grid_age_s will be wrong. Both run on the Jetson here, so time.time()
        # and header stamps share a clock. Revisit if that ever changes.
        result = self.refiner.refine(action16, snap, pose_now=pose_now, now=now)

        if self._log_file is not None:
            try:
                rec = result.log_dict()
                rec["t"] = now
                rec["shadow"] = self.shadow
                self._log_file.write(json.dumps(rec) + "\n")
                self._log_file.flush()
            except Exception:
                pass

        if result.reason == "blocked" and not self.shadow:
            self.node.get_logger().warn(
                f"REFINER BLOCKED: min clearance {result.min_clearance_raw:.2f} m "
                "— slowing hard. (Fully blocked corridors are the AR layer's "
                "job — this only buys time.)")

        return np.asarray(action16, dtype=np.float32).reshape(16) \
            if self.shadow else result.action

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
