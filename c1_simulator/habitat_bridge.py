"""HabitatBridge — wraps habitat-sim behind the Bridge interface. Runs in the `habitat` env.

UNTESTED in this form (no habitat-sim available where it was written): the loop logic is tested
via FakeBridge; this is the piece to debug on MSI. The risky part is the coordinate mapping
between Habitat (Y up, agent faces -Z) and our robot/world frame (x forward, y left, yaw CCW).
Those conversions are the two small functions below — if the map or heading looks mirrored on the
first run, flip the sign there and nowhere else.

World mapping used here:  world_x = habitat_x,  world_y = -habitat_z,  up = habitat_y.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.types import Observation, Pose
from c1_simulator.bridge import Bridge
from c4_safety.safety_occupancy import grid_from_depth, grid_to_occupancy


def _make_cfg(scene, width, height, hfov, sensor_height):
    import habitat_sim
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene
    sim_cfg.enable_physics = False

    specs = []
    for uuid, stype in (("color", habitat_sim.SensorType.COLOR),
                        ("depth", habitat_sim.SensorType.DEPTH)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = stype
        spec.resolution = [height, width]
        spec.position = [0.0, sensor_height, 0.0]
        spec.hfov = hfov                       # degrees
        specs.append(spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


class HabitatBridge(Bridge):
    def __init__(self, scene: str, width: int = 640, height: int = 480,
                 hfov: float = 90.0, sensor_height: float = 0.88):
        import habitat_sim
        self.habitat_sim = habitat_sim
        self.width, self.height, self.hfov = width, height, hfov
        self.sensor_height = sensor_height
        self.sim = habitat_sim.Simulator(_make_cfg(scene, width, height, hfov, sensor_height))
        if not self.sim.pathfinder.is_loaded:
            ns = habitat_sim.NavMeshSettings()
            ns.set_defaults()
            self.sim.recompute_navmesh(self.sim.pathfinder, ns)
        self._K = self._intrinsics()

    # ---- coordinate conversions (verify these first) ----
    def _pose_from_agent(self) -> Pose:
        import quaternion as _q
        st = self.sim.get_agent(0).get_state()
        pos = np.asarray(st.position, float)              # habitat [x, y_up, z]
        fwd = _q.rotate_vectors(st.rotation, np.array([0.0, 0.0, -1.0]))
        world_x, world_y = float(pos[0]), float(-pos[2])
        yaw = float(np.arctan2(-fwd[2], fwd[0]))          # heading in (world_x, world_y)
        return Pose(world_x, world_y, yaw)

    def _agent_state_for(self, pose: Pose, height_y: float):
        import quaternion as _q
        from habitat_sim.utils.common import quat_from_angle_axis
        # world -> habitat position (keep current height)
        hab_pos = np.array([pose.x, height_y, -pose.y], float)
        # rotation about Y so the agent's world heading == pose.yaw (see spec derivation)
        a = float(np.arctan2(-np.cos(pose.yaw), np.sin(pose.yaw)))
        rot = quat_from_angle_axis(a, np.array([0.0, 1.0, 0.0]))
        st = self.habitat_sim.AgentState()
        st.position = hab_pos
        st.rotation = rot
        return st

    def _intrinsics(self) -> np.ndarray:
        fx = (self.width / 2) / np.tan(np.deg2rad(self.hfov) / 2)
        return np.array([[fx, 0, self.width / 2], [0, fx, self.height / 2], [0, 0, 1]], float)

    # ---- Bridge API ----
    def reset(self, start: Optional[Pose] = None) -> Observation:
        agent = self.sim.initialize_agent(0)
        st = self.habitat_sim.AgentState()
        if start is None:
            st.position = self.sim.pathfinder.get_random_navigable_point()
        else:
            snap = self.sim.pathfinder.snap_point(np.array([start.x, 0.0, -start.y]))
            st.position = np.asarray(snap, float)
        agent.set_state(st)
        return self.get_obs()

    def get_obs(self) -> Observation:
        raw = self.sim.get_sensor_observations()
        rgb = np.asarray(raw["color"])[:, :, :3]
        depth = np.asarray(raw["depth"], float)
        pose = self._pose_from_agent()
        grid = grid_from_depth(depth, self._K, self.sensor_height, cam_pitch_deg=0.0)
        occ = grid_to_occupancy(grid, stamp=0.0)
        return Observation(images=[rgb], robot_pose=pose, occupancy=occ, detections=[], stamp=0.0)

    def move_to(self, pose: Pose) -> Observation:
        cur = np.asarray(self.sim.get_agent(0).get_state().position, float)
        target = np.array([pose.x, cur[1], -pose.y], float)     # keep current height
        reached = self.sim.pathfinder.try_step(cur, target)     # collision-aware slide
        st = self._agent_state_for(Pose(float(reached[0]), float(-reached[2]), pose.yaw), cur[1])
        self.sim.get_agent(0).set_state(st)
        return self.get_obs()

    def intrinsics(self) -> np.ndarray:
        return self._K

    def close(self) -> None:
        try:
            self.sim.close()
        except Exception:
            pass
