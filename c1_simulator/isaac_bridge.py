"""Isaac Sim bridge — component 1.

Gives the pipeline the one thing it needs from a simulator: a pose in, a picture out.
Implements the same four methods as fake_bridge.py, so `run_sim.py --bridge isaac` runs the
real FSM + OmniVLA + occupancy + logging inside an Isaac scene with nothing else changed.

VERIFIED ON MSI (2026-07-16, A40 node, isaac-sim 5.1.0 container):
  - SimulationApp({"headless": True}) launches and initialises the renderer
  - omni.usd.get_context().open_stage(<local .usd>) loads a scene
  - Camera.set_world_pose(..., camera_axes="world") is accepted  (the CONSTRUCTOR is not)
  - Camera.get_rgba() returns empty frames for a while after a move -> must poll

GOTCHAS baked in:
  - camera_axes="world" makes identity orientation mean "+X forward, +Y left, +Z up" — the same
    convention as the rest of SignWay. USD's default is -Z forward, which points at the floor.
  - Scenes must be LOCAL. MSI compute nodes cannot reach NVIDIA's S3 asset server, so a bare
    .usd whose references live on S3 loads as an empty grey room. Use the downloaded asset
    packs and pass assets_root.

MOTION MODEL: the camera flies — there is no physics body, so nothing stops it passing through
walls. That is deliberate for now: collision avoidance is component 4's job (the occupancy grid
+ replanner), and letting the sim silently block motion would hide whether the safety layer
actually works.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.types import Observation, Pose
from c1_simulator.bridge import Bridge


class IsaacBridge(Bridge):
    def __init__(self, scene: str, width: int = 640, height: int = 480,
                 sensor_height: float = 0.88, assets_root: Optional[str] = None,
                 headless: bool = True, grid_range: float = 6.0):
        # SimulationApp must exist before any other isaacsim/omni import — the extension system
        # does not exist until it boots. This is why the imports are inside __init__.
        from isaacsim import SimulationApp
        self._app = SimulationApp({"headless": headless})

        import carb
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera

        if assets_root:
            carb.settings.get_settings().set("/persistent/isaac/asset_root/default", assets_root)

        self._omni_usd = omni.usd
        self.width, self.height = width, height
        self.sensor_height = sensor_height

        omni.usd.get_context().open_stage(scene)
        self._stage = omni.usd.get_context().get_stage()
        self._world = World(stage_units_in_meters=1.0)
        self._n_signs = 0
        self._cam = Camera(prim_path="/World/signway_cam", resolution=(width, height),
                           position=np.array([0.0, 0.0, sensor_height]))
        self._world.reset()
        self._cam.initialize()
        self._cam.add_distance_to_image_plane_to_frame()      # depth -> occupancy
        for _ in range(60):
            self._world.step(render=True)

        self._pose = Pose(0.0, 0.0, 0.0)
        self.grid_range = grid_range     # occupancy only sees this far; a wall beyond it is invisible
        self._depth_reported = False
        self._last_depth_msg = ""
        self._K = self._compute_intrinsics()

    # ---- internals ----
    def _compute_intrinsics(self) -> np.ndarray:
        """Pinhole K from the USD camera's focal length + aperture."""
        try:
            f = float(self._cam.get_focal_length())
            ha = float(self._cam.get_horizontal_aperture())
            fx = self.width * f / ha
        except Exception:
            fx = (self.width / 2) / np.tan(np.deg2rad(90.0) / 2)   # sane fallback
        return np.array([[fx, 0, self.width / 2],
                         [0, fx, self.height / 2],
                         [0, 0, 1]], float)

    def _place(self, pose: Pose) -> None:
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        quat = euler_angles_to_quat(np.array([0.0, 0.0, pose.yaw]))   # yaw about world Z
        self._cam.set_world_pose(position=np.array([pose.x, pose.y, self.sensor_height]),
                                 orientation=quat, camera_axes="world")
        self._pose = pose

    def _render(self, tries: int = 30):
        """Step until a frame with data arrives.

        NOTE: we accept an ALL-BLACK frame. Black usually means the camera is inside geometry —
        the camera flies with no collision body, so if the safety layer fails to steer around a
        shelf we end up inside it. That is a real observation the pipeline should see and log,
        not an exception. Raising here just hid the actual bug (empty occupancy).
        """
        rgb = depth = None
        for i in range(tries):
            self._world.step(render=True)
            rgba = self._cam.get_rgba()
            if rgba is not None and rgba.size:
                rgb = rgba[:, :, :3].astype(np.uint8)
            d = self._cam.get_current_frame().get("distance_to_image_plane")
            if d is not None and np.asarray(d).size:
                depth = d
                if rgb is not None:
                    break            # colour AND depth — done
            # NOTE: do NOT give up on depth early. Bailing after a few polls made every frame
            # after the first return depth=None, so the occupancy grid was empty and the robot
            # drove blind. Depth is what component 4 exists for; wait for it.
        return rgb, depth

    def _observe(self) -> Observation:
        from c4_safety.safety_occupancy import grid_from_depth, grid_to_occupancy
        rgb, depth = self._render()
        if rgb is None:
            raise RuntimeError("Isaac returned no frame at all — renderer problem")

        occ = None
        if depth is not None and np.asarray(depth).size:
            d = np.asarray(depth, float)
            grid = grid_from_depth(d, self._K, self.sensor_height, cam_pitch_deg=0.0,
                                   x_max=self.grid_range)
            occ = grid_to_occupancy(grid)
            finite = d[np.isfinite(d)]
            self._last_depth_msg = (f"depth {finite.min():.2f}..{finite.max():.2f}m "
                                    f"-> {len(occ.cells)} cells")
            if not self._depth_reported:
                print(f"[isaac] depth ok: shape={d.shape} {self._last_depth_msg}", flush=True)
                self._depth_reported = True
        else:
            self._last_depth_msg = "depth MISSING -> occupancy EMPTY (robot is blind)"
            if not self._depth_reported:
                keys = list(self._cam.get_current_frame().keys())
                print(f"[isaac] WARNING: no depth. frame keys = {keys}", flush=True)
                self._depth_reported = True

        return Observation(images=[rgb], robot_pose=self._pose, occupancy=occ,
                           detections=[], stamp=0.0)

    def add_sign(self, label: str, position, yaw_deg: float, out_dir: str,
                 arrow: Optional[str] = None, width: float = 1.2, height: float = 0.6):
        """Hang a readable sign on a wall. The detector and VLM must SEE it in the pixels, so
        this is a real textured quad, not a coloured box."""
        import os as _os

        from c1_simulator.signs import add_sign_to_stage, make_sign_texture

        low = label.lower()
        if arrow is None:                       # infer the arrow from the words on the sign
            arrow = "left" if "left" in low else ("right" if "right" in low else None)
        # The label is what the detector/reasoner key on ("cafeteria left"); the board itself
        # shows "CAFETERIA <-" because the arrow already says the direction.
        display = label
        for w in ("left", "right"):
            display = display.replace(w, "").replace(w.upper(), "").strip()
        png = make_sign_texture(display or label,
                                _os.path.join(out_dir, f"sign_{self._n_signs}.png"), arrow=arrow)
        add_sign_to_stage(self._stage, f"/World/signway_sign_{self._n_signs}", png,
                          position=position, yaw_deg=yaw_deg, width=width, height=height)
        self._n_signs += 1
        print(f"[isaac] sign '{label}' at {tuple(round(float(v), 2) for v in position)} "
              f"facing {yaw_deg:.0f} deg -> {png}", flush=True)

    # ---- Bridge API ----
    def reset(self, start: Optional[Pose] = None) -> Observation:
        self._place(start or Pose(0.0, 0.0, 0.0))
        return self._observe()

    def get_obs(self) -> Observation:
        return self._observe()

    def move_to(self, pose: Pose) -> Observation:
        self._place(pose)
        return self._observe()

    def intrinsics(self) -> np.ndarray:
        return self._K

    def close(self) -> None:
        try:
            self._app.close()
        except Exception:
            pass