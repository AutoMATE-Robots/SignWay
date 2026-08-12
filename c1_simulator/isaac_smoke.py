#!/usr/bin/env python3
"""isaac_smoke.py — prove Isaac Sim works, in two stages.

Run in the Isaac container on an A40/L40S node (interactive-gpu / preempt-gpu).
NOT on msigpu — A100s have no RT cores; NVIDIA does not support Isaac on them.

    --stage 1   does the app launch headless? (renderer init — the hard part)
    --stage 2   put a camera at a pose, render, move it, render again -> two PNGs

IMPORTANT: everything here is built from primitives. No `add_default_ground_plane()`, no sample
scenes, no Nucleus. Those all fetch from NVIDIA's S3 asset server, and MSI compute nodes have no
outbound internet ("Could not find assets root folder"). A ground plane is just a flat box.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
ap.add_argument("--out", default="isaac_out")
args = ap.parse_args()

# SimulationApp must be created before any other isaacsim/omni import — the extension system
# doesn't exist until it boots.
from isaacsim import SimulationApp  # noqa: E402

sim_app = SimulationApp({"headless": True})
print("[stage 1] SimulationApp launched headless — renderer initialised OK", flush=True)

if args.stage == 1:
    sim_app.close()
    print("[stage 1] PASS")
    raise SystemExit(0)

# ---- stage 2 ----
import os  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import omni.usd  # noqa: E402
from pxr import UsdLux  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import VisualCuboid  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402

os.makedirs(args.out, exist_ok=True)

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# light (without one the render is black) — UsdLux is built in, nothing to download
key = UsdLux.DistantLight.Define(stage, "/World/key_light")
key.CreateIntensityAttr(3000.0)
dome = UsdLux.DomeLight.Define(stage, "/World/dome_light")
dome.CreateIntensityAttr(1000.0)

# "ground" = a big flat box. Same pixels as a real ground plane, zero assets.
VisualCuboid(prim_path="/World/ground", name="ground",
             position=np.array([0.0, 0.0, -0.05]),
             scale=np.array([20.0, 20.0, 0.1]),
             color=np.array([0.35, 0.35, 0.40]))
# the thing we look at: a red slab 3 m down +X
VisualCuboid(prim_path="/World/box", name="box",
             position=np.array([3.0, 0.0, 0.5]),
             scale=np.array([0.4, 2.0, 1.0]),
             color=np.array([0.85, 0.15, 0.15]))

# camera_axes="world" makes identity orientation mean "looking down +X, Y left, Z up" —
# the same convention the rest of SignWay uses. Without it, USD's default is -Z (straight down
# from here), which is why the first attempt photographed the floor.
cam = Camera(prim_path="/World/cam", resolution=(320, 240),
             position=np.array([0.0, 0.0, 1.0]))
world.reset()
cam.initialize()

for _ in range(60):                 # let the render product spin up
    world.step(render=True)


def shoot(name, pos, tries=60):
    """Move the camera to a world pose, render, save. This IS the Bridge's move_to()."""
    cam.set_world_pose(position=np.array(pos, dtype=float),
                       orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                       camera_axes="world")
    rgba = None
    for _ in range(tries):          # poll: the first frames after a move come back empty
        world.step(render=True)
        rgba = cam.get_rgba()
        if rgba is not None and rgba.size and rgba[:, :, :3].any():
            break
    if rgba is None or not rgba.size:
        print(f"[stage 2] FAIL: {name} — no frame after {tries} steps", flush=True)
        return False
    path = os.path.join(args.out, f"{name}.png")
    Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(path)
    print(f"[stage 2] {name}: shape={rgba.shape} mean={rgba[:, :, :3].mean():.1f} -> {path}",
          flush=True)
    return rgba[:, :, :3].astype(float)


a = shoot("pose_a", [0.0, 0.0, 1.0])     # 3 m from the box
b = shoot("pose_b", [1.5, 0.0, 1.0])     # 1.5 m closer — the box must look BIGGER

# The real check: moving the camera must change the picture. Identical frames mean the pose
# isn't reaching the renderer, which is the only thing the bridge actually needs.
if a is False or b is False:
    print("[stage 2] FAIL — no frames", flush=True)
else:
    diff = float(np.abs(a - b).mean())
    red_a = float((a[:, :, 0] > 120).mean() * 100)   # % of pixels that are red-ish
    red_b = float((b[:, :, 0] > 120).mean() * 100)
    print(f"[stage 2] mean abs diff between poses = {diff:.2f}", flush=True)
    print(f"[stage 2] red pixels: pose_a {red_a:.1f}%  ->  pose_b {red_b:.1f}%  "
          f"(closer should be MORE)", flush=True)
    print(f"[stage 2] {'PASS — pose-driven rendering works' if diff > 1.0 else 'FAIL — poses render identically'}",
          flush=True)
sim_app.close()