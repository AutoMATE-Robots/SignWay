#!/usr/bin/env python3
"""isaac_scene.py — load a REAL .usd scene, fly a camera through it, save frames.

This answers the only question that matters for SignWay: can Isaac give us a picture from a
given pose? If yes, the bridge is four small methods and the pipeline runs inside Isaac.

Assets note: MSI compute nodes cannot reach NVIDIA's S3 asset server (nvcr.io is allowlisted,
S3 is not), so scenes must be on local disk. Download on your laptop, scp to MSI, point --scene
at it. If the scene references textures/props it can't find, it'll still load — just grey.

    singularity exec --nv --home $SCRATCH/home \
      --bind $SCRATCH/cache/kit:/isaac-sim/kit/cache:rw \
      --bind $SCRATCH/cache/data:/isaac-sim/kit/data:rw \
      --bind $HOME/SignWay:/workspace/SignWay:rw \
      $SCRATCH/isaac-sim_5.1.0.sif /isaac-sim/python.sh \
      /workspace/SignWay/c1_simulator/isaac_scene.py \
      --scene /workspace/SignWay/assets/warehouse.usd \
      --out /workspace/SignWay/isaac_out
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True, help="path to a local .usd scene")
ap.add_argument("--out", default="isaac_out")
ap.add_argument("--height", type=float, default=1.0, help="camera height (m)")
ap.add_argument("--steps", type=int, default=6, help="how many poses to fly through")
ap.add_argument("--stride", type=float, default=1.5, help="metres between poses")
ap.add_argument("--assets-root", default=None,
                help="local Isaac assets root, if you downloaded the packs")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

sim_app = SimulationApp({"headless": True})

import os  # noqa: E402

import carb  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import omni.usd  # noqa: E402
from pxr import UsdLux  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402

if args.assets_root:   # only needed if you downloaded the asset packs
    carb.settings.get_settings().set("/persistent/isaac/asset_root/default", args.assets_root)

os.makedirs(args.out, exist_ok=True)
if not os.path.exists(args.scene):
    raise SystemExit(f"scene not found: {args.scene}")

print(f"[scene] opening {args.scene}", flush=True)
omni.usd.get_context().open_stage(args.scene)
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

# The scene may ship its own lights. Add a dim dome anyway so we never render pure black.
dome = UsdLux.DomeLight.Define(stage, "/World/signway_dome")
dome.CreateIntensityAttr(400.0)

cam = Camera(prim_path="/World/signway_cam", resolution=(640, 480),
             position=np.array([0.0, 0.0, args.height]))
world.reset()
cam.initialize()
for _ in range(60):
    world.step(render=True)

frames = []
for i in range(args.steps):
    x = i * args.stride
    # camera_axes="world": identity orientation = looking down +X, Y left, Z up (our convention)
    cam.set_world_pose(position=np.array([x, 0.0, args.height]),
                       orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                       camera_axes="world")
    rgba = None
    for _ in range(60):                       # poll — frames right after a move come back empty
        world.step(render=True)
        rgba = cam.get_rgba()
        if rgba is not None and rgba.size and rgba[:, :, :3].any():
            break
    if rgba is None or not rgba.size:
        print(f"[scene] step {i}: NO FRAME", flush=True)
        continue
    rgb = rgba[:, :, :3].astype(np.uint8)
    p = os.path.join(args.out, f"scene_{i:02d}.png")
    Image.fromarray(rgb).save(p)
    frames.append(rgb.astype(float))
    print(f"[scene] step {i}: x={x:.1f}m mean={rgb.mean():.1f} -> {p}", flush=True)

# The test: consecutive poses must produce DIFFERENT pictures.
if len(frames) >= 2:
    diffs = [float(np.abs(frames[i] - frames[i + 1]).mean()) for i in range(len(frames) - 1)]
    print(f"[scene] frame-to-frame diffs: {[round(d, 1) for d in diffs]}", flush=True)
    ok = sum(d > 2.0 for d in diffs) >= max(1, len(diffs) // 2)
    print(f"[scene] {'PASS — pose-driven rendering works' if ok else 'FAIL — frames barely change'}",
          flush=True)
else:
    print("[scene] FAIL — fewer than 2 frames", flush=True)

sim_app.close()