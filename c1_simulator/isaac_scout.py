#!/usr/bin/env python3
"""isaac_scout.py — find a sane spawn by LOOKING, not by guessing conventions.

The robot spawning nose-first into a wall makes every downstream number meaningless. This tries
several camera orientations from high above and saves one PNG per candidate, so you can see which
is actually a top-down map instead of trusting anyone's idea of USD axis conventions.

    ... /isaac-sim/python.sh /workspace/SignWay/c1_simulator/isaac_scout.py \
        --scene /assets/Isaac/Environments/Simple_Warehouse/full_warehouse.usd \
        --assets-root /assets --out /workspace/SignWay/isaac_out

Then: look at isaac_out/top_*.png, pick whichever is a real overhead view, read a clear spot off
it, and pass it to run_sim as --start X Y YAW.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--assets-root", default=None)
ap.add_argument("--out", default="isaac_out")
ap.add_argument("--height", type=float, default=40.0)
ap.add_argument("--probe-height", type=float, default=0.88)
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

sim_app = SimulationApp({"headless": True})

import os  # noqa: E402

import carb  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, UsdLux  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402

if args.assets_root:
    carb.settings.get_settings().set("/persistent/isaac/asset_root/default", args.assets_root)
os.makedirs(args.out, exist_ok=True)

omni.usd.get_context().open_stage(args.scene)
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(stage, "/World/scout_dome").CreateIntensityAttr(1200.0)

cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
bb = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
lo, hi = bb.GetMin(), bb.GetMax()
cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
print(f"[scout] extent x {lo[0]:.1f}..{hi[0]:.1f}  y {lo[1]:.1f}..{hi[1]:.1f}  "
      f"z {lo[2]:.1f}..{hi[2]:.1f}   centre ({cx:.1f},{cy:.1f})", flush=True)

cam = Camera(prim_path="/World/scout_cam", resolution=(1024, 1024))
world.reset()
cam.initialize()
cam.add_distance_to_image_plane_to_frame()
for _ in range(90):                     # generous warm-up: the depth annotator is slow to arrive
    world.step(render=True)


def grab(tries=90):
    rgba = depth = None
    for i in range(tries):
        world.step(render=True)
        r = cam.get_rgba()
        if r is not None and r.size:
            rgba = r
            depth = cam.get_current_frame().get("distance_to_image_plane")
            if depth is not None and np.asarray(depth).size:
                break
    return rgba, depth


def shoot(name, pos, quat, axes):
    cam.set_world_pose(position=np.array(pos, float), orientation=np.array(quat, float),
                       camera_axes=axes)
    rgba, depth = grab()
    if rgba is None or not rgba.size:
        print(f"[scout] {name}: no frame", flush=True)
        return
    p = os.path.join(args.out, f"{name}.png")
    Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(p)
    dmsg = "depth MISSING"
    if depth is not None and np.asarray(depth).size:
        d = np.asarray(depth, float)
        f = d[np.isfinite(d)]
        dmsg = f"depth {f.min():.2f}..{f.max():.2f}m"
    print(f"[scout] {name}: {dmsg} -> {p}", flush=True)


# ---- which orientation actually looks DOWN? try them all, look at the pictures ----
top = [cx, cy, args.height]
shoot("top_A_world_pitch+90", top, euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0])), "world")
shoot("top_B_world_pitch-90", top, euler_angles_to_quat(np.array([0.0, -np.pi / 2, 0.0])), "world")
shoot("top_C_usd_identity", top, [1.0, 0.0, 0.0, 0.0], "usd")
shoot("top_D_usd_pitch+90", top, euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0])), "usd")

# ---- clearance from the centre at robot height, four ways ----
print("[scout] clearance from centre at robot height:", flush=True)
for name, yaw in [("+X", 0.0), ("+Y", np.pi / 2), ("-X", np.pi), ("-Y", -np.pi / 2)]:
    cam.set_world_pose(position=np.array([cx, cy, args.probe_height]),
                       orientation=euler_angles_to_quat(np.array([0.0, 0.0, yaw])),
                       camera_axes="world")
    rgba, depth = grab()
    if depth is None or not np.asarray(depth).size:
        print(f"  facing {name:2s}: depth MISSING", flush=True)
        continue
    d = np.asarray(depth, float)
    f = d[np.isfinite(d)]
    mid = float(d[d.shape[0] // 2, d.shape[1] // 2])
    print(f"  facing {name:2s} (yaw={yaw:+.2f}): straight ahead {mid:6.2f}m   "
          f"view {f.min():.2f}..{f.max():.2f}m", flush=True)
    if rgba is not None and rgba.size:
        Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(
            os.path.join(args.out, f"probe_{name.replace('+','p').replace('-','m')}.png"))

print("[scout] done — look at top_*.png, pick the real overhead one, then --start X Y YAW",
      flush=True)
sim_app.close()