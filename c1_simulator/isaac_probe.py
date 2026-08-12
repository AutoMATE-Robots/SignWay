"""Find the corridors: sweep a grid of robot poses in ONE Isaac boot and report free space.

WHY THIS AND NOT isaac_scout.py: the hospital takes ~3 minutes to load its assets, so probing
one pose per boot is unusable. This boots once, teleports the camera over a grid, and for each
pose reports how far it can see in four directions. A corridor is the signature we want: a long
clear run one way, walls close on both sides.

It also never crashes on an empty depth array — the old scout died on f.min() when a probe saw
nothing at all, which is a perfectly normal thing for a probe to do.

    /isaac-sim/python.sh -m c1_simulator.isaac_probe --scene /assets/.../hospital.usd \
        --assets-root /assets --x -45 25 10 --y 0 36 6 --out probe_out
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--assets-root", default=None)
    ap.add_argument("--out", default="probe_out")
    ap.add_argument("--x", nargs=3, type=float, default=[-45, 25, 10], metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--y", nargs=3, type=float, default=[0, 36, 6], metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--z", type=float, default=0.88, help="sensor height")
    ap.add_argument("--min-corridor-m", type=float, default=6.0,
                    help="a view this clear counts as 'down a corridor'")
    ap.add_argument("--save-top", type=int, default=8, help="render this many best poses")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from isaacsim.simulation_app import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.numpy.rotations as rot_utils

    if args.assets_root:
        import carb
        carb.settings.get_settings().set("/persistent/isaac/asset_root/default", args.assets_root)
    open_stage(args.scene)
    for _ in range(60):
        app.update()

    # Use World.step(render=True), exactly as isaac_bridge does. app.update() alone does not
    # reliably tick the render pipeline that fills the annotators.
    world = World(stage_units_in_meters=1.0)
    cam = Camera(prim_path="/World/probe_cam", frequency=20, resolution=(320, 240))
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    world.reset()
    for _ in range(20):
        world.step(render=True)

    def look(x, y, yaw_deg, settle=8):
        """Depth is get_current_frame()["distance_to_image_plane"], NOT get_depth() (which
        returns None in this container).

        CRITICAL: step a FIXED number of frames and only then read. Polling until the array is
        non-empty looks sensible and is wrong — the annotator hands back the PREVIOUS pose's
        depth instantly, so it breaks on stale data and every pose reports its neighbour's view.
        The tell was a pose claiming 28m of clear space in a direction where the building ends
        after 12m. Non-empty is not the same as fresh.
        """
        cam.set_world_pose(np.array([x, y, args.z]),
                           rot_utils.euler_angles_to_quats(np.array([0, 0, yaw_deg]), degrees=True))
        for _ in range(settle):
            world.step(render=True)
        d = cam.get_current_frame().get("distance_to_image_plane")
        if d is None or not np.asarray(d).size:
            return None
        d = np.asarray(d, float)
        h, w = d.shape[:2]
        band = d[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)]   # straight ahead
        good = band[np.isfinite(band) & (band > 0.05)]
        return float(np.median(good)) if good.size else None

    xs = np.arange(args.x[0], args.x[1] + 1e-6, args.x[2])
    ys = np.arange(args.y[0], args.y[1] + 1e-6, args.y[2])
    print(f"[probe] sweeping {len(xs)}x{len(ys)} poses x 4 headings", flush=True)

    rows, n_blind = [], 0
    for y in ys:
        for x in xs:
            d = {yaw: look(x, y, yaw) for yaw in (0, 90, 180, 270)}
            vals = [v for v in d.values() if v is not None]
            if not vals:
                n_blind += 1
                continue                            # outside the building, or no depth
            best_yaw = max(d, key=lambda k: (d[k] or 0.0))
            best = d[best_yaw] or 0.0
            side = [d[(best_yaw + 90) % 360] or 99, d[(best_yaw + 270) % 360] or 99]
            # corridor = long clear run one way, walls close on both sides
            corridor = best >= args.min_corridor_m and max(side) < best / 2
            rows.append({"x": float(x), "y": float(y), "yaw": best_yaw, "ahead": best,
                         "left": side[0], "right": side[1], "corridor": corridor})
            tag = "  <-- CORRIDOR" if corridor else ""
            print(f"[probe] ({x:6.1f},{y:5.1f}) yaw {best_yaw:3d}  ahead {best:5.1f}m  "
                  f"sides {side[0]:4.1f}/{side[1]:4.1f}{tag}", flush=True)

    if not rows:
        print(f"[probe] NO depth at any of {n_blind} poses — the annotator never filled. "
              f"frame keys = {list(cam.get_current_frame().keys())}", flush=True)
        app.close()
        return
    print(f"[probe] {len(rows)} poses inside the building, {n_blind} blind", flush=True)
    rows.sort(key=lambda r: (not r["corridor"], -r["ahead"]))
    print(f"\n[probe] {sum(r['corridor'] for r in rows)} corridor-like poses found", flush=True)
    for r in rows[:args.save_top]:
        look(r["x"], r["y"], r["yaw"])
        rgba = cam.get_rgba()
        if rgba is None or not rgba.size:
            continue
        rgb = rgba[:, :, :3]
        from PIL import Image
        name = f"probe_x{r['x']:.0f}_y{r['y']:.0f}_yaw{r['yaw']}.png"
        Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(args.out, name))
        print(f"[probe] BEST ({r['x']:.1f},{r['y']:.1f}) yaw {r['yaw']} "
              f"ahead {r['ahead']:.1f}m -> {name}", flush=True)

    app.close()


if __name__ == "__main__":
    main()