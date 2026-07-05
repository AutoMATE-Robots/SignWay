#!/usr/bin/env python3
"""
odom_replay.py — odometry-anchored replay on real frames.

Uses the pre-extracted /frames + odom.csv (timestamp_ns, x, y, yaw). The goal is pinned
at the LAST odom row (the true end of the bag); at each frame the fixed world goal is
transformed into the current robot frame via that row's (x, y, yaw), so the bearing to
the goal evolves naturally as the robot drives. Renders a two-panel figure per frame:
  left  = the camera image
  right = world-frame top-down: full odom path, a robot marker that slides + rotates
          along it, Omni's predicted chunk (transformed into world coords) fanning out
          ahead, and the goal star fixed at the end.

Run in the `omnivla` env on the GPU node:
    export PYTHONNOUSERSITE=1
    python odom_replay.py --frames <.../frames> --odom <.../odom.csv> \
        --stride 15 --count 40 --backend omnivla --out odom_out --gif

Conventions: odom is world SE(2) (x, y, yaw CCW). Robot frame x=forward, y=left.
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omni_backend import make_backend
import occupancy_replan as occ


def build_grid(frame, demo_obstacle):
    """Occupancy for this frame. Empty by default. --demo-obstacle injects a synthetic
    box so the occupancy + replan visualization is visible before real depth is wired.
    TODO(depth): replace with occ.occupancy_from_depth(depth, K, cam_height, pitch) once
    you provide camera calibration and run Depth Anything V2 on the frame."""
    g = occ.Grid(x_max=4.0, y_half=2.0, res=0.05)
    if demo_obstacle:
        g.add_box(1.5, 2.0, -0.5, 0.5)   # synthetic obstacle ~1.5-2 m ahead
    return g


OMNI_RANGE_M = 30.0  # run_omnivla clips relative goal to 30 m; match it to stay in-distribution


# ── data loading ────────────────────────────────────────────────────────────────
def load_odom(path):
    ts, xs, ys, yaws = [], [], [], []
    with open(path) as f:
        for d in csv.DictReader(f):
            ts.append(int(d["timestamp_ns"]))
            xs.append(float(d["x"])); ys.append(float(d["y"])); yaws.append(float(d["yaw"]))
    return (np.array(ts, dtype=np.int64), np.array(xs), np.array(ys), np.array(yaws))


def _frame_ts(fname):
    runs = re.findall(r"\d+", os.path.basename(fname))
    return int(max(runs, key=len)) if runs else None


def list_frames(frames_dir):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        files += glob.glob(os.path.join(frames_dir, ext))
    files.sort()
    return files


def pair_frames_to_odom(frame_files, odom_ts):
    """Return list of (frame_file, odom_index). Timestamp mode if filenames carry ns
    timestamps, else proportional index mode."""
    parsed = [_frame_ts(f) for f in frame_files]
    looks_like_ns = parsed[0] is not None and np.median([p for p in parsed if p]) > 1e15
    pairs = []
    if looks_like_ns:
        for f, p in zip(frame_files, parsed):
            j = int(np.argmin(np.abs(odom_ts - p))) if p is not None else 0
            pairs.append((f, j, abs(int(odom_ts[j]) - (p or 0)) / 1e6))  # dt in ms
        mode = "timestamp"
    else:
        n_o = len(odom_ts)
        for i, f in enumerate(frame_files):
            j = min(n_o - 1, int(round(i * (n_o - 1) / max(1, len(frame_files) - 1))))
            pairs.append((f, j, float("nan")))
        mode = "index"
    return pairs, mode


# ── geometry ────────────────────────────────────────────────────────────────────
def world_goal_to_robot(gx, gy, gyaw, x, y, yaw, max_range=OMNI_RANGE_M):
    dx, dy = gx - x, gy - y
    fwd = dx * np.cos(yaw) + dy * np.sin(yaw)
    left = -dx * np.sin(yaw) + dy * np.cos(yaw)
    r = np.hypot(fwd, left)
    if r > max_range:
        fwd *= max_range / r
        left *= max_range / r
    dtheta = np.arctan2(np.sin(gyaw - yaw), np.cos(gyaw - yaw))
    return fwd, left, dtheta


def wp_robot_to_world(wp, x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    wx = x + wp[:, 0] * c - wp[:, 1] * s
    wy = y + wp[:, 0] * s + wp[:, 1] * c
    return wx, wy


# ── rendering ─────────────────────────────────────────────────────────────────
def render(frame_img, path_xy, traveled_n, robot, wp_robot, wp_world,
           goal_robot, goal_xy, lims, title, out_path):
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(18, 6))
    axL.imshow(frame_img); axL.axis("off"); axL.set_title("camera")

    # middle: OmniVLA's raw action grid in its own robot frame (forward up, left/right)
    axM.plot(-wp_robot[:, 1], wp_robot[:, 0], color="#1f77b4", lw=2.6, marker="o", ms=4,
             zorder=4, label="action chunk")
    axM.scatter([0], [0], marker="^", s=120, color="k", zorder=5)   # robot at origin
    gf, gl = goal_robot
    gnorm = max(np.hypot(gf, gl), 1e-6)
    disp = 1.3
    axM.annotate("", xy=(-gl / gnorm * disp, gf / gnorm * disp), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=2))  # goal bearing
    axM.axhline(0, color="0.85", lw=0.8); axM.axvline(0, color="0.85", lw=0.8)
    axM.set_xlim(-2.5, 2.5); axM.set_ylim(-0.3, 3.5)
    axM.set_aspect("equal"); axM.grid(alpha=0.3)
    axM.set_xlabel("← right        left →   (m)"); axM.set_ylabel("forward (m)")
    axM.set_title("OmniVLA action grid (robot frame)")
    axM.legend(fontsize=8, loc="upper right")

    # right: world map with the moving robot marker + fixed goal
    axR.plot(path_xy[:, 0], path_xy[:, 1], color="0.82", lw=1.5, zorder=1)            # full path
    axR.plot(path_xy[:traveled_n, 0], path_xy[:traveled_n, 1], color="0.45", lw=2.2, zorder=2)
    axR.plot(wp_world[0], wp_world[1], color="#1f77b4", lw=2.6, marker="o", ms=3, zorder=4,
             label="OmniVLA chunk")
    rx, ry, ryaw = robot
    axR.quiver([rx], [ry], [np.cos(ryaw)], [np.sin(ryaw)], color="k", scale=12,
               width=0.012, zorder=5)
    axR.scatter([rx], [ry], s=70, color="k", zorder=6)
    axR.plot([goal_xy[0]], [goal_xy[1]], marker="*", ms=20, color="#d62728",
             zorder=5, label="goal (bag end)")
    axR.set_xlim(lims[0], lims[1]); axR.set_ylim(lims[2], lims[3])
    axR.set_aspect("equal"); axR.grid(alpha=0.3)
    axR.set_xlabel("world x (m)"); axR.set_ylabel("world y (m)")
    axR.legend(fontsize=8, loc="best"); axR.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder of extracted frames")
    ap.add_argument("--odom", required=True, help="odom.csv (timestamp_ns,x,y,yaw)")
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--backend", default="omnivla", choices=["mock", "omnivla"])
    ap.add_argument("--out", default="odom_out")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--log", default=None,
                    help="also write a JSON-lines log (per-frame pose/goal/omni/occupancy/refined) "
                         "for the spatial viewer")
    ap.add_argument("--demo-obstacle", action="store_true",
                    help="inject a synthetic obstacle so occupancy+replan is visible before "
                         "real depth is wired")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ts, xs, ys, yaws = load_odom(args.odom)
    path_xy = np.column_stack([xs, ys])
    gx, gy, gyaw = xs[-1], ys[-1], yaws[-1]   # goal = last odom pose

    frame_files = list_frames(args.frames)
    if not frame_files:
        raise SystemExit(f"no images found in {args.frames}")
    pairs, mode = pair_frames_to_odom(frame_files, ts)
    sel = pairs[args.start::args.stride][:args.count]
    print(f"[odom_replay] {len(frame_files)} frames, {len(ts)} odom rows, pairing={mode}, "
          f"replaying {len(sel)}")
    print(f"[odom_replay] goal (bag end) world=({gx:.2f},{gy:.2f},{gyaw:.2f})")

    # fixed world limits from the whole path + goal, padded
    pad = 1.5
    xmin, xmax = min(xs.min(), gx) - pad, max(xs.max(), gx) + pad
    ymin, ymax = min(ys.min(), gy) - pad, max(ys.max(), gy) + pad
    lims = (xmin, xmax, ymin, ymax)

    backend = make_backend(args.backend)
    logf = open(args.log, "w") if args.log else None
    step_p = max(1, len(xs) // 400)   # downsample the full path for the world-map polyline
    meta = {"type": "meta",
            "goal_world": [float(gx), float(gy), float(gyaw)],
            "grid": {"res": 0.05, "x_max": 4.0, "y_half": 2.0},
            "path": np.stack([xs[::step_p], ys[::step_p]], 1).round(3).tolist(),
            "world_bounds": [float(xmin), float(xmax), float(ymin), float(ymax)]}
    if logf:
        logf.write(json.dumps(meta) + "\n")
    out_paths = []
    for k, (ffile, j, dt_ms) in enumerate(sel):
        x, y, yaw = xs[j], ys[j], yaws[j]
        fwd, left, dtheta = world_goal_to_robot(gx, gy, gyaw, x, y, yaw)
        frame = np.asarray(Image.open(ffile).convert("RGB"))
        wp = np.asarray(backend.predict_waypoints([frame, frame], (fwd, left, dtheta)),
                        float).reshape(-1, 2)
        wp_world = wp_robot_to_world(wp, x, y, yaw)

        grid = build_grid(frame, args.demo_obstacle)
        used_omni, refined, _occ_inf, _subgoal = occ.refine(grid, wp)

        title = f"t={k}  goal_robot=({fwd:.1f},{left:.1f})  dt={dt_ms:.0f}ms"
        p = os.path.join(args.out, f"step_{k:03d}.png")
        render(frame, path_xy, j + 1, (x, y, yaw), wp, wp_world, (fwd, left),
               (gx, gy), lims, title, p)
        out_paths.append(p)

        if logf:
            rr, cc = np.where(grid.occ)
            rec = {
                "i": k, "ts": int(ts[j]), "frame": os.path.basename(ffile),
                "pose": [float(x), float(y), float(yaw)],
                "goal_robot": [float(fwd), float(left), float(dtheta)],
                "omni": wp.round(3).tolist(),
                "occ_cells": np.stack([rr, cc], 1).tolist() if len(rr) else [],
                "refined": (None if refined is None else np.asarray(refined).round(3).tolist()),
                "used_omni": bool(used_omni),
            }
            logf.write(json.dumps(rec) + "\n")

        if k < 3 or k == len(sel) - 1:
            print(f"  step {k}: {os.path.basename(ffile)} -> odom[{j}] dt={dt_ms:.0f}ms "
                  f"goal_robot=({fwd:.2f},{left:.2f},{dtheta:.2f}) used_omni={used_omni}")

    if logf:
        logf.close()
        print(f"[odom_replay] log: {args.log}")

    if args.gif and out_paths:
        imgs = [Image.open(p) for p in out_paths]
        gif_path = os.path.join(args.out, "odom_replay.gif")
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=250, loop=0)
        print(f"[odom_replay] gif: {gif_path}")
    print(f"[odom_replay] {len(out_paths)} figures in {args.out}/")


if __name__ == "__main__":
    main()
