#!/usr/bin/env python3
"""
screen_bags.py -- fast health check for a whole directory of ROS2 bags BEFORE labeling/training.

For every bag it reports: image count, fps (from timestamps, no decoding), odom count, whether
the odom is LIVE (position actually changes) or DEAD (frozen pose, like the bad left-turn bag),
net yaw (turn direction), and final displacement. One screen, all bags, in seconds each --
image messages are never decoded, only counted and timestamped.

    python tools/screen_bags.py --root ros2_bags
    python tools/screen_bags.py --root ros2_bags --odom-topic /odom --image-topic /c1/image_raw
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def screen_bag(bag_path: Path, image_topic: str, odom_topic: str, distro: str):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bag_to_episode import _make_typestore, _open_reader

    ts = _make_typestore(distro)
    img_times, poses = [], []
    with _open_reader(bag_path, ts) as reader:
        conns = {c.topic: c for c in reader.connections}
        wanted = [conns[t] for t in (image_topic, odom_topic) if t in conns]
        missing = [t for t in (image_topic, odom_topic) if t not in conns]
        for conn, t, raw in reader.messages(connections=wanted):
            if conn.topic == image_topic:
                img_times.append(t * 1e-9)          # timestamp only -- no decode
            else:
                m = reader.deserialize(raw, conn.msgtype)
                p, q = m.pose.pose.position, m.pose.pose.orientation
                poses.append((float(p.x), float(p.y),
                              yaw_from_quat(q.x, q.y, q.z, q.w)))

    r = {"bag": bag_path.name, "missing": missing,
         "n_images": len(img_times), "n_odom": len(poses)}
    if len(img_times) > 1:
        r["fps"] = (len(img_times) - 1) / (img_times[-1] - img_times[0])
    if poses:
        xs = np.array([p[0] for p in poses]); ys = np.array([p[1] for p in poses])
        travel = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
        r["travel_m"] = travel
        r["live"] = travel > 0.10
        net = math.degrees(poses[-1][2] - poses[0][2])
        r["net_yaw_deg"] = (net + 180) % 360 - 180
        # final displacement in the start frame
        th0 = poses[0][2]
        dx, dy = xs[-1] - xs[0], ys[-1] - ys[0]
        c, s = math.cos(-th0), math.sin(-th0)
        r["fwd_m"], r["left_m"] = c * dx - s * dy, s * dx + c * dy
    return r


def verdict(r):
    if r["missing"]:
        return f"MISSING TOPICS {r['missing']}"
    if r["n_odom"] == 0:
        return "NO ODOM MSGS"
    if not r.get("live", False):
        return "DEAD ODOM (frozen pose) -- unusable"
    yaw = r["net_yaw_deg"]
    if yaw > 30:
        turn = "LEFT turn"
    elif yaw < -30:
        turn = "RIGHT turn"
    else:
        turn = "straight-ish"
    return f"LIVE -- {turn}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ros2_bags")
    ap.add_argument("--image-topic", default="/c1/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    args = ap.parse_args()

    root = Path(args.root)
    bags = sorted([d for d in root.iterdir()
                   if d.is_dir() and (d / "metadata.yaml").exists()])
    print(f"{len(bags)} bags under {root}\n")
    print(f"{'bag':32s} {'imgs':>5s} {'fps':>5s} {'odom':>5s} {'travel':>7s} "
          f"{'netYaw':>7s} {'fwd':>6s} {'left':>6s}  verdict")
    print("-" * 110)
    for b in bags:
        try:
            r = screen_bag(b, args.image_topic, args.odom_topic, args.ros_distro)
            print(f"{r['bag'][:32]:32s} {r['n_images']:>5d} "
                  f"{r.get('fps', 0):>5.1f} {r['n_odom']:>5d} "
                  f"{r.get('travel_m', 0):>6.1f}m "
                  f"{r.get('net_yaw_deg', 0):>+6.1f}° "
                  f"{r.get('fwd_m', 0):>+5.1f}m {r.get('left_m', 0):>+5.1f}m  "
                  f"{verdict(r)}")
        except Exception as e:
            print(f"{b.name[:32]:32s}  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
