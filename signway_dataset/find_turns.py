#!/usr/bin/env python3
"""
find_turns.py -- locate the turns in a bag from ODOMETRY, no video needed.

A turn is where yaw changes. Reading odom is cheap (no image decoding), so this
prints exact frame ranges in seconds rather than requiring an mp4 to be built
and scrubbed.

    python signway_dataset/find_turns.py --bag ros2_bags/figure_lidar_image

Prints, per detected turn, the frame range and total heading change, plus a
ready-made --chain-range argument with approach frames included.

CPU only: needs rosbags + numpy, not torch. Runs on a login node.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

YUV = {"yuv422", "uyvy", "yuv422_yuy2", "yuyv"}
CH = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4}


def _accept(msg) -> bool:
    enc = (getattr(msg, "encoding", "") or "").lower()
    if enc not in YUV and enc not in CH:
        return False
    return len(msg.data) >= msg.width * (2 if enc in YUV else CH[enc])


def read_stamps_odom(bag, image_topic, odom_topic, distro):
    from bag_to_episode import _make_typestore, _open_reader

    ts = _make_typestore(distro)
    stamps, odom = [], []
    with _open_reader(Path(bag), ts) as reader:
        conns = {c.topic: c for c in reader.connections}
        itopic = image_topic if image_topic in conns else next(
            (t for t in ("/c1/image_raw", "/image_raw") if t in conns), None)
        if itopic is None:
            raise SystemExit(f"no image topic; bag has {sorted(conns)}")
        if odom_topic not in conns:
            raise SystemExit(f"no {odom_topic}; bag has {sorted(conns)}")
        for conn, t, raw in reader.messages(
                connections=[conns[itopic], conns[odom_topic]]):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == itopic:
                if _accept(msg):
                    stamps.append(t / 1e9)
            else:
                p = msg.pose.pose
                q = p.orientation
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y ** 2 + q.z ** 2))
                odom.append((t / 1e9, float(p.position.x),
                             float(p.position.y), float(yaw)))
    return stamps, odom, itopic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--image-topic", default="/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    ap.add_argument("--min-turn-deg", type=float, default=25.0,
                    help="ignore heading changes smaller than this")
    ap.add_argument("--rate-thresh", type=float, default=4.0,
                    help="deg/s above which the robot counts as turning")
    ap.add_argument("--approach-s", type=float, default=4.0,
                    help="seconds of approach to include in the suggested range")
    args = ap.parse_args()

    from bag_to_episode import interp_odom

    print(f"reading {Path(args.bag).name} ...")
    stamps, odom, itopic = read_stamps_odom(
        args.bag, args.image_topic, args.odom_topic, args.ros_distro)
    n = len(stamps)
    if n < 10:
        raise SystemExit(f"only {n} frames")
    dt = float(np.median(np.diff(stamps)))
    fps = 1.0 / dt
    print(f"topic={itopic}  {n} frames  {fps:.1f} fps  "
          f"{n * dt:.1f} s  ({len(odom)} odom samples)")

    yaw = np.array([interp_odom(odom, t)[2] for t in stamps])
    yaw = np.unwrap(yaw)
    pos = np.array([interp_odom(odom, t)[:2] for t in stamps])

    k = max(3, int(round(0.5 / dt)) | 1)             # ~0.5 s smoothing, odd
    rate = np.degrees(np.gradient(yaw, dt))
    rate = np.convolve(rate, np.ones(k) / k, mode="same")

    turning = np.abs(rate) > args.rate_thresh
    runs, i = [], 0
    while i < n:
        if turning[i]:
            j = i
            while j < n and turning[j]:
                j += 1
            if (j - i) * dt > 0.6:                   # ignore blips
                runs.append((i, j - 1))
            i = j
        else:
            i += 1

    speed = np.r_[0, np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt]
    print(f"median speed {np.median(speed[speed > 0.05]):.2f} m/s")

    if not runs:
        print("\nno turns found -- lower --rate-thresh, or this bag is straight")
        return

    print(f"\n{len(runs)} turn(s):")
    print(f"{'#':>2} {'frames':>16} {'dur':>6} {'heading':>9}  suggested "
          f"--chain-range (with {args.approach_s:.0f}s approach)")
    print("-" * 92)
    approach = int(round(args.approach_s / dt))
    for m, (a, b) in enumerate(runs, 1):
        d = np.degrees(yaw[b] - yaw[a])
        if abs(d) < args.min_turn_deg:
            continue
        f0 = max(0, a - approach)
        f1 = min(n - 1, b + int(round(1.0 / dt)))
        step = max(1, int(round(0.1 / dt)))          # ~10 Hz sampling
        print(f"{m:>2} {a:>7}-{b:<8} {(b-a)*dt:5.1f}s {d:+8.1f}deg  "
              f"--chain-range {f0}:{f1}:{step}   "
              f"({'LEFT' if d > 0 else 'RIGHT'})")

    print(f"\nstride for this bag: {int(round(fps / 10))}  "
          f"(native {fps:.1f} fps -> 10 Hz)")
    print("Pick the turn you want, copy its --chain-range into "
          "run_vla_on_bag.py.")


if __name__ == "__main__":
    main()
