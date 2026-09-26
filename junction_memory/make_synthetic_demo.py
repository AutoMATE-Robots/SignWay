#!/usr/bin/env python3
"""
make_synthetic_demo.py
----------------------
End-to-end rehearsal of the real-bag workflow on a synthetic square bag:

  1. generates the square scenario (north, read sign, LEFT at A for goal_A,
     loop the block, retrace the same corridor, STRAIGHT through A for goal_E)
  2. injects the Hunter-style +20% rotational scale error (k_err = 1.2)
  3. writes a genuine rosbag2 (.db3) with /odom + /c1/image_raw (procedural
     corridor frames with a sign board on both approaches to A)
  4. runs the real toolchain in order:
        odom_memory_probe --bag ... --dump-csv
        calibrate_yaw --csv ... --expected-turn 360
        replay_ar_memory --bag ... --csv odom_calibrated.csv --annotations ...
  5. checks the outcome: 1 VLM call instead of 2, recall lead >> proximity.

Run:  python make_synthetic_demo.py --out demo
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_yaw import reintegrate                       # noqa: E402
from synth_traj import traj                                 # noqa: E402

HZ_ODOM = 20.0
HZ_IMG = 2.0
IMG_W, IMG_H = 320, 240
V = 0.8


# ----------------------------------------------------------------------------
def ground_truth():
    return traj([
        ("straight", 25.0), ("turn", 90, 1.5),
        ("straight", 38.0), ("turn", 90, 1.5),
        ("straight", 23.0), ("turn", 90, 1.5),
        ("straight", 38.0), ("turn", 90, 1.5),
        ("straight", 23.0), ("straight", 8.0),
    ], hz=HZ_ODOM, v=V, x0=40.0, y0=-25.0, yaw0=math.pi / 2)


def corrupt(t, x, y, yaw, k_err=1.2):
    """What the robot RECORDED: every rotation over-reported by k_err."""
    xc, yc, yawc = reintegrate(t, x, y, yaw, k_err)
    return xc, yc, yawc


# ----------------------------------------------------------------------------
# procedural camera: one-point corridor + a sign board near node A
# ----------------------------------------------------------------------------
def render_cam(xg, yg, yawg, s_now, s_A_first, s_A_second):
    img = np.full((IMG_H, IMG_W, 3), (46, 42, 40), np.uint8)
    cx, cy = IMG_W // 2, int(IMG_H * 0.52)
    cv2.rectangle(img, (0, cy), (IMG_W, IMG_H), (70, 66, 62), -1)   # floor
    for f in (0.12, 0.3, 0.55, 0.8):
        w = int(IMG_W * (1 - f) / 2)
        h = int((IMG_H - cy) * (1 - f))
        cv2.rectangle(img, (w, cy - int(cy * (1 - f))),
                      (IMG_W - w, cy + h), (58, 54, 52), 1)
    for sA, tag in ((s_A_first, 1), (s_A_second, 2)):
        d = sA - s_now
        if 2.0 < d < 20.0:
            sc = np.clip(6.0 / d, 0.25, 2.2)
            bw, bh = int(90 * sc), int(46 * sc)
            bx = cx - bw // 2 + int(18 * sc)
            by = cy - int(56 * sc)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (30, 30, 30), -1)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
            fs = 0.32 * sc
            cv2.putText(img, "goal_A <-", (bx + 4, by + int(bh * 0.42)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 235, 235), 1,
                        cv2.LINE_AA)
            cv2.putText(img, "goal_E ^", (bx + 4, by + int(bh * 0.86)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 235, 235), 1,
                        cv2.LINE_AA)
    cv2.putText(img, "/c1 synthetic", (6, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (140, 140, 140), 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------
# rosbag2 writer
# ----------------------------------------------------------------------------
def write_bag(bag_dir, t, x, y, yaw, s, s_A1, s_A2):
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    Odom = ts.types["nav_msgs/msg/Odometry"]
    Image = ts.types["sensor_msgs/msg/Image"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Pose = ts.types["geometry_msgs/msg/Pose"]
    PoseC = ts.types["geometry_msgs/msg/PoseWithCovariance"]
    Point = ts.types["geometry_msgs/msg/Point"]
    Quat = ts.types["geometry_msgs/msg/Quaternion"]
    Twist = ts.types["geometry_msgs/msg/Twist"]
    TwistC = ts.types["geometry_msgs/msg/TwistWithCovariance"]
    Vec3 = ts.types["geometry_msgs/msg/Vector3"]

    if Path(bag_dir).exists():
        shutil.rmtree(bag_dir)
    cov = np.zeros(36, dtype=np.float64)

    with Writer(Path(bag_dir), version=8) as w:
        co = w.add_connection("/odom", Odom.__msgtype__, typestore=ts)
        ci = w.add_connection("/c1/image_raw", Image.__msgtype__, typestore=ts)
        img_period = 1.0 / HZ_IMG
        next_img = t[0]
        for i in range(len(t)):
            tns = int(t[i] * 1e9)
            stamp = Time(sec=int(t[i]), nanosec=int((t[i] % 1) * 1e9))
            hdr = Header(stamp=stamp, frame_id="odom")
            q = Quat(x=0.0, y=0.0, z=math.sin(yaw[i] / 2),
                     w=math.cos(yaw[i] / 2))
            msg = Odom(header=hdr, child_frame_id="base_link",
                       pose=PoseC(pose=Pose(position=Point(x=float(x[i]),
                                                           y=float(y[i]), z=0.0),
                                            orientation=q), covariance=cov),
                       twist=TwistC(twist=Twist(linear=Vec3(x=V, y=0.0, z=0.0),
                                                angular=Vec3(x=0.0, y=0.0,
                                                             z=0.0)),
                                    covariance=cov))
            w.write(co, tns, ts.serialize_cdr(msg, Odom.__msgtype__))
            if t[i] >= next_img:
                next_img += img_period
                frame = render_cam(x[i], y[i], yaw[i], s[i], s_A1, s_A2)
                im = Image(header=Header(stamp=stamp, frame_id="c1"),
                           height=IMG_H, width=IMG_W, encoding="bgr8",
                           is_bigendian=0, step=IMG_W * 3,
                           data=frame.reshape(-1).astype(np.uint8))
                w.write(ci, tns, ts.serialize_cdr(im, Image.__msgtype__))
    print(f"[demo] wrote bag {bag_dir}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo")
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent

    # ground truth + recorded (corrupted) odom
    t, xg, yg, yawg = ground_truth()
    t = t + 1_000_000.0                       # bag-style epoch offset
    sg = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xg), np.diff(yg)))])
    s_A1 = 25.0
    s_A2 = float(sg[-1] - 8.0)
    xr, yr, yawr = corrupt(t, xg, yg, yawg, k_err=1.2)

    bag_dir = out / "demo_square_bag"
    write_bag(bag_dir, t, xr, yr, yawr, sg, s_A1, s_A2)

    # annotations, seconds-keyed off the ground-truth arc length
    t0 = float(t[0])
    t_at = lambda s_m: float(t[np.searchsorted(sg, s_m)] - t0)
    ann = {
        "bag": bag_dir.name,
        "goals": [{"from_s": 0.0, "goal": "goal_A"},
                  {"from_s": t_at(45.0), "goal": "goal_E"}],
        "signs": [
            {"arm_s": t_at(13.0), "decide_s": t_at(20.0), "decision": "left",
             "conf": 0.91,
             "directory": {"goal_A": "left", "goal_E": "straight"}},
            {"arm_s": t_at(sg[-1] - 8.0 - 12.0),
             "decide_s": t_at(sg[-1] - 8.0 - 5.0), "decision": "straight",
             "conf": 0.90,
             "directory": {"goal_A": "left", "goal_E": "straight"}},
        ],
    }
    ann_path = out / "demo_annotations.yaml"
    ann_path.write_text(yaml.safe_dump(ann, sort_keys=False))
    print(f"[demo] wrote {ann_path}")

    def sh(cmd):
        print("\n[demo] $ " + " ".join(str(c) for c in cmd))
        r = subprocess.run([sys.executable] + [str(c) for c in cmd],
                           cwd=str(here))
        if r.returncode != 0:
            sys.exit(r.returncode)

    # the real-bag toolchain, verbatim
    sh([here / "odom_memory_probe.py", "--bag", bag_dir,
        "--dump-csv", out / "odom.csv", "--json", out / "odom_probe.json",
        "--plot", out / "odom_probe.png"])
    sh([here / "calibrate_yaw.py", "--csv", out / "odom.csv",
        "--expected-turn", "360", "--out", out / "odom_calibrated.csv",
        "--plot", out / "odom_calibration.png"])
    fast = out / "out_fast"
    sh([here / "replay_ar_memory.py", "--csv", out / "odom_calibrated.csv",
        "--annotations", ann_path, "--no-video", "--out", fast])
    if not args.skip_video:
        vid = out / "out_video"
        sh([here / "replay_ar_memory.py", "--bag", bag_dir,
            "--csv", out / "odom_calibrated.csv", "--annotations", ann_path,
            "--out", vid])
        print(f"\n[demo] video: {vid/'replay.mp4'}")

    print("\n[demo] done. Fast-loop summary:")
    print((fast / "summary.txt").read_text())


if __name__ == "__main__":
    main()