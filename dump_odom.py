#!/usr/bin/env python3
"""Dump per-frame odom poses for a bag -> odom_<bag>.npz for fig_path.py.

CPU only, seconds per bag, no GPU -- it never decodes image data, only reads
image message TIMESTAMPS so odom can be interpolated onto the same frames the
dataset uses.

Reader: tries `rosbags` (pure Python, no ROS install) first, then rclpy +
rosbag2_py. Run it in whichever env can already run eval_openloop.py on a bag
-- that env demonstrably has a working reader.

  PYTHONPATH=~/SignWay/tools:$PYTHONPATH python dump_odom.py \
      --bag /users/1/munda057/SignWay/ros2_bags/rosbag2-keller-e2 \
      --stride 2 --out odom_e2.npz

Output arrays: frames (raw indices of the strided frames, matching the
"frames" array in series.npy), x, y, yaw.
"""
import argparse
from pathlib import Path

import numpy as np


def _yaw(qw, qx, qy, qz):
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)))


def _read_rosbags(bag, image_topic, odom_topic, distro="humble"):
    """Pure-Python reader; handles both .db3 and .mcap."""
    from rosbags.highlevel import AnyReader

    # Older rosbag2 .db3 files carry no embedded type definitions, so the
    # reader has to be handed the distro's typestore explicitly.
    try:
        from rosbags.typesys import Stores, get_typestore
        store = get_typestore(getattr(Stores, f"ROS2_{distro.upper()}"))
    except Exception:                                        # noqa: BLE001
        store = None

    stamps, odom = [], []
    with AnyReader([Path(bag)], default_typestore=store) as reader:
        want = {image_topic, odom_topic}
        conns = [c for c in reader.connections if c.topic in want]
        if not conns:
            raise RuntimeError(
                f"neither {image_topic} nor {odom_topic} in bag; topics are: "
                + ", ".join(sorted({c.topic for c in reader.connections})))
        for conn, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if conn.topic == image_topic:
                stamps.append(t)
            else:
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                odom.append((t, p.x, p.y, _yaw(q.w, q.x, q.y, q.z)))
    return stamps, odom


def _read_rclpy(bag, image_topic, odom_topic, distro="humble"):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    stamps, odom = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in (image_topic, odom_topic):
            continue
        msg = deserialize_message(data, get_message(types[topic]))
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if topic == image_topic:
            stamps.append(t)
        else:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            odom.append((t, p.x, p.y, _yaw(q.w, q.x, q.y, q.z)))
    return stamps, odom


def read_bag(bag, image_topic, odom_topic, distro="humble"):
    errors = []
    for name, fn in (("rosbags", _read_rosbags), ("rclpy", _read_rclpy)):
        try:
            out = fn(bag, image_topic, odom_topic, distro)
            print(f"[reader] {name}")
            return out
        except ImportError as e:
            errors.append(f"{name}: {e}")
    raise SystemExit(
        "no usable ROS bag reader in this env.\n  " + "\n  ".join(errors)
        + "\nRun in the env that runs eval_openloop.py, or: pip install rosbags")


def _interp(odom, t):
    """Fallback linear interpolation, used only if bag_to_episode has none."""
    ts = np.array([o[0] for o in odom])
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    t0, t1 = ts[i - 1], ts[i]
    f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    a, b = odom[i - 1], odom[i]
    dyaw = (b[3] - a[3] + np.pi) % (2 * np.pi) - np.pi      # shortest arc
    return (a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * dyaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--stride", type=int, required=True, help="2 or 3")
    ap.add_argument("--image-topic", default="/c1/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble",
                    help="typestore for bags without embedded type defs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # prefer the project's own interpolation so poses match the dataset exactly
    try:
        from bag_to_episode import interp_odom
        print("[interp] bag_to_episode.interp_odom")
    except Exception:                                        # noqa: BLE001
        interp_odom = _interp
        print("[interp] local fallback (bag_to_episode not importable)")

    stamps, odom = read_bag(args.bag, args.image_topic, args.odom_topic)
    print(f"[bag] {len(stamps)} image frames, {len(odom)} odom samples")
    if not stamps or not odom:
        raise SystemExit("empty topic(s) -- check --image-topic / --odom-topic")

    frames = list(range(0, len(stamps), args.stride))
    P = np.asarray([interp_odom(odom, stamps[f]) for f in frames],
                   dtype=np.float64)
    np.savez(args.out, frames=np.asarray(frames), x=P[:, 0], y=P[:, 1],
             yaw=P[:, 2])
    span = np.hypot(P[:, 0] - P[0, 0], P[:, 1] - P[0, 1]).max()
    turned = np.degrees(np.unwrap(P[:, 2])[-1] - P[0, 2])
    print(f"wrote {args.out}  ({len(frames)} strided frames, "
          f"{span:.1f} m extent, {turned:+.0f} deg net heading)")


if __name__ == "__main__":
    main()