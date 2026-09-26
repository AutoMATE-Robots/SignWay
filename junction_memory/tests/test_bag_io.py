#!/usr/bin/env python3
"""
test_bag_io.py — bag round-trip + the §7.3 no-typedefs fallback.

The real bag (rosbag2_2026_08_22-20_29_37) was written without embedded
message definitions, so AnyReader refuses it and the reader must fall back to
a distro typestore read from the db3 `schema` table. This test writes a tiny
bag, STRIPS its message definitions the same way, and proves the replay
tool's open_bag() still decodes odometry and images.
"""
import math
import sqlite3
from glob import glob
from pathlib import Path

import numpy as np
import pytest


def _write_tiny_bag(bag_dir: Path, n=120, with_images=True):
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    Odom = ts.types["nav_msgs/msg/Odometry"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Pose = ts.types["geometry_msgs/msg/Pose"]
    PoseC = ts.types["geometry_msgs/msg/PoseWithCovariance"]
    Point = ts.types["geometry_msgs/msg/Point"]
    Quat = ts.types["geometry_msgs/msg/Quaternion"]
    Twist = ts.types["geometry_msgs/msg/Twist"]
    TwistC = ts.types["geometry_msgs/msg/TwistWithCovariance"]
    Vec3 = ts.types["geometry_msgs/msg/Vector3"]
    Image = ts.types["sensor_msgs/msg/Image"]
    cov = np.zeros(36, dtype=np.float64)

    with Writer(bag_dir, version=8) as w:
        co = w.add_connection("/odom", Odom.__msgtype__, typestore=ts)
        ci = (w.add_connection("/c1/image_raw", Image.__msgtype__, typestore=ts)
              if with_images else None)
        for i in range(n):
            t = 100.0 + i * 0.05
            x, y, yaw = 0.4 * i * 0.05, 0.0, 0.0
            stamp = Time(sec=int(t), nanosec=int((t % 1) * 1e9))
            hdr = Header(stamp=stamp, frame_id="odom")
            msg = Odom(header=hdr, child_frame_id="base_link",
                       pose=PoseC(pose=Pose(
                           position=Point(x=x, y=y, z=0.0),
                           orientation=Quat(x=0.0, y=0.0,
                                            z=math.sin(yaw / 2),
                                            w=math.cos(yaw / 2))),
                           covariance=cov),
                       twist=TwistC(twist=Twist(linear=Vec3(x=0.4, y=0.0, z=0.0),
                                                angular=Vec3(x=0.0, y=0.0, z=0.0)),
                                    covariance=cov))
            w.write(co, int(t * 1e9), ts.serialize_cdr(msg, Odom.__msgtype__))
            if ci is not None and i % 10 == 0:
                frame = np.full((24, 32, 3), i % 255, np.uint8)
                im = Image(header=hdr, height=24, width=32, encoding="bgr8",
                           is_bigendian=0, step=96,
                           data=frame.reshape(-1))
                w.write(ci, int(t * 1e9), ts.serialize_cdr(im, Image.__msgtype__))


def _strip_typedefs(bag_dir: Path):
    """Reproduce the real bag's condition: no embedded message definitions,
    distro recoverable only from the db3 `schema` table (README §7.3)."""
    for db in glob(str(bag_dir / "*.db3")):
        c = sqlite3.connect(db)
        tables = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "message_definitions" in tables:
            c.execute("DELETE FROM message_definitions")
        cols = [r[1] for r in c.execute("PRAGMA table_info(topics)")]
        if "offered_qos_profiles" in cols:
            pass
        if "type_description_hash" in cols:
            c.execute("UPDATE topics SET type_description_hash=''")
        if "schema" not in tables:
            c.execute("CREATE TABLE schema (schema_version INTEGER, "
                      "ros_distro TEXT)")
            c.execute("INSERT INTO schema VALUES (3, 'humble')")
        else:
            c.execute("UPDATE schema SET ros_distro='humble'")
        c.commit()
        c.close()
    # metadata.yaml: drop version >=8 hints so AnyReader treats it as legacy
    import yaml
    meta = bag_dir / "metadata.yaml"
    doc = yaml.safe_load(meta.read_text())
    info = doc["rosbag2_bagfile_information"]
    info["version"] = 5
    info["ros_distro"] = "humble"
    for entry in info.get("topics_with_message_count", []):
        entry.get("topic_metadata", {}).pop("type_description_hash", None)
    for key in ("custom_data",):
        info.pop(key, None)
    meta.write_text(yaml.safe_dump(doc, sort_keys=False))


def test_defless_bag_falls_back_and_decodes(tmp_path):
    from rosbags.highlevel import AnyReader
    import replay_ar_memory as R

    bag = tmp_path / "tiny_bag"
    _write_tiny_bag(bag)
    _strip_typedefs(bag)

    # sanity: the naive path must fail exactly like the real bag does
    with pytest.raises(Exception, match="type definitions"):
        r = AnyReader([bag])
        r.open()
        try:
            conns = [c for c in r.connections
                     if c.msgtype == "nav_msgs/msg/Odometry"]
            for conn, tns, raw in r.messages(connections=conns):
                r.deserialize(raw, conn.msgtype)
                break
        finally:
            r.close()

    # the replay tool's opener must recover via the schema-table typestore
    reader = R.open_bag(bag)
    try:
        conns = [c for c in reader.connections
                 if c.msgtype == "nav_msgs/msg/Odometry"]
        assert conns
        n, first_x = 0, None
        for conn, tns, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            if first_x is None:
                first_x = m.pose.pose.position.x
            n += 1
        assert n == 120 and abs(first_x) < 1e-9
        img_conns = R.pick_image_connection(reader, None)
        for conn, tns, raw in reader.messages(connections=img_conns):
            msg = reader.deserialize(raw, conn.msgtype)
            img = R.decode_image(msg, conn.msgtype)
            assert img.shape == (24, 32, 3)
            break
    finally:
        reader.close()


def test_odom_loader_on_defless_bag(tmp_path):
    import replay_ar_memory as R
    bag = tmp_path / "tiny_bag2"
    _write_tiny_bag(bag, with_images=False)
    _strip_typedefs(bag)
    t, x, y, yaw = R.load_odom_from_bag(str(bag))
    assert len(t) == 120
    assert x[-1] > x[0]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
