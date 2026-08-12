#!/usr/bin/env python3
r"""
extract_cache.py -- one ros2 bag -> one compact .npz cache.

WHY: raw bags are ~3 GB each (1920x1280 @ 30 fps). The dataset only ever uses
224x224 frames plus the pose at each frame. Caching those is ~100x smaller, so
100 bags fit in a couple of GB instead of 300.

WHAT IS CACHED
    times   (N,)              float64  frame timestamps
    images  (N, 224, 224, 3)  uint8    resized with the SAME tf.image.resize the
                                       builder uses, so cached frames are
                                       byte-identical to the current pipeline
    poses   (N, 3)            float32  (x, y, theta) interpolated at each frame time

Actions are NOT baked in. flip_frame / decision / turn_done / stride are applied at
BUILD time from this cache, so re-annotating never means re-downloading a bag.

RUN (on MSI, oft env -- needs tensorflow for the resize):
    python extract_cache.py --bag /path/to/rosbag2-keller-t30 --out $SCRATCH/bag_cache/t30.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

for p in (os.path.expanduser("~/SignWay/tools"),
          os.path.expanduser("~/SignWay/signway_dataset")):
    if p not in sys.path:
        sys.path.insert(0, p)

from bag_to_episode import interp_odom, read_bag          # noqa: E402
from tfds_builder import IMAGE_TOPIC, ODOM_TOPIC, ROS_DISTRO, _resize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="path to the ros2 bag directory")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--image-topic", default=None,
                    help="force a topic; default auto-detects from known candidates")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f"[skip] {out.name} already exists")
        return

    bag = Path(args.bag)

    # Collection sessions used different camera topic names: the original bags publish
    # /c1/image_raw, the newer ones /image_raw. Try candidates until one works so both
    # generations cache with one command.
    candidates = ([args.image_topic] if args.image_topic
                  else list(dict.fromkeys([IMAGE_TOPIC, "/image_raw", "/c1/image_raw"])))
    frames, odom, used = None, None, None
    last_err = None
    for topic in candidates:
        try:
            frames, odom = read_bag(bag, topic, ODOM_TOPIC, ROS_DISTRO)
            used = topic
            break
        except (SystemExit, RuntimeError) as e:   # read_bag raises SystemExit
            if "not in bag" in str(e):
                last_err = e
                continue
            raise
    if frames is None:
        print(f"[FAIL] {bag.name}: no usable image topic. Tried {candidates}\n{last_err}")
        sys.exit(2)
    if not frames:
        print(f"[FAIL] {bag.name}: no frames on {used}")
        sys.exit(2)
    if not odom:
        print(f"[FAIL] {bag.name}: no odom on {ODOM_TOPIC}")
        sys.exit(2)

    times = np.asarray([t for t, _ in frames], dtype=np.float64)
    imgs = np.stack([_resize(rgb) for _, rgb in frames]).astype(np.uint8)
    # float64, NOT float32: odom x/y are absolute positions that reach tens of metres,
    # where float32 resolution is already ~4e-6. to_robot_frame subtracts two nearby
    # absolute positions, which amplifies that rounding into the waypoints (measured:
    # 3.6e-6 m of drift vs the uncached pipeline). float64 makes the cache exact, and
    # poses are negligible next to the images.
    poses = np.asarray([interp_odom(odom, t) for t in times], dtype=np.float64)

    # rough fps, so the right stride is obvious at annotation time
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    fps = (1.0 / dt) if dt > 0 else 0.0

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, times=times, images=imgs, poses=poses,
                        fps=np.float32(fps), bag_name=str(bag.name),
                        image_topic=str(used))
    mb = out.stat().st_size / 1e6
    print(f"[ok] {bag.name}: {len(times)} frames @ {fps:.1f}fps from {used} "
          f"-> {out.name} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()