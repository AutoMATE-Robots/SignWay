#!/usr/bin/env python3
r"""
make_mp4.py -- one ros2 bag (or one .npz cache) -> one mp4 for review/annotation.

Two sources:
  --bag   /path/to/rosbag2-...   full resolution, downscaled to --width (default 640)
  --cache /path/to/xxx.npz       instant, no download, but only 224x224 (squished)

Frame numbers are burned into the corner so you can read annotation indices straight
off the video. These are RAW frame indices -- the same convention flip_frame and
turn_done_frame use.

RUN
    python make_mp4.py --bag $SCRATCH/_tmp/bagdir --out $SCRATCH/mp4/t30.mp4
    python make_mp4.py --cache $SCRATCH/bag_cache/x.npz --out $SCRATCH/mp4/x.mp4
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


def load_from_bag(bag: Path, width: int, odom_topic: str, ros_distro: str):
    # NOTE: deliberately does NOT import tfds_builder -- that pulls in
    # tensorflow_datasets -> tensorflow, ~30s of startup for two string constants.
    # This script only needs rosbags + cv2.
    from bag_to_episode import read_bag
    ODOM_TOPIC, ROS_DISTRO = odom_topic, ros_distro
    last = None
    for topic in ("/image_raw", "/c1/image_raw"):
        try:
            frames, _ = read_bag(bag, topic, ODOM_TOPIC, ROS_DISTRO)
            break
        except (SystemExit, RuntimeError) as e:      # read_bag raises SystemExit
            if "not in bag" in str(e):
                last = e
                continue
            raise
    else:
        print(f"[FAIL] {bag.name}: no usable image topic ({last})")
        sys.exit(2)

    import cv2
    out = []
    for _, rgb in frames:
        h, w = rgb.shape[:2]
        nh = int(round(h * width / w))
        out.append(cv2.resize(rgb, (width, nh), interpolation=cv2.INTER_AREA))
    times = [t for t, _ in frames]
    return out, times


def load_from_cache(cache: Path):
    d = np.load(cache)
    return list(d["images"]), list(d["times"])


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bag")
    src.add_argument("--cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=640, help="output width (bag source only)")
    ap.add_argument("--fps", type=float, default=None, help="override playback fps")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.overwrite:
        print(f"[skip] {out.name} exists")
        return

    if a.bag:
        imgs, times = load_from_bag(Path(a.bag), a.width, a.odom_topic, a.ros_distro)
        label = Path(a.bag).name
    else:
        imgs, times = load_from_cache(Path(a.cache))
        label = Path(a.cache).stem

    if not imgs:
        print(f"[FAIL] {label}: no frames")
        sys.exit(2)

    import cv2
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.033
    fps = a.fps if a.fps else (1.0 / dt if dt > 0 else 30.0)
    h, w = imgs[0].shape[:2]

    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        print("[FAIL] cv2.VideoWriter would not open -- is ffmpeg/openh264 available?")
        sys.exit(3)

    for i, rgb in enumerate(imgs):
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
        # raw frame index, burned in -- this is what annotations refer to
        cv2.rectangle(bgr, (0, 0), (150, 34), (0, 0, 0), -1)
        cv2.putText(bgr, f"f{i}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(bgr)
    vw.release()

    mb = out.stat().st_size / 1e6
    print(f"[ok] {label}: {len(imgs)} frames @ {fps:.1f}fps -> {out.name} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()