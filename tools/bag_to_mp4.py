#!/usr/bin/env python3
"""
bag_to_mp4.py -- ROS2 bag -> mp4 for annotation, streaming (constant memory).

CRITICAL: frame indices in the mp4 must match the indices eval_openloop counts
when it reads the bag, or every annotated frame number is silently offset. This
applies the SAME accept/reject rule as bag_to_episode.read_bag -- a message
counts as a frame only if its encoding is supported and its first row is
complete -- and it writes one video frame per accepted message. Index i in the
mp4 is therefore index i in the eval.

A 33 GB bag is ~3000+ frames at 1920x1080; decoded that is far more than RAM,
so frames are written to the encoder as they are read and never accumulated.

    python tools/bag_to_mp4.py --bag ros2_bags/rosbag2-square --out square.mp4
    python tools/bag_to_mp4.py --bag <b> --out <o> --scale 0.5   # smaller file

Writes <out> and prints the frame count + fps, which are what the annotator and
the eval both key off.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

YUV = {"yuv422", "uyvy", "yuv422_yuy2", "yuyv"}
CH = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-topic", default=None,
                    help="default: auto-detect /c1/image_raw or /image_raw")
    ap.add_argument("--ros-distro", default="humble")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="resize factor for a smaller file (indices unchanged)")
    ap.add_argument("--fps", type=float, default=None,
                    help="override; default = measured from message timestamps")
    args = ap.parse_args()

    from bag_to_episode import _decode_image, _make_typestore, _open_reader

    ts = _make_typestore(args.ros_distro)
    bag = Path(args.bag)

    # pass 1: pick the topic and measure fps from timestamps (cheap, no decode)
    with _open_reader(bag, ts) as reader:
        conns = {c.topic: c for c in reader.connections}
        topic = args.image_topic
        if topic is None:
            for cand in ("/c1/image_raw", "/image_raw"):
                if cand in conns:
                    topic = cand
                    break
        if topic is None or topic not in conns:
            raise SystemExit(f"no image topic; bag has {sorted(conns)}")
        stamps = [t for _c, t, _r in reader.messages(connections=[conns[topic]])]
    if len(stamps) < 2:
        raise SystemExit("fewer than 2 messages on the image topic")
    dt = np.diff(np.array(stamps, dtype=float) / 1e9)
    fps = args.fps or float(1.0 / np.median(dt))
    print(f"topic={topic}  messages={len(stamps)}  measured fps={fps:.2f}")

    # pass 2: decode + write, streaming
    writer, n_written, n_seen = None, 0, 0
    with _open_reader(bag, ts) as reader:
        conns = {c.topic: c for c in reader.connections}
        for conn, _t, raw in reader.messages(connections=[conns[topic]]):
            msg = reader.deserialize(raw, conn.msgtype)
            enc = (getattr(msg, "encoding", "") or "").lower()
            n_seen += 1
            # SAME accept rule as read_bag -- keeps indices aligned
            if enc not in YUV and enc not in CH:
                continue
            if len(msg.data) < msg.width * (2 if enc in YUV else CH[enc]):
                continue
            rgb = _decode_image(msg)
            if rgb is None:
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if args.scale != 1.0:
                bgr = cv2.resize(bgr, None, fx=args.scale, fy=args.scale,
                                 interpolation=cv2.INTER_AREA)
            if writer is None:
                h, w = bgr.shape[:2]
                # even dimensions: H.264 requires them
                w -= w % 2
                h -= h % 2
                vw, vh = w, h
                writer = cv2.VideoWriter(args.out,
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (vw, vh))
                if not writer.isOpened():
                    raise SystemExit("VideoWriter failed to open")
                print(f"writing {w}x{h} @ {fps:.2f} fps -> {args.out}")
            writer.write(bgr[:vh, :vw])
            n_written += 1
            if n_written % 250 == 0:
                print(f"  {n_written} frames...", flush=True)

    if writer:
        writer.release()
    print(f"\ndone: {n_written} frames written ({n_seen} messages seen)")
    print(f"ANNOTATION INDICES 0..{n_written - 1} match the eval's frame indices.")
    print(f"\nmp4v may not play in QuickTime -- re-encode for annotation:\n"
          f"  ffmpeg -y -i {args.out} -c:v libx264 -pix_fmt yuv420p "
          f"-vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" "
          f"{Path(args.out).with_suffix('')}_h264.mp4")


if __name__ == "__main__":
    main()