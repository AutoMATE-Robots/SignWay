#!/usr/bin/env python3
"""
overlay_odom_path.py -- draw the path the robot ACTUALLY took onto its own
camera frame, straight from the bag. No policy, no GPU, no separate calibration
step: runs in the `ar` env in under a minute.

HOW. Odometry gives the robot's pose at every frame. Poses after frame F0,
expressed in F0's body frame, are the future path in metres on the ground
plane. A ground plane maps to the image by a pinhole projection, so with the
camera intrinsics plus its height and pitch every point on that path has an
exact pixel.

    q_0 = R(-yaw_0)(p_i - p_0)          # future pose in F0's body frame

CALIBRATION WITHOUT MEASURING. Height and pitch are the only unknowns, and both
are visible in the result: --sweep renders a grid of candidates as one contact
sheet. Pick the cell where the ribbon lies flat along the corridor floor. That
is faster and usually more accurate than a tape measure.

    # 1. sweep -- pick the cell that looks right
    python overlay_odom_path.py --bag ros2_bags/figure_lidar_image \\
        --frame 109 --sweep --out sweep.png

    # 2. render the chosen one
    python overlay_odom_path.py --bag ros2_bags/figure_lidar_image \\
        --frame 109 --cam-height 0.42 --cam-pitch 6 --horizon-s 8 \\
        --out fig_rgb.png
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# camera_info for this robot (k matrix), 1920x1280
FX, FY, CX, CY = 2018.008516, 2060.3797, 1039.039874, 584.01593
DIST = [-0.098641, 0.137388, -0.020093, 0.003674, 0.0]

YUV = {"yuv422", "uyvy", "yuv422_yuy2", "yuyv"}
CH = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4}


def _accept(msg) -> bool:
    enc = (getattr(msg, "encoding", "") or "").lower()
    if enc not in YUV and enc not in CH:
        return False
    return len(msg.data) >= msg.width * (2 if enc in YUV else CH[enc])


def read_bag(bag, image_topic, odom_topic, distro, want_frame):
    """Frame timestamps, odom, and the one decoded frame we need."""
    from bag_to_episode import _decode_image, _make_typestore, _open_reader

    ts = _make_typestore(distro)
    stamps, odom, frame = [], [], None
    idx = 0
    with _open_reader(Path(bag), ts) as reader:
        conns = {c.topic: c for c in reader.connections}
        itopic = image_topic if image_topic in conns else next(
            (t for t in ("/c1/image_raw", "/image_raw") if t in conns), None)
        if itopic is None:
            raise SystemExit(f"no image topic; bag has {sorted(conns)}")
        for conn, t, raw in reader.messages(
                connections=[conns[itopic], conns[odom_topic]]):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == itopic:
                if not _accept(msg):
                    continue
                stamps.append(t / 1e9)
                if idx == want_frame:
                    frame = _decode_image(msg)
                idx += 1
            else:
                p = msg.pose.pose
                q = p.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y ** 2 + q.z ** 2))
                odom.append((t / 1e9, float(p.position.x),
                             float(p.position.y), float(yaw)))
    return stamps, odom, frame, itopic


def path_in_body_frame(stamps, odom, f0, horizon_s):
    """Future odom poses expressed in frame f0's body frame -> [(fwd, left)]."""
    from bag_to_episode import interp_odom

    t0 = stamps[f0]
    x0, y0, th0 = interp_odom(odom, t0)
    c0, s0 = math.cos(-th0), math.sin(-th0)
    out = []
    for t in stamps[f0:]:
        if t - t0 > horizon_s:
            break
        x, y, _ = interp_odom(odom, t)
        dx, dy = x - x0, y - y0
        out.append((c0 * dx - s0 * dy, s0 * dx + c0 * dy))
    return out


def project(pts, h, pitch_deg, x_off=0.0):
    """(fwd, left) on the ground -> (u, v). None when behind the camera."""
    th = math.radians(pitch_deg)
    ct, st = math.cos(th), math.sin(th)
    out = []
    for f, l in pts:
        f = f - x_off
        xc, yc, zc = -l, h * ct - f * st, h * st + f * ct
        out.append(None if zc <= 1e-6 else (FX * xc / zc + CX, FY * yc / zc + CY))
    return out


def draw(img, pts, h, pitch, width_m=0.5, color=(232, 155, 76), alpha=0.55,
         grid=False):
    import cv2

    im = img.copy()
    if grid:
        for x in (1, 2, 3, 4, 6):
            seg = project([(x, -0.9), (x, 0.9)], h, pitch)
            if all(s is not None for s in seg):
                a, b = np.int32(seg[0]), np.int32(seg[1])
                cv2.line(im, tuple(a), tuple(b), (190, 190, 190), 1, cv2.LINE_AA)
                cv2.putText(im, f"{x}m", (int(b[0]) + 6, int(b[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1,
                            cv2.LINE_AA)
    if len(pts) < 2:
        return im
    left, right = [], []
    for i in range(len(pts)):
        j = min(i + 1, len(pts) - 1)
        k = max(i - 1, 0)
        dx, dy = pts[j][0] - pts[k][0], pts[j][1] - pts[k][1]
        n = math.hypot(dx, dy)
        if n < 1e-9:
            continue
        nx, ny = -dy / n, dx / n
        w = width_m / 2
        left.append((pts[i][0] + nx * w, pts[i][1] + ny * w))
        right.append((pts[i][0] - nx * w, pts[i][1] - ny * w))
    poly = [p for p in project(left + right[::-1], h, pitch) if p is not None]
    if len(poly) < 3:
        return im
    pp = np.array(poly, np.int32)
    ov = im.copy()
    cv2.fillPoly(ov, [pp], color, lineType=cv2.LINE_AA)
    im = cv2.addWeighted(ov, alpha, im, 1 - alpha, 0)
    cv2.polylines(im, [pp], True, color, 2, cv2.LINE_AA)
    cl = [p for p in project(pts, h, pitch) if p is not None]
    if len(cl) > 1:
        cv2.polylines(im, [np.array(cl, np.int32)], False, (255, 255, 255), 3,
                      cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon-s", type=float, default=8.0,
                    help="seconds of future path to draw")
    ap.add_argument("--cam-height", type=float, default=0.42)
    ap.add_argument("--cam-pitch", type=float, default=6.0)
    ap.add_argument("--cam-x-offset", type=float, default=0.0)
    ap.add_argument("--width-m", type=float, default=0.5,
                    help="ribbon width = robot footprint")
    ap.add_argument("--color", default="232,155,76", help="B,G,R")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--grid", action="store_true", help="ground rulers")
    ap.add_argument("--sweep", action="store_true",
                    help="contact sheet over height x pitch -- pick by eye")
    ap.add_argument("--undistort", action="store_true", default=True)
    ap.add_argument("--image-topic", default="/image_raw")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--ros-distro", default="humble")
    args = ap.parse_args()

    import cv2

    print(f"reading {Path(args.bag).name} ...")
    stamps, odom, frame, itopic = read_bag(
        args.bag, args.image_topic, args.odom_topic, args.ros_distro, args.frame)
    if frame is None:
        raise SystemExit(f"frame {args.frame} not found ({len(stamps)} frames)")
    print(f"{len(stamps)} frames, {len(odom)} odom samples, topic={itopic}")

    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if args.undistort:
        K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], np.float64)
        img = cv2.undistort(img, K, np.array(DIST, np.float64), None, K)

    pts = path_in_body_frame(stamps, odom, args.frame, args.horizon_s)
    if pts:
        ext = max(p[0] for p in pts)
        lat = max(abs(p[1]) for p in pts)
        print(f"path: {len(pts)} poses over {args.horizon_s}s, "
              f"{ext:.2f} m forward, {lat:.2f} m lateral")
    color = tuple(int(v) for v in args.color.split(","))

    if args.sweep:
        heights = [0.30, 0.40, 0.50, 0.60]
        pitches = [0.0, 4.0, 8.0, 12.0]
        cells = []
        for h in heights:
            row = []
            for p in pitches:
                c = draw(img, pts, h, p, args.width_m, color, args.alpha,
                         grid=True)
                c = cv2.resize(c, (480, 320))
                cv2.putText(c, f"h={h:.2f} pitch={p:.0f}", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3,
                            cv2.LINE_AA)
                cv2.putText(c, f"h={h:.2f} pitch={p:.0f}", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                            cv2.LINE_AA)
                row.append(c)
            cells.append(np.hstack(row))
        cv2.imwrite(args.out, np.vstack(cells))
        print(f"\nsweep -> {args.out}")
        print("Pick the cell where the ribbon lies flat along the corridor "
              "floor and runs\nwhere the robot actually drove, then rerun with "
              "those --cam-height/--cam-pitch.")
        return

    out = draw(img, pts, args.cam_height, args.cam_pitch, args.width_m, color,
               args.alpha, grid=args.grid)
    cv2.imwrite(args.out, out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
