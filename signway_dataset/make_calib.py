#!/usr/bin/env python3
"""
make_calib.py -- turn camera_info + one click + one tape measurement into the
calibration overlay_traj.py needs.

camera_info gives the INTRINSICS (fx, fy, cx, cy, distortion). Ground-plane
projection also needs the EXTRINSICS, which camera_info never carries:

    camera height   above the floor -- measure with a tape, once.
    camera pitch    downward tilt -- recovered here from the image.

PITCH FROM THE VANISHING POINT. Parallel lines on the ground converge at the
horizon. For a camera pitched down by theta, a point infinitely far ahead
projects to v = cy - fy*tan(theta), so

    theta = atan( (cy - v_horizon) / fy ).

In a corridor the floor/wall junction lines give that point precisely, and it is
far easier to click accurately than a tilt is to measure.

    python make_calib.py --image frame_01850.png \\
        --fx 2018.008516 --fy 2060.3797 --cx 1039.039874 --cy 584.01593 \\
        --dist -0.098641,0.137388,-0.020093,0.003674,0.0 \\
        --cam-height 0.42 --out intrinsics.json

Click where the corridor's floor lines converge, press q. Then check the printed
FOV and the verification image before trusting it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def pick_vanishing_point(img):
    disp = img.copy()
    h, w = img.shape[:2]
    pt = {}
    win = "click the vanishing point (where floor/wall lines converge) | q=done"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pt["v"] = (x, y)
            d = img.copy()
            cv2.line(d, (0, y), (w, y), (60, 220, 60), 2)
            cv2.circle(d, (x, y), 8, (60, 220, 60), 2)
            cv2.putText(d, f"horizon v={y}", (20, max(30, y - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 220, 60), 2)
            cv2.imshow(win, d)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, disp)
    print("\nClick where the corridor's parallel lines converge, then press q.")
    while True:
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    if "v" not in pt:
        raise SystemExit("no point clicked")
    return pt["v"]


def verify(img, calib, out_path):
    """Draw ground-plane rulers. If these do not sit on the floor where the
    tiles say they should, the calibration is wrong -- fix it before drawing
    any trajectory."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from overlay_traj import draw_grid, load_calib

    tmp = Path(out_path).with_suffix(".tmp.json")
    tmp.write_text(json.dumps(calib))
    c = load_calib(tmp)
    tmp.unlink()
    out = draw_grid(img.copy(), c, xs=(1, 2, 3, 4, 6), y_half=0.9)
    p = str(Path(out_path).with_name("calib_check.png"))
    cv2.imwrite(p, out)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--dist", default=None,
                    help="k1,k2,p1,p2,k3 from camera_info 'd'")
    ap.add_argument("--cam-height", type=float, required=True,
                    help="camera height above the FLOOR, metres")
    ap.add_argument("--cam-x-offset", type=float, default=0.0,
                    help="camera forward offset from the body origin, metres")
    ap.add_argument("--pitch-deg", type=float, default=None,
                    help="skip the click and set the pitch directly")
    ap.add_argument("--out", default="intrinsics.json")
    args = ap.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    h, w = img.shape[:2]
    print(f"image {w}x{h}")

    dist = None
    if args.dist:
        dist = [float(v) for v in args.dist.split(",")]
        # Undistort first, then project with the same K. Tangential terms here
        # (p1 = -0.020) are large enough to bend straight floor lines
        # noticeably at the frame edges.
        K = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]],
                     np.float64)
        und = cv2.undistort(img, K, np.array(dist, np.float64), None, K)
        p_und = str(Path(args.out).with_name("undistorted.png"))
        cv2.imwrite(p_und, und)
        print(f"undistorted preview -> {p_und}")
        img = und

    if args.pitch_deg is not None:
        pitch = args.pitch_deg
        vp = None
    else:
        vp = pick_vanishing_point(img)
        pitch = math.degrees(math.atan((args.cy - vp[1]) / args.fy))
        print(f"\nvanishing point {vp} -> pitch {pitch:+.2f} deg "
              f"({'down' if pitch > 0 else 'UP -- check the click'})")
        dyaw = math.degrees(math.atan((vp[0] - args.cx) / args.fx))
        print(f"camera yaw vs corridor axis: {dyaw:+.2f} deg "
              f"(large values mean the robot was not aligned with the corridor)")

    calib = dict(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                 cam_height=args.cam_height, cam_pitch_deg=pitch,
                 cam_x_offset=args.cam_x_offset,
                 image_size=[w, h], dist=dist, vanishing_point=vp)
    Path(args.out).write_text(json.dumps(calib, indent=1))
    print(f"\nsaved -> {args.out}")

    fov_h = 2 * math.degrees(math.atan(w / 2 / args.fx))
    fov_v = 2 * math.degrees(math.atan(h / 2 / args.fy))
    print(f"FOV: {fov_h:.1f} deg horizontal, {fov_v:.1f} deg vertical")

    try:
        p = verify(img, calib, args.out)
        print(f"\nVERIFY -> {p}")
        print("The 1/2/3/4/6 m lines must lie flat on the floor at plausible "
              "spacing.\nIf they float or bunch up, the height or pitch is "
              "wrong -- nothing downstream\nwill be right until they look "
              "correct.")
    except Exception as e:                                   # noqa: BLE001
        print(f"(verification render skipped: {e})")


if __name__ == "__main__":
    main()
