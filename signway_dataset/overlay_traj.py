#!/usr/bin/env python3
"""
overlay_traj.py -- draw a body-frame trajectory onto the robot's own RGB frame.

Runs anywhere with opencv+numpy (your laptop), no GPU, no ROS -- so the figure
can be iterated on quickly.

THE GEOMETRY. The trajectory lies on the ground plane in the robot's body frame
(x forward, y left, z=0). The image comes from the robot's own camera, rigidly
mounted. A plane viewed by a pinhole camera maps to the image by a homography,
so every waypoint has an exact pixel -- no odometry and no external camera pose
are involved. Two ways to get that mapping:

  --calib intrinsics.json      needs fx, fy, cx, cy, camera height and pitch.
                               Use this if the bag carries /camera_info or the
                               camera has been calibrated.

  --calib homography.json      4+ correspondences between measured floor points
                               (in metres, robot frame) and their pixels. No
                               camera parameters needed at all. Build it with
                               --pick (below). This is the robust path and the
                               one to use if the intrinsics are uncertain.

BUILDING A HOMOGRAPHY (five minutes, once per camera mounting):
  1. Put tape crosses on the floor at measured positions in front of a
     stationary robot, e.g. (1.0, 0.0), (2.0, 0.0), (1.0, +0.5), (1.0, -0.5),
     (2.0, +0.5), (2.0, -0.5). x forward, y LEFT, metres, measured from the
     point on the floor directly below the camera.
  2. Grab one frame from that recording.
  3. python overlay_traj.py --pick --image cal.png --out-calib homography.json
     Click each cross, type its (x, y) when prompted. 6+ points, not collinear.
  4. Reuse that file for every figure from the same camera mounting.

DRAWING:
    python overlay_traj.py --image frame_01850.png --traj traj.json \
        --frame 1850 --calib homography.json --out fig_left.png

    # both prompts on one frame -- the decision-vs-timing figure
    python overlay_traj.py --image frame_01850.png --traj traj.json \
        --frame 1850 --calib homography.json --compare-prompts --out fig_left.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

# BGR. Matches the palette used by the plotting scripts.
COLORS = dict(turn_left=(232, 155, 76), turn_right=(76, 155, 232),
              straight=(95, 190, 120), stop=(80, 80, 230))


# --------------------------------------------------------------------------- #
# ground plane -> image
# --------------------------------------------------------------------------- #
def project_intrinsics(pts_xy, K, cam_height, cam_pitch_deg, cam_x_offset=0.0):
    """(x forward, y left) on the ground -> (u, v) pixels.

    Camera sits at height `cam_height` above the floor, `cam_x_offset` ahead of
    the body origin, pitched down by `cam_pitch_deg`.
    """
    fx, fy, cx, cy = K
    th = math.radians(cam_pitch_deg)
    ct, st = math.cos(th), math.sin(th)
    out = []
    for x, y in pts_xy:
        f = float(x) - cam_x_offset          # forward from camera
        # unrotated camera frame: right = -y, down = +height, forward = f
        xc = -float(y)
        yc = cam_height * ct - f * st
        zc = cam_height * st + f * ct
        if zc <= 1e-6:                        # behind the image plane
            out.append(None)
            continue
        out.append((fx * xc / zc + cx, fy * yc / zc + cy))
    return out


def project_homography(pts_xy, H):
    pts = np.asarray([[float(x), float(y)] for x, y in pts_xy],
                     dtype=np.float64).reshape(-1, 1, 2)
    img = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [tuple(p) for p in img]


def load_calib(path):
    d = json.loads(Path(path).read_text())
    if "H" in d:
        return ("H", np.asarray(d["H"], dtype=np.float64))
    need = ("fx", "fy", "cx", "cy", "cam_height", "cam_pitch_deg")
    miss = [k for k in need if k not in d]
    if miss:
        raise SystemExit(f"calibration missing {miss}")
    return ("K", ((d["fx"], d["fy"], d["cx"], d["cy"]),
                  d["cam_height"], d["cam_pitch_deg"],
                  d.get("cam_x_offset", 0.0)))


def project(pts_xy, calib):
    kind, val = calib
    if kind == "H":
        return project_homography(pts_xy, val)
    K, h, pitch, xoff = val
    return project_intrinsics(pts_xy, K, h, pitch, xoff)


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def densify(wp, n=60):
    """Smooth the 8 waypoints into a dense ground path (starts at the robot)."""
    pts = np.vstack([[0.0, 0.0], np.asarray(wp, dtype=float)])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if d[-1] <= 1e-6:
        return pts
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], 1)


def draw_ribbon(img, wp, calib, color, width_m=0.45, alpha=0.55,
                outline=True, taper=True):
    """Project a ground-plane ribbon of `width_m` around the path and fill it.

    A ribbon rather than a line because it reads as occupying space on the
    floor, which is the point: it shows where the ROBOT will be, not an abstract
    curve. Perspective does the rest -- far waypoints correctly appear narrower.
    """
    path = densify(wp)
    left, right = [], []
    for i, (x, y) in enumerate(path):
        if i < len(path) - 1:
            dx, dy = path[i + 1] - path[i]
        else:
            dx, dy = path[i] - path[i - 1]
        n = math.hypot(dx, dy)
        if n < 1e-9:
            continue
        # unit normal on the ground plane
        nx, ny = -dy / n, dx / n
        w = width_m / 2
        if taper:                              # slight taper looks less blunt
            w *= 1.0 - 0.25 * (i / max(1, len(path) - 1))
        left.append((x + nx * w, y + ny * w))
        right.append((x - nx * w, y - ny * w))

    poly_xy = left + right[::-1]
    proj = project(poly_xy, calib)
    pts = np.array([p for p in proj if p is not None], dtype=np.int32)
    if len(pts) < 3:
        print("  [warn] ribbon projected to nothing -- check the calibration")
        return img
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color, lineType=cv2.LINE_AA)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    if outline:
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    return img


def draw_centerline(img, wp, calib, color):
    proj = [p for p in project(densify(wp), calib) if p is not None]
    if len(proj) > 1:
        cv2.polylines(img, [np.array(proj, np.int32)], False,
                      (255, 255, 255), 3, cv2.LINE_AA)
        cv2.polylines(img, [np.array(proj, np.int32)], False, color, 1,
                      cv2.LINE_AA)
    ends = project([wp[-1]], calib)
    if ends and ends[0] is not None:
        cv2.circle(img, tuple(np.int32(ends[0])), 6, (255, 255, 255), -1,
                   cv2.LINE_AA)
        cv2.circle(img, tuple(np.int32(ends[0])), 4, color, -1, cv2.LINE_AA)
    return img


def draw_grid(img, calib, xs=(1, 2, 3, 4), y_half=1.0, color=(180, 180, 180)):
    """Ground-plane distance rulers -- a sanity check that the calibration is
    right, and a legible way to give the figure scale."""
    for x in xs:
        seg = project([(x, -y_half), (x, y_half)], calib)
        if all(p is not None for p in seg):
            a, b = np.int32(seg[0]), np.int32(seg[1])
            cv2.line(img, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
            cv2.putText(img, f"{x} m", (int(b[0]) + 6, int(b[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# interactive calibration picker
# --------------------------------------------------------------------------- #
def pick_calibration(image_path, out_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"cannot read {image_path}")
    disp = img.copy()
    clicks, ground = [], []
    win = "click a floor point, then type its (x,y) in the terminal | q=done"

    def on_mouse(ev, x, y, _flags, _p):
        if ev == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(disp, (x, y), 5, (60, 220, 60), -1, cv2.LINE_AA)
            cv2.putText(disp, str(len(clicks)), (x + 7, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), 2)
            cv2.imshow(win, disp)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    print("\nClick a floor point with a KNOWN position, then enter it as "
          "'x y' in metres\n(x forward, y LEFT, from the floor point under the "
          "camera). 6+ points, not all in a line.\nPress q in the window when "
          "done.\n")
    while True:
        cv2.imshow(win, disp)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        while len(ground) < len(clicks):
            i = len(ground)
            try:
                s = input(f"  point {i+1} at pixel {clicks[i]} -> x y : ")
                x, y = (float(v) for v in s.replace(",", " ").split())
                ground.append((x, y))
            except (ValueError, EOFError):
                print("    expected two numbers, e.g. '2.0 0.5'")
    cv2.destroyAllWindows()

    if len(ground) < 4:
        raise SystemExit(f"need at least 4 points, got {len(ground)}")
    src = np.asarray(ground, np.float64).reshape(-1, 1, 2)
    dst = np.asarray(clicks[:len(ground)], np.float64).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise SystemExit("homography fit failed -- are the points collinear?")
    back = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    err = np.linalg.norm(back - dst.reshape(-1, 2), axis=1)
    print(f"\nfit over {len(ground)} points: mean reprojection error "
          f"{err.mean():.1f} px, max {err.max():.1f} px")
    if err.mean() > 15:
        print("  that is high -- re-check the measured positions")
    Path(out_path).write_text(json.dumps(
        dict(H=H.tolist(), points_ground=ground,
             points_image=clicks[:len(ground)],
             mean_reproj_px=float(err.mean())), indent=1))
    print(f"saved -> {out_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--traj", default=None, help="traj.json from run_vla_on_bag")
    ap.add_argument("--frame", default=None, help="which frame key in traj.json")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--width-m", type=float, default=0.45,
                    help="ribbon width, metres (robot footprint)")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--compare-prompts", action="store_true",
                    help="draw every prompt stored for this frame")
    ap.add_argument("--grid", action="store_true",
                    help="ground-plane distance rulers (calibration check)")
    ap.add_argument("--pick", action="store_true",
                    help="interactive calibration builder")
    ap.add_argument("--out-calib", default="homography.json")
    args = ap.parse_args()

    if args.pick:
        pick_calibration(args.image, args.out_calib)
        return

    if not args.traj or not args.calib:
        raise SystemExit("need --traj and --calib (or --pick)")

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    calib = load_calib(args.calib)
    data = json.loads(Path(args.traj).read_text())["frames"]
    key = args.frame or sorted(data)[0]
    if key not in data:
        raise SystemExit(f"frame {key} not in traj.json (have {sorted(data)})")
    entry = data[key]

    if args.grid:
        img = draw_grid(img, calib)

    prompts = list(entry["wp"].keys()) if args.compare_prompts \
        else [entry.get("prompt", list(entry["wp"])[0])]
    for p in prompts:
        wp = entry["wp"][p]
        col = COLORS.get(p, (200, 200, 200))
        img = draw_ribbon(img, wp, calib, col, args.width_m, args.alpha)
        img = draw_centerline(img, wp, calib, col)
        print(f"  drew {p}: wp8 = ({wp[-1][0]:+.3f}, {wp[-1][1]:+.3f}) m")

    if len(prompts) > 1:                    # small legend for the compare case
        y = 30
        for p in prompts:
            cv2.rectangle(img, (18, y - 12), (44, y + 4), COLORS.get(p), -1)
            cv2.putText(img, p, (52, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            y += 30

    cv2.imwrite(str(args.out), img)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
