#!/usr/bin/env python3
"""
calibrate_yaw.py
----------------
Diagnose and correct a systematic rotational scale error in wheel odometry.

Symptom: the trajectory shape is sheared, every turn reads more (or less) than
its true angle, and a closed loop does not close. Translation is fine --
opposite sides of the loop come out equal -- but headings are wrong.

Cause: the yaw rate is derived from wheel speeds divided by an assumed track
width. If that parameter is wrong, or if the platform is skid-steer and the
GEOMETRIC width is used instead of the larger EFFECTIVE width, every rotation
is scaled by a constant factor.

Fix: because the error is a constant multiplier, it is invertible offline.
Recover per-step forward travel, rescale the yaw increments, re-integrate.

    python calibrate_yaw.py --csv odom.csv --expected-turn 360
    python calibrate_yaw.py --csv odom.csv --scale 0.83      # force a factor
"""
from __future__ import annotations

import argparse
import math

import numpy as np


def signed_step(x, y, yaw):
    """
    Per-step travel WITH SIGN. Negative when the robot is reversing.

    hypot() alone is a magnitude, so a reversing step would be re-integrated
    FORWARD along the heading -- which silently destroys every three-point
    turn. Project the displacement onto the heading to recover the direction.

    Valid for a non-holonomic base (motion along the heading axis), which
    covers Hunter. A holonomic base would need the lateral component too.
    """
    dx, dy = np.diff(x), np.diff(y)
    mag = np.hypot(dx, dy)
    proj = dx * np.cos(yaw[1:]) + dy * np.sin(yaw[1:])
    step = np.where(mag > 1e-9, np.sign(proj) * mag, 0.0)
    return np.concatenate([[0.0], step])


def reverse_fraction(x, y, yaw):
    st = signed_step(x, y, yaw)
    moving = np.abs(st) > 1e-6
    if not moving.any():
        return 0.0, 0.0
    return float((st[moving] < 0).mean()), float(-st[st < 0].sum())


def reintegrate(t, x, y, yaw, k):
    """
    Rebuild the trajectory with yaw increments multiplied by k.

    Per-step travel is taken from the recorded translation, which is correct --
    only the heading it was applied along was wrong. The step is SIGNED so that
    reversing manoeuvres survive.
    """
    ds = signed_step(x, y, yaw)
    yaw_u = np.unwrap(yaw)
    yaw_c = yaw_u[0] + k * (yaw_u - yaw_u[0])

    xc = np.empty_like(x)
    yc = np.empty_like(y)
    xc[0], yc[0] = x[0], y[0]
    for i in range(1, len(x)):
        xc[i] = xc[i - 1] + ds[i] * math.cos(yaw_c[i])
        yc[i] = yc[i - 1] + ds[i] * math.sin(yaw_c[i])
    return xc, yc, yaw_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="odom_calibrated.csv")
    ap.add_argument("--plot", default="odom_calibration.png")
    ap.add_argument("--expected-turn", type=float, default=360.0,
                    help="true total heading change over the bag, degrees "
                         "(4 right-angle corners = 360)")
    ap.add_argument("--scale", type=float, default=None,
                    help="force a scale factor instead of fitting one")
    args = ap.parse_args()

    a = np.loadtxt(args.csv, delimiter=",", skiprows=1)
    t, x, y, yaw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    yaw_u = np.unwrap(yaw)
    rev_frac, rev_m = reverse_fraction(x, y, yaw)
    measured = math.degrees(yaw_u[-1] - yaw_u[0])
    path = float(np.hypot(np.diff(x), np.diff(y)).sum())

    k = args.scale if args.scale is not None else args.expected_turn / measured

    W = 70
    print("=" * W)
    print("YAW SCALE CALIBRATION")
    print("=" * W)
    print(f"path length            {path:.1f} m")
    print(f"reverse motion         {100*rev_frac:.1f}% of steps, {rev_m:.1f} m driven backwards")
    if rev_frac > 0.001:
        print("                       (three-point turns present — signed steps required)")
    print(f"measured total turn    {measured:+.1f} deg")
    print(f"expected total turn    {args.expected_turn:+.1f} deg")
    print(f"scale factor k         {k:.4f}   "
          f"({100 * (1 / k - 1):+.1f}% over-reported)" if k < 1 else
          f"scale factor k         {k:.4f}")
    print()
    print("If the base is skid-steer, the effective track width exceeds the")
    print("geometric one. Using the geometric value over-reports rotation by")
    print("exactly this factor, so the corrected track width is:")
    print(f"    L_effective = L_geometric / {k:.4f} = L_geometric x {1/k:.3f}")

    xc, yc, yawc = reintegrate(t, x, y, yaw, k)

    gap_before = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    gap_after = float(np.hypot(xc[-1] - xc[0], yc[-1] - yc[0]))
    print()
    print(f"{'':22}{'before':>12}{'after':>12}")
    print(f"{'start-end gap':22}{gap_before:>11.2f}m{gap_after:>11.2f}m")
    print(f"{'total turn':22}{measured:>+11.1f}d"
          f"{math.degrees(yawc[-1] - yawc[0]):>+11.1f}d")

    np.savetxt(args.out, np.column_stack([t, xc, yc, yawc]),
               delimiter=",", header="t,x,y,yaw", comments="")
    print(f"\nwrote {args.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
        for a_, xx, yy, ttl in ((ax[0], x, y, f"raw  (turn {measured:+.0f}°)"),
                                (ax[1], xc, yc, f"corrected k={k:.3f}  "
                                                f"(turn {math.degrees(yawc[-1]-yawc[0]):+.0f}°)")):
            a_.plot(xx, yy, lw=1.2, color="#444")
            a_.scatter([xx[0]], [yy[0]], s=90, c="green", zorder=5, label="start")
            a_.scatter([xx[-1]], [yy[-1]], s=90, marker="X", c="red", zorder=5, label="end")
            a_.set_aspect("equal"); a_.grid(alpha=0.3); a_.set_title(ttl)
            a_.set_xlabel("x (m)"); a_.set_ylabel("y (m)"); a_.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(args.plot, dpi=140)
        print(f"wrote {args.plot}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")
    print("=" * W)


if __name__ == "__main__":
    main()