"""OmniVLA's control law — waypoints in, (linear_vel, angular_vel) out.

Ported from SignNav's run_omnivla.py rather than invented here. This matters:

OmniVLA emits FOUR numbers per waypoint — [dx, dy, hx, hy] — a displacement AND a heading as
(cos, sin). Earlier this code took only [dx, dy], teleported the robot to the 4th waypoint, and
set its heading to face that point. That invented heading turned the model's small lateral
wiggle into a large rotation, which compounded every step and spiralled the robot away from its
goal. The band-aid was a hand-tuned per-step turn cap. Both the invention and the band-aid are
gone: the model already says how fast to drive and how fast to turn, and `maxw` below IS the
turn limit, straight from the reference implementation.

Constants are SignNav's, unchanged: DT=1/3s control period, waypoint index 4 of 8, linear
velocity clipped to 0.5 then limited to 0.3, angular to 1.0 then limited to 0.3. The limiter
scales v and w together so the turning radius is preserved rather than clipping them apart.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

DT = 1.0 / 3.0            # control period (s)
WAYPOINT_SELECT = 4       # which of the 8 waypoints the controller tracks
MAXV, MAXW = 0.3, 0.3     # velocity limits (m/s, rad/s)
EPS = 1e-8


def clip_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def waypoints_to_velocity(wp, waypoint_select: int = WAYPOINT_SELECT, dt: float = DT,
                          maxv: float = MAXV, maxw: float = MAXW) -> Tuple[float, float]:
    """(8,4) chunk in metres+heading -> (linear_vel m/s, angular_vel rad/s)."""
    wp = np.asarray(wp, float)
    if wp.ndim != 2 or wp.shape[1] < 4:
        raise ValueError(f"need an (N,4) chunk [dx,dy,hx,hy], got {wp.shape}. The heading "
                         f"columns are not optional — see this module's docstring.")
    i = min(waypoint_select, len(wp) - 1)
    dx, dy, hx, hy = wp[i][:4]

    if abs(dx) < EPS and abs(dy) < EPS:
        # not translating -> rotate toward the heading the MODEL predicted
        v = 0.0
        w = 1.0 * clip_angle(float(np.arctan2(hy, hx))) / dt
    elif abs(dx) < EPS:
        v = 0.0
        w = 1.0 * float(np.sign(dy)) * np.pi / (2 * dt)
    else:
        v = float(dx) / dt
        w = float(np.arctan(dy / dx)) / dt

    v = float(np.clip(v, 0, 0.5))
    w = float(np.clip(w, -1.0, 1.0))

    # Limit v and w together so the turning radius survives; clipping them independently would
    # change the shape of the path the model asked for.
    if abs(v) <= maxv:
        if abs(w) <= maxw:
            return v, w
        rd = v / w if abs(w) > EPS else 0.0
        return maxw * np.sign(v) * abs(rd), maxw * np.sign(w)
    if abs(w) <= 0.001:
        return maxv * np.sign(v), 0.0
    rd = v / w
    if abs(rd) >= maxv / maxw:
        return maxv * np.sign(v), maxv * np.sign(w) / abs(rd)
    return maxw * np.sign(v) * abs(rd), maxw * np.sign(w)


def step_unicycle(x: float, y: float, yaw: float, v: float, w: float, dt: float):
    """Exact unicycle integration of a constant (v, w) over dt. Replaces teleporting the robot
    to a waypoint and guessing which way it should face."""
    if abs(w) < 1e-6:
        return x + v * dt * np.cos(yaw), y + v * dt * np.sin(yaw), yaw
    r = v / w
    nx = x + r * (np.sin(yaw + w * dt) - np.sin(yaw))
    ny = y - r * (np.cos(yaw + w * dt) - np.cos(yaw))
    return float(nx), float(ny), float(yaw + w * dt)
