#!/usr/bin/env python3
"""
synth_traj.py — synthetic trajectory generator shared by the demo and tests.

Lives outside the test package so `make_synthetic_demo.py` does not have to
import a test module (which breaks as soon as the tests move into tests/).
"""
from __future__ import annotations

import math

import numpy as np


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def traj(spec, hz=20.0, v=0.8, x0=0.0, y0=0.0, yaw0=0.0, t0=0.0):
    """Build (t, x, y, yaw) from a list of motion primitives.

    spec: list of ('straight', length_m)
                | ('reverse', length_m)          -- backs up ALONG the heading
                | ('turn', signed_deg, radius_m)
                | ('rturn', signed_deg, radius_m) -- arc driven in REVERSE
    """
    dt = 1.0 / hz
    ds = v * dt
    t, x, y, yaw = [t0], [x0], [y0], [yaw0]
    for item in spec:
        if item[0] in ("straight", "reverse"):
            sgn = 1.0 if item[0] == "straight" else -1.0
            n = max(1, int(round(item[1] / ds)))
            for _ in range(n):
                x.append(x[-1] + sgn * ds * math.cos(yaw[-1]))
                y.append(y[-1] + sgn * ds * math.sin(yaw[-1]))
                yaw.append(yaw[-1])
                t.append(t[-1] + dt)
        elif item[0] == "rturn":
            _, deg, radius = item
            arc = abs(math.radians(deg)) * radius
            n = max(2, int(round(arc / ds)))
            dpsi = math.radians(deg) / n
            for _ in range(n):
                yaw.append(yaw[-1] + dpsi)
                x.append(x[-1] - ds * math.cos(yaw[-1]))
                y.append(y[-1] - ds * math.sin(yaw[-1]))
                t.append(t[-1] + dt)
        else:
            _, deg, radius = item
            arc = abs(math.radians(deg)) * radius
            n = max(2, int(round(arc / ds)))
            dpsi = math.radians(deg) / n
            for _ in range(n):
                yaw.append(yaw[-1] + dpsi)
                x.append(x[-1] + ds * math.cos(yaw[-1]))
                y.append(y[-1] + ds * math.sin(yaw[-1]))
                t.append(t[-1] + dt)
    return (np.array(t), np.array(x), np.array(y),
            np.array([wrap_pi(a) for a in yaw]))


def square(x0=40.0, y0=-25.0, hz=20.0, v=0.8):
    """The paper's scenario-1 loop: north to A, left at A, around the block,
    retrace the first corridor northbound, straight through A."""
    return traj([
        ("straight", 25.0),            # -> A
        ("turn", 90, 1.5),             # left at A (VLM said left for goal_A)
        ("straight", 38.0),            # -> B
        ("turn", 90, 1.5),
        ("straight", 23.0),            # -> C
        ("turn", 90, 1.5),
        ("straight", 38.0),            # -> D (same corridor line as start)
        ("turn", 90, 1.5),
        ("straight", 23.0),            # northbound retrace toward A
        ("straight", 8.0),             # straight THROUGH A
    ], hz=hz, v=v, x0=x0, y0=y0, yaw0=math.pi / 2)