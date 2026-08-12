"""Oracle sign detection for the simulator: project a known sign into the image.

WHY THIS EXISTS: the real text detector (PaddleOCR/DBNet) cannot run inside the Isaac Sim
container — it ships a minimal Python we cannot pip into. But the readability gate is pure
numpy and the Gemini reasoner needs only urllib, so both run in-container happily. Only
detection is missing.

WHAT THIS IS AND IS NOT: it knows where the sign is because we put it there, and projects its
corners through the camera to get a pixel box. So DETECTION is an oracle — no misses, no false
positives. Everything downstream is real: the box is geometrically correct, readability is
MEASURED from the actual rendered pixels (height, blur, all of it), and the VLM reads the real
crop and really reasons.

That is a deliberate experimental choice, not a shortcut. It isolates the contribution — when
to reason — from the quality of a detector, which is a solved problem someone else can improve.
Swapping in DBNet later changes nothing downstream: it implements the same TextBackend protocol.

This is NOT MockDetector. That faked the whole thing: it compared robot pose to sign coordinates
and reported a detection if the distance was small, never touching an image.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from common.types import Pose

BBox = Tuple[float, float, float, float]


class ProjectedSignBackend:
    """Projects known signs into the camera. Implements c3_reasoning.detector.TextBackend."""

    def __init__(self, signs: Sequence[Tuple[str, float, float, float, float, float, float]],
                 K: np.ndarray, sensor_height: float = 0.88, text_frac: float = 0.55):
        """signs: (label, x, y, z, yaw_deg, width_m, height_m) — as passed to --sign.

        text_frac: the lettering occupies this fraction of the board's height. The gate cares
        about TEXT height, not board height — a big board with small print is not readable, and
        that distinction is the entire point of measuring pixels instead of distance.
        """
        self.signs = list(signs)
        self.K = np.asarray(K, float)
        self.sensor_height = sensor_height
        self.text_frac = text_frac
        self._pose = Pose(0.0, 0.0, 0.0)

    def set_pose(self, pose: Pose) -> None:
        self._pose = pose

    def _corners(self, sx, sy, sz, yaw_deg, w, h) -> np.ndarray:
        """The four corners of the board in world coordinates."""
        yaw = np.deg2rad(yaw_deg)
        # the board spans +-w/2 along its own left/right axis (perpendicular to its normal)
        px, py = -np.sin(yaw), np.cos(yaw)
        ht = h * self.text_frac / 2.0            # only the lettering counts
        return np.array([[sx + px * w / 2, sy + py * w / 2, sz - ht],
                         [sx - px * w / 2, sy - py * w / 2, sz - ht],
                         [sx - px * w / 2, sy - py * w / 2, sz + ht],
                         [sx + px * w / 2, sy + py * w / 2, sz + ht]], float)

    def _project(self, pts_world: np.ndarray):
        """World -> pixels. Returns None if any corner is behind the camera."""
        p = self._pose
        d = pts_world - np.array([p.x, p.y, self.sensor_height])
        c, s = np.cos(p.yaw), np.sin(p.yaw)
        fwd = d[:, 0] * c + d[:, 1] * s          # robot frame: x forward
        left = -d[:, 0] * s + d[:, 1] * c        #              y left
        up = d[:, 2]                             #              z up
        if np.any(fwd <= 0.05):                  # behind or level with the lens
            return None
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        u = fx * (-left) / fwd + cx              # pinhole: image x is to the right = -left
        v = fy * (-up) / fwd + cy                #          image y is down = -up
        return np.stack([u, v], axis=1)

    def detect_text(self, image) -> List[Tuple[BBox, float]]:
        a = np.asarray(image)
        h_img, w_img = a.shape[:2]
        out: List[Tuple[BBox, float]] = []
        for (_label, sx, sy, sz, yaw_deg, w, h) in self.signs:
            # A sign is only detectable if it is facing us — the back of a board has no text.
            n = np.array([np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))])
            to_robot = np.array([self._pose.x - sx, self._pose.y - sy])
            if np.dot(n, to_robot) <= 0:
                continue
            uv = self._project(self._corners(sx, sy, sz, yaw_deg, w, h))
            if uv is None:
                continue
            x0, y0 = uv[:, 0].min(), uv[:, 1].min()
            x1, y1 = uv[:, 0].max(), uv[:, 1].max()
            if x1 < 0 or y1 < 0 or x0 > w_img or y0 > h_img:
                continue                          # off-frame
            out.append(((float(x0), float(y0), float(x1), float(y1)), 1.0))
        return out
