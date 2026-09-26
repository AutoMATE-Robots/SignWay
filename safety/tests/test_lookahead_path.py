import math
import unittest

import numpy as np

from vla_path import build_vla_path
from lookahead_path import build_lookahead_path


class TestLookaheadPath(unittest.TestCase):
    def test_straight_vla_path_extends_to_three_meters(self):
        action = [
            0.05, 0.00,
            0.10, 0.00,
            0.15, 0.00,
            0.20, 0.00,
            0.25, 0.00,
            0.30, 0.00,
            0.35, 0.00,
            0.40, 0.00,
        ]
        vla = build_vla_path(action)
        lookahead = build_lookahead_path(vla, total_lookahead_m=3.0, sample_spacing_m=0.05)

        self.assertAlmostEqual(lookahead.actual_total_length_m, 3.0, places=6)
        self.assertAlmostEqual(lookahead.full_path_points[-1, 0], 3.0, places=6)
        self.assertAlmostEqual(lookahead.full_path_points[-1, 1], 0.0, places=6)
        self.assertAlmostEqual(lookahead.estimated_curvature_per_m, 0.0, places=6)
        self.assertGreater(len(lookahead.check_points), len(vla.path_points))

    def test_right_turn_continues_to_the_right(self):
        # Synthetic constant-curvature right turn: negative curvature.
        k = -0.8  # rad per meter
        ds = 0.05
        x = y = theta = 0.0
        points = []
        for _ in range(8):
            theta_next = theta + k * ds
            x += (math.sin(theta_next) - math.sin(theta)) / k
            y += (-math.cos(theta_next) + math.cos(theta)) / k
            theta = theta_next
            points.extend([x, y])

        vla = build_vla_path(points)
        lookahead = build_lookahead_path(vla, total_lookahead_m=3.0, sample_spacing_m=0.05)

        self.assertLess(lookahead.estimated_curvature_per_m, 0.0)
        self.assertLess(lookahead.full_path_points[-1, 1], vla.final_point_m[1])
        self.assertGreater(lookahead.extension_length_m, 0.0)

    def test_dense_check_points_never_skip_more_than_requested_spacing(self):
        action = [
            0.05, 0.00,
            0.10, 0.01,
            0.15, 0.02,
            0.20, 0.03,
            0.25, 0.04,
            0.30, 0.05,
            0.35, 0.06,
            0.40, 0.07,
        ]
        vla = build_vla_path(action)
        lookahead = build_lookahead_path(vla, total_lookahead_m=3.0, sample_spacing_m=0.05)

        gaps = np.linalg.norm(np.diff(lookahead.check_points, axis=0), axis=1)
        self.assertLessEqual(float(gaps.max()), 0.050001)

    def test_no_extension_when_vla_path_is_already_longer_than_target(self):
        action = [
            0.5, 0.0,
            1.0, 0.0,
            1.5, 0.0,
            2.0, 0.0,
            2.5, 0.0,
            3.0, 0.0,
            3.5, 0.0,
            4.0, 0.0,
        ]
        vla = build_vla_path(action)
        lookahead = build_lookahead_path(vla, total_lookahead_m=3.0, sample_spacing_m=0.05)

        self.assertAlmostEqual(lookahead.extension_length_m, 0.0, places=9)
        self.assertEqual(lookahead.extension_points.shape, (0, 2))
        self.assertAlmostEqual(lookahead.actual_total_length_m, vla.path_length_m, places=9)


if __name__ == "__main__":
    unittest.main()
