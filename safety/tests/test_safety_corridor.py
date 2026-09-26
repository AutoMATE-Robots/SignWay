import unittest
import numpy as np

from vla_path import build_vla_path
from lookahead_path import build_lookahead_path
from safety_corridor import build_safety_corridor


class SafetyCorridorTests(unittest.TestCase):
    def setUp(self):
        action = np.array([
            0.04, 0.00,
            0.08, 0.00,
            0.12, 0.00,
            0.16, 0.00,
            0.20, 0.00,
            0.24, 0.00,
            0.28, 0.00,
            0.32, 0.00,
        ])
        vla = build_vla_path(action)
        self.lookahead = build_lookahead_path(vla, total_lookahead_m=3.0)

    def test_default_corridor_is_three_meters_long_and_1_15_meters_wide(self):
        corridor = build_safety_corridor(self.lookahead)
        self.assertAlmostEqual(corridor.length_m, 3.0, places=9)
        self.assertAlmostEqual(corridor.effective_width_m, 1.15, places=9)
        self.assertAlmostEqual(corridor.half_width_m, 0.575, places=9)

    def test_corridor_uses_lookahead_length_by_default(self):
        vla = build_vla_path(np.array([
            0.04, 0.00, 0.08, 0.00, 0.12, 0.00, 0.16, 0.00,
            0.20, 0.00, 0.24, 0.00, 0.28, 0.00, 0.32, 0.00,
        ]))
        lookahead = build_lookahead_path(vla, total_lookahead_m=2.5)
        corridor = build_safety_corridor(lookahead)
        self.assertAlmostEqual(corridor.length_m, 2.5, places=9)

    def test_filled_points_cover_center_and_both_width_edges(self):
        corridor = build_safety_corridor(self.lookahead, sample_spacing_m=0.05)
        pts = corridor.filled_points_m

        self.assertTrue(np.any(np.all(np.isclose(pts, [0.0, 0.0]), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(pts, [3.0, 0.0]), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(pts, [0.0, 0.575]), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(pts, [0.0, -0.575]), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(pts, [3.0, 0.575]), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(pts, [3.0, -0.575]), axis=1)))

    def test_sample_spacing_never_exceeds_requested_spacing(self):
        corridor = build_safety_corridor(self.lookahead, sample_spacing_m=0.05)
        self.assertLessEqual(corridor.actual_x_spacing_m, 0.05 + 1e-12)
        self.assertLessEqual(corridor.actual_y_spacing_m, 0.05 + 1e-12)

    def test_margin_is_configurable(self):
        corridor = build_safety_corridor(
            self.lookahead,
            robot_width_m=0.75,
            safety_margin_m=0.10,
        )
        self.assertAlmostEqual(corridor.effective_width_m, 0.95, places=9)

    def test_bad_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            build_safety_corridor(self.lookahead, robot_width_m=0.0)
        with self.assertRaises(ValueError):
            build_safety_corridor(self.lookahead, safety_margin_m=-0.01)
        with self.assertRaises(ValueError):
            build_safety_corridor(self.lookahead, sample_spacing_m=0.0)


if __name__ == '__main__':
    unittest.main()
