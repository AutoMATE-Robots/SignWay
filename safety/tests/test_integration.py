import unittest

from vla_path import build_vla_path
from lookahead_path import build_lookahead_path


class TestVlaLookaheadIntegration(unittest.TestCase):
    def test_previous_vla_component_feeds_lookahead_directly(self):
        action = [
            0.06, 0.00,
            0.12, 0.00,
            0.18, 0.00,
            0.24, 0.00,
            0.30, 0.00,
            0.36, 0.00,
            0.42, 0.00,
            0.48, 0.00,
        ]
        vla = build_vla_path(action)
        lookahead = build_lookahead_path(vla)

        self.assertEqual(vla.waypoints.shape, (8, 2))
        self.assertAlmostEqual(lookahead.actual_total_length_m, 3.0, places=6)
        self.assertGreater(len(lookahead.check_points), 8)


if __name__ == "__main__":
    unittest.main()
