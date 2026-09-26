
import unittest
import numpy as np

from candidate_planner import (
    CandidatePlannerConfig,
    generate_candidate_paths,
    build_curved_safety_band,
    plan_best_candidate,
)
from occupancy_checker import GridGeometry, OccupancyValues
from vla_path import build_vla_path
from lookahead_path import build_lookahead_path


class CandidatePlannerTests(unittest.TestCase):
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
        ], dtype=float)
        self.vla = build_vla_path(action)
        self.lookahead = build_lookahead_path(
            self.vla,
            total_lookahead_m=3.0,
            sample_spacing_m=0.05,
        )
        self.config = CandidatePlannerConfig(
            candidate_count=40,
            max_lateral_offset_m=1.0,
            robot_width_m=0.75,
            safety_margin_m=0.20,
            sample_spacing_m=0.05,
        )
        self.geom = GridGeometry(
            resolution_m=0.05,
            robot_row=70,
            robot_col=80,
            forward_direction="-row",
            left_direction="-col",
        )
        self.values = OccupancyValues(free=0, occupied=100, unknown=-1)

    def test_generates_40_candidates_20_each_side(self):
        candidates = generate_candidate_paths(self.lookahead, self.config)
        self.assertEqual(len(candidates), 40)
        self.assertEqual(sum(c.side == "left" for c in candidates), 20)
        self.assertEqual(sum(c.side == "right" for c in candidates), 20)

    def test_candidates_start_at_reference_path_and_end_with_requested_offset(self):
        candidates = generate_candidate_paths(self.lookahead, self.config)
        left = max((c for c in candidates if c.side == "left"), key=lambda c: c.target_lateral_offset_m)
        right = min((c for c in candidates if c.side == "right"), key=lambda c: c.target_lateral_offset_m)

        self.assertTrue(np.allclose(left.centerline_points_m[0], self.lookahead.check_points[0]))
        self.assertTrue(np.allclose(right.centerline_points_m[0], self.lookahead.check_points[0]))
        self.assertAlmostEqual(left.target_lateral_offset_m, 1.0, places=6)
        self.assertAlmostEqual(right.target_lateral_offset_m, -1.0, places=6)

    def test_curved_safety_band_has_full_1_15m_width(self):
        candidate = generate_candidate_paths(self.lookahead, self.config)[0]
        band = build_curved_safety_band(candidate.centerline_points_m, self.config)
        expected_width = 0.75 + 2 * 0.20
        self.assertAlmostEqual(band.effective_width_m, expected_width, places=9)
        self.assertGreater(len(band.filled_points_m), len(candidate.centerline_points_m))

    def test_planner_rejects_collision_and_selects_a_safe_side(self):
        grid = np.zeros((160, 160), dtype=np.int16)

        # Put a block centered in the original forward route, 1.2-1.8 m ahead.
        # With -row forward and -col left, robot is at [70, 80].
        for x_m in np.arange(1.2, 1.81, 0.05):
            for y_m in np.arange(-0.25, 0.26, 0.05):
                row = int(round(self.geom.robot_row - x_m / self.geom.resolution_m))
                col = int(round(self.geom.robot_col - y_m / self.geom.resolution_m))
                grid[row, col] = 100

        result = plan_best_candidate(
            occupancy_grid=grid,
            lookahead=self.lookahead,
            geometry=self.geom,
            values=self.values,
            config=self.config,
        )

        self.assertIsNotNone(result.best_candidate)
        self.assertTrue(result.best_candidate.safe)
        self.assertEqual(result.safe_candidate_count + result.rejected_candidate_count, 40)
        self.assertGreater(result.rejected_candidate_count, 0)

    def test_scoring_prefers_smaller_change_when_clearance_is_similar(self):
        grid = np.zeros((160, 160), dtype=np.int16)

        result = plan_best_candidate(
            occupancy_grid=grid,
            lookahead=self.lookahead,
            geometry=self.geom,
            values=self.values,
            config=self.config,
        )

        self.assertIsNotNone(result.best_candidate)
        # In an empty grid there is no obstacle reason to make a large change.
        self.assertLessEqual(abs(result.best_candidate.target_lateral_offset_m), 0.10)

    def test_candidate_centerline_has_about_5cm_sampling(self):
        candidate = generate_candidate_paths(self.lookahead, self.config)[0]
        distances = np.linalg.norm(np.diff(candidate.centerline_points_m, axis=0), axis=1)
        self.assertLessEqual(float(np.max(distances)), 0.08)


if __name__ == "__main__":
    unittest.main()
