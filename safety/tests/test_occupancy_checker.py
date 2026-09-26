
import unittest
import numpy as np

from occupancy_checker import GridGeometry, OccupancyValues, check_corridor_against_ogm
from safety_corridor import SafetyCorridorInfo


def fake_corridor(points):
    pts = np.asarray(points, dtype=float)
    return SafetyCorridorInfo(
        filled_points_m=pts,
        corners_m=np.array([[0, -0.5], [0, 0.5], [3, 0.5], [3, -0.5]], dtype=float),
        length_m=3.0,
        robot_width_m=0.75,
        robot_length_m=1.10,
        safety_margin_m=0.20,
        effective_width_m=1.15,
        half_width_m=0.575,
        requested_spacing_m=0.05,
        actual_x_spacing_m=0.05,
        actual_y_spacing_m=0.05,
    )


class OccupancyCheckerTests(unittest.TestCase):
    def setUp(self):
        self.geom = GridGeometry(
            resolution_m=0.10,
            robot_row=50,
            robot_col=50,
            forward_direction="-row",
            left_direction="-col",
        )
        self.values = OccupancyValues(free=0, occupied=100, unknown=-1)

    def test_clear_corridor_is_safe(self):
        grid = np.zeros((100, 100), dtype=np.int16)
        corridor = fake_corridor([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = check_corridor_against_ogm(grid, corridor, self.geom, self.values)
        self.assertTrue(result.safe)
        self.assertEqual(result.occupied_count, 0)

    def test_occupied_cell_inside_corridor_is_unsafe(self):
        grid = np.zeros((100, 100), dtype=np.int16)
        grid[40, 50] = 100
        corridor = fake_corridor([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]])
        result = check_corridor_against_ogm(grid, corridor, self.geom, self.values)
        self.assertFalse(result.safe)
        self.assertGreaterEqual(result.occupied_count, 1)
        self.assertAlmostEqual(result.first_occupied_distance_m, 1.0, places=6)

    def test_left_axis_conversion(self):
        grid = np.zeros((100, 100), dtype=np.int16)
        grid[40, 45] = 100
        corridor = fake_corridor([[1.0, 0.5]])
        result = check_corridor_against_ogm(grid, corridor, self.geom, self.values)
        self.assertFalse(result.safe)

    def test_unknown_can_be_reported_without_forcing_unsafe(self):
        grid = np.zeros((100, 100), dtype=np.int16)
        grid[40, 50] = -1
        corridor = fake_corridor([[1.0, 0.0]])
        result = check_corridor_against_ogm(
            grid, corridor, self.geom, self.values, unknown_is_unsafe=False
        )
        self.assertTrue(result.safe)
        self.assertEqual(result.unknown_count, 1)

    def test_unknown_can_be_configured_as_unsafe(self):
        grid = np.zeros((100, 100), dtype=np.int16)
        grid[40, 50] = -1
        corridor = fake_corridor([[1.0, 0.0]])
        result = check_corridor_against_ogm(
            grid, corridor, self.geom, self.values, unknown_is_unsafe=True
        )
        self.assertFalse(result.safe)

    def test_out_of_bounds_is_reported(self):
        grid = np.zeros((20, 20), dtype=np.int16)
        geom = GridGeometry(
            resolution_m=0.10,
            robot_row=10,
            robot_col=10,
            forward_direction="-row",
            left_direction="-col",
        )
        corridor = fake_corridor([[3.0, 0.0]])
        result = check_corridor_against_ogm(
            grid, corridor, geom, self.values, out_of_bounds_is_unsafe=True
        )
        self.assertFalse(result.safe)
        self.assertEqual(result.out_of_bounds_count, 1)


if __name__ == "__main__":
    unittest.main()
