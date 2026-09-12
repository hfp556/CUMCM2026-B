import math
import unittest

import numpy as np

from Q1 import (
    convex_diameter,
    diameter_circle_coverage,
    localize_region,
    point_in_convex_polygon,
)
from Q2 import (
    classify_detection_point,
    first_possible_region,
    get_problem2_point_dynamic,
    optimize_second_point,
)


def angular_error_deg(point, origin, measured_deg):
    vector = np.asarray(point) - np.asarray(origin)
    true_deg = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
    return abs((true_deg - measured_deg + 180.0) % 360.0 - 180.0)


class Problem1GeometryTests(unittest.TestCase):
    def test_single_measurement_obeys_all_three_constraints(self):
        sensor = np.array([500.0, -300.0])
        region = localize_region([sensor], [35.0], circle_segments=180)

        self.assertGreater(len(region), 2)
        self.assertLessEqual(np.max(np.linalg.norm(region, axis=1)), 1800.0 + 1e-6)
        self.assertLessEqual(
            np.max(np.linalg.norm(region - sensor, axis=1)), 1500.0 + 1e-6
        )
        for point in region:
            if np.linalg.norm(point - sensor) > 1e-5:
                self.assertLessEqual(angular_error_deg(point, sensor, 35.0), 1.01)

    def test_wedge_wraparound_keeps_points_near_positive_x_axis(self):
        region = localize_region([(0.0, 0.0)], [359.5], circle_segments=180)

        self.assertGreater(len(region), 2)
        self.assertGreaterEqual(np.min(region[:, 0]), -1e-6)
        for point in region:
            if np.linalg.norm(point) > 1e-5:
                self.assertLessEqual(angular_error_deg(point, (0.0, 0.0), 359.5), 1.01)

    def test_diameter_matches_equilateral_triangle(self):
        triangle = np.array(
            [[0.0, 0.0], [2.0, 0.0], [1.0, math.sqrt(3.0)]], dtype=float
        )
        diameter, _ = convex_diameter(triangle)
        self.assertAlmostEqual(diameter, 2.0, places=10)

    def test_diameter_circle_does_not_always_cover_region(self):
        triangle = np.array(
            [[0.0, 0.0], [2.0, 0.0], [1.0, math.sqrt(3.0)]], dtype=float
        )
        result = diameter_circle_coverage(triangle)

        self.assertFalse(result.covers)
        self.assertAlmostEqual(result.diameter, 2.0, places=10)
        self.assertGreater(result.max_center_distance, result.radius)

    def test_diameter_circle_covers_rectangle(self):
        rectangle = np.array(
            [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]],
            dtype=float,
        )
        self.assertTrue(diameter_circle_coverage(rectangle).covers)


class Problem2ModelTests(unittest.TestCase):
    def test_first_region_has_no_artificial_1000m_inner_boundary(self):
        region = first_possible_region((200.0, 100.0), 0.0, circle_segments=180)
        near_source = np.array([300.0, 100.0])

        self.assertTrue(point_in_convex_polygon(near_source, region, tolerance=1e-6))

    def test_detection_classes_use_minimum_and_maximum_reception_radii(self):
        region = first_possible_region((0.0, 0.0), 0.0, circle_segments=120)

        robust = classify_detection_point(np.array([750.0, 0.0]), region)
        possible_only = classify_detection_point(np.array([2200.0, 0.0]), region)
        impossible = classify_detection_point(np.array([3100.0, 0.0]), region)

        self.assertTrue(robust.guaranteed)
        self.assertTrue(robust.possible)
        self.assertTrue(possible_only.possible)
        self.assertFalse(possible_only.guaranteed)
        self.assertFalse(impossible.possible)

    def test_strict_optimizer_uses_worst_case_diameter(self):
        result = optimize_second_point(
            (0.0, 0.0),
            0.0,
            candidate_spacing=250.0,
            source_sample_count=7,
            circle_segments=90,
        )

        self.assertIsNotNone(result.recommended_point)
        self.assertGreater(len(result.robust_candidate_points), 0)
        self.assertAlmostEqual(
            result.worst_case_diameter,
            min(item.worst_case_diameter for item in result.evaluations),
            places=9,
        )
        classification = classify_detection_point(
            result.recommended_point, result.first_region
        )
        self.assertTrue(classification.guaranteed)
        self.assertGreaterEqual(result.typical_crossing_angle_deg, 0.0)
        self.assertLessEqual(result.typical_crossing_angle_deg, 90.0)

    def test_dynamic_fallback_keeps_valid_preferred_side(self):
        point, _ = get_problem2_point_dynamic(
            np.array([-1200.0, -2550.0]), 0.0, attempt=0
        )
        np.testing.assert_allclose(point, np.array([0.0, -1750.0]), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
