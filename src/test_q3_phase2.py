"""Focused tests for the independent Phase 2 active-localization kernel."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from new import q3_active_localization as active


class Phase2GeometryTests(unittest.TestCase):
    def test_robust_candidate_uses_all_vertices(self):
        omega = np.array(
            [[-800.0, -10.0], [800.0, -10.0], [800.0, 10.0], [-800.0, 10.0]]
        )
        self.assertAlmostEqual(
            active.max_distance_to_region((0.0, 0.0), omega),
            np.hypot(800.0, 10.0),
            places=8,
        )
        self.assertTrue(active.is_robust_candidate((0.0, 0.0), omega))
        self.assertFalse(active.is_robust_candidate((0.0, 1000.0), omega))

    def test_posterior_retains_history_region(self):
        # This Omega is intentionally much smaller than the target disk.  A
        # history-discarding implementation would incorrectly return points
        # outside this rectangle after the new measurement.
        omega = np.array(
            [[500.0, -100.0], [900.0, -100.0], [900.0, 100.0], [500.0, 100.0]]
        )
        posterior = active.posterior_region(
            omega,
            (0.0, 0.0),
            (700.0, 10.0),
            0.0,
            circle_segments=36,
        )
        self.assertGreaterEqual(len(posterior), 3)
        self.assertGreaterEqual(float(np.min(posterior[:, 0])), 500.0 - 1e-5)
        self.assertLessEqual(float(np.max(posterior[:, 0])), 900.0 + 1e-5)
        self.assertTrue(
            all(active.point_in_convex_polygon(point, omega, tolerance=1e-5) for point in posterior)
        )

    def test_outside_target_candidates_are_allowed(self):
        omega = active.make_ellipse_region((1700.0, 0.0), (25.0, 12.0), vertices=20)
        points = active.generate_robust_candidates(
            omega,
            (1700.0, 0.0),
            300.0,
            spacing_m=50.0,
            circle_segments=24,
        )
        self.assertGreater(len(points), 0)
        self.assertTrue(np.any(np.linalg.norm(points, axis=1) > active.TARGET_RADIUS))


class Phase2OptimizationTests(unittest.TestCase):
    def test_random_error_is_order_independent_and_position_fixed(self):
        source = np.array([120.0, -35.0])
        first = np.array([0.0, 0.0])
        second = np.array([20.0, 10.0])
        errors_a = {}
        first_a = active._lookup_error(
            None, first, errors_a, np.random.default_rng(1), source=source, error_seed=77
        )
        second_a = active._lookup_error(
            None, second, errors_a, np.random.default_rng(1), source=source, error_seed=77
        )
        errors_b = {}
        second_b = active._lookup_error(
            None, second, errors_b, np.random.default_rng(999), source=source, error_seed=77
        )
        first_b = active._lookup_error(
            None, first, errors_b, np.random.default_rng(999), source=source, error_seed=77
        )
        self.assertAlmostEqual(first_a, first_b, places=12)
        self.assertAlmostEqual(second_a, second_b, places=12)
        self.assertGreaterEqual(first_a, -1.0)
        self.assertLessEqual(first_a, 1.0)

    def test_six_to_twenty_metres_is_direction_not_near(self):
        omega = np.array(
            [[0.0, -50.0], [40.0, -50.0], [40.0, 50.0], [0.0, 50.0]]
        )
        # The true source is 10 m from the initial robust candidate.  It is
        # deliberately above the simulator's 5 m near threshold but inside
        # the 20 m MEC guarantee radius.
        result = active.simulate_single_target(
            omega,
            (30.0, 0.0),
            (20.0, 0.0),
            120.0,
            source_sample_count=1,
            error_samples=(0.0,),
            spacing_m=80.0,
            circle_segments=16,
            max_steps=2,
        )
        self.assertNotEqual(result.reason, "near_then_cleared")

    def test_candidate_evaluation_enumerates_sources_and_errors(self):
        omega = active.make_ellipse_region((650.0, 0.0), (80.0, 20.0), vertices=16)
        evaluation = active.evaluate_candidate(
            omega,
            (400.0, 0.0),
            (650.0, 250.0),
            source_sample_count=5,
            error_samples=(-1.0, 0.0, 1.0),
            circle_segments=24,
        )
        self.assertTrue(evaluation.robust)
        self.assertEqual(evaluation.case_count, 15)
        self.assertEqual(evaluation.nonempty_case_count, 15)
        self.assertTrue(np.isfinite(evaluation.q_rho_m))
        self.assertTrue(np.isfinite(evaluation.q_diameter_m))

    def test_pareto_choices_are_budget_constrained(self):
        omega = active.make_ellipse_region((700.0, 0.0), (180.0, 25.0), vertices=16)
        result = active.solve_pareto(
            omega,
            (450.0, -50.0),
            budgets_m=(50.0, 150.0, 300.0),
            source_sample_count=5,
            error_samples=(-1.0, 0.0, 1.0),
            spacing_m=100.0,
            circle_segments=24,
            max_candidates=60,
        )
        self.assertEqual(result.budgets_m, (50.0, 150.0, 300.0))
        for choice in result.choices:
            if choice.evaluation is not None:
                self.assertLessEqual(choice.evaluation.movement_distance_m, choice.budget_m + 1e-5)
                self.assertTrue(choice.evaluation.robust)
        self.assertGreaterEqual(len(result.frontier), 1)

    def test_single_target_time_breakdown_and_clear_attempt(self):
        omega = active.make_ellipse_region((650.0, 0.0), (100.0, 20.0), vertices=16)
        source = np.array([720.0, 5.0])
        result = active.simulate_single_target(
            omega,
            source,
            (450.0, -50.0),
            300.0,
            source_sample_count=3,
            error_samples=(-1.0, 0.0, 1.0),
            spacing_m=120.0,
            circle_segments=24,
            max_steps=4,
        )
        self.assertTrue(result.success, result.reason)
        self.assertEqual(result.clear_attempts, 1)
        expected = (
            result.measurement_move_distance_m / active.SPEED_MPS
            + result.measurement_time_s
            + result.final_clear_move_distance_m / active.SPEED_MPS
            + active.CLEAR_TIME_S
        )
        self.assertAlmostEqual(result.total_time_s, expected, places=7)


if __name__ == "__main__":
    unittest.main()
