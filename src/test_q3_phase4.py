"""Focused offline tests for the Q3 Phase 4 active controller."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np


SRC = Path(__file__).resolve().parent
SRC_NEW = SRC / "new"
SRC_EXPERIMENTS_Q3 = SRC / "experiments" / "q3"
for path in (SRC_NEW, SRC_EXPERIMENTS_Q3):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Q3_active as q  # noqa: E402


def evaluation(point, movement, rho, diameter=100.0, robust=True):
    return SimpleNamespace(
        point=np.asarray(point, dtype=float),
        movement_distance_m=float(movement),
        q_rho_m=float(rho),
        q_diameter_m=float(diameter),
        robust=bool(robust),
    )


def nontrivial_state(radius=100.0):
    """Build a deterministic state without invoking geometry/network code."""
    state = q.ChannelState(
        1,
        observations=[np.array([0.0, 0.0])],
        bearings=[0.0],
        last_position=np.array([0.0, 0.0]),
        polygon=np.array(
            [[-radius, -10.0], [radius, -10.0], [radius, 10.0], [-radius, 10.0]]
        ),
    )
    state.mec_center = np.array([0.0, 0.0])
    state.mec_radius = float(radius)
    # The tests below inject the posterior refresh behavior.  This keeps them
    # focused on Phase 4 control flow rather than the Phase 1 geometry kernel.
    state.refresh = lambda: None
    return state


class Phase4SelectionTests(unittest.TestCase):
    def test_qrho_threshold_uses_shortest_actual_move_across_budget_choices(self):
        # The 350 m point has better Q_D but must lose to the 150 m point.
        first = evaluation([350.0, 0.0], 350.0, 19.0, diameter=30.0)
        second = evaluation([150.0, 0.0], 150.0, 20.0, diameter=80.0)
        pareto = SimpleNamespace(evaluations=(first, second), choices=())

        decision = q.select_active_candidate(pareto)

        self.assertIsNotNone(decision)
        np.testing.assert_allclose(decision.point, [150.0, 0.0])
        self.assertEqual(decision.budget_m, 200.0)

    def test_without_qrho_threshold_uses_first_feasible_budget_pareto_point(self):
        high_quality_late = evaluation([300.0, 0.0], 300.0, 40.0)
        low_budget = evaluation([80.0, 0.0], 80.0, 55.0)
        pareto = SimpleNamespace(
            evaluations=(high_quality_late, low_budget),
            choices=(
                q.BudgetChoice(100.0, low_budget),
                q.BudgetChoice(200.0, high_quality_late),
            ),
        )

        decision = q.select_active_candidate(pareto)

        self.assertIsNotNone(decision)
        self.assertEqual(decision.budget_m, 100.0)
        np.testing.assert_allclose(decision.point, [80.0, 0.0])


class Phase4ControlFlowTests(unittest.TestCase):
    def test_phase1_fallback_synchronizes_active_position_mirror(self):
        state = nontrivial_state()
        q.CURRENT_POS = np.array([10.0, 20.0])
        q._phase1.CURRENT_POS = np.array([-1.0, -1.0])
        fallback_endpoint = np.array([310.0, 420.0])

        def fake_homing(*args, **kwargs):
            np.testing.assert_allclose(q._phase1.CURRENT_POS, [10.0, 20.0])
            q._phase1.CURRENT_POS = fallback_endpoint.copy()
            return True

        with patch.object(q._phase1, "homing_clear", side_effect=fake_homing):
            self.assertTrue(q.homing_clear(state, [0.0, 0.0], 0.0, 100.0))

        np.testing.assert_allclose(q.CURRENT_POS, fallback_endpoint)
        np.testing.assert_allclose(q._phase1.CURRENT_POS, fallback_endpoint)

    def test_direction_after_active_move_is_recorded_and_loop_continues_before_homing(self):
        state = nontrivial_state()
        first = evaluation([100.0, 0.0], 100.0, 30.0)
        second = evaluation([180.0, 0.0], 180.0, 35.0)
        responses = [
            {"accepted": True, "measure_result": "direction", "svd_deg": 42.0},
            {"accepted": True, "measure_result": "near"},
        ]

        with patch.object(q, "plan_active_candidate", side_effect=[
            q.ActiveDecision(first, 100.0),
            q.ActiveDecision(second, 200.0),
        ]) as planner_mock:
            with patch.object(q, "measure", side_effect=responses):
                with patch.object(q, "try_clear", return_value=True):
                    with patch.object(
                        q,
                        "homing_bracket_fallback",
                        side_effect=AssertionError("homing used before active loop finished"),
                    ):
                        result = q.process_channel(state)

        self.assertTrue(result)
        self.assertEqual(planner_mock.call_count, 2)
        self.assertEqual(len(state.observations), 2)
        self.assertEqual(len(state.bearings), 2)
        self.assertAlmostEqual(state.bearings[-1], 42.0)

    def test_missing_candidate_enters_homing_bracket_fallback(self):
        state = nontrivial_state()
        with patch.object(q, "plan_active_candidate", return_value=None):
            with patch.object(q, "homing_bracket_fallback", return_value=True) as fallback:
                result = q.active_localization_clear(
                    state,
                    config=q.ActiveConfig(max_steps=1, time_limit_s=10.0),
                )

        self.assertTrue(result)
        self.assertEqual(fallback.call_count, 1)
        self.assertIn("no_candidate", fallback.call_args.args[-1])

    def test_planner_exception_enters_homing_bracket_fallback(self):
        state = nontrivial_state()
        with patch.object(q, "plan_active_candidate", side_effect=RuntimeError("boom")):
            with patch.object(q, "homing_bracket_fallback", return_value=False) as fallback:
                result = q.active_localization_clear(
                    state,
                    config=q.ActiveConfig(max_steps=1, time_limit_s=10.0),
                )

        self.assertFalse(result)
        self.assertEqual(fallback.call_count, 1)
        self.assertIn("planner_exception", fallback.call_args.args[-1])

    def test_homing_failure_continues_to_bounded_bracket_fallback(self):
        state = nontrivial_state()
        with patch.object(q, "homing_clear", return_value=False) as homing:
            with patch.object(q, "bracket_clear", return_value=True) as bracket:
                result = q.homing_bracket_fallback(state, reason="active_no_candidate")

        self.assertTrue(result)
        homing.assert_called_once()
        bracket.assert_called_once()

    def test_mec_radius_at_or_below_twenty_is_cleared_before_active_planner(self):
        state = nontrivial_state(radius=10.0)
        with patch.object(q, "try_clear", return_value=True) as clear:
            with patch.object(
                q,
                "plan_active_candidate",
                side_effect=AssertionError("active planner called before MEC clear"),
            ):
                result = q.process_channel(state)

        self.assertTrue(result)
        clear.assert_called_once()


class Phase4PlannerInjectionTests(unittest.TestCase):
    def test_real_solver_smoke_returns_feasible_decision_with_phase4_ladder(self):
        # Small convex prior plus a light configuration keeps this a genuine
        # solve_pareto integration check without running the research sweep.
        omega = np.array(
            [[70.0, -10.0], [130.0, -10.0], [130.0, 10.0], [70.0, 10.0]]
        )
        state = q.ChannelState(
            2,
            polygon=omega,
            last_position=np.array([0.0, 0.0]),
        )
        config = q.ActiveConfig(
            source_sample_count=1,
            error_samples=(0.0,),
            spacing_m=80.0,
            circle_segments=16,
            max_candidates=30,
            max_steps=1,
            time_limit_s=10.0,
        )

        decision = q.plan_active_candidate(
            state,
            np.array([0.0, 0.0]),
            config=config,
        )

        self.assertIsNotNone(decision)
        self.assertIn(decision.budget_m, q.ACTIVE_BUDGETS_M)
        self.assertTrue(decision.evaluation.robust)
        self.assertLessEqual(
            decision.evaluation.movement_distance_m,
            decision.budget_m + q.GEOMETRY_EPS,
        )
        self.assertEqual(decision.pareto.budgets_m, q.ACTIVE_BUDGETS_M)
        self.assertFalse(decision.pareto.sampling["continuous_optimum_claimed"])

    def test_planner_injection_uses_shared_ladder_and_allows_circle_outside_point(self):
        state = nontrivial_state(radius=100.0)
        outside = evaluation([1830.0, 0.0], 100.0, 35.0)
        captured = {}

        def fake_solver(omega, current, **kwargs):
            captured["omega"] = omega
            captured["current"] = current
            captured.update(kwargs)
            return SimpleNamespace(
                evaluations=(outside,),
                choices=(q.BudgetChoice(200.0, outside),),
            )

        decision = q.plan_active_candidate(
            state,
            np.array([1730.0, 0.0]),
            config=q.ActiveConfig(source_sample_count=1, circle_segments=16),
            planner=fake_solver,
        )

        self.assertIsNotNone(decision)
        np.testing.assert_allclose(decision.point, [1830.0, 0.0])
        self.assertEqual(captured["budgets_m"], q.ACTIVE_BUDGETS_M)
        self.assertIsNone(captured["target_radius"])


class Phase4BaselineTests(unittest.TestCase):
    def test_fast2_baseline_has_no_worktree_diff(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "diff", "--quiet", "--", "src/new/Q3_fast2.py"],
            cwd=root,
            check=False,
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
