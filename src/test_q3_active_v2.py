"""Regression checks for transplanting active processing onto current Phase 1."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from new import Q3_active_v2 as q


class ActiveV2Tests(unittest.TestCase):
    def test_phase1_functions_preserved_except_processing(self):
        root = Path(__file__).parent
        def definitions(path):
            tree = ast.parse((root / path).read_text(encoding='utf-8-sig'))
            return {n.name: ast.dump(n) for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        original = definitions('experiments/q3/Q3_phase1.py')
        merged = definitions('new/Q3_active_v2.py')
        intentionally_changed = {
            'ChannelServiceTask',
            'homing_clear',
            'channel_service_task',
            'process_channel',
            'process_pending_channels',
            'main',
        }
        for name, definition in original.items():
            if name not in intentionally_changed:
                self.assertEqual(definition, merged[name], name)

    def state(self, diameter):
        state = q.ChannelState(1, observations=[np.array([0., 0.])],
                               bearings=[0.], last_position=np.array([0., 0.]))
        state.polygon = np.array([[100., -30.], [100.+diameter, -30.],
                                  [100.+diameter, 30.], [100., 30.]])
        state.diameter = diameter
        state.mec_center = np.mean(state.polygon, axis=0)
        state.mec_radius = diameter / 2
        state.refresh = lambda: None
        return state

    def test_latest_speculative_clear_precedes_active(self):
        with patch.object(q, 'try_clear', return_value=True) as clear, \
             patch.object(q, 'active_localization_clear') as active:
            self.assertTrue(q.process_channel(self.state(80)))
            self.assertEqual(clear.call_args.args[2], 'speculative-centroid')
            active.assert_not_called()

    def test_active_starts_from_actual_robot_position(self):
        with patch.object(q, 'CURRENT_POS', np.array([200., 50.])), \
             patch.object(q, 'active_localization_clear', return_value=True) as active:
            self.assertTrue(q.process_channel(self.state(200)))
            np.testing.assert_allclose(active.call_args.args[1], [200., 50.])

    def test_conservative_retry_retains_phase1_fallback(self):
        with patch.object(q, 'iterative_clear', return_value=True) as fallback, \
             patch.object(q, 'active_localization_clear') as active:
            self.assertTrue(q.process_channel(self.state(200), 'conservative'))
            fallback.assert_called_once()
            active.assert_not_called()

    def test_no_candidate_uses_local_fallback(self):
        state = self.state(200)
        with patch.object(q, 'plan_active_candidate', return_value=None), \
             patch.object(q, 'homing_clear', return_value=True) as fallback:
            self.assertTrue(q.active_localization_clear(state, [200., 50.]))
            fallback.assert_called_once()

    def active_decision(self, point=(80.0, 20.0), q_rho=40.0):
        evaluation = SimpleNamespace(
            point=np.asarray(point, dtype=float),
            movement_distance_m=float(np.linalg.norm(point)),
            q_rho_m=float(q_rho),
            q_diameter_m=80.0,
            robust=True,
        )
        return q.ActiveDecision(evaluation, 100.0)

    def test_route_task_uses_actual_first_active_candidate(self):
        state = self.state(200)
        decision = self.active_decision(q_rho=19.0)
        with patch.object(q, 'plan_active_candidate', return_value=decision):
            task = q.channel_service_task(
                state, np.array([0.0, 0.0]), mode='normal'
            )

        self.assertEqual(task.action, 'active-one-step')
        self.assertTrue(task.active_plan_ready)
        self.assertIs(task.active_decision, decision)
        np.testing.assert_allclose(task.entry, decision.point)
        np.testing.assert_allclose(task.exit, np.mean(state.polygon, axis=0))

    def test_route_uses_service_proxy_for_nonclosing_active_step(self):
        state = self.state(200)
        decision = self.active_decision(q_rho=40.0)
        expected_entry = q._homing_join(state)[0]
        with patch.object(q, 'plan_active_candidate', return_value=decision):
            task = q.channel_service_task(
                state, np.array([0.0, 0.0]), mode='normal'
            )

        self.assertEqual(task.action, 'active-via-service-proxy')
        self.assertIs(task.active_decision, decision)
        np.testing.assert_allclose(task.entry, expected_entry)

    def test_route_task_models_no_candidate_fallback_from_homing_join(self):
        state = self.state(200)
        current = np.array([25.0, -10.0])
        with patch.object(q, 'plan_active_candidate', return_value=None):
            task = q.channel_service_task(state, current, mode='normal')

        self.assertEqual(task.action, 'active_fallback')
        self.assertTrue(task.active_plan_ready)
        self.assertEqual(task.active_failure_reason, 'active_no_candidate')
        self.assertIsNotNone(task.fallback_plan)
        np.testing.assert_allclose(task.entry, task.fallback_plan[0])

    def test_route_active_decision_is_reused_by_execution(self):
        state = self.state(200)
        decision = self.active_decision()
        q.CURRENT_POS = np.array([0.0, 0.0])
        with patch.object(q, 'plan_active_candidate', return_value=decision) as planner, \
             patch.object(
                 q,
                 'measure_and_update',
                 return_value={'accepted': True, 'measure_result': 'near'},
             ), \
             patch.object(q, 'try_clear', return_value=True):
            unresolved = q.process_pending_channels(
                {state.channel: state},
                set(),
                runtime_limit_s=None,
            )

        self.assertEqual(unresolved, [])
        self.assertEqual(planner.call_count, 1)

    def test_route_homing_fallback_plan_is_reused_by_execution(self):
        state = self.state(200)
        expected_point, expected_theta, expected_bracket = q._homing_join(state)
        q.CURRENT_POS = np.array([25.0, -10.0])
        with patch.object(q, 'plan_active_candidate', return_value=None) as planner, \
             patch.object(q, 'homing_clear', return_value=True) as homing:
            unresolved = q.process_pending_channels(
                {state.channel: state},
                set(),
                runtime_limit_s=None,
            )

        self.assertEqual(unresolved, [])
        self.assertEqual(planner.call_count, 1)
        np.testing.assert_allclose(homing.call_args.args[1], expected_point)
        self.assertEqual(homing.call_args.args[2], expected_theta)
        self.assertEqual(homing.call_args.args[3], expected_bracket)

    def test_homing_reuses_existing_direction_without_same_point_measure(self):
        state = self.state(200)
        with patch.object(
            q,
            'measure_and_update',
            return_value={'accepted': True, 'measure_result': 'near'},
        ) as measured, patch.object(q, 'try_clear', return_value=True):
            self.assertTrue(
                q.homing_clear(
                    state,
                    np.array([0.0, 0.0]),
                    0.0,
                    100.0,
                    initial_direction_known=True,
                )
            )

        np.testing.assert_allclose(measured.call_args.args[1], [q.HOMING_STEP, 0.0])
        self.assertEqual(measured.call_count, 1)

    def test_discovery_failure_retry_records_recovered_direction(self):
        states = {}
        failures = {(2, 7): np.array([10.0, 20.0])}
        response = {
            'accepted': True,
            'measure_result': 'direction',
            'svd_deg': 33.0,
        }
        with patch.object(q, 'measure', return_value=response):
            remaining = q.retry_discovery_failures(failures, states, set())

        self.assertEqual(remaining, {})
        self.assertIn(7, states)
        np.testing.assert_allclose(states[7].observations[-1], [10.0, 20.0])
        self.assertEqual(states[7].bearings[-1], 33.0)

    def test_discovery_failure_remains_explicit_after_retry(self):
        failures = {(1, 4): np.array([30.0, 40.0])}
        with patch.object(q, 'measure', return_value=None):
            remaining = q.retry_discovery_failures(failures, {}, set())

        self.assertEqual(set(remaining), {(1, 4)})


if __name__ == '__main__':
    unittest.main()
