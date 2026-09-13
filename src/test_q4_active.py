"""Regression and geometry checks for Q4 Active."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from new import Q3_active_v2 as q3
from new import Q4_active as q4
import q4_active_experiments as q4_experiments


def _point_in_triangle(point, triangle, tolerance=1e-7):
    point = np.asarray(point, dtype=float)
    a, b, c = np.asarray(triangle, dtype=float)
    matrix = np.column_stack((b - a, c - a))
    try:
        uv = np.linalg.solve(matrix, point - a)
    except np.linalg.LinAlgError:
        return False
    u, v = uv
    return u >= -tolerance and v >= -tolerance and u + v <= 1.0 + tolerance


class Q4ActiveTests(unittest.TestCase):
    def state(self, diameter=200.0):
        state = q4.ChannelState(
            1,
            observations=[np.array([0.0, 0.0])],
            bearings=[0.0],
            last_position=np.array([0.0, 0.0]),
        )
        state.polygon = np.array(
            [[100.0, -30.0], [100.0 + diameter, -30.0],
             [100.0 + diameter, 30.0], [100.0, 30.0]]
        )
        state.diameter = diameter
        state.mec_center = np.mean(state.polygon, axis=0)
        state.mec_radius = diameter / 2.0
        state.refresh = lambda: None
        return state

    def decision(self, point, budget=100.0):
        evaluation = SimpleNamespace(
            point=np.asarray(point, dtype=float),
            movement_distance_m=0.0,
            q_rho_m=40.0,
            q_diameter_m=80.0,
            robust=True,
        )
        return q4.ActiveDecision(evaluation, budget)

    def test_q3_state_geometry_and_routing_are_inherited(self):
        self.assertIs(q4.ChannelState, q3.ChannelState)
        self.assertIs(q4.channel_action_plan, q3.channel_action_plan)
        self.assertIs(q4.channel_service_task, q3.channel_service_task)
        self.assertIs(q4.service_route_exact, q3.service_route_exact)
        self.assertIs(q4.try_clear, q3.try_clear)
        self.assertIs(q4.add_observation, q3.add_observation)

    def test_directional_mesh_has_distance_and_angular_coverage(self):
        certificate = q4.directional_coverage_certificate()
        self.assertGreater(certificate["outer_inradius_m"], q4.TARGET_RADIUS)
        self.assertLess(
            certificate["max_triangle_edge_m"], q4.MIN_RECEPTION_RADIUS
        )
        self.assertEqual(certificate["full_probe_count"], 25)
        ordered = q4.directional_discovery_positions()
        self.assertEqual(len(ordered), 24)
        start = q4.scan_position(q4.SCAN_N - 1)
        path = [start] + ordered
        route_length = sum(
            np.linalg.norm(right - left)
            for left, right in zip(path, path[1:])
        )
        self.assertLess(route_length, 18_500.0)

        triangles = q4.directional_discovery_triangles()
        probes = [np.array([0.0, 0.0])]
        probes += [
            q4.directional_inner_position(index)
            for index in range(q4.DIRECTIONAL_RING_N)
        ]
        probes += [
            q4.directional_outer_position(index)
            for index in range(q4.DIRECTIONAL_RING_N)
        ]
        for radius in np.linspace(0.0, q4.TARGET_RADIUS, 10):
            for angle in np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False):
                source = radius * np.array([np.cos(angle), np.sin(angle)])
                containing = [
                    triangle
                    for triangle in triangles
                    if _point_in_triangle(source, triangle)
                ]
                self.assertTrue(containing, tuple(source))
                self.assertTrue(
                    any(
                        max(np.linalg.norm(vertex - source) for vertex in triangle)
                        <= q4.MIN_RECEPTION_RADIUS + 1e-7
                        for triangle in containing
                    )
                )
                in_range = [
                    probe
                    for probe in probes
                    if np.linalg.norm(probe - source)
                    <= q4.MIN_RECEPTION_RADIUS + 1e-7
                ]
                for heading in np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False):
                    direction = np.array([np.cos(heading), np.sin(heading)])
                    self.assertGreater(
                        max(float((probe - source) @ direction) for probe in in_range),
                        -1e-7,
                    )

    def test_directional_discovery_only_keeps_never_seen_channels_pending(self):
        states = {}
        cleared = set()
        calls = 0

        def fake_measure(x, y, channel):
            nonlocal calls
            calls += 1
            return {"accepted": True, "measure_result": "direction", "svd_deg": 12.0}

        with patch.object(q4, "measure", side_effect=fake_measure):
            result = q4.directional_discovery({4}, states, cleared)
        self.assertIn(4, states)
        self.assertFalse(result.unseen_channels)
        self.assertEqual(result.visited_probes, 1)
        self.assertEqual(calls, 1)

    def test_boundary_source_hidden_from_q3_is_found_by_directional_mesh(self):
        source = np.array([q4.TARGET_RADIUS, 0.0])
        transmit_heading = np.array([1.0, 0.0])  # emits east; Q3 ring is west

        def directional_measure(x, y, channel):
            point = np.array([x, y], dtype=float)
            offset = point - source
            visible = (
                np.linalg.norm(offset) <= q4.MIN_RECEPTION_RADIUS
                and float(offset @ transmit_heading) > 0.0
            )
            if not visible:
                return {"accepted": True, "measure_result": "no_signal"}
            bearing = np.degrees(np.arctan2(-offset[1], -offset[0])) % 360.0
            return {
                "accepted": True,
                "measure_result": "direction",
                "svd_deg": float(bearing),
            }

        self.assertTrue(
            all(
                directional_measure(*q4.scan_position(index), 1)["measure_result"]
                == "no_signal"
                for index in range(q4.SCAN_N)
            )
        )
        states = {}
        with patch.object(q4, "measure", side_effect=directional_measure):
            result = q4.directional_discovery({1}, states, set())
        self.assertIn(1, states)
        self.assertFalse(result.unseen_channels)
        self.assertTrue(result.completed)

    def test_directional_discovery_channel_order_is_serpentine(self):
        calls = []

        def no_signal(x, y, channel):
            calls.append(channel)
            return {"accepted": True, "measure_result": "no_signal"}

        with patch.object(q4, "measure", side_effect=no_signal):
            q4.directional_discovery({1, 2}, {}, set())
        self.assertEqual(calls[:4], [1, 2, 2, 1])

    def test_directional_experiment_simulator_distinguishes_front_and_back(self):
        source = q4_experiments.DirectionalSource(1, (0.0, 0.0), 1000.0, 0.0)
        simulator = q4_experiments.DirectionalOfflineSimulator([source])
        behind = simulator._measure(
            {"position": {"x": -100.0, "y": 0.0}, "channel": 1}
        )
        ahead = simulator._measure(
            {"position": {"x": 100.0, "y": 0.0}, "channel": 1}
        )
        self.assertEqual(behind["measure_result"], "no_signal")
        self.assertEqual(ahead["measure_result"], "direction")

    def test_directional_reacquisition_bisects_visible_to_blind_segment(self):
        state = self.state()
        probes = []

        def fake_measure_and_update(current, position):
            point = np.asarray(position, dtype=float)
            probes.append(point.copy())
            if len(probes) == 1:
                return {"accepted": True, "measure_result": "no_signal"}
            q4.add_observation(current, point, 5.0)
            return {"accepted": True, "measure_result": "direction", "svd_deg": 5.0}

        with patch.object(q4, "measure_and_update", side_effect=fake_measure_and_update):
            result = q4.directional_reacquisition(state, np.array([100.0, 0.0]))
        self.assertTrue(result.recovered)
        np.testing.assert_allclose(probes[0], [50.0, 0.0])
        np.testing.assert_allclose(probes[1], [25.0, 0.0])

    def test_active_no_signal_reacquires_then_replans(self):
        state = self.state()
        decisions = [self.decision([100.0, 0.0]), self.decision([50.0, 80.0])]
        measured = []

        def planner(*_args, **_kwargs):
            return decisions.pop(0)

        def fake_measure_and_update(current, position):
            point = np.asarray(position, dtype=float)
            measured.append(point.copy())
            if len(measured) == 1:
                return {"accepted": True, "measure_result": "no_signal"}
            if len(measured) == 2:
                q4.add_observation(current, point, 4.0)
                return {"accepted": True, "measure_result": "direction", "svd_deg": 4.0}
            return {"accepted": True, "measure_result": "near"}

        config = q4.ActiveConfig(max_steps=3, time_limit_s=10.0)
        with patch.object(q4, "measure_and_update", side_effect=fake_measure_and_update), \
             patch.object(q4, "try_clear", return_value=True):
            self.assertTrue(
                q4.active_localization_clear(
                    state, np.array([0.0, 0.0]), config=config, planner=planner
                )
            )
        np.testing.assert_allclose(measured[0], [100.0, 0.0])
        np.testing.assert_allclose(measured[1], [50.0, 0.0])
        np.testing.assert_allclose(measured[2], [50.0, 80.0])

    def test_homing_no_signal_brackets_from_last_visible_point(self):
        state = self.state()
        with patch.object(
            q4,
            "measure_and_update",
            return_value={"accepted": True, "measure_result": "no_signal"},
        ), patch.object(q4, "bracket_clear", return_value=True) as bracket:
            self.assertTrue(
                q4.homing_clear(
                    state, [0.0, 0.0], 0.0, 1500.0,
                    initial_direction_known=True,
                )
            )
        np.testing.assert_allclose(bracket.call_args.args[1], [0.0, 0.0])
        self.assertEqual(bracket.call_args.args[3], q4.HOMING_STEP)

    def test_iterative_reacquisition_stays_inside_its_bound(self):
        state = self.state()
        recovered = q4.ReacquisitionResult(
            "direction", np.array([0.0, 0.0]), 0.0, 1
        )
        with patch.object(
            q4,
            "measure_and_update",
            return_value={"accepted": True, "measure_result": "no_signal"},
        ) as measured, patch.object(
            q4, "directional_reacquisition", return_value=recovered
        ) as reacquire, patch.object(q4, "try_clear", return_value=False):
            self.assertFalse(q4.iterative_clear(state, max_iter=2))
        self.assertEqual(measured.call_count, 2)
        self.assertEqual(reacquire.call_count, 2)
        self.assertEqual(state.last_failure_reason, "iterative_limit")

    def test_bracket_no_signal_does_not_promote_blind_endpoint(self):
        state = self.state()
        probes = []

        def fake_measure(_state, position):
            probes.append(np.asarray(position, dtype=float).copy())
            if len(probes) == 1:
                return {"accepted": True, "measure_result": "no_signal"}
            return {"accepted": True, "measure_result": "near"}

        with patch.object(q4, "measure_and_update", side_effect=fake_measure), \
             patch.object(q4, "try_clear", side_effect=[False, True]):
            self.assertTrue(q4.bracket_clear(state, [0.0, 0.0], 0.0, 200.0))
        np.testing.assert_allclose(probes[0], [100.0, 0.0])
        np.testing.assert_allclose(probes[1], [50.0, 0.0])

    def test_legacy_q4_failures_are_not_present_as_code(self):
        path = Path(__file__).parent / "new" / "Q4_active.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        numeric_constants = {
            float(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        }
        self.assertNotIn("solve_tsp_nearest_neighbor", function_names)
        self.assertNotIn(1780.0, numeric_constants)
        self.assertNotIn(50.0, numeric_constants)


if __name__ == "__main__":
    unittest.main()
