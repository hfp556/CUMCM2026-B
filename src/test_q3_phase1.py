"""Offline regression tests for the deterministic Q3 Phase 1 changes."""

import math
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np


SRC = Path(__file__).resolve().parent
SRC_NEW = SRC / "new"
SRC_EXPERIMENTS_Q3 = SRC / "experiments" / "q3"
for path in (SRC_NEW, SRC_EXPERIMENTS_Q3):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import Q3_phase1 as q  # noqa: E402
from geo_common import minimum_distance_to_polygon  # noqa: E402


class Q3Phase1GeometryTests(unittest.TestCase):
    def test_scan_channel_order_is_serpentine_and_complete(self):
        ascending = q.scan_channel_order(0)
        descending = q.scan_channel_order(1)
        self.assertEqual(ascending, tuple(range(1, 21)))
        self.assertEqual(descending, tuple(range(20, 0, -1)))
        self.assertEqual(descending, tuple(reversed(ascending)))
        self.assertEqual(set(ascending), set(descending))

    def test_regular_heptagon_has_formulaic_angles_and_worst_case_coverage(self):
        self.assertEqual(q.SCAN_N, 7)
        self.assertAlmostEqual(q.SCAN_ANGLE_DEG, 360.0 / 7.0, places=12)
        self.assertGreater(q.SCAN_R, q.SCAN_R_CRITICAL)
        self.assertLessEqual(
            q.worst_regular_scan_distance(), q.MIN_RECEPTION_RADIUS + 1e-9
        )

        angles = []
        for index in range(q.SCAN_N):
            point = q.scan_position(index)
            self.assertAlmostEqual(np.linalg.norm(point), q.SCAN_R, places=9)
            angles.append(math.atan2(point[1], point[0]))
        gaps = [
            (angles[(index + 1) % q.SCAN_N] - angles[index]) % (2.0 * math.pi)
            for index in range(q.SCAN_N)
        ]
        for gap in gaps:
            self.assertAlmostEqual(gap, 2.0 * math.pi / q.SCAN_N, places=12)

    def test_polygon_distance_checks_edge_interiors(self):
        polygon = np.array(
            [[0.0, 100.0], [3000.0, 100.0], [3000.0, 200.0], [0.0, 200.0]]
        )
        point = np.array([1500.0, 0.0])
        vertex_distance = min(np.linalg.norm(vertex - point) for vertex in polygon)
        self.assertGreater(vertex_distance, q.MAX_RECEPTION_RADIUS)
        self.assertAlmostEqual(minimum_distance_to_polygon(point, polygon), 100.0)

    def test_small_diameter_does_not_replace_distance_impossibility_check(self):
        polygon = np.array(
            [[0.0, 100.0], [3000.0, 100.0], [3000.0, 200.0], [0.0, 200.0]]
        )
        state = q.ChannelState(1)
        state.observations = [np.array([0.0, 0.0])]
        state.bearings = [0.0]
        state.polygon = polygon
        state.diameter = 1.0
        state.refresh = lambda: None
        self.assertFalse(q.should_skip_scan(1, {1: state}, set(), [1500.0, 0.0]))

    def test_vertex_only_skip_predicate_is_wrong_but_polygon_distance_is_safe(self):
        """Experiment 4: an edge interior can be within reception range."""
        polygon = np.array(
            [[0.0, 100.0], [3000.0, 100.0], [3000.0, 200.0], [0.0, 200.0]]
        )
        position = np.array([1500.0, 0.0])
        old_predicate = min(
            np.linalg.norm(vertex - position) for vertex in polygon
        ) > q.MAX_RECEPTION_RADIUS
        self.assertTrue(old_predicate)
        self.assertLess(
            minimum_distance_to_polygon(position, polygon),
            q.MAX_RECEPTION_RADIUS,
        )

        state = q.ChannelState(2, polygon=polygon)
        self.assertFalse(q.should_skip_scan(2, {2: state}, set(), position))

    def test_refresh_cache_hits_until_observation_snapshot_changes(self):
        state = q.ChannelState(
            15,
            observations=[np.array([0.0, 0.0])],
            bearings=[0.0],
        )
        with patch.object(q, "region_of", wraps=q.region_of) as region_mock:
            state.refresh()
            state.refresh()
            self.assertEqual(region_mock.call_count, 1)

            q.add_observation(state, np.array([100.0, 0.0]), 20.0)
            self.assertEqual(region_mock.call_count, 2)
            state.refresh()
            self.assertEqual(region_mock.call_count, 2)


class Q3Phase1RoutingTests(unittest.TestCase):
    def test_homing_exit_is_polygon_vertex_mean_and_changes_route_order(self):
        state = q.ChannelState(
            1,
            observations=[np.array([0.0, 0.0])],
            bearings=[0.0],
            polygon=np.array(
                [[50.0, -50.0], [150.0, -50.0], [150.0, 50.0], [50.0, 50.0]]
            ),
            diameter=100.0,
            mec_center=np.array([100.0, 0.0]),
            mec_radius=50.0,
        )
        state.refresh = lambda: None
        actual_homing_task = q.channel_service_task(state)
        self.assertEqual(actual_homing_task.action, "homing")
        np.testing.assert_allclose(actual_homing_task.exit, np.mean(state.polygon, axis=0))
        homing_task = q.ChannelServiceTask(
            state, np.array([10.0, 0.0]), np.array([100.0, 0.0]), "homing"
        )
        np.testing.assert_allclose(
            np.mean(state.polygon, axis=0), np.array([100.0, 0.0])
        )

        other = q.ChannelState(2)
        direct_task = q.ChannelServiceTask(
            other, np.array([20.0, 0.0]), np.array([0.0, 0.0]), "homing"
        )
        entry_only_order = q.tsp_exact(
            [homing_task.entry, direct_task.entry], np.array([0.0, 0.0])
        )
        service_order = q.service_route_exact(
            [homing_task, direct_task], np.array([0.0, 0.0])
        )
        self.assertEqual(entry_only_order, [0, 1])
        self.assertEqual(service_order, [1, 0])

    def test_direct_clear_service_task_has_entry_equal_exit(self):
        state = q.ChannelState(
            3,
            observations=[np.array([0.0, 0.0])],
            bearings=[0.0],
            polygon=np.array(
                [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]
            ),
            diameter=20.0,
            mec_center=np.array([0.0, 0.0]),
            mec_radius=10.0,
        )
        state.refresh = lambda: None
        task = q.channel_service_task(state)
        self.assertEqual(task.action, "guaranteed_clear")
        np.testing.assert_allclose(task.entry, task.exit)

    def test_pending_replans_and_processes_only_each_round_first_task(self):
        old_position = q.CURRENT_POS.copy()
        q.CURRENT_POS = np.array([0.0, 0.0])
        states = {channel: q.ChannelState(channel) for channel in (1, 2, 3)}
        specs = {
            1: (np.array([10.0, 0.0]), np.array([100.0, 0.0])),
            2: (np.array([20.0, 0.0]), np.array([0.0, 0.0])),
            3: (np.array([500.0, 0.0]), np.array([500.0, 0.0])),
        }
        routes = []
        processed = []
        original_route = q.service_route_exact

        def make_task(state):
            entry, exit_point = specs[state.channel]
            return q.ChannelServiceTask(state, entry, exit_point, "homing")

        def traced_route(tasks, start):
            order = original_route(tasks, start)
            routes.append((list(tasks), np.asarray(start).copy(), order))
            return order

        def processor(state, mode):
            processed.append(state.channel)
            q.CURRENT_POS = specs[state.channel][1].copy()
            return True

        try:
            with patch.object(q, "channel_service_task", side_effect=make_task):
                with patch.object(q, "service_route_exact", side_effect=traced_route):
                    unresolved = q.process_pending_channels(
                        states,
                        set(),
                        start_time=0.0,
                        processor=processor,
                        runtime_limit_s=None,
                    )
        finally:
            q.CURRENT_POS = old_position

        self.assertEqual(unresolved, [])
        self.assertEqual([len(item[0]) for item in routes], [3, 2, 1])
        self.assertEqual(
            [
                item[0][item[2][0]].state.channel
                for item in routes
            ],
            processed,
        )
        np.testing.assert_allclose(routes[0][1], [0.0, 0.0])
        np.testing.assert_allclose(routes[1][1], specs[processed[0]][1])
        np.testing.assert_allclose(routes[2][1], specs[processed[1]][1])


class Q3Phase1PerformanceRepairTests(unittest.TestCase):
    @staticmethod
    def _small_speculative_state(channel=16):
        polygon = np.array(
            [[-40.0, -20.0], [40.0, -20.0], [40.0, 20.0], [-40.0, 20.0]]
        )
        state = q.ChannelState(
            channel,
            observations=[np.array([200.0, 0.0])],
            bearings=[180.0],
            polygon=polygon,
            diameter=80.0,
            mec_center=np.array([0.0, 0.0]),
            mec_radius=40.0,
        )
        # Isolate routing/action-plan decisions from polygon recomputation.
        state.refresh = lambda: None
        return state

    def test_routing_point_equals_speculative_first_action(self):
        state = self._small_speculative_state()
        route = q.routing_point(state)
        with patch.object(q, "try_clear", return_value=True) as clear_mock:
            self.assertTrue(q.process_channel(state))
        np.testing.assert_allclose(route, clear_mock.call_args.args[1])
        self.assertTrue(q.channel_action_plan(state).speculative)

    def test_routing_point_equals_homing_join_first_measure(self):
        polygon = np.array(
            [[-500.0, -100.0], [500.0, -100.0], [500.0, 100.0], [-500.0, 100.0]]
        )
        state = q.ChannelState(
            17,
            observations=[np.array([0.0, 0.0])],
            bearings=[0.0],
            polygon=polygon,
            diameter=1000.0,
            mec_center=np.array([0.0, 0.0]),
            mec_radius=500.0,
        )
        state.refresh = lambda: None
        route = q.routing_point(state)
        with patch.object(q, "homing_clear", return_value=True) as homing_mock:
            self.assertTrue(q.process_channel(state))
        np.testing.assert_allclose(route, homing_mock.call_args.args[1])

    def test_d60_skip_requires_reserved_immediate_enroute_clear(self):
        state = self._small_speculative_state(18)
        state.diameter = 50.0
        states = {state.channel: state}
        current = np.array([0.0, 0.0])
        next_scan = np.array([100.0, 0.0])
        self.assertTrue(
            q.should_skip_scan(
                state.channel,
                states,
                set(),
                current,
                next_scan=next_scan,
                current_position=current,
                planned_enroute_channels={state.channel},
            )
        )
        self.assertFalse(
            q.should_skip_scan(
                state.channel,
                states,
                set(),
                current,
                next_scan=next_scan,
                current_position=current,
                planned_enroute_channels=set(),
            )
        )
        state.enroute_clear_failed = True
        self.assertFalse(
            q.should_skip_scan(
                state.channel,
                states,
                set(),
                current,
                next_scan=next_scan,
                current_position=current,
                planned_enroute_channels={state.channel},
            )
        )

    def test_enroute_success_marks_channel_cleared_without_extra_processing(self):
        state = self._small_speculative_state(19)
        route = q.routing_point(state)
        with patch.object(q, "try_clear", return_value=True) as clear_mock:
            self.assertTrue(q.attempt_enroute_clear(state, [0.0, 0.0], [100.0, 0.0]))
        self.assertEqual(state.status, "cleared")
        self.assertTrue(state.enroute_clear_attempted)
        np.testing.assert_allclose(route, clear_mock.call_args.args[1])
        self.assertTrue(q.should_skip_scan(state.channel, {state.channel: state}, {19}, route))

    def test_enroute_failure_does_not_enter_fallback_and_later_measurement_survives(self):
        state = self._small_speculative_state(20)
        with patch.object(q, "try_clear", return_value=False), patch.object(
            q, "homing_clear"
        ) as homing_mock, patch.object(q, "bracket_clear") as bracket_mock, patch.object(
            q, "iterative_clear"
        ) as iterative_mock:
            self.assertFalse(
                q.attempt_enroute_clear(state, [0.0, 0.0], [100.0, 0.0])
            )
        homing_mock.assert_not_called()
        bracket_mock.assert_not_called()
        iterative_mock.assert_not_called()
        self.assertTrue(state.enroute_clear_failed)
        self.assertEqual(state.status, "retry_pending")
        with patch.object(
            q,
            "measure",
            return_value={"accepted": True, "measure_result": "direction", "svd_deg": 42.0},
        ):
            q.measure_and_update(state, np.array([300.0, 0.0]))
        self.assertEqual(len(state.observations), 2)
        self.assertEqual(state.bearings[-1], 42.0)

    def test_enroute_preflight_rejection_does_not_poison_tried_set(self):
        """A reserved candidate may be reconsidered if no clear was issued."""
        state = self._small_speculative_state(22)
        tried = set()
        cleared = set()
        state.diameter = 100.0  # reservation became ineligible before dispatch
        with patch.object(q, "try_clear") as clear_mock:
            self.assertFalse(
                q._attempt_and_record_enroute(
                    state, [0.0, 0.0], [100.0, 0.0], tried, cleared
                )
            )
        clear_mock.assert_not_called()
        self.assertNotIn(state.channel, tried)
        self.assertNotIn(state.channel, cleared)

        state.diameter = 80.0
        with patch.object(q, "try_clear", return_value=True) as clear_mock:
            self.assertTrue(
                q._attempt_and_record_enroute(
                    state, [0.0, 0.0], [100.0, 0.0], tried, cleared
                )
            )
        self.assertIn(state.channel, tried)
        self.assertIn(state.channel, cleared)
        np.testing.assert_allclose(
            clear_mock.call_args.args[1], np.mean(state.polygon, axis=0)
        )

    def test_enroute_rejected_clear_does_not_consume_confirmed_path(self):
        """None or rejected clear responses must not spend leg budget."""
        for response in (None, {"accepted": False}):
            with self.subTest(response=response):
                state = self._small_speculative_state(27)
                tried = set()
                cleared = set()
                leg_context = {
                    "origin": np.array([100.0, 0.0]),
                    "next_scan": np.array([200.0, 0.0]),
                    "path_m": 0.0,
                    "budget_m": q.ENROUTE_CUMULATIVE_BUDGET,
                }
                with patch.object(
                    q, "CURRENT_POS", np.array([100.0, 0.0])
                ), patch.object(q, "_post", return_value=response):
                    self.assertFalse(
                        q._attempt_and_record_enroute(
                            state,
                            [100.0, 0.0],
                            [200.0, 0.0],
                            tried,
                            cleared,
                            leg_context=leg_context,
                        )
                    )
                self.assertEqual(leg_context["path_m"], 0.0)
                self.assertIn(state.channel, tried)
                self.assertNotIn(state.channel, cleared)

    def test_enroute_accepted_no_target_consumes_confirmed_path(self):
        """An accepted no-target clear still moves to its requested point."""
        state = self._small_speculative_state(28)
        tried = set()
        cleared = set()
        leg_context = {
            "origin": np.array([100.0, 0.0]),
            "next_scan": np.array([200.0, 0.0]),
            "path_m": 0.0,
            "budget_m": q.ENROUTE_CUMULATIVE_BUDGET,
        }
        with patch.object(q, "CURRENT_POS", np.array([100.0, 0.0])), patch.object(
            q,
            "_post",
            return_value={"accepted": True, "clear_result": "no_target_in_range"},
        ):
            self.assertFalse(
                q._attempt_and_record_enroute(
                    state,
                    [100.0, 0.0],
                    [200.0, 0.0],
                    tried,
                    cleared,
                    leg_context=leg_context,
                )
            )
            np.testing.assert_allclose(q.CURRENT_POS, [0.0, 0.0])
        self.assertAlmostEqual(leg_context["path_m"], 100.0)
        self.assertIn(state.channel, tried)
        self.assertNotIn(state.channel, cleared)

    def test_enroute_cumulative_budget_stops_chain_after_individual_limits(self):
        origin = np.array([0.0, 0.0])
        next_scan = np.array([1000.0, 0.0])

        def translated(channel, center):
            state = self._small_speculative_state(channel)
            offset = np.asarray(center, dtype=float)
            state.polygon = state.polygon + offset
            state.mec_center = offset.copy()
            state.diameter = 50.0
            return state

        states = {
            23: translated(23, [0.0, 200.0]),
            24: translated(24, [1000.0, 200.0]),
        }
        for state in states.values():
            self.assertLess(
                q._enroute_detour(state.mec_center, origin, next_scan),
                q.ENROUTE_DETOUR,
            )

        first, first_plan, _ = q.select_enroute_candidate(
            states, set(), origin, next_scan, leg_origin=origin, leg_path_m=0.0
        )
        self.assertIsNotNone(first)
        remaining = {
            channel: state
            for channel, state in states.items()
            if state.channel != first.channel
        }
        current = first_plan.point
        path_m = float(np.linalg.norm(origin - current))
        second = next(iter(remaining.values()))
        self.assertLess(
            q._enroute_detour(second.mec_center, current, next_scan),
            q.ENROUTE_DETOUR,
        )
        self.assertGreater(
            q.enroute_cumulative_extra(
                second.mec_center,
                current,
                next_scan,
                leg_origin=origin,
                leg_path_m=path_m,
            ),
            q.ENROUTE_CUMULATIVE_BUDGET,
        )
        candidate, _, _ = q.select_enroute_candidate(
            remaining,
            set(),
            current,
            next_scan,
            leg_origin=origin,
            leg_path_m=path_m,
        )
        self.assertIsNone(candidate)

    def test_enroute_cumulative_budget_allows_chain_inside_cap(self):
        origin = np.array([0.0, 0.0])
        next_scan = np.array([1000.0, 0.0])

        def translated(channel, center):
            state = self._small_speculative_state(channel)
            offset = np.asarray(center, dtype=float)
            state.polygon = state.polygon + offset
            state.mec_center = offset.copy()
            state.diameter = 50.0
            return state

        states = {
            25: translated(25, [500.0, 100.0]),
            26: translated(26, [500.0, -100.0]),
        }
        first, first_plan, _ = q.select_enroute_candidate(
            states, set(), origin, next_scan, leg_origin=origin, leg_path_m=0.0
        )
        self.assertIsNotNone(first)
        remaining = {
            channel: state
            for channel, state in states.items()
            if state.channel != first.channel
        }
        current = first_plan.point
        path_m = float(np.linalg.norm(origin - current))
        candidate, _, _ = q.select_enroute_candidate(
            remaining,
            set(),
            current,
            next_scan,
            leg_origin=origin,
            leg_path_m=path_m,
        )
        self.assertIsNotNone(candidate)
        self.assertLessEqual(
            q.enroute_cumulative_extra(
                candidate.mec_center,
                current,
                next_scan,
                leg_origin=origin,
                leg_path_m=path_m,
            ),
            q.ENROUTE_CUMULATIVE_BUDGET + q.GEOMETRY_EPS,
        )

    def test_speculative_d90_failure_is_requeued_and_can_clear_later(self):
        state = self._small_speculative_state(21)
        calls = []

        def processor(current, mode):
            calls.append(mode)
            if len(calls) == 1:
                return q.process_channel(current, mode)
            return True

        with patch.object(q, "try_clear", return_value=False) as clear_mock, patch.object(
            q,
            "measure_and_update",
            return_value={"accepted": True, "measure_result": "no_signal"},
        ):
            cleared = set()
            unresolved = q.process_pending_channels(
                {state.channel: state},
                cleared,
                start_time=0.0,
                processor=processor,
                runtime_limit_s=None,
            )
        self.assertEqual(calls, ["normal", "conservative"])
        self.assertEqual(clear_mock.call_count, 1)
        self.assertEqual(cleared, {state.channel})
        self.assertEqual(unresolved, [])
        self.assertEqual(state.retry_count, 1)


class Q3Phase1StateMachineTests(unittest.TestCase):
    def test_failed_channel_is_requeued_and_can_succeed(self):
        state = q.ChannelState(7, last_position=np.array([0.0, 0.0]))
        cleared = set()
        calls = []

        def processor(current, mode):
            calls.append(mode)
            if len(calls) == 1:
                current.last_failure_reason = "temporary_clear_failed"
                return False
            return True

        unresolved = q.process_pending_channels(
            {state.channel: state},
            cleared,
            start_time=time.time(),
            processor=processor,
        )

        self.assertEqual(calls, ["normal", "conservative"])
        self.assertEqual(cleared, {7})
        self.assertEqual(unresolved, [])
        self.assertEqual(state.status, "cleared")
        self.assertEqual(state.retry_count, 1)

    def test_no_signal_then_direction_is_recorded_by_common_update_path(self):
        state = q.ChannelState(
            3,
            observations=[np.array([0.0, 0.0])],
            bearings=[0.0],
        )
        responses = [
            {"accepted": True, "measure_result": "no_signal"},
            {"accepted": True, "measure_result": "direction", "svd_deg": 42.5},
        ]

        with patch.object(q, "measure", side_effect=responses):
            first = q.measure_and_update(state, np.array([100.0, 0.0]))
            second = q.measure_and_update(state, np.array([200.0, 0.0]))

        self.assertEqual(first["measure_result"], "no_signal")
        self.assertEqual(second["measure_result"], "direction")
        self.assertEqual(len(state.observations), 2)
        self.assertEqual(len(state.bearings), 2)
        np.testing.assert_allclose(state.observations[-1], [200.0, 0.0])
        self.assertAlmostEqual(state.bearings[-1], 42.5)

    def test_no_signal_then_direction_runs_fallback_and_keeps_planning(self):
        """Experiment 2: exercise the real no-signal recovery branch."""
        state = q.ChannelState(
            4,
            observations=[np.array([0.0, 0.0])],
            bearings=[0.0],
        )
        responses = [
            {"accepted": True, "measure_result": "no_signal"},
            {"accepted": True, "measure_result": "direction", "svd_deg": 42.5},
            {"accepted": True, "measure_result": "near"},
        ]

        with patch.object(q, "measure", side_effect=responses) as measure_mock:
            with patch.object(q, "try_clear", side_effect=[False, True]) as clear_mock:
                result = q.iterative_clear(state, max_iter=3)

        self.assertTrue(result)
        self.assertEqual(measure_mock.call_count, 3)
        self.assertEqual(clear_mock.call_count, 2)
        self.assertEqual(len(state.observations), 2)
        self.assertEqual(len(state.bearings), 2)
        self.assertAlmostEqual(state.bearings[-1], 42.5)
        self.assertGreater(len(state.polygon), 0)
        self.assertTrue(np.all(np.isfinite(state.polygon)))

    def test_failed_clear_is_remeasured_and_retried_after_region_update(self):
        """Experiment 3: a rejected clear does not discard a recoverable channel."""
        source = np.array([100.0, 100.0])
        observations = []
        bearings = []
        for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
            sensor = 1000.0 * np.array([math.cos(angle), math.sin(angle)])
            observations.append(sensor)
            bearings.append(
                math.degrees(math.atan2(*(source - sensor)[::-1]))
            )

        state = q.ChannelState(
            6,
            observations=observations,
            bearings=bearings,
            last_position=observations[-1],
        )
        state.refresh()
        initial_polygon = state.polygon.copy()
        initial_radius = state.mec_radius
        self.assertLessEqual(initial_radius, q.CLEAR_RADIUS + q.GEOMETRY_EPS)
        center = state.mec_center.copy()
        new_bearing = math.degrees(math.atan2(*(source - center)[::-1]))

        responses = [
            {"accepted": True, "clear_result": "no_target_in_range"},
            {
                "accepted": True,
                "measure_result": "direction",
                "svd_deg": new_bearing,
            },
            {"accepted": True, "clear_result": "success"},
        ]
        requests = []

        def fake_post(path, payload):
            requests.append((path, payload))
            if path == "/clear" and len(requests) == 1:
                self.assertEqual(state.status, "processing")
                self.assertNotIn(state.channel, cleared)
            return responses[len(requests) - 1]

        cleared = set()
        with patch.object(q, "_post", side_effect=fake_post):
            unresolved = q.process_pending_channels(
                {state.channel: state},
                cleared,
                start_time=0.0,
                runtime_limit_s=None,
            )

        self.assertEqual([path for path, _ in requests], ["/clear", "/measure", "/clear"])
        self.assertEqual(cleared, {state.channel})
        self.assertEqual(unresolved, [])
        self.assertEqual(state.status, "cleared")
        self.assertEqual(len(state.observations), 9)
        self.assertGreater(len(state.polygon), 0)
        self.assertFalse(np.array_equal(state.polygon, initial_polygon))
        self.assertLessEqual(state.mec_radius, initial_radius + q.GEOMETRY_EPS)


class Q3Phase1BoundaryTests(unittest.TestCase):
    def test_measure_allows_outer_position_but_clear_has_strict_1820_boundary(self):
        requests = []

        def fake_post(path, payload):
            requests.append((path, payload))
            if path == "/measure":
                return {
                    "accepted": True,
                    "measure_result": "direction",
                    "svd_deg": 0.0,
                }
            return {"accepted": True, "clear_result": "success"}

        with patch.object(q, "_post", side_effect=fake_post):
            self.assertEqual(q.measure(2000.0, 0.0, 1)["measure_result"], "direction")
            self.assertTrue(q.try_clear(1, np.array([1820.0, 0.0]), "edge"))
            request_count = len(requests)
            self.assertFalse(q.try_clear(1, np.array([1820.001, 0.0]), "outside"))

        self.assertEqual(len(requests), request_count)
        self.assertEqual(requests[0][0], "/measure")
        self.assertEqual(requests[1][0], "/clear")

    def test_mec_19_999_20_and_20_001_boundaries(self):
        class CachedState:
            def __init__(self, center, radius):
                self.mec_center = center
                self.mec_radius = radius

            def refresh(self):
                return None

        for radius, should_be_guaranteed in (
            (19.999, True),
            (20.0, True),
            (20.001, False),
        ):
            center, measured_radius = q.minimum_enclosing_circle(
                np.array([[-radius, 0.0], [radius, 0.0]])
            )
            self.assertAlmostEqual(measured_radius, radius, places=12)
            result = q.guaranteed_clear_circle(CachedState(center, measured_radius))
            self.assertEqual(result is not None, should_be_guaranteed)

    def test_mec_tolerance_is_small_and_explicit(self):
        class CachedState:
            def __init__(self, radius):
                self.mec_center = np.array([0.0, 0.0])
                self.mec_radius = radius

            def refresh(self):
                return None

        self.assertIsNotNone(
            q.guaranteed_clear_circle(
                CachedState(q.CLEAR_RADIUS + 0.5 * q.GEOMETRY_EPS)
            )
        )
        self.assertIsNone(
            q.guaranteed_clear_circle(
                CachedState(q.CLEAR_RADIUS + 2.0 * q.GEOMETRY_EPS)
            )
        )

    def test_regular_heptagon_boundary_and_adjacent_voronoi_worst_points(self):
        """Experiment 8: boundary/intersection points retain coverage margin."""
        def nearest_scan_distance(point):
            return min(
                float(np.linalg.norm(np.asarray(point) - q.scan_position(index)))
                for index in range(q.SCAN_N)
            )

        self.assertGreater(q.SCAN_R, q.SCAN_R_CRITICAL)
        self.assertLess(q.worst_regular_scan_distance(), q.MIN_RECEPTION_RADIUS)
        self.assertGreater(
            q.MIN_RECEPTION_RADIUS - q.worst_regular_scan_distance(), 0.01
        )

        midpoint_distances = []
        for index in range(q.SCAN_N):
            midpoint_angle = (index + 0.5) * q.SCAN_ANGLE_DEG
            point = q.TARGET_RADIUS * q.uvec(midpoint_angle)
            midpoint_distances.append(nearest_scan_distance(point))
            self.assertLessEqual(
                midpoint_distances[-1], q.MIN_RECEPTION_RADIUS - 0.01
            )
            first = q.scan_position(index)
            second = q.scan_position((index + 1) % q.SCAN_N)
            self.assertAlmostEqual(
                np.linalg.norm(point - first),
                np.linalg.norm(point - second),
                places=9,
            )

            # Exercise both floating-point sides of the angular Voronoi boundary.
            for direction in (0.0, -1.0, 1.0):
                angle = math.radians(midpoint_angle)
                if direction:
                    angle = math.nextafter(angle, angle + direction * 1.0)
                numerical_worst = q.TARGET_RADIUS * np.array(
                    [math.cos(angle), math.sin(angle)]
                )
                self.assertLessEqual(
                    nearest_scan_distance(numerical_worst),
                    q.MIN_RECEPTION_RADIUS + 1e-9,
                )

        # The target center is a second radial endpoint in the analytic bound;
        # all seven scan points remain strictly within the 1000 m lower radius.
        self.assertAlmostEqual(nearest_scan_distance([0.0, 0.0]), q.SCAN_R, places=9)
        self.assertLessEqual(max(midpoint_distances), q.worst_regular_scan_distance() + 1e-9)

    def test_homing_fallback_can_measure_beyond_2100(self):
        state = q.ChannelState(5)
        positions = []

        def fake_measure_and_update(current, position):
            positions.append(np.asarray(position, dtype=float).copy())
            return {"measure_result": "no_signal"}

        with patch.object(q, "measure_and_update", side_effect=fake_measure_and_update):
            with patch.object(q, "iterative_clear", return_value=False):
                self.assertFalse(
                    q.homing_clear(state, np.array([2050.0, 0.0]), 0.0, 100.0)
                )

        self.assertTrue(any(np.linalg.norm(position) > 2100.0 for position in positions))


class Q3Phase1RecoveryAndNetworkTests(unittest.TestCase):
    def test_empty_intersection_drops_oldest_constraint_with_bounded_recovery(self):
        """Experiment 9: contradictory bearings recover without false clearing."""
        observations = [
            np.array([-1500.0, 0.0]),
            np.array([-1000.0, 0.0]),
        ]
        bearings = [-180.0, -150.0]
        localize = q.localize_region
        self.assertEqual(len(localize(observations, bearings)), 0)

        calls = []

        def traced_localize(points, angles):
            calls.append((len(points), tuple(angles)))
            return localize(points, angles)

        with patch.object(q, "localize_region", side_effect=traced_localize):
            recovered = q.region_of(observations, bearings)

        self.assertEqual([size for size, _ in calls], [2, 1])
        self.assertGreater(len(recovered), 0)
        self.assertTrue(np.all(np.isfinite(recovered)))

        consistent_state = q.ChannelState(
            11,
            observations=[np.array([0.0, 0.0])],
            bearings=[0.0],
        )
        consistent_state.refresh()
        self.assertEqual(consistent_state.recovery_dropped_constraints, 0)
        consistent_state.refresh()
        self.assertEqual(consistent_state.recovery_dropped_constraints, 0)

        state = q.ChannelState(
            12,
            observations=observations,
            bearings=bearings,
        )
        state.refresh()
        self.assertGreater(len(state.polygon), 0)
        self.assertEqual(state.recovery_dropped_constraints, 1)
        state.refresh()
        self.assertEqual(state.recovery_dropped_constraints, 1)
        state.observations.append(np.array([-1000.0, 0.0]))
        state.bearings.append(-150.0)
        state.refresh()
        self.assertEqual(state.recovery_dropped_constraints, 2)
        self.assertIsNone(q.guaranteed_clear_circle(state))
        cleared = set()
        attempts = []

        def bounded_failure(current, mode):
            attempts.append(mode)
            self.assertIsNone(q.guaranteed_clear_circle(current))
            current.last_failure_reason = "empty_region_recovered_but_unresolved"
            return False

        unresolved = q.process_pending_channels(
            {state.channel: state},
            cleared,
            start_time=0.0,
            processor=bounded_failure,
            runtime_limit_s=None,
        )
        self.assertEqual(len(attempts), q.MAX_CHANNEL_ATTEMPTS)
        self.assertEqual(cleared, set())
        self.assertEqual([item.channel for item in unresolved], [state.channel])
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.retry_count, q.MAX_CHANNEL_ATTEMPTS)

    def test_post_retry_reuses_request_id_and_updates_position_only_when_accepted(self):
        """Experiment 10: transport retry is idempotent and position-safe."""
        old_position = q.CURRENT_POS.copy()
        q.CURRENT_POS = np.array([-321.0, 654.0])
        calls = []
        responses = [
            None,
            {
                "accepted": True,
                "measure_result": "direction",
                "svd_deg": 12.0,
                "virtual_time_s": 7.0,
            },
        ]

        def fake_post(path, payload):
            calls.append((path, payload))
            return responses[len(calls) - 1]

        try:
            with patch.object(q, "post", side_effect=fake_post):
                with patch.object(q.time, "sleep"):
                    result = q.measure(123.0, 456.0, 8)
            self.assertEqual(result["measure_result"], "direction")
            self.assertEqual(len(calls), 2)
            self.assertIs(calls[0][1], calls[1][1])
            self.assertEqual(
                calls[0][1]["request_id"], calls[1][1]["request_id"]
            )
            np.testing.assert_allclose(q.CURRENT_POS, [123.0, 456.0])
        finally:
            q.CURRENT_POS = old_position

    def test_none_and_rejected_measure_clear_do_not_move_current_position(self):
        """Experiment 10: API failures cannot advance CURRENT_POS."""
        old_position = q.CURRENT_POS.copy()
        q.CURRENT_POS = np.array([11.0, -22.0])
        try:
            with patch.object(q, "_post", return_value=None):
                self.assertIsNone(q.measure(100.0, 200.0, 9))
                self.assertFalse(q.try_clear(9, np.array([100.0, 200.0]), "none"))
            np.testing.assert_allclose(q.CURRENT_POS, [11.0, -22.0])

            rejected = {"accepted": False, "measure_result": "direction"}
            with patch.object(q, "_post", return_value=rejected):
                self.assertIsNone(q.measure(300.0, 400.0, 9))
                self.assertFalse(
                    q.try_clear(9, np.array([300.0, 400.0]), "rejected")
                )
            np.testing.assert_allclose(q.CURRENT_POS, [11.0, -22.0])
        finally:
            q.CURRENT_POS = old_position

    def test_bounded_network_failure_requeues_then_marks_channel_unresolved(self):
        """Experiment 10: repeated transport failure reaches a finite fallback bound."""
        old_position = q.CURRENT_POS.copy()
        q.CURRENT_POS = np.array([77.0, 88.0])
        state = q.ChannelState(14, last_position=np.array([100.0, 100.0]))
        cleared = set()
        try:
            with patch.object(q, "post", return_value=None) as post_mock:
                with patch.object(q.time, "sleep"):
                    unresolved = q.process_pending_channels(
                        {state.channel: state},
                        cleared,
                        start_time=0.0,
                        processor=q.process_channel,
                        runtime_limit_s=None,
                    )
            self.assertEqual(
                post_mock.call_count,
                q.MAX_CHANNEL_ATTEMPTS * (2 + 1),
            )
            self.assertEqual(cleared, set())
            self.assertEqual([item.channel for item in unresolved], [14])
            self.assertEqual(state.status, "failed")
            self.assertEqual(state.retry_count, q.MAX_CHANNEL_ATTEMPTS)
            self.assertIn("network_or_rejected", state.last_failure_reason)
            np.testing.assert_allclose(q.CURRENT_POS, [77.0, 88.0])
        finally:
            q.CURRENT_POS = old_position


if __name__ == "__main__":
    unittest.main()
