"""Regression tests for the local API-compatible Q3 test platform."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.local_test_platform.server import (
    InjectedTransportFailure,
    PlatformState,
    RequestProblem,
    VERSION_SCRIPTS,
)


def base(request_id: str) -> dict:
    return {
        "arena_id": "default",
        "robot_id": "test-robot",
        "request_id": request_id,
    }


class PlatformStateTests(unittest.TestCase):
    def make_state(self, **extra) -> PlatformState:
        config = {
            "seed": 17,
            "sources": [
                {
                    "channel": 3,
                    "position": [100.0, 0.0],
                    "reception_radius": 1200.0,
                }
            ],
            "error_mode": "zero",
        }
        config.update(extra)
        state = PlatformState(seed=17, source_count=10)
        state.configure(config)
        return state

    def test_official_flow_and_virtual_time(self) -> None:
        state = self.make_state()
        enter = state.handle_official("/enter", base("enter-1"))
        self.assertTrue(enter["accepted"])
        self.assertEqual(1200, enter["remaining_real_duration_s"])

        measure_payload = {
            **base("measure-1"),
            "position": {"x": 0, "y": 0},
            "channel": 3,
        }
        measured = state.handle_official("/measure", measure_payload)
        self.assertEqual("direction", measured["measure_result"])
        self.assertEqual(6.0, measured["virtual_time_s"])
        self.assertEqual(0.0, measured["svd_deg"])

        clear_payload = {
            **base("clear-1"),
            "position": {"x": 100, "y": 0},
            "channel": 3,
        }
        cleared = state.handle_official("/clear", clear_payload)
        self.assertEqual("success", cleared["clear_result"])
        self.assertEqual(31.0, cleared["virtual_time_s"])

        exited = state.handle_official("/exit", base("exit-1"))
        self.assertEqual("user_exit", exited["exit_reason"])
        self.assertEqual(1, state.snapshot()["summary"]["cleared_count"])

    def test_idempotent_retry_does_not_repeat_action(self) -> None:
        state = self.make_state()
        state.handle_official("/enter", base("enter-1"))
        payload = {
            **base("measure-1"),
            "position": {"x": 20, "y": 0},
            "channel": 3,
        }
        first = state.handle_official("/measure", payload)
        second = state.handle_official("/measure", payload)
        self.assertEqual(first, second)
        self.assertEqual(1, len(state.snapshot()["actions"]))
        self.assertEqual(3, len(state.snapshot()["transactions"]))

    def test_request_id_reuse_with_different_action_is_rejected(self) -> None:
        state = self.make_state()
        state.handle_official("/enter", base("enter-1"))
        original = {
            **base("same-id"),
            "position": {"x": 0, "y": 0},
            "channel": 3,
        }
        changed = {
            **base("same-id"),
            "position": {"x": 1, "y": 0},
            "channel": 3,
        }
        state.handle_official("/measure", original)
        response = state.handle_official("/measure", changed)
        self.assertFalse(response["accepted"])
        self.assertEqual("request_id_reuse", response["error"])
        self.assertEqual(1, len(state.snapshot()["actions"]))

    def test_invalid_action_schema_is_a_bad_request(self) -> None:
        state = self.make_state()
        payload = {
            **base("bad-1"),
            "position": {"x": 0, "y": 0},
            "channel": 3,
            "extra": True,
        }
        with self.assertRaises(RequestProblem):
            state.handle_official("/measure", payload)

    def test_transport_failure_consumes_no_action_then_allows_retry(self) -> None:
        state = self.make_state(fail_transport_every=2)
        state.handle_official("/enter", base("enter-1"))
        first = {
            **base("measure-1"),
            "position": {"x": 0, "y": 0},
            "channel": 3,
        }
        second = {
            **base("measure-2"),
            "position": {"x": 10, "y": 0},
            "channel": 3,
        }
        state.handle_official("/measure", first)
        with self.assertRaises(InjectedTransportFailure):
            state.handle_official("/measure", second)
        self.assertEqual(1, len(state.snapshot()["actions"]))
        state.handle_official("/measure", second)
        self.assertEqual(2, len(state.snapshot()["actions"]))
        self.assertEqual(1, state.snapshot()["summary"]["transport_retries"])

    def test_seeded_scenario_is_repeatable(self) -> None:
        state = PlatformState(seed=42, source_count=12)
        first = state.snapshot()["scenario"]["sources"]
        state.configure({"seed": 42, "source_count": 12, "error_mode": "hash"})
        self.assertEqual(first, state.snapshot()["scenario"]["sources"])

    def test_batch_mode_uses_twelve_seeds_for_only_selected_version(self) -> None:
        state = PlatformState(seed=42, source_count=10)
        with patch("threading.Thread.start"):
            snapshot = state.start_batch(
                {
                    "version": "phase1",
                    "seed": 900,
                    "source_count": 10,
                    "error_mode": "hash",
                }
            )
        self.assertEqual("phase1", snapshot["run"]["version"])
        self.assertEqual("batch", snapshot["run"]["mode"])
        self.assertEqual(list(range(900, 912)), snapshot["batch"]["seeds"])
        self.assertEqual(12, snapshot["batch"]["total"])

    def test_active_platform_option_uses_active_v2_entry(self) -> None:
        self.assertEqual("Q3_active_v2.py", VERSION_SCRIPTS["active"].name)

    def test_batch_summary_averages_virtual_time(self) -> None:
        state = PlatformState(seed=42, source_count=10)
        state.batch = {
            "total": 12,
            "completed": 2,
            "current_index": 2,
            "current_seed": None,
            "seeds": list(range(12)),
            "results": [
                {
                    "seed": 0,
                    "returncode": 0,
                    "successful": True,
                    "total_virtual_time_s": 100.0,
                    "total_move_m": 500.0,
                    "measure_count": 10,
                    "clear_attempt_count": 2,
                },
                {
                    "seed": 1,
                    "returncode": 0,
                    "successful": False,
                    "total_virtual_time_s": 140.0,
                    "total_move_m": 700.0,
                    "measure_count": 14,
                    "clear_attempt_count": 4,
                },
            ],
        }
        summary = state._batch_summary()
        self.assertEqual(120.0, summary["average_virtual_time_s"])
        self.assertEqual(600.0, summary["average_move_m"])
        self.assertEqual(15.0, summary["average_action_count"])
        self.assertEqual(1, summary["successful"])


if __name__ == "__main__":
    unittest.main()
