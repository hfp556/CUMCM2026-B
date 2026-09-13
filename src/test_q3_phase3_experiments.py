"""Small tests for the Phase 3 shared offline harness."""

from pathlib import Path
import sys
import unittest

import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q3_phase3_experiments import (  # noqa: E402
    FairCandidatePoolPlanner,
    OfflineSimulator,
    Scenario,
    Source,
    aggregate_runs,
    adversarial_cases,
    boundary_cases,
    make_scenario,
    run_experiment6,
    run_experiment7,
    slender_cases,
    run_version,
)


class OfflineSimulatorTests(unittest.TestCase):
    def test_exact_time_and_request_idempotency(self):
        scenario = Scenario(7, (Source(1, (100.0, 0.0), 1000.0),))
        sim = OfflineSimulator(scenario)
        enter = {"request_id": "enter-1"}
        self.assertTrue(sim.post("/enter", enter)["accepted"])

        measure = {
            "request_id": "measure-1",
            "position": {"x": 0.0, "y": 0.0},
            "channel": 1,
        }
        first = sim.post("/measure", measure)
        self.assertEqual(first["measure_result"], "direction")
        self.assertAlmostEqual(sim.virtual_time_s, 5.0)
        duplicate = sim.post("/measure", measure)
        self.assertEqual(duplicate, first)
        self.assertAlmostEqual(sim.virtual_time_s, 5.0)

        clear = {
            "request_id": "clear-1",
            "position": {"x": 100.0, "y": 0.0},
            "channel": 1,
        }
        result = sim.post("/clear", clear)
        self.assertEqual(result["clear_result"], "success")
        self.assertAlmostEqual(sim.virtual_time_s, 5.0 + 20.0 + 5.0)
        self.assertEqual(sim.position.tolist(), [100.0, 0.0])

    def test_error_is_fixed_for_same_rounded_position(self):
        source = Source(1, (1000.0, 0.0), 1000.0)
        sim = OfflineSimulator(Scenario(11, (source,)))
        self.assertAlmostEqual(
            sim.direction_error(source, (12.341, -8.761)),
            sim.direction_error(source, (12.344, -8.764)),
        )
        self.assertNotEqual(
            sim.direction_error(source, (12.341, -8.761)),
            sim.direction_error(source, (12.35, -8.761)),
        )

    def test_transport_failure_is_retried_without_duplicate_action(self):
        source = Source(1, (0.0, 0.0), 1000.0)
        sim = OfflineSimulator(Scenario(3, (source,)), fail_transport_every=1)
        sim.post("/enter", {"request_id": "enter"})
        payload = {
            "request_id": "m1",
            "position": {"x": 0.0, "y": 0.0},
            "channel": 1,
        }
        self.assertIsNone(sim.post("/measure", payload))
        response = sim.post("/measure", payload)
        self.assertEqual(response["measure_result"], "near")
        self.assertEqual(len(sim.actions), 1)
        self.assertEqual(sim.transport_failures, 1)
        self.assertEqual(sim.transport_retries, 1)


class ScenarioAndRunnerTests(unittest.TestCase):
    def test_random_scene_is_reproducible_and_in_target_disk(self):
        first = make_scenario(1234, n=10)
        second = make_scenario(1234, n=10)
        self.assertEqual(first, second)
        self.assertEqual(len(first.sources), 10)
        self.assertEqual(len({source.channel for source in first.sources}), 10)
        for source in first.sources:
            self.assertLessEqual(np.linalg.norm(source.point()), 1800.0 + 1e-9)
            self.assertGreaterEqual(source.reception_radius, 1000.0)
            self.assertLessEqual(source.reception_radius, 1500.0)

    def test_aggregate_records_expose_game_omission_and_percentiles(self):
        complete = {
            "source_count": 2,
            "discovered_count": 2,
            "cleared_count": 2,
            "any_missed_channel": False,
            "total_virtual_time_s": 20.0,
            "source_time_s": {"1": 8.0, "2": 12.0},
            "total_move_m": 10.0,
            "measure_count": 2,
            "clear_attempt_count": 2,
            "clear_failure_count": 0,
            "transport_failures": 0,
            "transport_retries": 0,
            "outside_measure_count": 0,
            "completed_without_exception": True,
        }
        incomplete = dict(complete)
        incomplete.update(
            source_count=2,
            discovered_count=1,
            cleared_count=1,
            any_missed_channel=True,
            total_virtual_time_s=30.0,
            source_time_s={"1": 15.0},
        )
        summary = aggregate_runs([complete, incomplete])
        self.assertEqual(summary["game_count"], 2)
        self.assertEqual(summary["source_count"], 4)
        self.assertAlmostEqual(summary["discovery_rate"], 0.75)
        self.assertAlmostEqual(summary["clear_rate"], 0.75)
        self.assertEqual(summary["games_with_any_missed_channel"], 1)
        self.assertEqual(summary["per_source_time_s"]["count"], 3)
        self.assertEqual(summary["successful_game_count"], 1)

    def test_fixed_experiment_case_sets_and_fairness_metadata(self):
        self.assertGreaterEqual(len(slender_cases()), 4)
        boundary = boundary_cases()
        self.assertGreaterEqual(len(boundary), 3)
        self.assertEqual(len({case["scenario"].seed for case in boundary}), len(boundary))
        adversarial = adversarial_cases()
        self.assertEqual(len(adversarial), 12)
        self.assertEqual(
            {case["case_id"] for case in adversarial},
            {
                "01_boundary_source",
                "02_minimum_reception_radius",
                "03_heptagon_worst_scan_point",
                "04_nearly_parallel_bearings",
                "05_slender_region",
                "06_mec_radius_near_twenty",
                "07_extreme_bearing_errors",
                "08_forced_first_process_failure",
                "09_forced_first_clear_failure",
                "10_intermittent_network_failure",
                "11_sources_one_side",
                "12_sources_distributed_boundary",
            },
        )
        allow = FairCandidatePoolPlanner(False)
        restricted = FairCandidatePoolPlanner(True)
        self.assertIn("unrestricted pool", allow.summary()["fairness"])
        self.assertFalse(allow.target_only)
        self.assertTrue(restricted.target_only)

    def test_controlled_slender_experiment_excludes_discovery_scan(self):
        result = run_experiment6(slender_cases()[:1])
        case = result["cases"][0]
        self.assertEqual(result["active_config_label"], "production_default")
        for version in ("fast2", "active"):
            run = case["raw_by_version"][version]
            self.assertTrue(run["completed_without_exception"])
            self.assertGreater(run["initial_region"]["slender_aspect_ratio"], 5.0)
            self.assertLess(run["measure_count"], 30)

    def test_boundary_outside_evidence_counts_only_active_selected_points(self):
        result = run_experiment7(boundary_cases()[:1])
        case = result["cases"][0]
        selected = case["raw_pair"]["allow_outside"]["planner_summary"][
            "selected_outside_count"
        ]
        self.assertEqual(case["active_selected_outside_count"], selected)
        self.assertEqual(case["outside_measure_used_in_allow"], selected > 0)

    def test_one_real_main_smoke_for_all_versions(self):
        # The source is deliberately central enough that the seven/eight-point
        # discovery routes receive it under the minimum reception radius.
        scenario = Scenario(
            99,
            (Source(1, (100.0, 80.0), 1000.0),),
        )
        for version in ("fast2", "phase1", "active"):
            with self.subTest(version=version):
                result = run_version(version, scenario)
                self.assertTrue(result["completed_without_exception"])
                self.assertTrue(result["stdout_captured"])
                self.assertGreater(result["stdout_line_count"], 0)
                self.assertGreaterEqual(result["discovered_count"], 1)
                self.assertGreaterEqual(result["clear_attempt_count"], 1)
                self.assertLess(result["total_virtual_time_s"], 360000.0)


if __name__ == "__main__":
    unittest.main()
