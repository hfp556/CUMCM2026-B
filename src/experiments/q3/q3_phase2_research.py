"""Offline numerical studies for Phase 2 mathematics A/B.

The script intentionally does not import or run the formal Q3 controller.  It
uses :mod:`q3_active_localization` to produce:

* movement-budget -> worst posterior MEC-radius/diameter Pareto rows;
* a receding-horizon single-target trade-off study;
* adversarial thin/near-collinear and target-boundary/outside cases;
* fixed-seed random scenes and candidate-grid/circle-resolution sensitivity.

The numerical posterior enumerates sampled possible sources and errors
``{-1,0,+1} deg``.  The continuous quantities are suprema over all
``G in Omega`` and ``e in [-1,1]``; finite samples and polygonal circles are
explicit approximations and are not presented as strict global optima.

Example::

    python src/experiments/q3/q3_phase2_research.py --output src/experiments/q3/q3_phase2_results.json

Use ``--quick`` for a small smoke study.  The default study is still designed
to finish on a laptop in a practical amount of time and uses a fixed seed for
reproducibility.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

try:
    from src.new import q3_active_localization as active
except ImportError:  # Direct execution with ``src/new`` on sys.path.
    import q3_active_localization as active


DEFAULT_SEED = 20260913


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _evaluation_row(evaluation):
    if evaluation is None:
        return None
    return {
        "point": evaluation.point.tolist(),
        "point_radius_m": float(np.linalg.norm(evaluation.point)),
        "movement_distance_m": float(evaluation.movement_distance_m),
        "max_source_distance_m": float(evaluation.max_source_distance_m),
        "q_rho_m": float(evaluation.q_rho_m),
        "q_diameter_m": float(evaluation.q_diameter_m),
        "worst_source": None if evaluation.worst_source is None else evaluation.worst_source.tolist(),
        "worst_error_deg": evaluation.worst_error_deg,
        "case_count": int(evaluation.case_count),
        "nonempty_case_count": int(evaluation.nonempty_case_count),
    }


def _pareto_rows(result):
    rows = []
    for choice in result.choices:
        row = {"budget_m": float(choice.budget_m), "choice": _evaluation_row(choice.evaluation)}
        rows.append(row)
    return rows


def _frontier_rows(result):
    return [_evaluation_row(evaluation) for evaluation in result.frontier]


def _run_pareto(
    omega,
    start,
    budgets,
    *,
    spacing_m,
    circle_segments,
    source_sample_count,
    max_candidates,
    target_radius=None,
    candidate_points=None,
    source_samples=None,
):
    started = time.perf_counter()
    result = active.solve_pareto(
        omega,
        start,
        budgets_m=budgets,
        candidate_points=candidate_points,
        source_samples=source_samples,
        source_sample_count=source_sample_count,
        error_samples=(-1.0, 0.0, 1.0),
        spacing_m=spacing_m,
        circle_segments=circle_segments,
        target_radius=target_radius,
        max_candidates=max_candidates,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    outside = [
        evaluation
        for evaluation in result.evaluations
        if float(np.linalg.norm(evaluation.point)) > active.TARGET_RADIUS + 1e-7
    ]
    inside = [
        evaluation
        for evaluation in result.evaluations
        if float(np.linalg.norm(evaluation.point)) <= active.TARGET_RADIUS + 1e-7
    ]
    key = lambda item: (item.q_rho_m, item.q_diameter_m, item.movement_distance_m)
    return {
        "sampling": dict(result.sampling),
        "elapsed_ms": float(elapsed_ms),
        "candidate_count": int(len(result.candidate_points)),
        "outside_candidate_count": int(len(outside)),
        "best_outside_candidate": _evaluation_row(min(outside, key=key) if outside else None),
        "best_inside_candidate": _evaluation_row(min(inside, key=key) if inside else None),
        "pareto_by_budget": _pareto_rows(result),
        "frontier": _frontier_rows(result),
    }


def _simulate(
    omega,
    source,
    start,
    budgets,
    *,
    error_seed,
    spacing_m,
    circle_segments,
    source_sample_count,
    max_steps,
    target_radius=None,
    measurement_error_deg=None,
):
    summaries = []
    runs = []
    for budget in budgets:
        result = active.simulate_single_target(
            omega,
            source,
            start,
            budget,
            receive_radius_m=active.MIN_RECEPTION_RADIUS,
            measurement_error_deg=measurement_error_deg,
            # Keep the same (source, position) error field for every budget;
            # otherwise a strategy comparison would mix movement effects with
            # unrelated random measurement errors.
            error_seed=error_seed,
            source_sample_count=source_sample_count,
            error_samples=(-1.0, 0.0, 1.0),
            spacing_m=spacing_m,
            circle_segments=circle_segments,
            max_steps=max_steps,
            target_radius=target_radius,
        )
        runs.append(
            {
                "budget_m": float(budget),
                "success": bool(result.success),
                "reason": result.reason,
                "total_time_s": float(result.total_time_s),
                "measurement_count": int(result.measurement_count),
                "clear_attempts": int(result.clear_attempts),
                "movement_distance_m": float(result.movement_distance_m),
                "measurement_move_distance_m": float(result.measurement_move_distance_m),
                "final_clear_move_distance_m": float(result.final_clear_move_distance_m),
                "outside_target_measurements": int(result.outside_target_measurements),
            }
        )
        summaries.append(active.summarize_simulations([result], budget_m=budget))
    return {"runs": runs, "summaries": summaries}


def _random_tradeoff(
    scenarios,
    budgets,
    *,
    spacing_m,
    circle_segments,
    source_sample_count,
    max_steps,
):
    records = []
    for index, scenario in enumerate(scenarios):
        simulation = _simulate(
            scenario["omega"],
            scenario["source"],
            scenario["start"],
            budgets,
            error_seed=1000 + index * 37,
            spacing_m=spacing_m,
            circle_segments=circle_segments,
            source_sample_count=source_sample_count,
            max_steps=max_steps,
            measurement_error_deg=None,
        )
        records.append(
            {
                "name": scenario["name"],
                "omega_vertices": int(len(scenario["omega"])),
                "source": scenario["source"].tolist(),
                "start": scenario["start"].tolist(),
                "simulation": simulation,
            }
        )

    by_budget = []
    for budget in budgets:
        bucket = []
        for record in records:
            bucket.extend(
                run
                for run in record["simulation"]["runs"]
                if abs(run["budget_m"] - budget) <= 1e-9
            )
        # Rehydrate the small result shape needed by summarize_simulations;
        # the full per-run values remain in ``records`` above.
        rehydrated = []
        for run in bucket:
            rehydrated.append(
                active.SimulationResult(
                    budget_m=run["budget_m"],
                    success=run["success"],
                    total_time_s=run["total_time_s"],
                    measurement_count=run["measurement_count"],
                    clear_attempts=run["clear_attempts"],
                    clear_failures=0,
                    movement_distance_m=run["movement_distance_m"],
                    measurement_move_distance_m=run["measurement_move_distance_m"],
                    final_clear_move_distance_m=run["final_clear_move_distance_m"],
                    measurement_time_s=run["measurement_count"] * active.MEASURE_TIME_S,
                    final_clear_time_s=(
                        run["final_clear_move_distance_m"] / active.SPEED_MPS + active.CLEAR_TIME_S
                        if run["success"]
                        else 0.0
                    ),
                    steps=0,
                    reason=run["reason"],
                    outside_target_measurements=run["outside_target_measurements"],
                )
            )
        by_budget.append(active.summarize_simulations(rehydrated, budget_m=budget))
    return {"scenarios": records, "aggregate_by_budget": by_budget}


def _data_driven_recommendation(summaries, *, success_threshold=0.95):
    """Choose a practical budget from measured time, never from a lambda."""

    eligible = [
        summary
        for summary in summaries
        if summary["success_rate"] >= success_threshold
        and summary["time_s"]["average"] is not None
    ]
    if eligible:
        selected = min(
            eligible,
            key=lambda summary: (summary["time_s"]["average"], summary["budget_m"]),
        )
        rule = f"minimum measured average time among success_rate >= {success_threshold:.2f}"
    else:
        finite = [summary for summary in summaries if summary["time_s"]["average"] is not None]
        if not finite:
            return {"selected_budget_m": None, "rule": "no successful run"}
        selected = max(
            finite,
            key=lambda summary: (summary["success_rate"], -summary["time_s"]["average"]),
        )
        rule = "no budget met success threshold; maximize success_rate then minimize measured average time"
    return {
        "selected_budget_m": float(selected["budget_m"]),
        "rule": rule,
        "success_rate": float(selected["success_rate"]),
        "average_time_s": selected["time_s"]["average"],
        "p95_time_s": selected["time_s"]["p95"],
        "worst_time_s": selected["time_s"]["worst"],
    }


def run_research(seed: int = DEFAULT_SEED, quick: bool = False):
    """Run all Phase 2 A/B studies and return JSON-serializable data."""

    if quick:
        budgets = (200.0, 400.0)
        spacing_m = 300.0
        circle_segments = 16
        source_sample_count = 2
        max_candidates = 20
        max_steps = 3
        random_count = 2
    else:
        budgets = (100.0, 200.0, 300.0, 400.0)
        spacing_m = 220.0
        circle_segments = 24
        source_sample_count = 3
        max_candidates = 40
        max_steps = 4
        random_count = 4

    slender_omega, slender_source, slender_start = active.make_slender_region()
    boundary_omega, boundary_source, boundary_start = active.make_boundary_region()

    # Use one candidate pool and one source sample set for both sides of the
    # circle-boundary ablation.  The target-only search is then a literal
    # subset of the unrestricted search, so any quality difference cannot be
    # caused by separate grid generation or candidate down-sampling.
    boundary_candidate_pool = active.generate_robust_candidates(
        boundary_omega,
        boundary_start,
        max(budgets),
        spacing_m=spacing_m,
        circle_segments=circle_segments,
        max_candidates=max_candidates,
    )
    boundary_source_samples = active.sample_region_sources(
        boundary_omega,
        source_sample_count,
        seed=seed,
    )

    slender_pareto = _run_pareto(
        slender_omega,
        slender_start,
        budgets,
        spacing_m=spacing_m,
        circle_segments=circle_segments,
        source_sample_count=source_sample_count,
        max_candidates=max_candidates,
    )
    slender_simulation = _simulate(
        slender_omega,
        slender_source,
        slender_start,
        budgets,
        error_seed=seed,
        spacing_m=spacing_m,
        circle_segments=circle_segments,
        source_sample_count=source_sample_count,
        max_steps=max_steps,
        measurement_error_deg=0.0,
    )

    boundary_unrestricted = _run_pareto(
        boundary_omega,
        boundary_start,
        budgets,
        spacing_m=spacing_m,
        circle_segments=circle_segments,
        source_sample_count=source_sample_count,
        max_candidates=max_candidates,
        target_radius=None,
        candidate_points=boundary_candidate_pool,
        source_samples=boundary_source_samples,
    )
    boundary_inside = _run_pareto(
        boundary_omega,
        boundary_start,
        budgets,
        spacing_m=spacing_m,
        circle_segments=circle_segments,
        source_sample_count=source_sample_count,
        max_candidates=max_candidates,
        target_radius=active.TARGET_RADIUS,
        candidate_points=boundary_candidate_pool,
        source_samples=boundary_source_samples,
    )
    boundary_sim_unrestricted = _simulate(
        boundary_omega,
        boundary_source,
        boundary_start,
        budgets,
        error_seed=seed + 10,
        spacing_m=spacing_m,
        circle_segments=circle_segments,
        source_sample_count=source_sample_count,
        max_steps=max_steps,
        target_radius=None,
        measurement_error_deg=0.0,
    )
    boundary_sim_inside = _simulate(
        boundary_omega,
        boundary_source,
        boundary_start,
        budgets,
        error_seed=seed + 20,
        spacing_m=spacing_m,
        circle_segments=circle_segments,
        source_sample_count=source_sample_count,
        max_steps=max_steps,
        target_radius=active.TARGET_RADIUS,
        measurement_error_deg=0.0,
    )

    random_scenarios = active.make_random_regions(seed=seed, count=random_count)
    random_tradeoff = _random_tradeoff(
        random_scenarios,
        budgets,
        spacing_m=spacing_m,
        circle_segments=circle_segments,
        source_sample_count=source_sample_count,
        max_steps=max_steps,
    )

    sensitivity = []
    # The same thin prior is evaluated under three grid and circle resolutions;
    # changes in Q values quantify numerical, not physical, sensitivity.
    for spacing in (75.0, 150.0, 250.0):
        for segments in (24, 48, 72):
            started = time.perf_counter()
            result = active.solve_pareto(
                slender_omega,
                slender_start,
                budgets_m=(300.0,),
                source_sample_count=min(source_sample_count, 5),
                error_samples=(-1.0, 0.0, 1.0),
                spacing_m=spacing,
                circle_segments=segments,
                # Keep enough candidates that changing circle resolution does
                # not also change the truncation subset; this isolates the
                # intended numerical-sensitivity comparison.
                max_candidates=max(80, max_candidates),
            )
            selected = result.choice_for_budget(300.0)
            sensitivity.append(
                {
                    "spacing_m": spacing,
                    "circle_segments": segments,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "candidate_count": int(len(result.candidate_points)),
                    "choice": _evaluation_row(selected),
                }
            )

    return {
        "metadata": {
            "seed": int(seed),
            "quick": bool(quick),
            "budgets_m": [float(value) for value in budgets],
            "speed_mps": active.SPEED_MPS,
            "measure_service_s": active.MEASURE_TIME_S,
            "clear_service_s": active.CLEAR_TIME_S,
            "timing_definition": "sum(move/5 + 5 per measure) + final_move/5 + 5 successful clear",
            "posterior_definition": "Omega_after = Omega intersect B(S,1500) intersect W(S,bearing(G-S)+e,+/-1deg)",
            "robust_candidate_definition": "max_{G in Omega} ||S-G|| <= 1000",
            "continuous_optimum_claimed": False,
            "discrete_approximation": {
                "source_sampling": "vertices/edge midpoints/centroid plus fixed-seed convex combinations",
                "error_samples_deg": [-1.0, 0.0, 1.0],
                "circle_clipping": "inscribed regular polygon",
                "boundary_ablation": "shared candidate pool and shared source samples; target-only is a filtered subset",
            },
        },
        "slender_near_collinear": {
            "omega_vertices": int(len(slender_omega)),
            "source": slender_source.tolist(),
            "start": slender_start.tolist(),
            "pareto": slender_pareto,
            "simulation": slender_simulation,
            "recommendation": _data_driven_recommendation(slender_simulation["summaries"]),
        },
        "boundary_outside_ablation": {
            "omega_vertices": int(len(boundary_omega)),
            "source": boundary_source.tolist(),
            "start": boundary_start.tolist(),
            "allow_outside_target": {
                "pareto": boundary_unrestricted,
                "simulation": boundary_sim_unrestricted,
                "recommendation": _data_driven_recommendation(boundary_sim_unrestricted["summaries"]),
            },
            "restrict_to_target_disk": {
                "pareto": boundary_inside,
                "simulation": boundary_sim_inside,
                "recommendation": _data_driven_recommendation(boundary_sim_inside["summaries"]),
            },
        },
        "random_scenes": {
            **random_tradeoff,
            "recommendation": _data_driven_recommendation(random_tradeoff["aggregate_by_budget"]),
        },
        "numerical_sensitivity": sensitivity,
    }


def _print_summary(results):
    metadata = results["metadata"]
    print(f"Phase 2 research seed={metadata['seed']} quick={metadata['quick']}")
    print("scenario/budget success_rate average_s p95_s worst_s avg_measures avg_move_m")
    sections = [
        ("slender", results["slender_near_collinear"]["simulation"]["summaries"]),
        ("boundary unrestricted", results["boundary_outside_ablation"]["allow_outside_target"]["simulation"]["summaries"]),
        ("boundary target-only", results["boundary_outside_ablation"]["restrict_to_target_disk"]["simulation"]["summaries"]),
        ("random aggregate", results["random_scenes"]["aggregate_by_budget"]),
    ]
    for name, summaries in sections:
        for summary in summaries:
            timing = summary["time_s"]
            measures = summary["measurements"]
            movement = summary["movement_distance_m"]
            print(
                f"{name}/{summary['budget_m']:g} "
                f"{summary['success_rate']:.3f} "
                f"{timing['average'] if timing['average'] is not None else float('nan'):.2f} "
                f"{timing['p95'] if timing['p95'] is not None else float('nan'):.2f} "
                f"{timing['worst'] if timing['worst'] is not None else float('nan'):.2f} "
                f"{measures['average'] if measures['average'] is not None else float('nan'):.2f} "
                f"{movement['average'] if movement['average'] is not None else float('nan'):.2f}"
            )
    boundary_choice = results["boundary_outside_ablation"]["allow_outside_target"]["pareto"]["pareto_by_budget"][-1]["choice"]
    if boundary_choice is not None:
        print(
            "boundary unrestricted largest-budget candidate radius="
            f"{boundary_choice['point_radius_m']:.1f} m"
        )
    print("Numerical sensitivity rows:", len(results["numerical_sensitivity"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/experiments/q3/q3_phase2_results.json"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--quick", action="store_true", help="run a smaller smoke study")
    args = parser.parse_args(argv)

    results = run_research(seed=args.seed, quick=args.quick)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_value(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_summary(results)
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
