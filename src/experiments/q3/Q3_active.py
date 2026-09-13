"""Q3 Phase 4 controller with bounded active localization.

This is an independent entry point built on the verified Phase 1 controller
and the offline Phase 2 localization kernel.  ``Q3_phase1.py`` and the
``Q3_fast2.py`` baseline are left untouched.  The active planner is a
discrete numerical heuristic: it evaluates a shared candidate pool with the
100, 200, 300 and 400 m budget ladder, but it does not claim a continuous
optimum or a theoretical guarantee.

The controller keeps Phase 1's channel state machine and bounded retry
semantics.  A robust candidate is measured, every returned direction goes
through :func:`add_observation`, and ``near`` is immediately offered to the
existing clear path.  Planner failures, no-candidate results and active-loop
bounds use the retained homing/bracket fallback.
"""

from dataclasses import dataclass
import inspect
import math
import sys
import time

import numpy as np

try:  # ``import src.experiments.q3.Q3_active``
    from . import Q3_phase1 as _phase1
    from src.new.q3_active_localization import (
        BudgetChoice,
        CandidateEvaluation,
        ParetoResult,
        bearing_deg,
        candidate_points_for_budget,
        evaluate_candidate,
        generate_robust_candidates,
        is_robust_candidate,
        max_distance_to_region,
        pareto_frontier,
        posterior_region,
        robust_candidate_domain,
        sample_region_sources,
        solve_pareto,
    )
except ImportError:  # Flat imports used by the offline experiment harness.
    import Q3_phase1 as _phase1
    from q3_active_localization import (
        BudgetChoice,
        CandidateEvaluation,
        ParetoResult,
        bearing_deg,
        candidate_points_for_budget,
        evaluate_candidate,
        generate_robust_candidates,
        is_robust_candidate,
        max_distance_to_region,
        pareto_frontier,
        posterior_region,
        robust_candidate_domain,
        sample_region_sources,
        solve_pareto,
    )


# Re-export the Phase 1 public controller surface.  Explicit wrappers for
# network and observation functions below keep module-level monkeypatching
# useful for small offline tests and keep CURRENT_POS synchronized with the
# imported Phase 1 kernel.
TARGET_RADIUS = _phase1.TARGET_RADIUS
CLEAR_RADIUS = _phase1.CLEAR_RADIUS
CLEAR_MAX_DISTANCE = _phase1.CLEAR_MAX_DISTANCE
MIN_RECEPTION_RADIUS = _phase1.MIN_RECEPTION_RADIUS
MAX_RECEPTION_RADIUS = _phase1.MAX_RECEPTION_RADIUS
ANGLE_ERROR_DEG = _phase1.ANGLE_ERROR_DEG
GEOMETRY_EPS = _phase1.GEOMETRY_EPS
MAX_COORDINATE_ABS = _phase1.MAX_COORDINATE_ABS
SCAN_N = _phase1.SCAN_N
SCAN_ANGLE_DEG = _phase1.SCAN_ANGLE_DEG
SCAN_R_CRITICAL = _phase1.SCAN_R_CRITICAL
SCAN_COVERAGE_SAFETY_M = _phase1.SCAN_COVERAGE_SAFETY_M
SCAN_R = _phase1.SCAN_R
HOMING_STEP = _phase1.HOMING_STEP
BRACKET_END = _phase1.BRACKET_END
ENROUTE_DETOUR = _phase1.ENROUTE_DETOUR
MAX_CHANNEL_ATTEMPTS = _phase1.MAX_CHANNEL_ATTEMPTS

ChannelState = _phase1.ChannelState
base = _phase1.base
post = _phase1.post
convex_diameter = _phase1.convex_diameter
localize_region = _phase1.localize_region
minimum_distance_to_polygon = _phase1.minimum_distance_to_polygon
minimum_enclosing_circle = _phase1.minimum_enclosing_circle
uvec = _phase1.uvec
angdiff = _phase1.angdiff
scan_position = _phase1.scan_position
worst_regular_scan_distance = _phase1.worst_regular_scan_distance
region_of = _phase1.region_of
guaranteed_clear_circle = _phase1.guaranteed_clear_circle
should_skip_scan = _phase1.should_skip_scan
join_depth = _phase1.join_depth
tsp_exact = _phase1.tsp_exact
routing_point = _phase1.routing_point
retry_mode = _phase1.retry_mode

LAST_VT = float(_phase1.LAST_VT)
CURRENT_POS = np.asarray(_phase1.CURRENT_POS, dtype=float).copy()


def _sync_from_phase1():
    global LAST_VT, CURRENT_POS
    LAST_VT = float(_phase1.LAST_VT)
    CURRENT_POS = np.asarray(_phase1.CURRENT_POS, dtype=float).copy()


def _track(response):
    result = _phase1._track(response)
    _sync_from_phase1()
    return result


def _post(path, payload, retries=2):
    return _phase1._post(path, payload, retries=retries)


def measure(x, y, channel):
    """Forward a measure request while exposing Phase 4's position state."""
    _phase1.CURRENT_POS = np.asarray(CURRENT_POS, dtype=float).copy()
    response = _phase1.measure(x, y, channel)
    _sync_from_phase1()
    return response


def try_clear(channel, position, tag=""):
    """Forward the strict Phase 1 clear boundary and synchronize position."""
    _phase1.CURRENT_POS = np.asarray(CURRENT_POS, dtype=float).copy()
    response = _phase1.try_clear(channel, position, tag)
    _sync_from_phase1()
    return response


def add_observation(state, position, bearing):
    """Use the verified single observation/update path."""
    return _phase1.add_observation(state, position, bearing)


def measure_and_update(state, position):
    """Measure and retain every valid direction in the channel history."""
    point = np.asarray(position, dtype=float)
    response = measure(point[0], point[1], state.channel)
    if response is None:
        return None
    state.last_position = point.copy()
    if response.get("measure_result") == "direction":
        add_observation(state, point, response["svd_deg"])
    return response


# Phase 1 fallback implementations are intentionally exposed as names in
# this module so callers/tests can replace them with bounded stubs.  Keep the
# two modules' position mirrors synchronized around every fallback: the Phase
# 1 implementations issue requests through their own module globals, while
# the active controller uses this module's mirror for later routing/detours.
def _run_phase1_fallback(function, *args, **kwargs):
    _phase1.CURRENT_POS = np.asarray(CURRENT_POS, dtype=float).copy()
    try:
        return function(*args, **kwargs)
    finally:
        _sync_from_phase1()


def iterative_clear(*args, **kwargs):
    return _run_phase1_fallback(_phase1.iterative_clear, *args, **kwargs)


def bracket_clear(*args, **kwargs):
    return _run_phase1_fallback(_phase1.bracket_clear, *args, **kwargs)


def homing_clear(*args, **kwargs):
    return _run_phase1_fallback(_phase1.homing_clear, *args, **kwargs)


ACTIVE_BUDGETS_M = (100.0, 200.0, 300.0, 400.0)
"""The only production budget ladder: 100 -> 200 -> 300 -> 400 m."""


@dataclass(frozen=True)
class ActiveConfig:
    """Bounded, injectable numerical settings for the active controller.

    Defaults are intentionally lighter than the offline research study so a
    live controller and unit tests do not launch a heavy search.  The solver
    still receives the exact four-step production budget ladder by default.
    ``planner`` may be supplied to inject a deterministic test planner.
    """

    budgets_m: tuple = ACTIVE_BUDGETS_M
    source_sample_count: int = 5
    error_samples: tuple = (-1.0, 0.0, 1.0)
    spacing_m: float = 150.0
    circle_segments: int = 16
    max_constraint_vertices: int = 96
    max_candidates: int = 80
    min_reception_radius: float = MIN_RECEPTION_RADIUS
    max_reception_radius: float = MAX_RECEPTION_RADIUS
    angle_error_deg: float = ANGLE_ERROR_DEG
    max_steps: int = 4
    time_limit_s: float = 12.0
    planner: object = None

    def __post_init__(self):
        budgets = tuple(float(value) for value in self.budgets_m)
        if any(value <= 0 for value in budgets) or any(
            right <= left for left, right in zip(budgets, budgets[1:])
        ):
            raise ValueError("active budgets must be positive and increasing")
        if len(budgets) != 4:
            raise ValueError("active budgets must contain four ladder values")
        if self.source_sample_count < 1:
            raise ValueError("source_sample_count must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.time_limit_s < 0 or not math.isfinite(float(self.time_limit_s)):
            raise ValueError("time_limit_s must be finite and non-negative")
        object.__setattr__(self, "budgets_m", budgets)
        object.__setattr__(
            self,
            "error_samples",
            tuple(float(value) for value in self.error_samples),
        )


DEFAULT_ACTIVE_CONFIG = ActiveConfig()


@dataclass(frozen=True)
class ActiveDecision:
    """Selected finite candidate and the minimum budget that admits it."""

    evaluation: object
    budget_m: float
    pareto: object = None

    @property
    def point(self):
        return None if self.evaluation is None else np.asarray(
            self.evaluation.point, dtype=float
        ).copy()


def _evaluation_value(evaluation, name, default=math.inf):
    value = getattr(evaluation, name, None)
    if value is None:
        aliases = {
            "movement_distance_m": "movement_distance",
            "q_rho_m": "q_rho",
            "q_diameter_m": "q_diameter",
        }
        alias = aliases.get(name)
        value = getattr(evaluation, alias, default) if alias else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _evaluation_point(evaluation):
    try:
        point = np.asarray(evaluation.point, dtype=float)
    except Exception:
        return None
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return None
    return point


def _is_finite_evaluation(evaluation):
    point = _evaluation_point(evaluation)
    return (
        point is not None
        and math.isfinite(_evaluation_value(evaluation, "movement_distance_m"))
        and math.isfinite(_evaluation_value(evaluation, "q_rho_m"))
    )


def _shortest_qrho_threshold_evaluation(evaluations, max_budget=math.inf):
    """Choose a discrete q_rho <= 20 evaluation by actual movement."""
    eligible = []
    for evaluation in evaluations:
        if not _is_finite_evaluation(evaluation):
            continue
        if not bool(getattr(evaluation, "robust", True)):
            continue
        if _evaluation_value(evaluation, "movement_distance_m") > max_budget + GEOMETRY_EPS:
            continue
        if _evaluation_value(evaluation, "q_rho_m") <= CLEAR_RADIUS + GEOMETRY_EPS:
            eligible.append(evaluation)
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            _evaluation_value(item, "movement_distance_m"),
            _evaluation_value(item, "q_diameter_m"),
            _evaluation_value(item, "q_rho_m"),
            float(_evaluation_point(item)[0]),
            float(_evaluation_point(item)[1]),
        ),
    )


def select_active_candidate(pareto, budgets=ACTIVE_BUDGETS_M):
    """Select a Phase 4 point from one shared discrete Pareto evaluation.

    First, all evaluated candidates with ``q_rho <= 20`` compete globally by
    shortest *actual* movement.  If none reaches that threshold, the first
    budget with a feasible Pareto choice wins; its solver-selected point is
    retained.  ``Q_D`` only breaks ties in the discrete-threshold case and never
    replaces movement as the primary criterion.
    """
    if pareto is None:
        return None
    ladder = tuple(float(value) for value in budgets)
    if len(ladder) != 4 or any(
        right <= left for left, right in zip(ladder, ladder[1:])
    ):
        raise ValueError("active budget ladder must contain four increasing values")

    raw_choices = getattr(pareto, "choices", ())
    choices = [] if raw_choices is None else list(raw_choices)
    raw_evaluations = getattr(pareto, "evaluations", ())
    evaluations = [] if raw_evaluations is None else list(raw_evaluations)
    # A small injected result may expose only budget choices.  Include those
    # evaluations in the global q_rho threshold competition as well.
    for choice in choices:
        evaluation = getattr(choice, "evaluation", None)
        if evaluation is not None and all(evaluation is not item for item in evaluations):
            evaluations.append(evaluation)
    threshold_evaluation = _shortest_qrho_threshold_evaluation(
        evaluations, max(ladder)
    )
    if threshold_evaluation is not None:
        distance = _evaluation_value(threshold_evaluation, "movement_distance_m")
        eligible_budgets = [
            value for value in ladder if distance <= value + GEOMETRY_EPS
        ]
        return ActiveDecision(
            threshold_evaluation,
            min(eligible_budgets) if eligible_budgets else distance,
            pareto,
        )

    # Prefer an explicit choice object for each budget, preserving the
    # solver's Pareto point.  A small injected fake may expose only a mapping.
    by_budget = {}
    for choice in choices:
        budget = getattr(choice, "budget_m", None)
        evaluation = getattr(choice, "evaluation", None)
        if budget is not None and evaluation is not None:
            by_budget[float(budget)] = evaluation
    if not by_budget:
        mapping = getattr(pareto, "recommended_by_budget", None)
        if isinstance(mapping, dict):
            by_budget = {
                float(budget): evaluation
                for budget, evaluation in mapping.items()
                if evaluation is not None
            }
            for evaluation in by_budget.values():
                if all(evaluation is not item for item in evaluations):
                    evaluations.append(evaluation)
            threshold_evaluation = _shortest_qrho_threshold_evaluation(
                evaluations, max(ladder)
            )
            if threshold_evaluation is not None:
                distance = _evaluation_value(
                    threshold_evaluation, "movement_distance_m"
                )
                eligible_budgets = [
                    value for value in ladder if distance <= value + GEOMETRY_EPS
                ]
                return ActiveDecision(
                    threshold_evaluation,
                    min(eligible_budgets) if eligible_budgets else distance,
                    pareto,
                )
    if not by_budget:
        # Last-resort adapter for a light fake exposing only evaluations.  It
        # mirrors solve_pareto's per-budget lexicographic selection while
        # retaining the first feasible ladder budget.
        for budget in ladder:
            feasible = [
                evaluation
                for evaluation in evaluations
                if _is_finite_evaluation(evaluation)
                and bool(getattr(evaluation, "robust", True))
                and _evaluation_value(evaluation, "movement_distance_m")
                <= budget + GEOMETRY_EPS
            ]
            if feasible:
                by_budget[budget] = min(
                    feasible,
                    key=lambda item: (
                        _evaluation_value(item, "q_rho_m"),
                        _evaluation_value(item, "q_diameter_m"),
                        _evaluation_value(item, "movement_distance_m"),
                        float(_evaluation_point(item)[0]),
                        float(_evaluation_point(item)[1]),
                    ),
                )
                break
    for budget in ladder:
        evaluation = by_budget.get(float(budget))
        if evaluation is None or not _is_finite_evaluation(evaluation):
            continue
        movement = _evaluation_value(evaluation, "movement_distance_m")
        if movement <= budget + GEOMETRY_EPS:
            return ActiveDecision(evaluation, budget, pareto)
    return None


def _call_planner(planner, state, position, config):
    """Call an injected planner without hiding planner-internal exceptions."""
    if planner is None:
        planner = config.planner if config.planner is not None else solve_pareto
    state.refresh()
    if len(state.polygon) == 0:
        return None
    kwargs = {
        "budgets_m": config.budgets_m,
        "source_sample_count": config.source_sample_count,
        "error_samples": config.error_samples,
        "spacing_m": config.spacing_m,
        "min_reception_radius": config.min_reception_radius,
        "max_reception_radius": config.max_reception_radius,
        "angle_error_deg": config.angle_error_deg,
        "circle_segments": config.circle_segments,
        "max_candidates": config.max_candidates,
        # Deliberately None: candidate measurements may be outside the
        # 1800 m source circle; only robust reception and budget constrain it.
        "target_radius": None,
    }
    try:
        signature = inspect.signature(planner)
    except (TypeError, ValueError):
        return planner(state.polygon.copy(), np.asarray(position, dtype=float).copy(), **kwargs)
    try:
        signature.bind(state.polygon.copy(), np.asarray(position, dtype=float), **kwargs)
    except TypeError:
        # Lightweight injected test doubles commonly accept only the core
        # three arguments.  This fallback is signature-based, so a TypeError
        # raised *inside* a compatible planner is never silently swallowed.
        try:
            signature.bind(
                state.polygon.copy(),
                np.asarray(position, dtype=float),
                config.budgets_m,
            )
        except TypeError:
            return planner(state.polygon.copy(), np.asarray(position, dtype=float).copy())
        return planner(
            state.polygon.copy(),
            np.asarray(position, dtype=float).copy(),
            config.budgets_m,
        )
    return planner(
        state.polygon.copy(), np.asarray(position, dtype=float).copy(), **kwargs
    )


def plan_active_candidate(state, position, config=None, planner=None):
    """Run one bounded discrete plan and return an :class:`ActiveDecision`."""
    config = DEFAULT_ACTIVE_CONFIG if config is None else config
    pareto = _call_planner(planner, state, position, config)
    if isinstance(pareto, ActiveDecision):
        return pareto
    if isinstance(pareto, CandidateEvaluation):
        # Lightweight test planners may return a single evaluation directly.
        # Keep the same production ladder rule when wrapping that result.
        pareto = type(
            "_SingleEvaluationPareto",
            (),
            {"evaluations": (pareto,), "choices": ()},
        )()
    return select_active_candidate(pareto, config.budgets_m)


def _fallback_position_theta(state, position=None):
    if position is None:
        position = state.last_position
    if position is None and state.observations:
        position = state.observations[-1]
    if position is None:
        position = CURRENT_POS
    point = np.asarray(position, dtype=float)
    if state.bearings:
        theta = float(state.bearings[-1])
    else:
        theta = 0.0
    return point, theta


def homing_bracket_fallback(state, position=None, reason="active_fallback"):
    """Invoke the retained homing/bracket emergency path, boundedly."""
    point, theta = _fallback_position_theta(state, position)
    state.last_failure_reason = reason
    if not state.bearings:
        try:
            return bool(iterative_clear(state))
        except Exception:
            _phase1._fail(state, reason)
            return False

    homing_result = False
    try:
        homing_result = bool(
            homing_clear(state, point, theta, MAX_RECEPTION_RADIUS)
        )
    except Exception:
        homing_result = False
    if homing_result:
        return True

    # A production homing implementation normally enters bracket_clear when
    # it observes a reversal.  Calling it explicitly after an injected or
    # bounded homing failure keeps the fallback contract testable and gives a
    # second bounded recovery route before process_pending_channels retries.
    if state.bearings:
        try:
            return bool(
                bracket_clear(state, point, float(state.bearings[-1]), MAX_RECEPTION_RADIUS)
            )
        except Exception:
            pass
    _phase1._fail(state, reason)
    return False


def _handle_active_measurement(state, position, response):
    """Handle a just-completed active measurement; return True only on clear."""
    if response is None:
        return None
    result = response.get("measure_result")
    if result == "near":
        try:
            return bool(try_clear(state.channel, position, "active-near"))
        except Exception:
            return False
    if result == "direction":
        # measure_and_update has already called add_observation.  Returning
        # False here means the active loop should recompute Omega/MEC.
        return False
    return None


def active_localization_clear(
    state,
    start_position=None,
    *,
    config=None,
    planner=None,
):
    """Run bounded receding-horizon active localization for one channel.

    The current posterior and MEC are refreshed at the top of every round.
    A ``rho <= 20`` region is always handled by the strict MEC clear path
    before asking the active planner for another point.
    """
    config = DEFAULT_ACTIVE_CONFIG if config is None else config
    position, _ = _fallback_position_theta(state, start_position)
    started = time.monotonic()

    for step in range(config.max_steps + 1):
        if time.monotonic() - started > float(config.time_limit_s):
            return homing_bracket_fallback(state, position, "active_time_guard")
        state.refresh()
        if len(state.polygon) == 0:
            return homing_bracket_fallback(state, position, "active_empty_region")

        # Strict MEC criterion remains first priority for every round.
        clear_circle = guaranteed_clear_circle(state)
        if clear_circle is not None:
            center, _ = clear_circle
            try:
                if try_clear(state.channel, center, "active-mec"):
                    return True
            except Exception:
                pass
            try:
                response = measure_and_update(state, center)
            except Exception:
                return homing_bracket_fallback(state, position, "active_mec_exception")
            if response is None:
                return homing_bracket_fallback(state, center, "active_mec_measure_failed")
            handled = _handle_active_measurement(state, center, response)
            if handled is True:
                return True
            if response.get("measure_result") == "direction":
                position = np.asarray(center, dtype=float)
                continue
            return homing_bracket_fallback(state, center, "active_mec_unexpected_result")

        if step >= config.max_steps:
            return homing_bracket_fallback(state, position, "active_step_guard")

        try:
            decision = plan_active_candidate(state, position, config, planner)
        except Exception:
            return homing_bracket_fallback(state, position, "active_planner_exception")
        if decision is None or decision.point is None:
            return homing_bracket_fallback(state, position, "active_no_candidate")
        candidate = np.asarray(decision.point, dtype=float)
        if candidate.shape != (2,) or not np.all(np.isfinite(candidate)):
            return homing_bracket_fallback(state, position, "active_invalid_candidate")
        movement = float(np.linalg.norm(candidate - position))
        if movement > float(decision.budget_m) + GEOMETRY_EPS:
            return homing_bracket_fallback(state, position, "active_budget_violation")

        # There is no separate movement endpoint in the simulator API.  The
        # measure request performs the move and updates CURRENT_POS only after
        # acceptance; the candidate itself is retained as the new position.
        try:
            response = measure_and_update(state, candidate)
        except Exception:
            return homing_bracket_fallback(state, position, "active_measure_exception")
        if response is None:
            return homing_bracket_fallback(state, position, "active_measure_failed")
        position = candidate.copy()
        handled = _handle_active_measurement(state, position, response)
        if handled is True:
            return True
        if response.get("measure_result") == "direction":
            # The new direction is already in the common observation history.
            # Recompute Omega/MEC at the next loop head.
            continue
        return homing_bracket_fallback(state, position, "active_unexpected_result")

    return homing_bracket_fallback(state, position, "active_step_guard")


def process_channel(state, mode="normal", *, active_config=None, planner=None):
    """Process one channel, preferring active localization in normal mode."""
    state.status = "processing"
    state.refresh()

    if not state.observations:
        if state.last_position is None:
            return _phase1._fail(state, "no_observation")
        try:
            response = measure_and_update(state, state.last_position)
        except Exception:
            return _phase1._fail(state, "measure_network_or_rejected")
        if response is None:
            return _phase1._fail(state, "measure_network_or_rejected")
        if response.get("measure_result") == "near":
            try:
                if try_clear(state.channel, state.last_position, "rediscovered-near"):
                    return True
            except Exception:
                pass
            return _phase1._fail(state, "near_clear_failed")
        if response.get("measure_result") != "direction":
            return _phase1._fail(state, "no_signal_without_bearing")

    state.refresh()
    if len(state.polygon) == 0:
        return homing_bracket_fallback(state, reason="empty_region_without_bearing")

    clear_circle = guaranteed_clear_circle(state)
    if clear_circle is not None:
        center, _ = clear_circle
        try:
            if try_clear(state.channel, center, "mec"):
                return True
        except Exception:
            pass
        try:
            response = measure_and_update(state, center)
        except Exception:
            return homing_bracket_fallback(state, center, "mec_clear_exception")
        if response is not None:
            handled = _handle_active_measurement(state, center, response)
            if handled is True:
                return True
            if response.get("measure_result") == "direction":
                # Continue below; this branch intentionally does not discard
                # the new direction after a failed MEC clear.
                pass
            else:
                return homing_bracket_fallback(state, center, "mec_clear_failed")
        else:
            return homing_bracket_fallback(state, center, "mec_measure_failed")

    if mode != "normal":
        # Preserve Phase 1 retry-mode semantics: conservative/emergency
        # retries use the bounded existing fallback instead of expanding the
        # active search after a prior failure.
        if mode == "conservative":
            try:
                return bool(iterative_clear(state))
            except Exception:
                return homing_bracket_fallback(state, reason="conservative_exception")
        return homing_bracket_fallback(state, reason="emergency_mode")

    print(
        f"  [{state.channel}] active localization: "
        f"D={state.diameter:.1f} m, rho={state.mec_radius:.1f} m"
    )
    return active_localization_clear(
        state,
        state.last_position,
        config=active_config,
        planner=planner,
    )


def process_pending_channels(
    states,
    cleared,
    start_time=None,
    processor=None,
    runtime_limit_s=17 * 60,
    *,
    active_config=None,
    planner=None,
):
    """Phase 1 retry queue with Phase 4's processor as the default."""
    if processor is None:
        processor = lambda state, mode: process_channel(
            state, mode, active_config=active_config, planner=planner
        )
    return _phase1.process_pending_channels(
        states,
        cleared,
        start_time=start_time,
        processor=processor,
        runtime_limit_s=runtime_limit_s,
    )


def main(active_config=None, planner=None):
    """Run the Phase 1 discovery route with Phase 4 channel processing."""
    start_time = time.time()
    response = _track(_post("/enter", base(_phase1._uid("enter"))))
    if not response or response.get("accepted") is not True:
        print("enter failed")
        return
    print("entered: Q3 Phase 4 (regular heptagon + active localization)")

    cleared = set()
    states = {}

    def scan_point(position, index):
        print(
            f"\nscan {index + 1}/{SCAN_N}: "
            f"({position[0]:.0f}, {position[1]:.0f})"
        )
        for channel in range(1, 21):
            if should_skip_scan(channel, states, cleared, position):
                continue
            result = measure(position[0], position[1], channel)
            if result is None:
                continue
            measurement = result.get("measure_result")
            if measurement == "direction":
                state = states.setdefault(channel, ChannelState(channel))
                add_observation(state, position, result["svd_deg"])
                print(f"  channel {channel}: direction {result['svd_deg']} deg")
            elif measurement == "near":
                state = states.setdefault(channel, ChannelState(channel))
                state.last_position = np.asarray(position, dtype=float).copy()
                if try_clear(channel, position, "scan-near"):
                    state.status = "cleared"
                    cleared.add(channel)
                else:
                    state.last_failure_reason = "scan_near_clear_failed"
                    state.status = "retry_pending"

    first = scan_position(0)
    scan_point(first, 0)
    sx = sy = 0.0
    for state in states.values():
        for bearing in state.bearings:
            vector = uvec(bearing)
            sx += vector[0]
            sy += vector[1]
    source_group_angle = math.degrees(math.atan2(sy, sx)) if (sx or sy) else 0.0
    plus = abs(angdiff(SCAN_ANGLE_DEG, source_group_angle))
    minus = abs(angdiff(-SCAN_ANGLE_DEG, source_group_angle))
    scan_direction = 1 if plus <= minus else -1
    terminal_angle = SCAN_ANGLE_DEG * scan_direction
    print(
        f"source-group bearing {source_group_angle:.0f} deg; "
        f"terminal scan angle {terminal_angle:+.2f} deg"
    )

    tried_enroute = set()
    for index in range(1, SCAN_N):
        if time.time() - start_time > 17 * 60:
            print("real-time guard reached during scanning")
            break
        position = scan_position(index, scan_direction)
        scan_point(position, index)
        if index < 2 or index + 1 >= SCAN_N:
            continue
        next_scan = scan_position(index + 1, scan_direction)
        while True:
            candidate = None
            best_detour = float("inf")
            for state in states.values():
                if state.channel in cleared or state.channel in tried_enroute:
                    continue
                clear_circle = guaranteed_clear_circle(state)
                if clear_circle is None:
                    continue
                center, _ = clear_circle
                detour = (
                    float(np.linalg.norm(CURRENT_POS - center))
                    + float(np.linalg.norm(center - next_scan))
                    - float(np.linalg.norm(CURRENT_POS - next_scan))
                )
                if detour < best_detour:
                    candidate, best_detour = state, detour
            if candidate is None or best_detour > ENROUTE_DETOUR:
                break
            tried_enroute.add(candidate.channel)
            print(
                f"en-route guaranteed clear: channel {candidate.channel}, "
                f"detour {best_detour:.0f} m"
            )
            if process_channel(
                candidate,
                active_config=active_config,
                planner=planner,
            ):
                candidate.status = "cleared"
                cleared.add(candidate.channel)
            else:
                candidate.retry_count += 1
                candidate.status = "retry_pending"

    print(
        f"\ndiscovery complete: {len(states)} channels discovered, "
        f"{len(cleared)} cleared, virtual time {LAST_VT:.0f} s"
    )
    unresolved = process_pending_channels(
        states,
        cleared,
        start_time,
        active_config=active_config,
        planner=planner,
    )
    discovered_channels = set(states)
    success = discovered_channels == cleared
    if success:
        print(f"\nsuccess: all {len(cleared)} discovered channels cleared")
    else:
        print("\nnot successful: discovered channels remain uncleared")
        print(f"unresolved channels: {sorted(state.channel for state in unresolved)}")
    print(f"cleared channels: {sorted(cleared)}")
    print(f"total virtual time: {LAST_VT:.1f} s ({LAST_VT / 60.0:.1f} min)")
    _track(_post("/exit", base(_phase1._uid("exit"))))


__all__ = [
    "ACTIVE_BUDGETS_M",
    "ActiveConfig",
    "ActiveDecision",
    "ANGLE_ERROR_DEG",
    "BudgetChoice",
    "CandidateEvaluation",
    "ChannelState",
    "CLEAR_MAX_DISTANCE",
    "CLEAR_RADIUS",
    "CURRENT_POS",
    "DEFAULT_ACTIVE_CONFIG",
    "GEOMETRY_EPS",
    "LAST_VT",
    "MAX_RECEPTION_RADIUS",
    "MIN_RECEPTION_RADIUS",
    "ParetoResult",
    "SCAN_ANGLE_DEG",
    "SCAN_N",
    "SCAN_R",
    "TARGET_RADIUS",
    "active_localization_clear",
    "add_observation",
    "bracket_clear",
    "candidate_points_for_budget",
    "convex_diameter",
    "evaluate_candidate",
    "generate_robust_candidates",
    "guaranteed_clear_circle",
    "homing_bracket_fallback",
    "homing_clear",
    "is_robust_candidate",
    "localize_region",
    "main",
    "max_distance_to_region",
    "minimum_distance_to_polygon",
    "minimum_enclosing_circle",
    "measure",
    "measure_and_update",
    "pareto_frontier",
    "plan_active_candidate",
    "posterior_region",
    "process_channel",
    "process_pending_channels",
    "region_of",
    "retry_mode",
    "robust_candidate_domain",
    "sample_region_sources",
    "select_active_candidate",
    "solve_pareto",
    "try_clear",
]


if __name__ == "__main__":
    main()
