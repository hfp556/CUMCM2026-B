"""Q4 Active: Q3 Active V2 plus directional discovery/reacquisition.

The Q3 controller remains the single source of truth for channel state,
geometry, strict MEC clearing, active localization, service routing, en-route
clears, retry modes and HTTP position tracking.  Q4 adds only the behavior
that changes when a source may transmit into an unknown 180-degree half-plane:

* a distance-and-angle discovery mesh;
* bounded directional reacquisition after a previously visible channel goes
  blind; and
* homing/bracket/iterative fallbacks that do not interpret ``no_signal`` as
  evidence that the source must be farther away.

The legacy Q4 nearest-neighbour tour, D<=50 shortcut and 1780 m robot boundary
are deliberately absent.
"""

from dataclasses import dataclass
import math
import time

import numpy as np

try:
    from . import Q3_active_v2 as _q3
    from . import api_utils as _api
except ImportError:  # Allow ``python src/new/Q4_active.py``.
    import Q3_active_v2 as _q3
    import api_utils as _api


# Q3 Active V2 is reused, not forked.  These aliases make the inherited
# architecture explicit and keep both questions on exactly the same state and
# geometry implementations.
ChannelState = _q3.ChannelState
ChannelActionPlan = _q3.ChannelActionPlan
ChannelServiceTask = _q3.ChannelServiceTask
ActiveConfig = _q3.ActiveConfig
ActiveDecision = _q3.ActiveDecision
DEFAULT_ACTIVE_CONFIG = _q3.DEFAULT_ACTIVE_CONFIG

TARGET_RADIUS = _q3.TARGET_RADIUS
CLEAR_RADIUS = _q3.CLEAR_RADIUS
CLEAR_MAX_DISTANCE = _q3.CLEAR_MAX_DISTANCE
MIN_RECEPTION_RADIUS = _q3.MIN_RECEPTION_RADIUS
MAX_RECEPTION_RADIUS = _q3.MAX_RECEPTION_RADIUS
GEOMETRY_EPS = _q3.GEOMETRY_EPS
MAX_CHANNEL_ATTEMPTS = _q3.MAX_CHANNEL_ATTEMPTS
SCAN_N = _q3.SCAN_N
SCAN_ANGLE_DEG = _q3.SCAN_ANGLE_DEG
SCAN_R = _q3.SCAN_R
HOMING_STEP = _q3.HOMING_STEP
BRACKET_END = _q3.BRACKET_END
ENROUTE_CUMULATIVE_BUDGET = _q3.ENROUTE_CUMULATIVE_BUDGET
_ACTIVE_DECISION_UNSET = _q3._ACTIVE_DECISION_UNSET

base = _q3.base
post = _q3.post
measure = _q3.measure
try_clear = _q3.try_clear
measure_and_update = _q3.measure_and_update
add_observation = _q3.add_observation
guaranteed_clear_circle = _q3.guaranteed_clear_circle
channel_action_plan = _q3.channel_action_plan
channel_service_task = _q3.channel_service_task
service_route_exact = _q3.service_route_exact
retry_mode = _q3.retry_mode
scan_position = _q3.scan_position
scan_channel_order = _q3.scan_channel_order
should_skip_scan = _q3.should_skip_scan
select_enroute_candidate = _q3.select_enroute_candidate
_attempt_and_record_enroute = _q3._attempt_and_record_enroute
_record_discovery_measurement = _q3._record_discovery_measurement
retry_discovery_failures = _q3.retry_discovery_failures
plan_active_candidate = _q3.plan_active_candidate
_evaluation_value = _q3._evaluation_value
_fallback_position_theta = _q3._fallback_position_theta
_has_direction_at_position = _q3._has_direction_at_position
_homing_join = _q3._homing_join
_fail = _q3._fail
uvec = _q3.uvec
angdiff = _q3.angdiff
_track = _q3._track
_post = _q3._post
_uid = _q3._uid
_invoke_processor = _q3._invoke_processor


# A 12+12 staggered polar mesh supplies the angular certificate.  Its inner
# vertex at angle zero is exactly Q3's first scan point and is therefore reused;
# the other 24 locations are supplemental.  The old Q4's 23-point grid is not
# involved.  The outer 12-gon's inradius is >1800 m, and every triangle edge is
# strictly shorter than the worst-case 1000 m reception radius.
DIRECTIONAL_RING_N = 12
DIRECTIONAL_INNER_R = SCAN_R
DIRECTIONAL_OUTER_R = 1880.0
DIRECTIONAL_RING_STEP_DEG = 360.0 / DIRECTIONAL_RING_N
REACQUISITION_BISECTIONS = 3
BLIND_POINT_EPS_M = 1.0
EXIT_TIME_RESERVE_S = 2.0


def _install_official_runtime():
    """Prepare shared official transport/logging for the Q4 script entry."""
    details = _api.configure(run_name="q4_active")
    _api.install_console_log()
    print(f"Q4 official simulator endpoint: {details['base_url']}")
    print(f"Q4 request log: {details['jsonl_path']}")
    print(f"Q4 console log: {details['console_path']}")
    return details


def _runtime_limit_from_enter(response):
    """Use the /enter budget without an artificial 16/17-minute cap."""
    return _api.runtime_limit_from_enter(
        response,
        reserve_s=EXIT_TIME_RESERVE_S,
    )


def directional_inner_position(index):
    angle = math.radians(DIRECTIONAL_RING_STEP_DEG * int(index))
    return DIRECTIONAL_INNER_R * np.array([math.cos(angle), math.sin(angle)])


def directional_outer_position(index):
    angle = math.radians(
        DIRECTIONAL_RING_STEP_DEG * (int(index) + 0.5)
    )
    return DIRECTIONAL_OUTER_R * np.array([math.cos(angle), math.sin(angle)])


def directional_discovery_positions(scan_direction=1):
    """Return only the supplemental probes, ordered from Q3's last scan point.

    The angle-zero inner vertex has already been visited by the inherited Q3
    scan.  The other eleven inner vertices, twelve staggered outer vertices and
    the origin complete the certified mesh.  The origin is visited last.
    """
    direction = 1 if scan_direction >= 0 else -1
    # Fixed-endpoint ring weave.  From Q3's terminal point it visits the two
    # nearest missing inner vertices, traverses the outer ring in the opposite
    # direction, then finishes the remaining inner arc before ending at the
    # origin.  A deterministic 2-opt check gives 18,401.94 m versus 19,329.61 m
    # for the former inner-then-outer order, with identical probes.
    points = [
        directional_inner_position((-2 * direction) % DIRECTIONAL_RING_N),
        directional_inner_position((-1 * direction) % DIRECTIONAL_RING_N),
    ]
    outer_start = DIRECTIONAL_RING_N - 1 if direction > 0 else 0
    for step in range(DIRECTIONAL_RING_N):
        index = (outer_start - step * direction) % DIRECTIONAL_RING_N
        points.append(directional_outer_position(index))
    for step in range(1, DIRECTIONAL_RING_N - 2):
        points.append(
            directional_inner_position((step * direction) % DIRECTIONAL_RING_N)
        )
    points.append(np.array([0.0, 0.0]))
    return points


def directional_discovery_triangles():
    """Triangulate the certified distance-and-angle discovery domain."""
    center = np.array([0.0, 0.0])
    inner = [directional_inner_position(index) for index in range(DIRECTIONAL_RING_N)]
    outer = [directional_outer_position(index) for index in range(DIRECTIONAL_RING_N)]
    triangles = []
    for index in range(DIRECTIONAL_RING_N):
        nxt = (index + 1) % DIRECTIONAL_RING_N
        previous_outer = (index - 1) % DIRECTIONAL_RING_N
        triangles.append(np.array([center, inner[index], inner[nxt]]))
        triangles.append(
            np.array([inner[index], outer[previous_outer], outer[index]])
        )
        triangles.append(np.array([inner[index], outer[index], inner[nxt]]))
    return triangles


def directional_coverage_certificate():
    """Return analytic margins for the Q4 discovery mesh.

    Every point in the target disk lies strictly inside the outer polygon and
    in one of the returned triangles.  A triangle has diameter equal to its
    longest edge, so all three probes are within the worst-case 1000 m receive
    radius.  Because the source is in their convex hull, the probes cannot all
    lie behind any line through the source; at least one is on the transmitting
    side of every 180-degree orientation.
    """
    max_edge = 0.0
    for triangle in directional_discovery_triangles():
        for left in range(3):
            for right in range(left + 1, 3):
                max_edge = max(
                    max_edge,
                    float(np.linalg.norm(triangle[left] - triangle[right])),
                )
    inradius = DIRECTIONAL_OUTER_R * math.cos(math.pi / DIRECTIONAL_RING_N)
    return {
        "outer_inradius_m": inradius,
        "target_margin_m": inradius - TARGET_RADIUS,
        "max_triangle_edge_m": max_edge,
        "distance_margin_m": MIN_RECEPTION_RADIUS - max_edge,
        "full_probe_count": 1 + 2 * DIRECTIONAL_RING_N,
        "supplemental_probe_count": 2 * DIRECTIONAL_RING_N,
    }


@dataclass(frozen=True)
class ReacquisitionResult:
    status: str
    position: np.ndarray | None = None
    bearing: float | None = None
    attempts: int = 0

    @property
    def recovered(self):
        return self.status == "direction"

    @property
    def cleared(self):
        return self.status == "cleared"


def _known_visible_positions(state, blind_position):
    blind = np.asarray(blind_position, dtype=float)
    candidates = []
    seen = set()
    for position in reversed(state.observations):
        point = np.asarray(position, dtype=float)
        key = tuple(np.round(point, 9))
        if key in seen or np.linalg.norm(point - blind) <= GEOMETRY_EPS:
            continue
        seen.add(key)
        candidates.append(point.copy())
    candidates.sort(key=lambda point: float(np.linalg.norm(point - blind)))
    return candidates


def directional_reacquisition(state, blind_position, max_bisections=None):
    """Recover a known directional channel without treating blindness as range.

    For each previously visible point, bisect the segment from that point to
    the new blind point.  When both endpoints are in reception range, the whole
    segment is in range by convexity, so the bisection searches the transmitter
    half-plane boundary rather than walking farther away.  The known visible
    endpoint is measured last as the bounded safety probe.
    """
    blind = np.asarray(blind_position, dtype=float)
    limit = REACQUISITION_BISECTIONS if max_bisections is None else int(max_bisections)
    attempts = 0
    for known in _known_visible_positions(state, blind):
        visible = known.copy()
        hidden = blind.copy()
        for _ in range(max(limit, 0)):
            probe = (visible + hidden) / 2.0
            response = measure_and_update(state, probe)
            attempts += 1
            if response is None:
                break
            result = response.get("measure_result")
            if result == "near":
                if try_clear(state.channel, probe, "directional-reacquire-near"):
                    return ReacquisitionResult("cleared", probe.copy(), None, attempts)
                return ReacquisitionResult("failed", probe.copy(), None, attempts)
            if result == "direction":
                return ReacquisitionResult(
                    "direction", probe.copy(), float(state.bearings[-1]), attempts
                )
            if result == "no_signal":
                hidden = probe
                continue
            break

        # Revisit the recorded visible endpoint.  Under the fixed-source Q4
        # model it is the deterministic final recovery point, while making the
        # simulator's actual robot position agree with the returned position.
        response = measure_and_update(state, visible)
        attempts += 1
        if response is None:
            continue
        result = response.get("measure_result")
        if result == "near":
            if try_clear(state.channel, visible, "directional-reacquire-near"):
                return ReacquisitionResult("cleared", visible.copy(), None, attempts)
            continue
        if result == "direction":
            return ReacquisitionResult(
                "direction", visible.copy(), float(state.bearings[-1]), attempts
            )

    _fail(state, "directional_reacquisition_failed")
    return ReacquisitionResult("failed", blind.copy(), None, attempts)


def iterative_clear(state, max_iter=8):
    """Q3 conservative fallback with directional blindness recovery."""
    for iteration in range(max_iter):
        state.refresh()
        if len(state.polygon) == 0:
            return _fail(state, "empty_region")
        print(
            f"  [{state.channel}] Q4 fallback iteration {iteration + 1}: "
            f"D={state.diameter:.1f} m, rho={state.mec_radius:.1f} m"
        )
        clear_circle = guaranteed_clear_circle(state)
        if clear_circle is not None:
            probe, _ = clear_circle
            if try_clear(state.channel, probe, "q4-mec-iter"):
                return True
        else:
            probe = np.asarray(state.mec_center, dtype=float)

        response = measure_and_update(state, probe)
        if response is None:
            return _fail(state, "measure_network_or_rejected")
        result = response.get("measure_result")
        if result == "near":
            if try_clear(state.channel, probe, "q4-iter-near"):
                return True
            return _fail(state, "near_clear_failed")
        if result == "direction":
            continue
        if result == "no_signal":
            reacquired = directional_reacquisition(state, probe)
            if reacquired.cleared:
                return True
            if reacquired.recovered:
                # The recovered direction is already in the common posterior.
                # Recompute on the next bounded iteration instead of nesting
                # homing -> iterative -> homing without a shared guard.
                continue
            return _fail(state, "iterative_directional_reacquisition_failed")
        return _fail(state, "iterative_unexpected_result")
    return _fail(state, "iterative_limit")


def bracket_clear(state, position, theta, bracket):
    """Bracket a source while treating a blind probe as a possible overshoot."""
    cur = np.asarray(position, dtype=float)
    cur_theta = float(theta)
    bracket = float(bracket)
    for _ in range(10):
        if bracket <= BRACKET_END:
            final = cur + (bracket / 2.0) * uvec(cur_theta)
            if try_clear(state.channel, final, "q4-bracket-end"):
                return True
            response = measure_and_update(state, final)
            if response is None:
                return _fail(state, "bracket_measure_network_or_rejected")
            if response.get("measure_result") == "near":
                if try_clear(state.channel, final, "q4-bracket-near"):
                    return True
            return iterative_clear(state)

        step = min(bracket, max(BRACKET_END, bracket / 2.0))
        nxt = cur + step * uvec(cur_theta)
        response = measure_and_update(state, nxt)
        if response is None:
            return iterative_clear(state)
        result = response.get("measure_result")
        if result == "near":
            if try_clear(state.channel, nxt, "q4-bracket-near"):
                return True
            return iterative_clear(state)
        if result == "direction":
            theta_new = float(state.bearings[-1])
            if abs(angdiff(theta_new, cur_theta)) > 90.0:
                bracket = step
            else:
                bracket = max(bracket - step, 0.0)
            cur, cur_theta = nxt, theta_new
            continue
        if result == "no_signal":
            # The robot did move to nxt, but nxt is not promoted to the visible
            # bracket endpoint.  Shrink toward the last direction-bearing point.
            bracket = step
            continue
        return iterative_clear(state)
    return iterative_clear(state)


def homing_clear(
    state,
    start_position,
    theta,
    bracket0,
    *,
    initial_direction_known=False,
):
    """Q4 homing: a forward blind point starts a bracket, not a range escape."""
    cur = np.asarray(start_position, dtype=float)
    cur_theta = float(theta)
    if not initial_direction_known:
        response = measure_and_update(state, cur)
        if response is None:
            return _fail(state, "homing_measure_network_or_rejected")
        result = response.get("measure_result")
        if result == "near":
            if try_clear(state.channel, cur, "q4-home-near"):
                return True
            return _fail(state, "near_clear_failed")
        if result == "direction":
            theta_new = float(state.bearings[-1])
            if abs(angdiff(theta_new, cur_theta)) > 90.0:
                return bracket_clear(state, cur, theta_new, bracket0)
            cur_theta = theta_new
        elif result == "no_signal":
            reacquired = directional_reacquisition(state, cur)
            if reacquired.cleared:
                return True
            if not reacquired.recovered:
                return _fail(state, "homing_directional_reacquisition_failed")
            cur = np.asarray(reacquired.position, dtype=float)
            cur_theta = float(reacquired.bearing)

    for _ in range(8):
        nxt = cur + HOMING_STEP * uvec(cur_theta)
        response = measure_and_update(state, nxt)
        if response is None:
            return iterative_clear(state)
        result = response.get("measure_result")
        if result == "near":
            if try_clear(state.channel, nxt, "q4-home-near"):
                return True
            return iterative_clear(state)
        if result == "direction":
            theta_new = float(state.bearings[-1])
            if abs(angdiff(theta_new, cur_theta)) > 90.0:
                return bracket_clear(state, nxt, theta_new, HOMING_STEP)
            cur, cur_theta = nxt, theta_new
            continue
        if result == "no_signal":
            return bracket_clear(state, cur, cur_theta, HOMING_STEP)
        return iterative_clear(state)
    return iterative_clear(state)


def homing_bracket_fallback(
    state,
    position=None,
    reason="q4_active_fallback",
    *,
    fallback_plan=None,
):
    """Bounded Q4 fallback using the directional-aware homing and bracket."""
    if fallback_plan is None:
        point, theta = _fallback_position_theta(state, position)
        bracket = MAX_RECEPTION_RADIUS
    else:
        point, theta, bracket = fallback_plan
        point = np.asarray(point, dtype=float)
        theta = float(theta)
        bracket = float(bracket)
    state.last_failure_reason = reason
    print(
        f"  [{state.channel}] Q4 active -> directional fallback: {reason} "
        f"at ({point[0]:.1f}, {point[1]:.1f})"
    )
    if not state.bearings:
        return iterative_clear(state)
    try:
        if homing_clear(
            state,
            point,
            theta,
            bracket,
            initial_direction_known=_has_direction_at_position(state, point),
        ):
            return True
    except Exception:
        pass
    if state.bearings:
        try:
            return bool(bracket_clear(state, point, float(state.bearings[-1]), bracket))
        except Exception:
            pass
    return _fail(state, reason)


def _is_blocked_candidate(point, blind_points):
    candidate = np.asarray(point, dtype=float)
    return any(
        float(np.linalg.norm(candidate - np.asarray(blind, dtype=float)))
        <= BLIND_POINT_EPS_M
        for blind in blind_points
    )


def active_localization_clear(
    state,
    start_position=None,
    *,
    config=None,
    planner=None,
    initial_decision=_ACTIVE_DECISION_UNSET,
    initial_failure_reason=None,
    initial_fallback_plan=None,
):
    """Q3 receding-horizon active localization with Q4 reacquisition."""
    config = DEFAULT_ACTIVE_CONFIG if config is None else config
    position, _ = _fallback_position_theta(state, start_position)
    started = time.monotonic()
    blind_points = []

    for step in range(config.max_steps + 1):
        if time.monotonic() - started > float(config.time_limit_s):
            return homing_bracket_fallback(state, position, "active_time_guard")
        state.refresh()
        if len(state.polygon) == 0:
            return homing_bracket_fallback(state, position, "active_empty_region")

        clear_circle = guaranteed_clear_circle(state)
        if clear_circle is not None:
            center, _ = clear_circle
            if try_clear(state.channel, center, "q4-active-mec"):
                return True
            response = measure_and_update(state, center)
            if response is None:
                return homing_bracket_fallback(state, center, "active_mec_measure_failed")
            result = response.get("measure_result")
            if result == "near" and try_clear(state.channel, center, "q4-active-near"):
                return True
            if result == "direction":
                position = np.asarray(center, dtype=float)
                continue
            if result == "no_signal":
                reacquired = directional_reacquisition(state, center)
                if reacquired.cleared:
                    return True
                if reacquired.recovered:
                    position = np.asarray(reacquired.position, dtype=float)
                    blind_points.append(np.asarray(center, dtype=float))
                    continue
            return homing_bracket_fallback(state, center, "active_mec_directional_blind")

        if step >= config.max_steps:
            return homing_bracket_fallback(state, position, "active_step_guard")

        cached = step == 0 and initial_decision is not _ACTIVE_DECISION_UNSET
        if cached:
            decision = initial_decision
        else:
            try:
                decision = plan_active_candidate(state, position, config, planner)
            except Exception:
                return homing_bracket_fallback(state, position, "active_planner_exception")
        if decision is None or decision.point is None:
            return homing_bracket_fallback(
                state,
                position,
                initial_failure_reason or "active_no_candidate",
                fallback_plan=initial_fallback_plan if cached else None,
            )
        candidate = np.asarray(decision.point, dtype=float)
        if candidate.shape != (2,) or not np.all(np.isfinite(candidate)):
            return homing_bracket_fallback(state, position, "active_invalid_candidate")
        if _is_blocked_candidate(candidate, blind_points):
            return homing_bracket_fallback(
                state, position, "active_repeated_directional_blind_spot"
            )
        movement = float(np.linalg.norm(candidate - position))
        if movement > float(decision.budget_m) + GEOMETRY_EPS:
            return homing_bracket_fallback(state, position, "active_budget_violation")
        evaluation = decision.evaluation
        print(
            f"  [{state.channel}] Q4 active step {step + 1} "
            f"({'route-cache' if cached else 'replan'}): "
            f"budget={decision.budget_m:.0f} m, move={movement:.1f} m, "
            f"Qrho={_evaluation_value(evaluation, 'q_rho_m'):.1f} m, "
            f"point=({candidate[0]:.1f}, {candidate[1]:.1f})"
        )
        response = measure_and_update(state, candidate)
        if response is None:
            return homing_bracket_fallback(state, position, "active_measure_failed")
        position = candidate.copy()
        result = response.get("measure_result")
        if result == "near":
            if try_clear(state.channel, position, "q4-active-near"):
                return True
            return homing_bracket_fallback(state, position, "active_near_clear_failed")
        if result == "direction":
            continue
        if result == "no_signal":
            blind_points.append(position.copy())
            reacquired = directional_reacquisition(state, position)
            if reacquired.cleared:
                return True
            if reacquired.recovered:
                position = np.asarray(reacquired.position, dtype=float)
                continue
            return homing_bracket_fallback(
                state, position, "active_directional_reacquisition_failed"
            )
        return homing_bracket_fallback(state, position, "active_unexpected_result")

    return homing_bracket_fallback(state, position, "active_step_guard")


def process_channel(
    state,
    mode="normal",
    *,
    active_config=None,
    planner=None,
    initial_active_decision=_ACTIVE_DECISION_UNSET,
    initial_active_failure_reason=None,
    initial_active_fallback_plan=None,
):
    """Q3 channel processor with only directional no-signal branches changed."""
    state.status = "processing"
    state.refresh()
    if not state.observations:
        if state.last_position is None:
            return _fail(state, "no_observation")
        response = measure_and_update(state, state.last_position)
        if response is None:
            return _fail(state, "measure_network_or_rejected")
        if response.get("measure_result") == "near":
            if try_clear(state.channel, state.last_position, "q4-rediscovered-near"):
                return True
            return _fail(state, "near_clear_failed")
        if response.get("measure_result") != "direction":
            return _fail(state, "no_signal_without_bearing")

    state.refresh()
    if len(state.polygon) == 0:
        if not state.bearings:
            return _fail(state, "empty_region_without_bearing")
        return homing_clear(
            state, state.observations[0], state.bearings[0], MAX_RECEPTION_RADIUS
        )

    print(
        f"  [{state.channel}] D={state.diameter:.1f} m, "
        f"rho={state.mec_radius:.1f} m ({len(state.observations)} observations)"
    )
    plan = channel_action_plan(state)
    if plan.guaranteed:
        if try_clear(state.channel, plan.point, "q4-mec"):
            return True
        response = measure_and_update(state, plan.point)
        if response is not None and response.get("measure_result") == "near":
            if try_clear(state.channel, plan.point, "q4-mec-near"):
                return True
        if response is not None and response.get("measure_result") == "no_signal":
            recovered = directional_reacquisition(state, plan.point)
            if recovered.cleared:
                return True
        return iterative_clear(state)

    if plan.speculative:
        if try_clear(state.channel, plan.point, "q4-speculative-centroid"):
            return True
        state.last_failure_reason = "speculative_clear_failed"
        response = measure_and_update(state, plan.point)
        if response is None:
            return _fail(state, "speculative_clear_measure_network_or_rejected")
        result = response.get("measure_result")
        if result == "near":
            if try_clear(state.channel, plan.point, "q4-speculative-near"):
                return True
            return _fail(state, "speculative_near_clear_failed")
        if result == "direction":
            return iterative_clear(state)
        if result == "no_signal":
            recovered = directional_reacquisition(state, plan.point)
            if recovered.cleared:
                return True
            if recovered.recovered:
                return iterative_clear(state)
            return homing_bracket_fallback(
                state, plan.point, "speculative_directional_reacquisition_failed"
            )
        return _fail(state, "speculative_clear_unexpected_result")

    if mode == "conservative":
        return iterative_clear(state)
    if mode == "normal":
        return active_localization_clear(
            state,
            _q3.CURRENT_POS.copy(),
            config=active_config,
            planner=planner,
            initial_decision=initial_active_decision,
            initial_failure_reason=initial_active_failure_reason,
            initial_fallback_plan=initial_active_fallback_plan,
        )
    join, theta, bracket = _homing_join(state)
    print(f"  [{state.channel}] using {mode} Q4 homing/bracket fallback")
    return homing_clear(state, join, theta, bracket)


def process_pending_channels(
    states,
    cleared,
    start_time=None,
    processor=None,
    runtime_limit_s=None,
    *,
    active_config=None,
    planner=None,
):
    """Inherited bounded retry queue with the Q4 processor injected."""
    if processor is None:
        processor = process_channel
    if start_time is None:
        start_time = time.monotonic()
    pending = [state for state in states.values() if state.channel not in cleared]
    for state in pending:
        state.status = "retry_pending" if state.retry_count else "discovered"

    while pending:
        if (
            runtime_limit_s is not None
            and time.monotonic() - start_time > float(runtime_limit_s)
        ):
            for state in pending:
                state.status = "retry_pending"
                state.last_failure_reason = "real_time_guard"
            break
        tasks = [
            channel_service_task(
                state,
                _q3.CURRENT_POS,
                mode=retry_mode(state),
                active_config=active_config,
                planner=planner,
            )
            for state in pending
        ]
        order = service_route_exact(tasks, _q3.CURRENT_POS)
        selected_index = order[0]
        task = tasks[selected_index]
        state = pending.pop(selected_index)
        mode = retry_mode(state)
        print(
            f"\n=== processing Q4 channel {state.channel} "
            f"(attempt {state.retry_count + 1}, mode={mode}) ==="
        )
        state.status = "processing"
        if processor is process_channel:
            result = process_channel(
                state,
                mode,
                active_config=active_config,
                planner=planner,
                initial_active_decision=(
                    task.active_decision if task.active_plan_ready else _ACTIVE_DECISION_UNSET
                ),
                initial_active_failure_reason=task.active_failure_reason,
                initial_active_fallback_plan=task.fallback_plan,
            )
        else:
            result = _invoke_processor(processor, state, mode)
        if result:
            state.status = "cleared"
            state.last_failure_reason = None
            cleared.add(state.channel)
            continue
        if not state.last_failure_reason:
            state.last_failure_reason = "processor_failed"
        state.retry_count += 1
        if state.retry_count < MAX_CHANNEL_ATTEMPTS:
            state.status = "retry_pending"
            pending.append(state)
            print(f"  [{state.channel}] retry pending: {state.last_failure_reason}")
        else:
            state.status = "failed"
            print(
                f"  [{state.channel}] bounded Q4 fallbacks exhausted: "
                f"{state.last_failure_reason}"
            )
    return [state for state in states.values() if state.channel not in cleared]


@dataclass(frozen=True)
class DirectionalDiscoveryResult:
    unseen_channels: frozenset
    unresolved_measurements: dict
    completed: bool
    visited_probes: int


def directional_discovery(
    unseen_channels,
    states,
    cleared,
    *,
    scan_direction=1,
    start_time=None,
    runtime_limit_s=None,
    tried_enroute=None,
):
    """Run angular probes and reuse Q3's cheap en-route clear opportunities."""
    unseen = set(unseen_channels) - set(states) - set(cleared)
    failures = {}
    completed = True
    visited = 0
    probes = directional_discovery_positions(scan_direction)
    for probe_index, position in enumerate(probes):
        if not unseen:
            break
        if (
            start_time is not None
            and runtime_limit_s is not None
            and time.monotonic() - start_time > float(runtime_limit_s)
        ):
            completed = False
            break
        visited += 1
        print(
            f"\ndirectional discovery {probe_index + 1}/{len(probes)}: "
            f"({position[0]:.0f}, {position[1]:.0f}), "
            f"{len(unseen)} unseen channels"
        )
        channel_order = sorted(tuple(unseen), reverse=bool(probe_index % 2))
        for channel in channel_order:
            response = measure(position[0], position[1], channel)
            if not _record_discovery_measurement(
                states, cleared, channel, position, response
            ):
                failures[(probe_index, channel)] = np.asarray(position, dtype=float).copy()
                continue
            if channel in states or channel in cleared:
                unseen.discard(channel)

        # The directional mesh has long inter-probe legs.  Reuse the inherited
        # Q3 one-shot/cumulative-detour policy instead of postponing every
        # already-localized channel until the mesh is complete.
        if tried_enroute is not None and probe_index + 1 < len(probes):
            next_probe = np.asarray(probes[probe_index + 1], dtype=float)
            leg_context = {
                "origin": np.asarray(_q3.CURRENT_POS, dtype=float).copy(),
                "next_scan": next_probe.copy(),
                "path_m": 0.0,
                "budget_m": ENROUTE_CUMULATIVE_BUDGET,
            }
            while True:
                candidate, _, _ = select_enroute_candidate(
                    states,
                    cleared,
                    _q3.CURRENT_POS,
                    next_probe,
                    tried_enroute,
                    leg_origin=leg_context["origin"],
                    leg_path_m=leg_context["path_m"],
                    cumulative_budget=leg_context["budget_m"],
                )
                if candidate is None:
                    break
                _attempt_and_record_enroute(
                    candidate,
                    _q3.CURRENT_POS,
                    next_probe,
                    tried_enroute,
                    cleared,
                    leg_context=leg_context,
                )

    remaining_failures = {}
    for key, position in failures.items():
        _, channel = key
        if channel in states or channel in cleared:
            continue
        response = measure(position[0], position[1], channel)
        if not _record_discovery_measurement(states, cleared, channel, position, response):
            remaining_failures[key] = position.copy()
        if channel in states or channel in cleared:
            unseen.discard(channel)
    return DirectionalDiscoveryResult(
        frozenset(unseen),
        remaining_failures,
        completed and not remaining_failures,
        visited,
    )


def main():
    start_time = time.monotonic()
    response = _track(_post("/enter", base(_uid("enter-q4-active"))))
    if not response or response.get("accepted") is not True:
        print("enter failed")
        return
    runtime_limit_s = _runtime_limit_from_enter(response)
    print(
        f"Q4 real-time budget: {runtime_limit_s:.1f} s "
        f"(/enter remaining minus {EXIT_TIME_RESERVE_S:.1f} s exit reserve)"
    )
    if runtime_limit_s <= 0.0:
        print("no usable real time remains; exiting without starting discovery")
        _track(_post("/exit", base(_uid("exit-q4-no-time"))))
        return
    certificate = directional_coverage_certificate()
    print(
        "entered: Q4 Active (Q3 Active V2 + directional discovery/reacquisition)"
    )
    print(
        "directional mesh: "
        f"{certificate['full_probe_count']} total probes, "
        f"distance margin {certificate['distance_margin_m']:.2f} m, "
        f"target margin {certificate['target_margin_m']:.2f} m"
    )

    cleared = set()
    states = {}
    discovery_failures = {}
    scan_completed = True

    def scan_point(position, index, next_scan=None, planned_enroute=None):
        print(f"\nscan {index + 1}/{SCAN_N}: ({position[0]:.0f}, {position[1]:.0f})")
        planned_channels = (
            {planned_enroute.channel} if planned_enroute is not None else set()
        )
        for channel in scan_channel_order(index):
            if should_skip_scan(
                channel,
                states,
                cleared,
                position,
                next_scan=next_scan,
                current_position=position,
                planned_enroute_channels=planned_channels,
            ):
                continue
            result = measure(position[0], position[1], channel)
            if not _record_discovery_measurement(states, cleared, channel, position, result):
                discovery_failures[(index, channel)] = np.asarray(position, dtype=float).copy()

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

    tried_enroute = set()
    for index in range(1, SCAN_N):
        if time.monotonic() - start_time > runtime_limit_s:
            scan_completed = False
            break
        position = scan_position(index, scan_direction)
        next_scan = (
            scan_position(index + 1, scan_direction) if index < SCAN_N - 1 else None
        )
        planned_enroute = None
        if index >= 2 and next_scan is not None:
            planned_enroute, _, _ = select_enroute_candidate(
                states, cleared, position, next_scan, tried_enroute
            )
        scan_point(position, index, next_scan, planned_enroute)
        if index < 2 or next_scan is None:
            continue
        leg_context = {
            "origin": np.asarray(_q3.CURRENT_POS, dtype=float).copy(),
            "next_scan": np.asarray(next_scan, dtype=float).copy(),
            "path_m": 0.0,
            "budget_m": ENROUTE_CUMULATIVE_BUDGET,
        }
        if planned_enroute is not None:
            _attempt_and_record_enroute(
                planned_enroute,
                _q3.CURRENT_POS,
                next_scan,
                tried_enroute,
                cleared,
                leg_context=leg_context,
            )
        while True:
            candidate, _, _ = select_enroute_candidate(
                states,
                cleared,
                _q3.CURRENT_POS,
                next_scan,
                tried_enroute,
                leg_origin=leg_context["origin"],
                leg_path_m=leg_context["path_m"],
                cumulative_budget=leg_context["budget_m"],
            )
            if candidate is None:
                break
            _attempt_and_record_enroute(
                candidate,
                _q3.CURRENT_POS,
                next_scan,
                tried_enroute,
                cleared,
                leg_context=leg_context,
            )

    if time.monotonic() - start_time <= runtime_limit_s:
        discovery_failures = retry_discovery_failures(
            discovery_failures, states, cleared
        )
    elif discovery_failures:
        scan_completed = False
        print("real-time guard reached before discovery retries")
    unseen = set(range(1, 21)) - set(states) - set(cleared)
    directional = directional_discovery(
        unseen,
        states,
        cleared,
        scan_direction=scan_direction,
        start_time=start_time,
        runtime_limit_s=runtime_limit_s,
        tried_enroute=tried_enroute,
    )
    # A failed inner-ring measurement is irrelevant once that channel is seen
    # elsewhere, but remains explicit for a still-unseen channel because it
    # weakens the angular coverage certificate.
    discovery_failures = {
        key: value
        for key, value in discovery_failures.items()
        if key[1] not in states and key[1] not in cleared
    }
    unseen_label = (
        "certified unseen/empty"
        if directional.completed
        and directional.visited_probes == certificate["supplemental_probe_count"]
        else "still unseen (coverage incomplete)"
    )
    print(
        f"\ndiscovery complete: {len(states)} channels discovered, "
        f"{len(cleared)} cleared, {len(directional.unseen_channels)} channels "
        f"{unseen_label}, virtual time {_q3.LAST_VT:.0f} s"
    )

    unresolved = process_pending_channels(
        states,
        cleared,
        start_time,
        runtime_limit_s=runtime_limit_s,
    )
    directional_probe_requirement_met = (
        not directional.unseen_channels
        or directional.visited_probes == certificate["supplemental_probe_count"]
    )
    discovery_complete = (
        scan_completed
        and not discovery_failures
        and directional.completed
        and directional_probe_requirement_met
    )
    success = discovery_complete and set(states) == cleared
    if success:
        print(f"\nsuccess: all {len(cleared)} discovered Q4 channels cleared")
    else:
        print("\nnot successful: Q4 discovery or channel processing remains incomplete")
        if discovery_failures:
            print("unresolved inner discovery measurements: " + str(sorted(discovery_failures)))
        if directional.unresolved_measurements:
            print(
                "unresolved directional discovery measurements: "
                + str(sorted(directional.unresolved_measurements))
            )
        print(f"unresolved channels: {sorted(state.channel for state in unresolved)}")
    print(f"cleared channels: {sorted(cleared)}")
    print(f"total virtual time: {_q3.LAST_VT:.1f} s ({_q3.LAST_VT / 60.0:.1f} min)")
    _track(_post("/exit", base(_uid("exit-q4-active"))))


if __name__ == "__main__":
    _install_official_runtime()
    main()
