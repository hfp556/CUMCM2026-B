"""Q3 Active V2: latest Phase 1 controller plus the active processing layer.

Scanning, caching, speculative/en-route clear, service routing and retries are
preserved from Q3_phase1. Normal unresolved localization uses the bounded
Q3_active planner. All actions share this module\'s single position state.
"""

from dataclasses import dataclass, field
import inspect
import itertools
import math
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from api_utils import (
        base,
        configure as configure_api,
        install_console_log,
        post,
        runtime_limit_from_enter,
    )
    from geo_common import (
        convex_diameter,
        localize_region,
        minimum_distance_to_polygon,
        minimum_enclosing_circle,
    )
except ImportError:  # Allow ``import src.new.Q3_phase1`` in offline tests.
    from .api_utils import (
        base,
        configure as configure_api,
        install_console_log,
        post,
        runtime_limit_from_enter,
    )
    from .geo_common import (
        convex_diameter,
        localize_region,
        minimum_distance_to_polygon,
        minimum_enclosing_circle,
    )



try:
    from .q3_active_localization import BudgetChoice, CandidateEvaluation, solve_pareto
except ImportError:
    from q3_active_localization import BudgetChoice, CandidateEvaluation, solve_pareto

TARGET_RADIUS = 1800.0
CLEAR_RADIUS = 20.0
CLEAR_MAX_DISTANCE = TARGET_RADIUS + CLEAR_RADIUS
MIN_RECEPTION_RADIUS = 1000.0
MAX_RECEPTION_RADIUS = 1500.0
ANGLE_ERROR_DEG = 1.0
GEOMETRY_EPS = 1e-9
MAX_COORDINATE_ABS = 2_000_000.0  # simulator interface limit per coordinate
EXIT_TIME_RESERVE_S = 2.0

SCAN_N = 7
SCAN_ANGLE_DEG = 360.0 / SCAN_N
# For a boundary point halfway between adjacent regular-n-gon scan points,
# d² = TARGET_RADIUS² + r² - 2*TARGET_RADIUS*r*cos(pi/n).  The smaller
# equality root is the approximately 997.201 m critical radius.
SCAN_R_CRITICAL = (
    TARGET_RADIUS * math.cos(math.pi / SCAN_N)
    - math.sqrt(
        MIN_RECEPTION_RADIUS**2
        - (TARGET_RADIUS * math.sin(math.pi / SCAN_N)) ** 2
    )
)
# Moving a little outward is the safe side of the smaller root and still keeps
# the origin strictly inside every 1000 m detection disk.
SCAN_COVERAGE_SAFETY_M = 0.10
SCAN_R = SCAN_R_CRITICAL + SCAN_COVERAGE_SAFETY_M

HOMING_STEP = 200.0
BRACKET_END = 36.0
ENROUTE_DETOUR = 250.0
MAX_CHANNEL_ATTEMPTS = 3
# These two values are deliberately kept separate: 90 m restores the
# baseline's low-cost *speculative* centroid attempt, while 60 m is only used
# as a scan skip when that attempt is already reserved for this route step.
SPECULATIVE_DIRECT_DIAMETER = 90.0
SMALL_DIAMETER_SKIP = 60.0
# A single candidate is still limited by ENROUTE_DETOUR.  This independent
# per-leg cap prevents several individually cheap detours from chaining into
# an unbounded route extension.
ENROUTE_CUMULATIVE_BUDGET = 250.0

_req_counter = itertools.count()
LAST_VT = 0.0
CURRENT_POS = np.array([0.0, 0.0])
_ACTIVE_DECISION_UNSET = object()


@dataclass
class ChannelState:
    channel: int
    observations: list = field(default_factory=list)
    bearings: list = field(default_factory=list)
    status: str = "discovered"
    retry_count: int = 0
    last_failure_reason: str | None = None
    last_position: np.ndarray | None = None
    polygon: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    diameter: float = float("inf")
    mec_center: np.ndarray | None = None
    mec_radius: float = float("inf")
    # Cumulative count across distinct observation/bearing snapshots.
    recovery_dropped_constraints: int = 0
    # A scan-stage en-route clear is intentionally one-shot.  A failed
    # speculative attempt must not suppress later measurements or trigger a
    # homing/bracket detour while the discovery scan is still running.
    enroute_clear_attempted: bool = False
    enroute_clear_failed: bool = False
    _last_region_input_signature: tuple | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def refresh(self):
        # A precomputed polygon is useful for deterministic/offline callers
        # that do not have raw observations.  Live channels always take the
        # observation path below, so this does not alter the normal flow.
        if not self.observations and len(self.polygon) > 0:
            self.diameter, _ = convex_diameter(self.polygon)
            self.mec_center, self.mec_radius = minimum_enclosing_circle(self.polygon)
            self._last_region_input_signature = None
            return
        signature = _region_input_signature(self.observations, self.bearings)
        # Geometry is expensive (circle clipping, diameter and MEC).  A
        # repeated refresh with the same value snapshot cannot change any of
        # those values, so keep the previous result.  The signature is based
        # on values rather than list identity so external test/recovery code
        # that appends an observation still invalidates the cache.
        if signature == self._last_region_input_signature:
            return
        diagnostics = {}
        self.polygon = region_of(
            self.observations, self.bearings, diagnostics=diagnostics
        )
        if signature != self._last_region_input_signature:
            self.recovery_dropped_constraints += int(
                diagnostics.get("dropped_constraints", 0)
            )
            self._last_region_input_signature = signature
        if len(self.polygon) == 0:
            self.diameter = float("inf")
            self.mec_center = None
            self.mec_radius = float("inf")
            return
        self.diameter, _ = convex_diameter(self.polygon)
        self.mec_center, self.mec_radius = minimum_enclosing_circle(self.polygon)


@dataclass(frozen=True)
class ChannelActionPlan:
    """The first action and route proxy for one channel.

    ``guaranteed_clear`` is the strict MEC case.  ``speculative_clear`` is
    the restored D<=90 centroid attempt and is intentionally not a theorem.
    Larger regions use the same homing join point that ``process_channel``
    actually sends to the simulator.
    """

    point: np.ndarray
    action: str
    guaranteed: bool = False
    speculative: bool = False


@dataclass(frozen=True)
class ChannelServiceTask:
    """Directed post-processing task used by the open-path route planner.

    ``active_decision`` caches the first active measurement selected from the
    real route origin.  Only that first step is modelled; the inexpensive
    polygon centroid remains the predicted service exit and the route is
    rebuilt from the actual robot position after every channel.
    """

    state: ChannelState
    entry: np.ndarray
    exit: np.ndarray
    action: str
    active_decision: object = None
    active_plan_ready: bool = False
    active_failure_reason: str | None = None
    fallback_plan: object = None

def _uid(prefix):
    return f"{prefix}-{next(_req_counter)}-{int(time.time() * 1000)}"


def _install_official_runtime():
    """Prepare shared official transport/logging for the Q3 script entry."""
    details = configure_api(run_name="q3_active_v2")
    install_console_log()
    print(f"Q3 official simulator endpoint: {details['base_url']}")
    print(f"Q3 request log: {details['jsonl_path']}")
    print(f"Q3 console log: {details['console_path']}")
    return details


def _runtime_limit_from_enter(response):
    return runtime_limit_from_enter(response, reserve_s=EXIT_TIME_RESERVE_S)


def _post(path, payload, retries=2):
    """Retry only transport failures, reusing the same idempotent payload."""
    for attempt in range(retries + 1):
        try:
            response = post(path, payload)
        except Exception:
            # ``api_utils.post`` already converts ordinary HTTP failures to
            # ``None``; keeping this guard here also makes transport mocks and
            # alternate clients obey the same bounded retry contract.
            response = None
        if response is not None:
            return response
        if attempt < retries:
            time.sleep(0.3)
    return None


def _track(response):
    global LAST_VT
    if response is not None and response.get("accepted") is True:
        virtual_time = response.get("virtual_time_s")
        if virtual_time is not None and float(virtual_time) > LAST_VT:
            LAST_VT = float(virtual_time)
    return response


def _valid_position(position):
    point = np.asarray(position, dtype=float)
    return (
        point.shape == (2,)
        and np.all(np.isfinite(point))
        and bool(np.all(np.abs(point) <= MAX_COORDINATE_ABS))
    )


def measure(x, y, channel):
    """Measure anywhere allowed by the interface; no target-radius restriction."""
    global CURRENT_POS
    position = np.array([float(x), float(y)])
    if not _valid_position(position):
        return None
    payload = base(_uid(f"m-{channel}"))
    payload["position"] = {"x": float(x), "y": float(y)}
    payload["channel"] = channel
    response = _track(_post("/measure", payload))
    if response is None or response.get("accepted") is not True:
        return None
    CURRENT_POS = position
    return response


def try_clear(channel, position, tag=""):
    """Attempt clear only where a source inside the target disk could exist."""
    global CURRENT_POS
    point = np.asarray(position, dtype=float)
    if not _valid_position(point):
        return False
    if float(np.linalg.norm(point)) > CLEAR_MAX_DISTANCE + GEOMETRY_EPS:
        return False
    payload = base(_uid(f"clear-{tag}-{channel}"))
    payload["position"] = {"x": float(point[0]), "y": float(point[1])}
    payload["channel"] = channel
    response = _track(_post("/clear", payload))
    if response is None or response.get("accepted") is not True:
        return False
    CURRENT_POS = point.copy()
    return response.get("clear_result") == "success"


def uvec(degrees):
    angle = math.radians(degrees)
    return np.array([math.cos(angle), math.sin(angle)])


def angdiff(first, second):
    difference = (first - second + 180.0) % 360.0 - 180.0
    return 180.0 if difference == -180.0 else difference


def scan_position(index, direction=1):
    angle = math.radians(SCAN_ANGLE_DEG * index * direction)
    return SCAN_R * np.array([math.cos(angle), math.sin(angle)])


def worst_regular_scan_distance():
    """Analytic worst coverage distance over the target disk."""
    boundary_gap = math.sqrt(
        TARGET_RADIUS**2
        + SCAN_R**2
        - 2.0 * TARGET_RADIUS * SCAN_R * math.cos(math.pi / SCAN_N)
    )
    return max(SCAN_R, boundary_gap)


def _region_input_signature(observations, bearings):
    """Return a stable value snapshot for deduplicating repeated refreshes."""
    point_signature = []
    for point in observations:
        array = np.asarray(point, dtype=float)
        point_signature.append((array.shape, array.tobytes()))
    return tuple(point_signature), tuple(float(angle) for angle in bearings)


def region_of(observations, bearings, diagnostics=None):
    """Intersect observations, dropping oldest constraints only in recovery.

    When supplied, ``diagnostics`` receives the number of discarded oldest
    constraints under the ``dropped_constraints`` key.
    """
    points = [np.asarray(point, dtype=float) for point in observations]
    angles = list(bearings)
    dropped_constraints = 0
    if diagnostics is not None:
        diagnostics["dropped_constraints"] = 0
    while points:
        polygon = localize_region(points, angles)
        if len(polygon) > 0:
            if diagnostics is not None:
                diagnostics["dropped_constraints"] = dropped_constraints
            return polygon
        points = points[1:]
        angles = angles[1:]
        dropped_constraints += 1
    if diagnostics is not None:
        diagnostics["dropped_constraints"] = dropped_constraints
    return np.empty((0, 2))


def add_observation(state, position, bearing):
    """Single entry point for every valid direction observation and state update."""
    point = np.asarray(position, dtype=float)
    angle = float(bearing)
    if not _valid_position(point) or not math.isfinite(angle):
        raise ValueError("observation position and bearing must be finite")
    state.observations.append(point.copy())
    state.bearings.append(angle)
    state.last_position = point.copy()
    state.refresh()


def measure_and_update(state, position):
    point = np.asarray(position, dtype=float)
    response = measure(point[0], point[1], state.channel)
    if response is None:
        return None
    state.last_position = point.copy()
    if response.get("measure_result") == "direction":
        add_observation(state, point, response["svd_deg"])
    return response


def guaranteed_clear_circle(state):
    state.refresh()
    if (
        state.mec_center is None
        or not math.isfinite(float(state.mec_radius))
        or state.mec_radius > CLEAR_RADIUS + GEOMETRY_EPS
    ):
        return None
    return state.mec_center.copy(), state.mec_radius


def should_skip_scan(
    channel,
    states,
    cleared,
    position,
    *,
    next_scan=None,
    current_position=None,
    planned_enroute_channels=None,
):
    """Return whether this scan request is provably or deliberately skipped.

    The strict distance test is independent of route planning.  The old
    ``D<=60`` shortcut is intentionally narrower: it is allowed only for a
    channel reserved for the immediately following en-route clear, so a
    failed/abandoned speculative attempt cannot silently remove future
    observations.
    """
    if channel in cleared:
        return True
    state = states.get(channel)
    if state is None or not state.observations:
        return False
    state.refresh()
    if len(state.polygon) == 0:
        return False
    # Vertex distances are insufficient: an edge interior can be much closer.
    # This strict impossibility test remains valid for every region size.
    if minimum_distance_to_polygon(position, state.polygon) > MAX_RECEPTION_RADIUS:
        return True

    if state.diameter > SMALL_DIAMETER_SKIP:
        return False
    if state.enroute_clear_failed or state.enroute_clear_attempted:
        return False
    if next_scan is None or current_position is None:
        return False
    planned = set(planned_enroute_channels or ())
    if channel not in planned:
        return False
    plan = channel_action_plan(state)
    if not plan.speculative and not plan.guaranteed:
        return False
    return _enroute_detour(plan.point, current_position, next_scan) <= (
        ENROUTE_DETOUR + GEOMETRY_EPS
    )


def _fail(state, reason):
    state.last_failure_reason = reason
    if state.status == "processing":
        state.status = "retry_pending"
    return False


def iterative_clear(state, max_iter=8):
    """Bounded conservative fallback; direct clear uses only the MEC guarantee."""
    for iteration in range(max_iter):
        state.refresh()
        if len(state.polygon) == 0:
            return _fail(state, "empty_region")
        print(
            f"  [{state.channel}] fallback iteration {iteration + 1}: "
            f"D={state.diameter:.1f} m, rho={state.mec_radius:.1f} m"
        )
        clear_circle = guaranteed_clear_circle(state)
        if clear_circle is not None:
            center, _ = clear_circle
            if try_clear(state.channel, center, "mec-iter"):
                return True
            response = measure_and_update(state, center)
        else:
            # This is a retained fallback probe, not the Phase 2 active planner.
            response = measure_and_update(state, state.mec_center)

        if response is None:
            return _fail(state, "measure_network_or_rejected")
        result = response.get("measure_result")
        if result == "near":
            if try_clear(state.channel, state.last_position, "iter-near"):
                return True
            return _fail(state, "near_clear_failed")
        if result == "direction":
            continue  # measure_and_update already recorded and recomputed Omega

        if not state.bearings:
            return _fail(state, "no_signal_without_bearing")
        probe = state.last_position + 100.0 * uvec(state.bearings[-1])
        second = measure_and_update(state, probe)
        if second is None:
            return _fail(state, "second_measure_network_or_rejected")
        if second.get("measure_result") == "near":
            if try_clear(state.channel, probe, "iter-advance-near"):
                return True
            return _fail(state, "near_clear_failed")
        if second.get("measure_result") == "direction":
            # Critical Phase 1 fix: preserve the new bearing and continue planning.
            continue
        return _fail(state, "repeated_no_signal")
    return _fail(state, "iterative_limit")


def bracket_clear(state, position, theta, bracket):
    cur = np.asarray(position, dtype=float)
    cur_theta = float(theta)
    for _ in range(10):
        if bracket <= BRACKET_END:
            final = cur + (bracket / 2.0) * uvec(cur_theta)
            if try_clear(state.channel, final, "bracket-end"):
                return True
            response = measure_and_update(state, final)
            if response is None:
                return _fail(state, "bracket_measure_network_or_rejected")
            if response.get("measure_result") == "near":
                if try_clear(state.channel, final, "bracket-near"):
                    return True
                return _fail(state, "near_clear_failed")
            # A direction has already been retained; let the common fallback replan.
            return iterative_clear(state)

        step = min(bracket, max(BRACKET_END, bracket / 2.0))
        nxt = cur + step * uvec(cur_theta)
        response = measure_and_update(state, nxt)
        if response is None:
            return iterative_clear(state)
        result = response.get("measure_result")
        if result == "near":
            if try_clear(state.channel, nxt, "bracket-near"):
                return True
            return iterative_clear(state)
        if result == "direction":
            theta_new = state.bearings[-1]
            bracket = step if abs(angdiff(theta_new, cur_theta)) > 90.0 else max(
                bracket - step, 0.0
            )
            cur, cur_theta = nxt, theta_new
        else:
            cur = nxt
            bracket = max(bracket - step, 0.0)
    return iterative_clear(state)


def homing_clear(
    state,
    start_position,
    theta,
    bracket0,
    *,
    initial_direction_known=False,
):
    """Retained emergency fallback; there is no artificial 2100 m boundary."""
    cur = np.asarray(start_position, dtype=float)
    cur_theta = float(theta)
    if initial_direction_known:
        print(
            f"  [{state.channel}] homing reuses direction at current position; "
            "skipping duplicate measure"
        )
    else:
        response = measure_and_update(state, cur)
        if response is None:
            return _fail(state, "homing_measure_network_or_rejected")
        result = response.get("measure_result")
        if result == "near":
            if try_clear(state.channel, cur, "home-near"):
                return True
            return _fail(state, "near_clear_failed")
        if result == "direction":
            theta_new = state.bearings[-1]
            if abs(angdiff(theta_new, theta)) > 90.0:
                return bracket_clear(state, cur, theta_new, bracket0)
            cur_theta = theta_new

    for _ in range(8):
        nxt = cur + HOMING_STEP * uvec(cur_theta)
        response = measure_and_update(state, nxt)
        if response is None:
            return iterative_clear(state)
        result = response.get("measure_result")
        if result == "near":
            if try_clear(state.channel, nxt, "home-near"):
                return True
            return iterative_clear(state)
        if result == "direction":
            theta_new = state.bearings[-1]
            if abs(angdiff(theta_new, cur_theta)) > 90.0:
                return bracket_clear(state, nxt, theta_new, HOMING_STEP)
            cur, cur_theta = nxt, theta_new
        else:
            cur = nxt
    return iterative_clear(state)


def join_depth(lower, upper):
    """Existing homing fallback heuristic, retained for baseline continuity."""
    return lower + 0.35 * (upper - lower)


def _polygon_centroid(state):
    return np.mean(state.polygon, axis=0).astype(float, copy=False)


def _homing_join(state):
    """Return the join point/heading/bracket used by the homing fallback."""
    if len(state.polygon) == 0 or not state.observations or not state.bearings:
        if state.observations and state.bearings:
            return (
                np.asarray(state.observations[0], dtype=float).copy(),
                float(state.bearings[0]),
                MAX_RECEPTION_RADIUS,
            )
        if state.last_position is not None:
            return np.asarray(state.last_position, dtype=float).copy(), 0.0, 0.0
        return np.array([0.0, 0.0]), 0.0, 0.0

    centroid = _polygon_centroid(state)
    best = min(
        range(len(state.observations)),
        key=lambda index: float(
            np.linalg.norm(np.asarray(state.observations[index]) - centroid)
        ),
    )
    origin = np.asarray(state.observations[best], dtype=float)
    theta = float(state.bearings[best])
    direction = uvec(theta)
    depths = (state.polygon - origin) @ direction
    lower, upper = float(np.min(depths)), float(np.max(depths))
    depth = join_depth(lower, upper)
    return origin + depth * direction, theta, max(depth - lower, 0.0)


def channel_action_plan(state):
    """Choose the next real action and expose the same point to TSP routing.

    Priority is strict MEC clear, then the baseline-compatible D<=90
    speculative centroid clear, then the homing join point.  Keeping this in
    one function prevents the route proxy from drifting away from the point
    ``process_channel`` actually visits.
    """
    state.refresh()
    if (
        state.mec_center is not None
        and math.isfinite(float(state.mec_radius))
        and state.mec_radius <= CLEAR_RADIUS + GEOMETRY_EPS
    ):
        return ChannelActionPlan(
            state.mec_center.copy(),
            "guaranteed_clear",
            guaranteed=True,
        )
    if len(state.polygon) > 0 and state.diameter <= SPECULATIVE_DIRECT_DIAMETER:
        return ChannelActionPlan(
            _polygon_centroid(state),
            "speculative_clear",
            speculative=True,
        )
    point, theta, bracket = _homing_join(state)
    return ChannelActionPlan(point, "homing", guaranteed=False, speculative=False)


def channel_service_task(
    state,
    start_point=None,
    *,
    mode=None,
    active_config=None,
    planner=None,
):
    """Build the directed route task for one pending channel.

    Guaranteed/speculative clear actions finish at their entry point.  A
    normal large-region channel uses the actual first active candidate as its
    route entry only when the finite planner predicts one-measure closure
    (Qrho<=20 m).  Otherwise the homing join remains the stable service-locality
    proxy while the active decision is still cached for execution.  Both
    active and homing are estimated to finish at the current polygon's vertex
    centroid; no multi-step active cost model is asserted here.
    """
    plan = channel_action_plan(state)
    entry = plan.point.copy()
    action = plan.action
    active_decision = None
    active_plan_ready = False
    active_failure_reason = None
    fallback_plan = None

    if mode is None:
        mode = retry_mode(state)
    if plan.action == "homing" and mode == "normal" and start_point is not None:
        active_plan_ready = True
        try:
            active_decision = plan_active_candidate(
                state,
                np.asarray(start_point, dtype=float),
                config=active_config,
                planner=planner,
            )
        except Exception:
            active_failure_reason = "active_planner_exception"
        if active_decision is None:
            if active_failure_reason is None:
                active_failure_reason = "active_no_candidate"
            fallback_plan = _homing_join(state)
            entry = np.asarray(fallback_plan[0], dtype=float).copy()
            action = "active_fallback"
        elif active_decision.point is not None:
            if (
                _evaluation_value(active_decision.evaluation, "q_rho_m")
                <= CLEAR_RADIUS + GEOMETRY_EPS
            ):
                entry = np.asarray(active_decision.point, dtype=float).copy()
                action = "active-one-step"
            else:
                action = "active-via-service-proxy"

    if plan.action == "homing" and len(state.polygon) > 0:
        exit_point = _polygon_centroid(state).copy()
    else:
        exit_point = entry.copy()
    return ChannelServiceTask(
        state,
        entry,
        exit_point,
        action,
        active_decision=active_decision,
        active_plan_ready=active_plan_ready,
        active_failure_reason=active_failure_reason,
        fallback_plan=fallback_plan,
    )


def _enroute_detour(point, current_position, next_scan):
    point = np.asarray(point, dtype=float)
    current_position = np.asarray(current_position, dtype=float)
    next_scan = np.asarray(next_scan, dtype=float)
    return (
        float(np.linalg.norm(current_position - point))
        + float(np.linalg.norm(point - next_scan))
        - float(np.linalg.norm(current_position - next_scan))
    )


def enroute_cumulative_extra(
    point, current_position, next_scan, leg_origin=None, leg_path_m=0.0
):
    """Extra path for a candidate relative to the leg's original direct path.

    ``leg_path_m`` is the actual distance already travelled through earlier
    en-route candidates on this same leg.  If omitted, the current position is
    the leg origin, so the result reduces to the usual single-candidate
    detour.  The candidate-to-``next_scan`` tail is included when deciding
    whether another candidate can still fit within the cumulative budget.
    """
    point = np.asarray(point, dtype=float)
    current_position = np.asarray(current_position, dtype=float)
    next_scan = np.asarray(next_scan, dtype=float)
    if leg_origin is None:
        leg_origin = current_position
    leg_origin = np.asarray(leg_origin, dtype=float)
    direct = float(np.linalg.norm(leg_origin - next_scan))
    return (
        float(leg_path_m)
        + float(np.linalg.norm(current_position - point))
        + float(np.linalg.norm(point - next_scan))
        - direct
    )


def select_enroute_candidate(
    states,
    cleared,
    current_position,
    next_scan,
    tried_enroute=None,
    *,
    leg_origin=None,
    leg_path_m=0.0,
    cumulative_budget=ENROUTE_CUMULATIVE_BUDGET,
):
    """Select one low-cost clear candidate without entering full fallback.

    Both strict MEC and D<=90 centroid actions are eligible.  The returned
    state is not mutated; the caller performs the one-shot attempt.
    """
    tried = set(tried_enroute or ())
    best_state = None
    best_plan = None
    best_detour = float("inf")
    for state in states.values():
        if state.channel in cleared or state.channel in tried:
            continue
        if state.enroute_clear_attempted or state.enroute_clear_failed:
            continue
        plan = channel_action_plan(state)
        if not plan.guaranteed and not plan.speculative:
            continue
        detour = _enroute_detour(plan.point, current_position, next_scan)
        cumulative_extra = enroute_cumulative_extra(
            plan.point,
            current_position,
            next_scan,
            leg_origin=leg_origin,
            leg_path_m=leg_path_m,
        )
        if (
            cumulative_budget is not None
            and cumulative_extra > float(cumulative_budget) + GEOMETRY_EPS
        ):
            continue
        if detour < best_detour:
            best_state, best_plan, best_detour = state, plan, detour
    if best_state is None or best_detour > ENROUTE_DETOUR + GEOMETRY_EPS:
        return None, None, float("inf")
    return best_state, best_plan, best_detour


def attempt_enroute_clear(
    state,
    current_position,
    next_scan,
    *,
    leg_origin=None,
    leg_path_m=0.0,
    cumulative_budget=ENROUTE_CUMULATIVE_BUDGET,
):
    """Make exactly one en-route clear attempt and return its result.

    A failed attempt only marks the channel for later retry.  In particular,
    this function never calls homing/bracket/iterative fallback while the
    discovery scan is active.
    """
    if state.enroute_clear_attempted or state.enroute_clear_failed:
        return False
    plan = channel_action_plan(state)
    if not plan.guaranteed and not plan.speculative:
        return False
    detour = _enroute_detour(plan.point, current_position, next_scan)
    if detour > ENROUTE_DETOUR + GEOMETRY_EPS:
        return False
    cumulative_extra = enroute_cumulative_extra(
        plan.point,
        current_position,
        next_scan,
        leg_origin=leg_origin,
        leg_path_m=leg_path_m,
    )
    if (
        cumulative_budget is not None
        and cumulative_extra > float(cumulative_budget) + GEOMETRY_EPS
    ):
        return False
    state.enroute_clear_attempted = True
    state.status = "processing"
    if try_clear(
        state.channel,
        plan.point,
        "enroute-mec" if plan.guaranteed else "enroute-speculative",
    ):
        state.status = "cleared"
        state.last_failure_reason = None
        return True
    state.enroute_clear_failed = True
    state.last_failure_reason = "enroute_clear_failed"
    state.status = "retry_pending"
    return False


def _attempt_and_record_enroute(
    state,
    current_position,
    next_scan,
    tried_enroute,
    cleared,
    *,
    leg_context=None,
):
    """Record an en-route candidate only after a clear request was issued."""
    plan = channel_action_plan(state)
    before = np.asarray(CURRENT_POS, dtype=float).copy()
    success = attempt_enroute_clear(
        state,
        current_position,
        next_scan,
        leg_origin=None if leg_context is None else leg_context["origin"],
        leg_path_m=0.0 if leg_context is None else leg_context["path_m"],
        cumulative_budget=(
            ENROUTE_CUMULATIVE_BUDGET
            if leg_context is None
            else leg_context.get("budget_m", ENROUTE_CUMULATIVE_BUDGET)
        ),
    )
    if state.enroute_clear_attempted:
        tried_enroute.add(state.channel)
        if leg_context is not None:
            # Charge only confirmed movement.  ``try_clear`` updates
            # CURRENT_POS after an accepted request, including an accepted
            # no_target_in_range result; rejected/transport-failed requests
            # leave it unchanged and therefore consume no route budget.
            after = np.asarray(CURRENT_POS, dtype=float).copy()
            leg_context["path_m"] += float(np.linalg.norm(after - before))
    if success:
        cleared.add(state.channel)
    return success


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


def _has_direction_at_position(state, position):
    """Whether the fixed-error bearing at ``position`` is already recorded."""
    if not state.observations or not state.bearings:
        return False
    point = np.asarray(position, dtype=float)
    observation = np.asarray(state.observations[-1], dtype=float)
    return bool(
        point.shape == (2,)
        and observation.shape == (2,)
        and np.all(np.isfinite(point))
        and np.linalg.norm(point - observation) <= GEOMETRY_EPS
    )


def homing_bracket_fallback(
    state,
    position=None,
    reason="active_fallback",
    *,
    fallback_plan=None,
):
    """Invoke the retained homing/bracket emergency path, boundedly."""
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
        f"  [{state.channel}] active -> homing fallback: {reason} "
        f"at ({point[0]:.1f}, {point[1]:.1f})"
    )
    if not state.bearings:
        try:
            return bool(iterative_clear(state))
        except Exception:
            _fail(state, reason)
            return False

    homing_result = False
    try:
        homing_result = bool(
            homing_clear(
                state,
                point,
                theta,
                bracket,
                initial_direction_known=_has_direction_at_position(state, point),
            )
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
                bracket_clear(state, point, float(state.bearings[-1]), bracket)
            )
        except Exception:
            pass
    _fail(state, reason)
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


def _log_active_decision(state, decision, step, *, cached=False):
    evaluation = decision.evaluation
    movement = _evaluation_value(evaluation, "movement_distance_m")
    q_rho = _evaluation_value(evaluation, "q_rho_m")
    q_diameter = _evaluation_value(evaluation, "q_diameter_m")
    point = np.asarray(decision.point, dtype=float)
    source = "route-cache" if cached else "replan"
    print(
        f"  [{state.channel}] active step {step + 1} ({source}): "
        f"budget={decision.budget_m:.0f} m, move={movement:.1f} m, "
        f"Qrho={q_rho:.1f} m, QD={q_diameter:.1f} m, "
        f"point=({point[0]:.1f}, {point[1]:.1f})"
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

        cached = step == 0 and initial_decision is not _ACTIVE_DECISION_UNSET
        if cached:
            decision = initial_decision
        else:
            try:
                decision = plan_active_candidate(state, position, config, planner)
            except Exception:
                return homing_bracket_fallback(
                    state, position, "active_planner_exception"
                )
        if decision is None or decision.point is None:
            return homing_bracket_fallback(
                state,
                position,
                initial_failure_reason or "active_no_candidate",
                fallback_plan=initial_fallback_plan if cached else None,
            )
        _log_active_decision(state, decision, step, cached=cached)
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
    state.status = "processing"
    state.refresh()

    if not state.observations:
        if state.last_position is None:
            return _fail(state, "no_observation")
        response = measure_and_update(state, state.last_position)
        if response is None:
            return _fail(state, "measure_network_or_rejected")
        if response.get("measure_result") == "near":
            if try_clear(state.channel, state.last_position, "rediscovered-near"):
                return True
            return _fail(state, "near_clear_failed")
        if response.get("measure_result") != "direction":
            return _fail(state, "no_signal_without_bearing")

    state.refresh()
    if len(state.polygon) == 0:
        if not state.observations or not state.bearings:
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
        if try_clear(state.channel, plan.point, "mec"):
            return True
        response = measure_and_update(state, plan.point)
        if response is not None and response.get("measure_result") == "near":
            if try_clear(state.channel, plan.point, "mec-near"):
                return True
        return iterative_clear(state)

    if plan.speculative:
        # Restore the baseline's useful D<=90 direct attempt, but keep its
        # status explicit: unlike MEC rho<=20 this is only a heuristic.
        if try_clear(state.channel, plan.point, "speculative-centroid"):
            return True
        state.last_failure_reason = "speculative_clear_failed"
        response = measure_and_update(state, plan.point)
        if response is None:
            return _fail(state, "speculative_clear_measure_network_or_rejected")
        if response.get("measure_result") == "near":
            if try_clear(state.channel, plan.point, "speculative-near"):
                return True
            return _fail(state, "speculative_near_clear_failed")
        if response.get("measure_result") == "direction":
            # The fresh direction is already in state; let the bounded common
            # fallback continue from the updated posterior.
            return iterative_clear(state)
        return _fail(state, "speculative_clear_no_signal")

    if mode == "conservative":
        return iterative_clear(state)

    if mode == "normal":
        return active_localization_clear(
            state,
            CURRENT_POS.copy(),
            config=active_config,
            planner=planner,
            initial_decision=initial_active_decision,
            initial_failure_reason=initial_active_failure_reason,
            initial_fallback_plan=initial_active_fallback_plan,
        )

    # Retain the latest Phase 1 emergency homing join and bracket.
    join, theta, bracket = _homing_join(state)
    print(f"  [{state.channel}] using {mode} homing/bracket fallback")
    return homing_clear(state, join, theta, bracket)


def retry_mode(state):
    reason = state.last_failure_reason or ""
    if state.retry_count == 0 or "network_or_rejected" in reason:
        return "normal"
    if "empty_region" in reason or "without_bearing" in reason:
        return "emergency"
    if state.retry_count == 1 or "clear_failed" in reason:
        return "conservative"
    return "emergency"


def routing_point(state):
    # Keep TSP's proxy exactly equal to the first point that
    # ``process_channel`` will visit for the current state.
    return channel_action_plan(state).point.copy()


def service_route_exact(tasks, start_point):
    """Held--Karp open path for directed entry/exit service tasks.

    The state cost ends at each task's predicted exit.  Per-task service time
    is omitted because it is independent of permutation; it therefore cannot
    affect the order.  Returning task indices preserves the ordering contract
    of ``tsp_exact`` while using the actual post-service location for every
    transition.
    """
    tasks = list(tasks)
    n = len(tasks)
    if n <= 1:
        return list(range(n))
    entries = [np.asarray(task.entry, dtype=float) for task in tasks]
    exits = [np.asarray(task.exit, dtype=float) for task in tasks]
    start = np.asarray(start_point, dtype=float)
    distances_from_start = np.array(
        [float(np.linalg.norm(start - entry)) for entry in entries]
    )
    distances = np.array(
        [
            [float(np.linalg.norm(exits[i] - entries[j])) for j in range(n)]
            for i in range(n)
        ]
    )
    size = 1 << n
    costs = np.full((size, n), np.inf)
    parents = np.full((size, n), -1, dtype=np.int32)
    for index in range(n):
        costs[1 << index, index] = distances_from_start[index]
    for mask in range(1, size):
        for nxt in range(n):
            if mask & (1 << nxt):
                continue
            candidates = costs[mask] + distances[:, nxt]
            previous = int(np.argmin(candidates))
            new_mask = mask | (1 << nxt)
            if candidates[previous] < costs[new_mask, nxt]:
                costs[new_mask, nxt] = candidates[previous]
                parents[new_mask, nxt] = previous
    mask = size - 1
    current = int(np.argmin(costs[mask]))
    order = []
    while current != -1:
        order.append(current)
        previous = int(parents[mask, current])
        mask ^= 1 << current
        current = previous
    return list(reversed(order))


# Descriptive alias for callers that refer to this planner as a directed TSP.
tsp_service_exact = service_route_exact


def tsp_exact(points, start_point):
    n = len(points)
    if n <= 1:
        return list(range(n))
    points = [np.asarray(point, dtype=float) for point in points]
    distances = np.array(
        [
            [float(np.linalg.norm(points[i] - points[j])) for j in range(n)]
            for i in range(n)
        ]
    )
    size = 1 << n
    costs = np.full((size, n), np.inf)
    parents = np.full((size, n), -1, dtype=np.int32)
    for index in range(n):
        costs[1 << index, index] = float(
            np.linalg.norm(np.asarray(start_point) - points[index])
        )
    for mask in range(1, size):
        for nxt in range(n):
            if mask & (1 << nxt):
                continue
            candidates = costs[mask] + distances[:, nxt]
            previous = int(np.argmin(candidates))
            new_mask = mask | (1 << nxt)
            if candidates[previous] < costs[new_mask, nxt]:
                costs[new_mask, nxt] = candidates[previous]
                parents[new_mask, nxt] = previous
    mask = size - 1
    current = int(np.argmin(costs[mask]))
    order = []
    while current != -1:
        order.append(current)
        previous = int(parents[mask, current])
        mask ^= 1 << current
        current = previous
    return list(reversed(order))


def _invoke_processor(processor, state, mode):
    """Call a retry callback while accepting one-argument test doubles.

    The production callback receives ``(state, mode)`` so the failure reason
    can select a fallback.  A one-argument callback remains useful for small
    offline state-machine tests and does not weaken the production path.
    """
    try:
        signature = inspect.signature(processor)
    except (TypeError, ValueError):
        return processor(state, mode)
    try:
        signature.bind(state, mode)
    except TypeError:
        signature.bind(state)
        return processor(state)
    return processor(state, mode)


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
    """Bounded retry state machine; failed channels are never silently dropped."""
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
                CURRENT_POS,
                mode=retry_mode(state),
                active_config=active_config,
                planner=planner,
            )
            for state in pending
        ]
        order = service_route_exact(tasks, CURRENT_POS)
        selected_index = order[0]
        task = tasks[selected_index]
        state = pending.pop(selected_index)
        mode = retry_mode(state)
        print(
            f"\n=== processing channel {state.channel} "
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
                    task.active_decision
                    if task.active_plan_ready
                    else _ACTIVE_DECISION_UNSET
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
            print(
                f"  [{state.channel}] retry pending: {state.last_failure_reason}"
            )
        else:
            state.status = "failed"
            print(
                f"  [{state.channel}] bounded fallbacks exhausted: "
                f"{state.last_failure_reason}"
            )
    return [state for state in states.values() if state.channel not in cleared]


def scan_channel_order(scan_index):
    """Return the serpentine channel order for one discovery scan point."""
    if int(scan_index) % 2 == 0:
        return tuple(range(1, 21))
    return tuple(range(20, 0, -1))


def _record_discovery_measurement(states, cleared, channel, position, result):
    """Apply one accepted discovery response; reject malformed responses."""
    if result is None:
        return False
    measurement = result.get("measure_result")
    if measurement == "no_signal":
        return True
    if measurement == "direction":
        bearing = result.get("svd_deg")
        try:
            bearing = float(bearing)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(bearing):
            return False
        state = states.setdefault(channel, ChannelState(channel))
        add_observation(state, position, bearing)
        print(f"  channel {channel}: direction {bearing} deg")
        return True
    if measurement == "near":
        state = states.setdefault(channel, ChannelState(channel))
        state.last_position = np.asarray(position, dtype=float).copy()
        if try_clear(channel, position, "scan-near"):
            state.status = "cleared"
            cleared.add(channel)
        else:
            state.last_failure_reason = "scan_near_clear_failed"
            state.status = "retry_pending"
        return True
    return False


def retry_discovery_failures(failures, states, cleared):
    """Retry each unresolved scan/channel pair once and return what remains."""
    if not failures:
        return {}
    print(f"\nretrying {len(failures)} unresolved discovery measurements")
    remaining = {}
    for key, position in failures.items():
        scan_index, channel = key
        if channel in cleared:
            continue
        print(f"  retry scan {scan_index + 1}, channel {channel}")
        result = measure(position[0], position[1], channel)
        if not _record_discovery_measurement(
            states, cleared, channel, position, result
        ):
            remaining[key] = np.asarray(position, dtype=float).copy()
            print(
                f"  discovery retry unresolved: scan {scan_index + 1}, "
                f"channel {channel}"
            )
    return remaining


def main():
    start_time = time.monotonic()
    response = _track(_post("/enter", base(_uid("enter"))))
    if not response or response.get("accepted") is not True:
        print("enter failed")
        return
    runtime_limit_s = _runtime_limit_from_enter(response)
    print(
        f"Q3 real-time budget: {runtime_limit_s:.1f} s "
        f"(/enter remaining minus {EXIT_TIME_RESERVE_S:.1f} s exit reserve)"
    )
    if runtime_limit_s <= 0.0:
        print("no real execution time remains; exiting")
        _track(_post("/exit", base(_uid("exit-no-time"))))
        return
    print("entered: Q3 Active V2 (Phase 1 safety + active localization)")

    cleared = set()
    states = {}
    discovery_failures = {}
    scan_completed = True

    def scan_point(position, index, next_scan=None, planned_enroute=None):
        print(
            f"\nscan {index + 1}/{SCAN_N}: "
            f"({position[0]:.0f}, {position[1]:.0f})"
        )
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
            if not _record_discovery_measurement(
                states, cleared, channel, position, result
            ):
                discovery_failures[(index, channel)] = np.asarray(
                    position, dtype=float
                ).copy()
                print(
                    f"  channel {channel}: discovery measurement failed; "
                    "queued for retry"
                )

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
        if time.monotonic() - start_time > runtime_limit_s:
            print("real-time guard reached during scanning")
            scan_completed = False
            break
        position = scan_position(index, scan_direction)
        next_scan = (
            scan_position(index + 1, scan_direction)
            if index < SCAN_N - 1
            else None
        )
        # Reserve at most one D<=60 candidate before this scan.  Its scan
        # measurement may be skipped only because this very step will attempt
        # its low-cost clear on the way to ``next_scan``.
        planned_enroute = None
        if index >= 2 and next_scan is not None:
            planned_enroute, _, _ = select_enroute_candidate(
                states,
                cleared,
                position,
                next_scan,
                tried_enroute,
            )
        scan_point(position, index, next_scan, planned_enroute)

        if index < 2 or next_scan is None:
            continue

        # Execute the reserved candidate first so a D<=60 skip is always
        # followed by the promised one-shot clear attempt.  Other D<=90/MEC
        # candidates may then be tried once each while the detour remains low.
        leg_context = {
            "origin": np.asarray(CURRENT_POS, dtype=float).copy(),
            "next_scan": np.asarray(next_scan, dtype=float).copy(),
            "path_m": 0.0,
            "budget_m": ENROUTE_CUMULATIVE_BUDGET,
        }
        if planned_enroute is not None:
            _attempt_and_record_enroute(
                planned_enroute,
                CURRENT_POS,
                next_scan,
                tried_enroute,
                cleared,
                leg_context=leg_context,
            )
        while True:
            candidate, _, best_detour = select_enroute_candidate(
                states,
                cleared,
                CURRENT_POS,
                next_scan,
                tried_enroute,
                leg_origin=leg_context["origin"],
                leg_path_m=leg_context["path_m"],
                cumulative_budget=leg_context["budget_m"],
            )
            if candidate is None:
                break
            print(
                f"en-route clear ({'guaranteed' if candidate.mec_radius <= CLEAR_RADIUS + GEOMETRY_EPS else 'speculative'}): "
                f"channel {candidate.channel}, "
                f"detour {best_detour:.0f} m"
            )
            _attempt_and_record_enroute(
                candidate,
                CURRENT_POS,
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
        print("real-time guard reached before discovery retries")
    print(
        f"\ndiscovery complete: {len(states)} channels discovered, "
        f"{len(cleared)} cleared, {len(discovery_failures)} measurements "
        f"unresolved, virtual time {LAST_VT:.0f} s"
    )
    unresolved = process_pending_channels(
        states,
        cleared,
        start_time,
        runtime_limit_s=runtime_limit_s,
    )

    discovered_channels = set(states)
    discovery_complete = scan_completed and not discovery_failures
    success = discovery_complete and discovered_channels == cleared
    if success:
        print(f"\nsuccess: all {len(cleared)} discovered channels cleared")
    else:
        print("\nnot successful: discovery or channel processing remains incomplete")
        if not scan_completed:
            print("discovery scan did not visit every planned scan point")
        if discovery_failures:
            print(
                "unresolved discovery measurements: "
                + str(sorted(discovery_failures))
            )
        print(f"unresolved channels: {sorted(state.channel for state in unresolved)}")
        for state in unresolved:
            print(
                f"  channel {state.channel}: status={state.status}, "
                f"retries={state.retry_count}, reason={state.last_failure_reason}"
            )
    print(f"cleared channels: {sorted(cleared)}")
    print(f"total virtual time: {LAST_VT:.1f} s ({LAST_VT / 60.0:.1f} min)")
    _track(_post("/exit", base(_uid("exit"))))


if __name__ == "__main__":
    _install_official_runtime()
    main()
