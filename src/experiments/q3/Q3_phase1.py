"""Q3 Phase 1: deterministic fixes on top of the Q3_fast2 baseline.

This file deliberately leaves Q3_fast2.py unchanged for later ablation tests.
It implements only Phase 1 of Q3优化流程.md.  Active-localization point
selection remains a later research task; homing/bracket and centroid probing are
retained only as bounded fallbacks until that work is available.
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
    from src.new.api_utils import base, post
    from src.new.geo_common import (
        convex_diameter,
        localize_region,
        minimum_distance_to_polygon,
        minimum_enclosing_circle,
    )
except ImportError:  # Allow direct execution with ``src/new`` on sys.path.
    from api_utils import base, post
    from geo_common import (
        convex_diameter,
        localize_region,
        minimum_distance_to_polygon,
        minimum_enclosing_circle,
    )


TARGET_RADIUS = 1800.0
CLEAR_RADIUS = 20.0
CLEAR_MAX_DISTANCE = TARGET_RADIUS + CLEAR_RADIUS
MIN_RECEPTION_RADIUS = 1000.0
MAX_RECEPTION_RADIUS = 1500.0
ANGLE_ERROR_DEG = 1.0
GEOMETRY_EPS = 1e-9
MAX_COORDINATE_ABS = 2_000_000.0  # simulator interface limit per coordinate

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
    """Directed post-processing task used by the open-path route planner."""

    state: ChannelState
    entry: np.ndarray
    exit: np.ndarray
    action: str

def _uid(prefix):
    return f"{prefix}-{next(_req_counter)}-{int(time.time() * 1000)}"


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


def homing_clear(state, start_position, theta, bracket0):
    """Retained emergency fallback; there is no artificial 2100 m boundary."""
    cur = np.asarray(start_position, dtype=float)
    cur_theta = float(theta)
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


def channel_service_task(state):
    """Build the directed route task for one pending channel.

    Guaranteed/speculative clear actions finish at their entry point.  A
    homing fallback is estimated to finish at the current polygon's vertex
    centroid, which is the same arithmetic mean used elsewhere in Phase 1.
    """
    plan = channel_action_plan(state)
    entry = plan.point.copy()
    if plan.action == "homing" and len(state.polygon) > 0:
        exit_point = _polygon_centroid(state).copy()
    else:
        exit_point = entry.copy()
    return ChannelServiceTask(state, entry, exit_point, plan.action)


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


def process_channel(state, mode="normal"):
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

    # Phase 2 has not yet supplied an active-localization planner.  Until then,
    # homing/bracket remains an explicit bounded fallback, never a MEC guarantee.
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
    states, cleared, start_time=None, processor=None, runtime_limit_s=17 * 60
):
    """Bounded retry state machine; failed channels are never silently dropped."""
    if processor is None:
        processor = process_channel
    if start_time is None:
        start_time = time.time()
    pending = [state for state in states.values() if state.channel not in cleared]
    for state in pending:
        state.status = "retry_pending" if state.retry_count else "discovered"

    while pending:
        if (
            runtime_limit_s is not None
            and time.time() - start_time > float(runtime_limit_s)
        ):
            for state in pending:
                state.status = "retry_pending"
                state.last_failure_reason = "real_time_guard"
            break
        tasks = [channel_service_task(state) for state in pending]
        order = service_route_exact(tasks, CURRENT_POS)
        state = pending.pop(order[0])
        mode = retry_mode(state)
        print(
            f"\n=== processing channel {state.channel} "
            f"(attempt {state.retry_count + 1}, mode={mode}) ==="
        )
        state.status = "processing"
        if _invoke_processor(processor, state, mode):
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


def main():
    start_time = time.time()
    response = _track(_post("/enter", base(_uid("enter"))))
    if not response or response.get("accepted") is not True:
        print("enter failed")
        return
    print(
        "entered: Q3 Phase 1 (regular heptagon + deterministic fixes + MEC)"
    )

    cleared = set()
    states = {}

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

    print(
        f"\ndiscovery complete: {len(states)} channels discovered, "
        f"{len(cleared)} cleared, virtual time {LAST_VT:.0f} s"
    )
    unresolved = process_pending_channels(states, cleared, start_time)

    discovered_channels = set(states)
    success = discovered_channels == cleared
    if success:
        print(f"\nsuccess: all {len(cleared)} discovered channels cleared")
    else:
        print("\nnot successful: discovered channels remain uncleared")
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
    main()
