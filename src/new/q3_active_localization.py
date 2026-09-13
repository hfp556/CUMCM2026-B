"""Independent Phase 2 active-localization research kernel for Q3.

This module is deliberately independent from ``Q3_phase1.py`` and
``Q3_fast2.py``.  It is a numerical research component only; it does not
send simulator requests and it does not change the formal Q3 controller.

Mathematical model
------------------

Let ``Omega`` be the current convex possible-source region, ``P`` the dog
position, and ``S`` a proposed next measurement point.  The robust candidate
domain is the continuous set

    R(Omega) = { S : max_{G in Omega} ||S-G|| <= 1000 m }.

For a convex polygonal Omega the maximum is attained at a vertex, so the
test in :func:`max_distance_to_region` is an exact test for the polygonal
Omega (the construction of a displayed domain uses a polygonal approximation
of the circular boundaries).

If the true source is ``G`` and the measurement error is ``e`` in
``[-1 deg, +1 deg]``, the posterior used here is

    Omega_after = Omega intersect B(S, 1500)
                 intersect W(S, bearing(G-S)+e, +/-1 deg).

The first ``Omega`` in this expression is intentional: history is retained;
the posterior is never rebuilt from the target disk alone.  The continuous
quality objectives are

    Q_rho(S) = sup_{G in Omega, e in [-1,1]} rho(Omega_after),
    Q_D(S)   = sup_{G in Omega, e in [-1,1]} diam(Omega_after),

where ``rho`` is the minimum-enclosing-circle radius.  The research code
approximates these suprema by deterministic source samples and a finite error
grid, and approximates circular clipping by a high-resolution convex polygon;
therefore its argmin/Pareto results are numerical evidence, not a proof of
continuous global optimality.

No fixed lambda is used.  :func:`solve_pareto` evaluates movement distance,
worst posterior ``Q_rho`` and worst posterior ``Q_D`` separately.  For every
movement budget it selects the smallest ``Q_rho`` (then ``Q_D`` and movement
as deterministic tie-breakers), and also returns the non-dominated frontier.

The single-target simulator in this module is receding-horizon: after each
measurement it replaces Omega with the posterior, recomputes a Pareto choice,
and continues until the MEC radius is at most 20 m.  Its time accounting is
explicit: every measurement costs 5 s, every movement costs distance/5 s,
and the final successful clear costs final movement/5 + 5 s.  The default
offline model has no failed clear attempts; a successful run consequently has
``clear_attempts == 1``.  Random errors are a stable function of
``(error_seed, true_source, measurement_position)`` so different budgets can
be compared on the same error field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

try:  # ``python src/new/q3_active_localization.py``
    from geo_common import (
        clip_polygon_by_convex_polygon,
        clip_polygon_by_halfplane,
        convex_diameter,
        generate_circle_polygon,
        minimum_enclosing_circle,
        point_in_convex_polygon,
    )
except ImportError:  # ``import src.new.q3_active_localization``
    from .geo_common import (
        clip_polygon_by_convex_polygon,
        clip_polygon_by_halfplane,
        convex_diameter,
        generate_circle_polygon,
        minimum_enclosing_circle,
        point_in_convex_polygon,
    )


TARGET_RADIUS = 1800.0
MIN_RECEPTION_RADIUS = 1000.0
MAX_RECEPTION_RADIUS = 1500.0
ANGLE_ERROR_DEG = 1.0
CLEAR_RADIUS = 20.0
# The simulator's ``measure_result == "near"`` threshold is separate from
# the 20 m MEC guarantee radius.  A 6--20 m measurement still supplies a
# direction and must update the posterior.
NEAR_RADIUS = 5.0
SPEED_MPS = 5.0
MEASURE_TIME_S = 5.0
CLEAR_TIME_S = 5.0
GEOMETRY_EPS = 1e-7
DEFAULT_BUDGETS_M = (50.0, 100.0, 150.0, 200.0, 300.0, 400.0)


def _as_point(value, name="point") -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must contain two finite coordinates")
    return point.copy()


def _as_polygon(value, name="omega") -> np.ndarray:
    polygon = np.asarray(value, dtype=float)
    if polygon.size == 0:
        raise ValueError(f"{name} must not be empty")
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    if not np.all(np.isfinite(polygon)):
        raise ValueError(f"{name} must contain finite coordinates")
    # The callers normally provide a convex polygon.  Repeated closing
    # vertices are harmless but make distance and clipping code needlessly
    # awkward, so remove only exact/near duplicates here.
    points = [polygon[0].copy()]
    for point in polygon[1:]:
        if float(np.linalg.norm(point - points[-1])) > GEOMETRY_EPS:
            points.append(point.copy())
    if len(points) > 1 and float(np.linalg.norm(points[0] - points[-1])) <= GEOMETRY_EPS:
        points.pop()
    return np.asarray(points, dtype=float)


def _empty_polygon() -> np.ndarray:
    return np.empty((0, 2), dtype=float)


def _dedupe_points(points: Iterable[Sequence[float]], decimals: int = 7) -> np.ndarray:
    unique = []
    seen = set()
    for value in points:
        point = np.asarray(value, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            continue
        key = tuple(np.round(point, decimals))
        if key not in seen:
            seen.add(key)
            unique.append(point.copy())
    return np.asarray(unique, dtype=float) if unique else _empty_polygon()


def max_distance_to_region(point, omega) -> float:
    """Return ``max_{G in Omega} ||point-G||`` for a convex polygon Omega.

    Squared distance is convex, hence its maximum over a compact convex
    polygon is attained at a vertex.  This is the exact robust-candidate test
    for the supplied polygon representation, not a grid approximation.
    """

    point = _as_point(point)
    polygon = _as_polygon(omega)
    return float(np.max(np.linalg.norm(polygon - point, axis=1)))


def is_robust_candidate(
    point,
    omega,
    min_reception_radius: float = MIN_RECEPTION_RADIUS,
    tolerance: float = GEOMETRY_EPS,
) -> bool:
    """Whether every possible source in polygon ``omega`` is detectable."""

    if min_reception_radius <= 0:
        raise ValueError("min_reception_radius must be positive")
    return max_distance_to_region(point, omega) <= min_reception_radius + tolerance


def robust_candidate_domain(
    omega,
    *,
    min_reception_radius: float = MIN_RECEPTION_RADIUS,
    circle_segments: int = 96,
    max_constraint_vertices: int = 96,
) -> np.ndarray:
    """Return a conservative polygonal display of the robust candidate set.

    Continuously, ``R(Omega) = intersection_{G in Omega} B(G,1000)``.  For a
    convex polygon it is enough to intersect the disks centred at its
    vertices.  The returned polygon approximates each disk by an inscribed
    regular polygon, so it is used only to seed candidates.  Every returned
    candidate is rechecked with :func:`max_distance_to_region` against *all*
    original vertices before being accepted.
    """

    polygon = _as_polygon(omega)
    if min_reception_radius <= 0:
        raise ValueError("min_reception_radius must be positive")
    if circle_segments < 16:
        raise ValueError("circle_segments must be at least 16")
    if max_constraint_vertices < 1:
        raise ValueError("max_constraint_vertices must be positive")

    # A finely sampled localize_region polygon can have hundreds of adjacent
    # vertices.  Reducing only this seed construction keeps the research
    # solver fast; the exact robust check still uses every original vertex.
    if len(polygon) > max_constraint_vertices:
        indices = np.linspace(
            0, len(polygon), max_constraint_vertices, endpoint=False, dtype=int
        )
        constraints = polygon[np.unique(indices)]
    else:
        constraints = polygon

    region = generate_circle_polygon(constraints[0], min_reception_radius, circle_segments)
    for vertex in constraints[1:]:
        disk = generate_circle_polygon(vertex, min_reception_radius, circle_segments)
        region = clip_polygon_by_convex_polygon(region, disk)
        if len(region) == 0:
            return _empty_polygon()
    return region


def _wedge_normals(theta_deg: float, angle_error_deg: float):
    lower = math.radians(theta_deg - angle_error_deg)
    upper = math.radians(theta_deg + angle_error_deg)
    lower_direction = np.array([math.cos(lower), math.sin(lower)])
    upper_direction = np.array([math.cos(upper), math.sin(upper)])
    # W(S,theta,+/-alpha) is to the left of the lower ray and to the right of
    # the upper ray, matching geo_common.localize_region.
    return (
        np.array([-lower_direction[1], lower_direction[0]]),
        np.array([upper_direction[1], -upper_direction[0]]),
    )


def bearing_deg(origin, target) -> float:
    """Return the [0,360) bearing from ``origin`` to ``target``."""

    origin = _as_point(origin, "origin")
    target = _as_point(target, "target")
    vector = target - origin
    if float(np.linalg.norm(vector)) <= GEOMETRY_EPS:
        raise ValueError("bearing is undefined for coincident points")
    return float(math.degrees(math.atan2(vector[1], vector[0])) % 360.0)


def posterior_region(
    omega,
    sensor,
    true_source,
    error_deg: float,
    *,
    max_reception_radius: float = MAX_RECEPTION_RADIUS,
    angle_error_deg: float = ANGLE_ERROR_DEG,
    circle_segments: int = 96,
) -> np.ndarray:
    """Compute the history-preserving posterior for one possible outcome.

    The implementation clips the supplied ``omega`` in place conceptually:
    it never calls ``localize_region`` on the target disk, which would discard
    historical constraints.  The measurement direction is
    ``bearing(sensor,true_source) + error_deg`` and the accepted wedge has
    half-angle ``angle_error_deg``.
    """

    polygon = _as_polygon(omega)
    sensor = _as_point(sensor, "sensor")
    true_source = _as_point(true_source, "true_source")
    if max_reception_radius <= 0:
        raise ValueError("max_reception_radius must be positive")
    if not 0 < angle_error_deg < 90:
        raise ValueError("angle_error_deg must be between 0 and 90 degrees")
    if not -angle_error_deg - GEOMETRY_EPS <= float(error_deg) <= angle_error_deg + GEOMETRY_EPS:
        raise ValueError("error_deg must be within the stated angle-error interval")

    distance = float(np.linalg.norm(true_source - sensor))
    # A near response does not provide a direction wedge.  In the offline
    # simulator it is cleared immediately; returning the history region here
    # is the least surprising mathematical convention for direct callers.
    if distance <= GEOMETRY_EPS:
        return polygon.copy()
    if distance > max_reception_radius + GEOMETRY_EPS:
        return _empty_polygon()

    measured_theta = bearing_deg(sensor, true_source) + float(error_deg)
    lower_normal, upper_normal = _wedge_normals(measured_theta, angle_error_deg)
    posterior = clip_polygon_by_halfplane(polygon, sensor, lower_normal)
    posterior = clip_polygon_by_halfplane(posterior, sensor, upper_normal)
    if len(posterior) == 0:
        return _empty_polygon()
    # geo_common uses a polygonal circle approximation for localize_region as
    # well.  A high segment count keeps the numerical error small and explicit.
    reception_disk = generate_circle_polygon(
        sensor, max_reception_radius, circle_segments
    )
    posterior = clip_polygon_by_convex_polygon(posterior, reception_disk)
    return posterior if len(posterior) else _empty_polygon()


def sample_region_sources(
    omega,
    count: int = 13,
    *,
    seed: int = 0,
    include_boundary: bool = True,
) -> np.ndarray:
    """Deterministically sample possible true sources from a convex polygon.

    Vertices, edge midpoints and the centroid cover thin/boundary regions;
    fixed-seed convex-combination samples add interior coverage.  This is a
    finite approximation to the continuous ``G in Omega`` supremum.
    """

    polygon = _as_polygon(omega)
    if count < 1:
        raise ValueError("count must be positive")
    raw = [np.mean(polygon, axis=0)]
    if include_boundary:
        for index, point in enumerate(polygon):
            raw.append(point)
            raw.append((point + polygon[(index + 1) % len(polygon)]) / 2.0)

    rng = np.random.default_rng(seed)
    if len(polygon) > 1:
        for _ in range(max(0, count * 3)):
            weights = rng.dirichlet(np.ones(len(polygon)))
            raw.append(np.sum(polygon * weights[:, None], axis=0))
    samples = _dedupe_points(raw)
    if len(samples) > count:
        # Preserve the centroid first and spread boundary/interior samples
        # deterministically across the generated list.
        indices = np.linspace(0, len(samples), count, endpoint=False, dtype=int)
        samples = samples[np.unique(indices)]
    return samples


def _posterior_quality(posterior: np.ndarray):
    if len(posterior) == 0:
        return math.inf, math.inf
    radius = float(minimum_enclosing_circle(posterior)[1])
    diameter = float(convex_diameter(posterior)[0])
    return radius, diameter


@dataclass(frozen=True)
class CandidateEvaluation:
    """Discrete estimate of a candidate's worst posterior qualities."""

    point: np.ndarray
    movement_distance_m: float
    max_source_distance_m: float
    robust: bool
    q_rho_m: float
    q_diameter_m: float
    mean_rho_m: float
    median_rho_m: float
    worst_source: Optional[np.ndarray]
    worst_error_deg: Optional[float]
    case_count: int
    nonempty_case_count: int

    @property
    def movement_distance(self) -> float:
        return self.movement_distance_m

    @property
    def worst_rho(self) -> float:
        return self.q_rho_m

    @property
    def worst_diameter(self) -> float:
        return self.q_diameter_m


def evaluate_candidate(
    omega,
    current_position,
    candidate,
    *,
    source_samples: Optional[Sequence[Sequence[float]]] = None,
    source_sample_count: int = 13,
    error_samples: Sequence[float] = (-1.0, 0.0, 1.0),
    min_reception_radius: float = MIN_RECEPTION_RADIUS,
    max_reception_radius: float = MAX_RECEPTION_RADIUS,
    angle_error_deg: float = ANGLE_ERROR_DEG,
    circle_segments: int = 96,
    require_robust: bool = True,
) -> CandidateEvaluation:
    """Evaluate one point by finite approximations to ``Q_rho`` and ``Q_D``.

    Every supplied source sample and every supplied error is enumerated.  A
    candidate outside the robust set may still be evaluated when
    ``require_robust=False``; such an evaluation is marked ``robust=False``
    and is not eligible for :func:`solve_pareto` by default.
    """

    polygon = _as_polygon(omega)
    current = _as_point(current_position, "current_position")
    point = _as_point(candidate, "candidate")
    if min_reception_radius <= 0 or max_reception_radius <= 0:
        raise ValueError("reception radii must be positive")
    if min_reception_radius > max_reception_radius:
        raise ValueError("min_reception_radius cannot exceed max_reception_radius")
    errors = tuple(float(error) for error in error_samples)
    if not errors:
        raise ValueError("error_samples must not be empty")
    if any(abs(error) > angle_error_deg + GEOMETRY_EPS for error in errors):
        raise ValueError("all error_samples must lie within +/- angle_error_deg")

    robust_distance = max_distance_to_region(point, polygon)
    robust = robust_distance <= min_reception_radius + GEOMETRY_EPS
    if require_robust and not robust:
        raise ValueError("candidate is outside the robust detection domain")

    if source_samples is None:
        sources = sample_region_sources(polygon, source_sample_count)
    else:
        sources = _dedupe_points(source_samples)
        if len(sources) == 0:
            raise ValueError("source_samples must contain at least one point")
    # Caller-provided samples are expected to be possible sources.  We do not
    # silently project them into Omega because that could hide a bad experiment.
    for source in sources:
        if not point_in_convex_polygon(source, polygon, tolerance=1e-5):
            raise ValueError("every source sample must lie in omega")

    rho_values = []
    diameter_values = []
    worst_pair = None
    nonempty = 0
    for source in sources:
        source_distance = float(np.linalg.norm(source - point))
        for error in errors:
            if source_distance > max_reception_radius + GEOMETRY_EPS:
                rho, diameter = math.inf, math.inf
            else:
                posterior = posterior_region(
                    polygon,
                    point,
                    source,
                    error,
                    max_reception_radius=max_reception_radius,
                    angle_error_deg=angle_error_deg,
                    circle_segments=circle_segments,
                )
                rho, diameter = _posterior_quality(posterior)
                if len(posterior):
                    nonempty += 1
            rho_values.append(rho)
            diameter_values.append(diameter)
            pair = (rho, diameter)
            if worst_pair is None or pair > worst_pair[:2]:
                worst_pair = (rho, diameter, source.copy(), error)

    if worst_pair is None:
        raise RuntimeError("candidate evaluation produced no posterior cases")
    finite_rho = np.asarray([value for value in rho_values if math.isfinite(value)])
    mean_rho = float(np.mean(finite_rho)) if len(finite_rho) else math.inf
    median_rho = float(np.median(finite_rho)) if len(finite_rho) else math.inf
    return CandidateEvaluation(
        point=point,
        movement_distance_m=float(np.linalg.norm(point - current)),
        max_source_distance_m=robust_distance,
        robust=robust,
        q_rho_m=float(max(rho_values)),
        q_diameter_m=float(max(diameter_values)),
        mean_rho_m=mean_rho,
        median_rho_m=median_rho,
        worst_source=worst_pair[2],
        worst_error_deg=float(worst_pair[3]),
        case_count=len(rho_values),
        nonempty_case_count=nonempty,
    )


def _circle_intersections(first, second, radius: float):
    first = _as_point(first)
    second = _as_point(second)
    delta = second - first
    distance = float(np.linalg.norm(delta))
    if distance <= GEOMETRY_EPS or distance > 2.0 * radius + GEOMETRY_EPS:
        return []
    midpoint = (first + second) / 2.0
    height_squared = radius * radius - (distance / 2.0) ** 2
    if height_squared < -GEOMETRY_EPS:
        return []
    height = math.sqrt(max(height_squared, 0.0))
    normal = np.array([-delta[1], delta[0]]) / distance
    return [midpoint + height * normal, midpoint - height * normal]


def generate_robust_candidates(
    omega,
    current_position,
    movement_budget_m: float,
    *,
    spacing_m: float = 100.0,
    min_reception_radius: float = MIN_RECEPTION_RADIUS,
    target_radius: Optional[float] = None,
    circle_segments: int = 72,
    max_constraint_vertices: int = 96,
    max_candidates: Optional[int] = 500,
    exclude_points: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    """Generate a finite candidate set inside the robust domain and budget.

    ``target_radius=None`` is the default and intentionally permits points
    outside the 1800 m source disk.  Pass ``target_radius=1800`` only for the
    boundary/outside ablation.  All accepted grid/domain points are checked
    against every original Omega vertex, so the finite search is conservative
    even when the displayed circular intersection is coarse.
    """

    polygon = _as_polygon(omega)
    current = _as_point(current_position, "current_position")
    if movement_budget_m < 0 or not math.isfinite(float(movement_budget_m)):
        raise ValueError("movement_budget_m must be finite and non-negative")
    if spacing_m <= 0 or not math.isfinite(float(spacing_m)):
        raise ValueError("spacing_m must be finite and positive")
    if min_reception_radius <= 0:
        raise ValueError("min_reception_radius must be positive")
    if target_radius is not None and target_radius <= 0:
        raise ValueError("target_radius must be positive when supplied")

    # The box intersection of all B(vertex,r) is a cheap necessary bounding
    # box for the robust domain and avoids scanning irrelevant positions.
    lower = np.max(polygon - min_reception_radius, axis=0)
    upper = np.min(polygon + min_reception_radius, axis=0)
    lower = np.maximum(lower, current - movement_budget_m)
    upper = np.minimum(upper, current + movement_budget_m)
    if np.any(lower > upper + GEOMETRY_EPS):
        return _empty_polygon()

    raw = []
    # Bound grid size for high-resolution localize_region polygons.  The
    # actual spacing remains part of the experiment metadata when this cap is
    # active; boundary/intersection points below supplement the grid.
    x_count = max(1, int(math.ceil((upper[0] - lower[0]) / spacing_m)) + 1)
    y_count = max(1, int(math.ceil((upper[1] - lower[1]) / spacing_m)) + 1)
    x_count = min(x_count, 81)
    y_count = min(y_count, 81)
    for x in np.linspace(lower[0], upper[0], x_count):
        for y in np.linspace(lower[1], upper[1], y_count):
            raw.append(np.array([x, y], dtype=float))

    domain = robust_candidate_domain(
        polygon,
        min_reception_radius=min_reception_radius,
        circle_segments=circle_segments,
        max_constraint_vertices=max_constraint_vertices,
    )
    if len(domain):
        raw.extend(domain)
        # Projections onto domain edges are important when the closest
        # feasible point is in the middle of an edge (small budgets).
        for index, start in enumerate(domain):
            end = domain[(index + 1) % len(domain)]
            edge = end - start
            length_squared = float(np.dot(edge, edge))
            if length_squared > GEOMETRY_EPS**2:
                fraction = float(np.dot(current - start, edge) / length_squared)
                fraction = min(1.0, max(0.0, fraction))
                raw.append(start + fraction * edge)
            for fraction in (0.25, 0.5, 0.75):
                raw.append(start + fraction * edge)

    # Pairwise circle intersections give useful corners of the true robust
    # set, particularly for small movement budgets and thin Omega.
    pair_vertices = polygon
    if len(pair_vertices) > 64:
        indices = np.linspace(0, len(pair_vertices), 64, endpoint=False, dtype=int)
        pair_vertices = pair_vertices[np.unique(indices)]
    for first, second in itertools.combinations(pair_vertices, 2):
        raw.extend(_circle_intersections(first, second, min_reception_radius))

    # Include central/radial probes around P.  They do not impose a target
    # circle and are useful when a grid misses a very small intersection.
    raw.extend([current, np.mean(polygon, axis=0), (lower + upper) / 2.0])
    if movement_budget_m > GEOMETRY_EPS:
        for radius_fraction in (0.25, 0.5, 0.75, 1.0):
            radius = radius_fraction * movement_budget_m
            for angle in np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False):
                raw.append(
                    current + radius * np.array([math.cos(angle), math.sin(angle)])
                )

    excluded = [] if exclude_points is None else [_as_point(p) for p in exclude_points]
    accepted = []
    for point in _dedupe_points(raw):
        if float(np.linalg.norm(point - current)) > movement_budget_m + GEOMETRY_EPS:
            continue
        if target_radius is not None and float(np.linalg.norm(point)) > target_radius + GEOMETRY_EPS:
            continue
        if any(float(np.linalg.norm(point - other)) <= 1e-5 for other in excluded):
            continue
        if is_robust_candidate(
            point,
            polygon,
            min_reception_radius=min_reception_radius,
            tolerance=GEOMETRY_EPS,
        ):
            accepted.append(point)

    if not accepted:
        return _empty_polygon()
    accepted_array = np.asarray(accepted, dtype=float)
    distances = np.linalg.norm(accepted_array - current, axis=1)
    order = np.lexsort((accepted_array[:, 1], accepted_array[:, 0], distances))
    accepted_array = accepted_array[order]
    if max_candidates is not None and max_candidates > 0 and len(accepted_array) > max_candidates:
        # Keep nearest points and then spread the remaining points across the
        # distance ordering, retaining candidate diversity for the Pareto scan.
        indices = np.linspace(0, len(accepted_array), max_candidates, endpoint=False, dtype=int)
        accepted_array = accepted_array[np.unique(indices)]
    return accepted_array


@dataclass(frozen=True)
class BudgetChoice:
    budget_m: float
    evaluation: Optional[CandidateEvaluation]

    @property
    def point(self):
        return None if self.evaluation is None else self.evaluation.point.copy()


@dataclass(frozen=True)
class ParetoResult:
    omega: np.ndarray
    current_position: np.ndarray
    budgets_m: tuple
    candidate_points: np.ndarray
    evaluations: tuple
    frontier: tuple
    choices: tuple
    source_samples: np.ndarray
    error_samples: tuple
    sampling: Mapping[str, object] = field(default_factory=dict)

    @property
    def recommended_by_budget(self):
        return {choice.budget_m: choice.evaluation for choice in self.choices}

    def choice_for_budget(self, budget_m: float) -> Optional[CandidateEvaluation]:
        for choice in self.choices:
            if abs(choice.budget_m - float(budget_m)) <= GEOMETRY_EPS:
                return choice.evaluation
        return None


def _dominates(first: CandidateEvaluation, second: CandidateEvaluation) -> bool:
    first_values = np.array([first.movement_distance_m, first.q_rho_m, first.q_diameter_m])
    second_values = np.array([second.movement_distance_m, second.q_rho_m, second.q_diameter_m])
    return bool(np.all(first_values <= second_values + GEOMETRY_EPS) and np.any(first_values < second_values - GEOMETRY_EPS))


def _select_best(evaluations: Sequence[CandidateEvaluation]) -> Optional[CandidateEvaluation]:
    if not evaluations:
        return None
    # No lambda: Q_rho is the primary stopping-related objective.  Q_D and
    # movement are transparent lexicographic tie-breakers only.
    return min(
        evaluations,
        key=lambda item: (
            item.q_rho_m,
            item.q_diameter_m,
            item.movement_distance_m,
            float(item.point[0]),
            float(item.point[1]),
        ),
    )


def solve_pareto(
    omega,
    current_position,
    budgets_m: Sequence[float] = DEFAULT_BUDGETS_M,
    *,
    candidate_points: Optional[Sequence[Sequence[float]]] = None,
    source_samples: Optional[Sequence[Sequence[float]]] = None,
    source_sample_count: int = 13,
    error_samples: Sequence[float] = (-1.0, 0.0, 1.0),
    spacing_m: float = 100.0,
    min_reception_radius: float = MIN_RECEPTION_RADIUS,
    max_reception_radius: float = MAX_RECEPTION_RADIUS,
    angle_error_deg: float = ANGLE_ERROR_DEG,
    circle_segments: int = 96,
    target_radius: Optional[float] = None,
    max_candidates: Optional[int] = 500,
    exclude_points: Optional[Sequence[Sequence[float]]] = None,
) -> ParetoResult:
    """Evaluate candidates and solve movement-budget Pareto subproblems.

    For each budget ``b`` this computes ``min Q_rho(S)`` over robust points
    with ``||S-P|| <= b``.  The returned ``frontier`` is non-dominated in the
    three measured quantities (movement, Q_rho, Q_D).  It is intentionally a
    discrete numerical search; no weighted objective or asserted global
    optimum is hidden in the implementation.
    """

    polygon = _as_polygon(omega)
    current = _as_point(current_position, "current_position")
    budgets = tuple(float(value) for value in budgets_m)
    if not budgets:
        raise ValueError("budgets_m must not be empty")
    if any(value < 0 or not math.isfinite(value) for value in budgets):
        raise ValueError("budgets_m must be finite and non-negative")
    max_budget = max(budgets)
    if source_samples is None:
        sources = sample_region_sources(polygon, source_sample_count)
    else:
        sources = _dedupe_points(source_samples)
    if len(sources) == 0:
        raise ValueError("source_samples must not be empty")
    for source in sources:
        if not point_in_convex_polygon(source, polygon, tolerance=1e-5):
            raise ValueError("every source sample must lie in omega")
    errors = tuple(float(value) for value in error_samples)

    if candidate_points is None:
        points = generate_robust_candidates(
            polygon,
            current,
            max_budget,
            spacing_m=spacing_m,
            min_reception_radius=min_reception_radius,
            target_radius=target_radius,
            circle_segments=max(16, min(circle_segments, 96)),
            max_candidates=max_candidates,
            exclude_points=exclude_points,
        )
    else:
        points = _dedupe_points(candidate_points)
        filtered = []
        excluded = [] if exclude_points is None else [_as_point(p) for p in exclude_points]
        for point in points:
            if float(np.linalg.norm(point - current)) > max_budget + GEOMETRY_EPS:
                continue
            if target_radius is not None and float(np.linalg.norm(point)) > target_radius + GEOMETRY_EPS:
                continue
            if any(float(np.linalg.norm(point - other)) <= 1e-5 for other in excluded):
                continue
            if is_robust_candidate(point, polygon, min_reception_radius=min_reception_radius):
                filtered.append(point)
        points = np.asarray(filtered, dtype=float) if filtered else _empty_polygon()

    evaluations = []
    for point in points:
        evaluations.append(
            evaluate_candidate(
                polygon,
                current,
                point,
                source_samples=sources,
                error_samples=errors,
                min_reception_radius=min_reception_radius,
                max_reception_radius=max_reception_radius,
                angle_error_deg=angle_error_deg,
                circle_segments=max(16, min(circle_segments, 96)),
                require_robust=True,
            )
        )

    evaluations.sort(key=lambda item: (item.movement_distance_m, item.q_rho_m, item.q_diameter_m))
    frontier = []
    for evaluation in evaluations:
        if not any(_dominates(other, evaluation) for other in evaluations if other is not evaluation):
            frontier.append(evaluation)
    frontier.sort(key=lambda item: item.movement_distance_m)

    choices = []
    for budget in budgets:
        feasible = [
            evaluation
            for evaluation in evaluations
            if evaluation.movement_distance_m <= budget + GEOMETRY_EPS
        ]
        choices.append(BudgetChoice(budget, _select_best(feasible)))

    return ParetoResult(
        omega=polygon.copy(),
        current_position=current.copy(),
        budgets_m=budgets,
        candidate_points=points.copy(),
        evaluations=tuple(evaluations),
        frontier=tuple(frontier),
        choices=tuple(choices),
        source_samples=sources.copy(),
        error_samples=errors,
        sampling={
            "source_sample_count": int(len(sources)),
            "error_samples_deg": list(errors),
            "candidate_spacing_m": float(spacing_m),
            "circle_segments": int(circle_segments),
            "candidate_count": int(len(points)),
            "target_radius_filter_m": target_radius,
            "continuous_optimum_claimed": False,
        },
    )


# Descriptive aliases used by notebooks/experiments.
pareto_frontier = solve_pareto
candidate_points_for_budget = generate_robust_candidates


@dataclass
class SimulationResult:
    budget_m: float
    success: bool
    total_time_s: float
    measurement_count: int
    clear_attempts: int
    clear_failures: int
    movement_distance_m: float
    measurement_move_distance_m: float
    final_clear_move_distance_m: float
    measurement_time_s: float
    final_clear_time_s: float
    steps: int
    reason: str
    outside_target_measurements: int
    trace: list = field(default_factory=list)

    @property
    def measurements(self) -> int:
        return self.measurement_count

    @property
    def movement_distance(self) -> float:
        return self.movement_distance_m


def _stable_error_from_key(seed: int, point: np.ndarray, source: np.ndarray) -> float:
    """Map (seed, source, position) to a reproducible error without RNG order."""

    # Unlike Python's hash(), this integer mixer is stable across processes
    # and versions.  Including source and position models the fixed simulator
    # error for a (source, position) pair.
    values = [int(round(float(value) * 1_000_000.0)) for value in (*source, *point)]
    state = (int(seed) & 0xFFFFFFFFFFFFFFFF) ^ 0x9E3779B97F4A7C15
    for value in values:
        state ^= value & 0xFFFFFFFFFFFFFFFF
        state = (state * 0xBF58476D1CE4E5B9 + 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 27
    # Convert the top 53 bits to [0,1), then to [-1,1] degrees.
    unit = ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)
    return (2.0 * unit - 1.0) * ANGLE_ERROR_DEG


def _lookup_error(
    measurement_error_deg,
    point: np.ndarray,
    error_map: dict,
    rng: np.random.Generator,
    *,
    source: Optional[np.ndarray] = None,
    error_seed: int = 0,
) -> float:
    key = tuple(np.round(point, 6))
    if key in error_map:
        return error_map[key]
    if callable(measurement_error_deg):
        error = float(measurement_error_deg(point.copy()))
    elif measurement_error_deg is None:
        if source is None:
            # Kept for direct private-call compatibility; the simulator always
            # supplies source and therefore uses the order-independent path.
            error = float(rng.uniform(-ANGLE_ERROR_DEG, ANGLE_ERROR_DEG))
        else:
            error = _stable_error_from_key(error_seed, point, source)
    else:
        error = float(measurement_error_deg)
    if abs(error) > ANGLE_ERROR_DEG + GEOMETRY_EPS:
        raise ValueError("measurement error must be within +/-1 degree")
    error_map[key] = error
    return error


def _finalize_success(
    *,
    budget_m: float,
    position: np.ndarray,
    center: np.ndarray,
    measurement_count: int,
    measurement_move_distance_m: float,
    steps: int,
    outside_target_measurements: int,
    elapsed_before_clear_s: float,
    trace: list,
) -> SimulationResult:
    final_move = float(np.linalg.norm(center - position))
    final_clear_time = final_move / SPEED_MPS + CLEAR_TIME_S
    return SimulationResult(
        budget_m=float(budget_m),
        success=True,
        total_time_s=float(elapsed_before_clear_s + final_clear_time),
        measurement_count=int(measurement_count),
        clear_attempts=1,
        clear_failures=0,
        movement_distance_m=float(measurement_move_distance_m + final_move),
        measurement_move_distance_m=float(measurement_move_distance_m),
        final_clear_move_distance_m=final_move,
        measurement_time_s=float(measurement_count * MEASURE_TIME_S),
        final_clear_time_s=float(final_clear_time),
        steps=int(steps),
        reason="cleared",
        outside_target_measurements=int(outside_target_measurements),
        trace=trace,
    )


def simulate_single_target(
    omega,
    true_source,
    start_position,
    movement_budget_m: float,
    *,
    receive_radius_m: float = MIN_RECEPTION_RADIUS,
    measurement_error_deg=0.0,
    error_seed: int = 0,
    source_sample_count: int = 9,
    error_samples: Sequence[float] = (-1.0, 0.0, 1.0),
    spacing_m: float = 100.0,
    circle_segments: int = 72,
    max_steps: int = 8,
    target_radius: Optional[float] = None,
) -> SimulationResult:
    """Run a bounded receding-horizon single-target offline simulation.

    ``target_radius=None`` permits circle-outside measurements.  The actual
    source and one fixed error per measurement position determine the observed
    outcome.  Repeated positions reuse the same error via ``error_map``;
    normally they are excluded after the first visit to prevent a no-progress
    loop.  The planner itself still evaluates the supplied ``error_samples``
    grid to approximate worst-case ``Q_rho``/``Q_D``.
    """

    polygon = _as_polygon(omega)
    source = _as_point(true_source, "true_source")
    position = _as_point(start_position, "start_position")
    if not point_in_convex_polygon(source, polygon, tolerance=1e-5):
        raise ValueError("true_source must lie in the initial omega")
    if movement_budget_m < 0 or not math.isfinite(float(movement_budget_m)):
        raise ValueError("movement_budget_m must be finite and non-negative")
    if receive_radius_m <= 0:
        raise ValueError("receive_radius_m must be positive")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    omega_current = polygon.copy()
    elapsed = 0.0
    measurement_count = 0
    measurement_move = 0.0
    final_clear_move = 0.0
    outside_count = 0
    steps = 0
    trace = []
    visited = []
    error_map = {}
    rng = np.random.default_rng(error_seed)

    def failed(reason: str) -> SimulationResult:
        return SimulationResult(
            budget_m=float(movement_budget_m),
            success=False,
            total_time_s=float(elapsed),
            measurement_count=int(measurement_count),
            clear_attempts=0,
            clear_failures=0,
            movement_distance_m=float(measurement_move + final_clear_move),
            measurement_move_distance_m=float(measurement_move),
            final_clear_move_distance_m=float(final_clear_move),
            measurement_time_s=float(measurement_count * MEASURE_TIME_S),
            final_clear_time_s=0.0,
            steps=int(steps),
            reason=reason,
            outside_target_measurements=int(outside_count),
            trace=trace,
        )

    for step in range(max_steps + 1):
        steps = step
        if len(omega_current) == 0:
            return failed("empty_posterior")
        center, rho = minimum_enclosing_circle(omega_current)
        if float(rho) <= CLEAR_RADIUS + GEOMETRY_EPS:
            # Verify the physical source before claiming a numerical MEC clear.
            if float(np.linalg.norm(source - center)) > CLEAR_RADIUS + 1e-4:
                return failed("mec_numerical_source_mismatch")
            result = _finalize_success(
                budget_m=movement_budget_m,
                position=position,
                center=center,
                measurement_count=measurement_count,
                measurement_move_distance_m=measurement_move,
                steps=steps,
                outside_target_measurements=outside_count,
                elapsed_before_clear_s=elapsed,
                trace=trace,
            )
            return result
        if step >= max_steps:
            return failed("max_steps")

        # Planner samples come only from the observable Omega.  Do not append
        # the hidden true source here: doing so leaks the answer and makes the
        # trade-off study systematically optimistic.  The true source is used
        # below only to generate the actual simulated measurement outcome.
        sources = sample_region_sources(omega_current, source_sample_count, seed=error_seed + step)
        result = solve_pareto(
            omega_current,
            position,
            budgets_m=(movement_budget_m,),
            source_samples=sources,
            error_samples=error_samples,
            spacing_m=spacing_m,
            min_reception_radius=MIN_RECEPTION_RADIUS,
            max_reception_radius=MAX_RECEPTION_RADIUS,
            angle_error_deg=ANGLE_ERROR_DEG,
            circle_segments=circle_segments,
            target_radius=target_radius,
            max_candidates=350,
            exclude_points=visited,
        )
        choice = result.choice_for_budget(movement_budget_m)
        if choice is None:
            return failed("no_robust_candidate_within_budget")
        # ``choice_for_budget`` returns the selected CandidateEvaluation.
        candidate = choice.point.copy()
        distance = float(np.linalg.norm(candidate - position))
        measurement_move += distance
        elapsed += distance / SPEED_MPS + MEASURE_TIME_S
        measurement_count += 1
        if target_radius is not None and float(np.linalg.norm(candidate)) > target_radius + GEOMETRY_EPS:
            # This branch is mostly diagnostic; solve_pareto already filters.
            outside_count += 1
        elif float(np.linalg.norm(candidate)) > TARGET_RADIUS + GEOMETRY_EPS:
            outside_count += 1
        visited.append(candidate.copy())

        source_distance = float(np.linalg.norm(source - candidate))
        trace.append(
            {
                "step": int(step + 1),
                "position": candidate.tolist(),
                "move_m": distance,
                "source_distance_m": source_distance,
                "q_rho_m": float(choice.q_rho_m),
                "q_diameter_m": float(choice.q_diameter_m),
            }
        )
        position = candidate.copy()

        if source_distance > receive_radius_m + GEOMETRY_EPS:
            return failed("no_signal")
        if source_distance <= NEAR_RADIUS + GEOMETRY_EPS:
            # Near response: the measurement has already happened; clear at
            # the current point, costing the required 5 s and no extra move.
            center = position.copy()
            elapsed += CLEAR_TIME_S
            return SimulationResult(
                budget_m=float(movement_budget_m),
                success=True,
                total_time_s=float(elapsed),
                measurement_count=int(measurement_count),
                clear_attempts=1,
                clear_failures=0,
                movement_distance_m=float(measurement_move),
                measurement_move_distance_m=float(measurement_move),
                final_clear_move_distance_m=0.0,
                measurement_time_s=float(measurement_count * MEASURE_TIME_S),
                final_clear_time_s=CLEAR_TIME_S,
                steps=int(step + 1),
                reason="near_then_cleared",
                outside_target_measurements=int(outside_count),
                trace=trace,
            )

        error = _lookup_error(
            measurement_error_deg,
            position,
            error_map,
            rng,
            source=source,
            error_seed=error_seed,
        )
        posterior = posterior_region(
            omega_current,
            position,
            source,
            error,
            max_reception_radius=MAX_RECEPTION_RADIUS,
            angle_error_deg=ANGLE_ERROR_DEG,
            circle_segments=circle_segments,
        )
        if len(posterior) == 0:
            return failed("empty_posterior")
        omega_current = posterior

    return failed("max_steps")


def _summary_stat(values: Sequence[float]):
    values = np.asarray([float(value) for value in values if math.isfinite(float(value))])
    if len(values) == 0:
        return {"average": None, "p95": None, "worst": None}
    return {
        "average": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "worst": float(np.max(values)),
    }


def summarize_simulations(results: Sequence[SimulationResult], budget_m: Optional[float] = None):
    """Aggregate successful-run average/P95/worst and explicit failure counts."""

    if not results:
        raise ValueError("results must not be empty")
    successful = [result for result in results if result.success and math.isfinite(result.total_time_s)]
    summary = {
        "budget_m": float(budget_m if budget_m is not None else results[0].budget_m),
        "runs": int(len(results)),
        "successes": int(len(successful)),
        "failures": int(len(results) - len(successful)),
        "success_rate": float(len(successful) / len(results)),
        # Time/measurement/distance statistics are over successful clears;
        # failures are reported separately and are never silently treated as
        # fast runs.
        "time_s": _summary_stat([result.total_time_s for result in successful]),
        "measurements": _summary_stat([result.measurement_count for result in successful]),
        "movement_distance_m": _summary_stat([result.movement_distance_m for result in successful]),
        "clear_attempts": _summary_stat([result.clear_attempts for result in successful]),
        "outside_measurements": _summary_stat([result.outside_target_measurements for result in successful]),
        "time_breakdown_s": {
            "measurement_movement": _summary_stat([result.measurement_move_distance_m / SPEED_MPS for result in successful]),
            "measure_service": _summary_stat([result.measurement_time_s for result in successful]),
            "final_clear_movement": _summary_stat([result.final_clear_move_distance_m / SPEED_MPS for result in successful]),
            "final_clear_service": _summary_stat([CLEAR_TIME_S for _ in successful]),
        },
        "failure_reasons": {},
    }
    for result in results:
        if not result.success:
            summary["failure_reasons"][result.reason] = summary["failure_reasons"].get(result.reason, 0) + 1
    return summary


def make_ellipse_region(center, semi_axes, *, angle_deg: float = 0.0, vertices: int = 24) -> np.ndarray:
    """Create a deterministic convex elliptical prior for offline studies."""

    center = _as_point(center, "center")
    axes = np.asarray(semi_axes, dtype=float)
    if axes.shape != (2,) or np.any(axes <= 0) or not np.all(np.isfinite(axes)):
        raise ValueError("semi_axes must contain two positive values")
    if vertices < 4:
        raise ValueError("vertices must be at least 4")
    angle = math.radians(float(angle_deg))
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    parameter = np.linspace(0.0, 2.0 * math.pi, vertices, endpoint=False)
    local = np.column_stack((axes[0] * np.cos(parameter), axes[1] * np.sin(parameter)))
    return center + local @ rotation.T


def make_slender_region() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Near-collinear adversarial region, true source and starting position."""

    omega = make_ellipse_region((1050.0, 0.0), (520.0, 12.0), angle_deg=0.0, vertices=32)
    source = np.array([1430.0, 5.0])
    start = np.array([850.0, -80.0])
    return omega, source, start


def make_boundary_region() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin boundary prior where a useful candidate may lie outside r=1800."""

    omega = make_ellipse_region((1700.0, 0.0), (80.0, 28.0), angle_deg=12.0, vertices=28)
    source = np.array([1750.0, 12.0])
    start = np.array([1600.0, 120.0])
    return omega, source, start


def make_random_regions(seed: int = 20260913, count: int = 8):
    """Generate fixed-seed random elliptical priors inside the target disk."""

    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    scenarios = []
    for index in range(count):
        radius = 1250.0 * math.sqrt(float(rng.uniform()))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        center = radius * np.array([math.cos(angle), math.sin(angle)])
        axes = np.array([rng.uniform(30.0, 260.0), rng.uniform(12.0, 120.0)])
        orientation = float(rng.uniform(0.0, 180.0))
        omega = make_ellipse_region(center, axes, angle_deg=orientation, vertices=20)
        # Keep every prior vertex comfortably within the source disk.
        envelope = float(np.max(np.linalg.norm(omega, axis=1)))
        if envelope > 1780.0:
            omega = center + (omega - center) * (1780.0 / envelope)
        weights = rng.dirichlet(np.ones(len(omega)))
        source = np.sum(omega * weights[:, None], axis=0)
        start = rng.uniform(-400.0, 400.0, size=2)
        scenarios.append(
            {
                "name": f"random_{index:02d}",
                "omega": omega,
                "source": source,
                "start": start,
            }
        )
    return scenarios


__all__ = [
    "ANGLE_ERROR_DEG",
    "BudgetChoice",
    "CandidateEvaluation",
    "CLEAR_RADIUS",
    "NEAR_RADIUS",
    "DEFAULT_BUDGETS_M",
    "MAX_RECEPTION_RADIUS",
    "MIN_RECEPTION_RADIUS",
    "ParetoResult",
    "SimulationResult",
    "TARGET_RADIUS",
    "bearing_deg",
    "candidate_points_for_budget",
    "evaluate_candidate",
    "generate_robust_candidates",
    "is_robust_candidate",
    "make_boundary_region",
    "make_ellipse_region",
    "make_random_regions",
    "make_slender_region",
    "max_distance_to_region",
    "pareto_frontier",
    "posterior_region",
    "robust_candidate_domain",
    "sample_region_sources",
    "simulate_single_target",
    "solve_pareto",
    "summarize_simulations",
]
