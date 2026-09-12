"""问题二：基于第一次完整可能区域的第二检测点优化。

严格模型与 Q3 快速启发式被有意分开：

* :func:`optimize_second_point` 用于问题二建模，以最坏定位区域直径为目标；
* :func:`get_problem2_point_dynamic` 仅是 Q3 可实时调用的快速启发式。
"""

from dataclasses import dataclass
import math
from typing import Optional
import warnings

import numpy as np

from Q1 import convex_diameter, localize_region, point_in_convex_polygon


MIN_RECEPTION_RADIUS = 1000.0
MAX_RECEPTION_RADIUS = 1500.0
TARGET_RADIUS = 1800.0


@dataclass(frozen=True)
class DetectionClassification:
    possible: bool
    guaranteed: bool
    minimum_distance_to_region: float
    maximum_distance_to_region: float


@dataclass(frozen=True)
class CandidateEvaluation:
    point: np.ndarray
    worst_case_diameter: float
    typical_crossing_angle_deg: float
    movement_distance: float


@dataclass(frozen=True)
class SecondPointOptimizationResult:
    first_region: np.ndarray
    possible_candidate_points: np.ndarray
    robust_candidate_points: np.ndarray
    source_samples: np.ndarray
    evaluations: tuple
    recommended_point: Optional[np.ndarray]
    worst_case_diameter: float
    typical_crossing_angle_deg: float


def _as_point(point, name="point"):
    result = np.asarray(point, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain two finite coordinates")
    return result


def _point_to_segment_distance(point, start, end):
    edge = end - start
    squared_length = float(np.dot(edge, edge))
    if squared_length == 0.0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, edge) / squared_length)
    fraction = min(1.0, max(0.0, fraction))
    projection = start + fraction * edge
    return float(np.linalg.norm(point - projection))


def _minimum_distance_to_polygon(point, polygon):
    if point_in_convex_polygon(point, polygon):
        return 0.0
    return min(
        _point_to_segment_distance(
            point, polygon[index], polygon[(index + 1) % len(polygon)]
        )
        for index in range(len(polygon))
    )


def first_possible_region(
    S1,
    theta1,
    *,
    target_center=(0.0, 0.0),
    target_radius=TARGET_RADIUS,
    max_reception_radius=MAX_RECEPTION_RADIUS,
    angle_error_deg=1.0,
    circle_segments=360,
):
    """返回 Ω1 = B(O,1800) ∩ B(S1,1500) ∩ W(S1,θ1,±1°)。"""
    return localize_region(
        [_as_point(S1, "S1")],
        [theta1],
        R_max=max_reception_radius,
        R_target=target_radius,
        angle_error_deg=angle_error_deg,
        target_center=target_center,
        circle_segments=circle_segments,
    )


def classify_detection_point(
    S2,
    first_region,
    *,
    min_reception_radius=MIN_RECEPTION_RADIUS,
    max_reception_radius=MAX_RECEPTION_RADIUS,
    tolerance=1e-7,
):
    """判断 S2 属于可能检测区域还是稳健候选区域。

    可能检测区域为 ``Ω1 ⊕ B(0,1500)``，即存在某个可能源位置能被
    1500 m 接收半径覆盖。稳健候选区域为
    ``∩_{G∈Ω1} B(G,1000)``，其中任意可能源在最小接收半径下均可检测。
    对凸多边形，距离 S2 最远的区域点一定可以在顶点中取得。
    """
    point = _as_point(S2, "S2")
    polygon = np.asarray(first_region, dtype=float)
    if len(polygon) < 3:
        raise ValueError("first_region must be a non-empty polygon")
    minimum_distance = _minimum_distance_to_polygon(point, polygon)
    maximum_distance = float(np.max(np.linalg.norm(polygon - point, axis=1)))
    return DetectionClassification(
        possible=minimum_distance <= max_reception_radius + tolerance,
        guaranteed=maximum_distance <= min_reception_radius + tolerance,
        minimum_distance_to_region=minimum_distance,
        maximum_distance_to_region=maximum_distance,
    )


def generate_candidate_points(
    first_region,
    *,
    spacing=150.0,
    min_reception_radius=MIN_RECEPTION_RADIUS,
    max_reception_radius=MAX_RECEPTION_RADIUS,
):
    """在连续可能检测区域的包围盒中离散采样候选点。

    返回 ``(possible_points, robust_points)``。点云只是连续候选区域的
    数值离散，不意味着机器狗被限制在目标圆内。
    """
    polygon = np.asarray(first_region, dtype=float)
    if len(polygon) < 3:
        raise ValueError("first_region must be a non-empty polygon")
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    lower = np.min(polygon, axis=0) - max_reception_radius
    upper = np.max(polygon, axis=0) + max_reception_radius
    x_values = np.arange(lower[0], upper[0] + 0.5 * spacing, spacing)
    y_values = np.arange(lower[1], upper[1] + 0.5 * spacing, spacing)

    # 加入几何中心，避免规则网格恰好漏掉较小的稳健区域。
    raw_points = [np.mean(polygon, axis=0)]
    raw_points.extend(np.array([x, y]) for x in x_values for y in y_values)

    possible = []
    robust = []
    seen = set()
    for point in raw_points:
        key = tuple(np.round(point, 9))
        if key in seen:
            continue
        seen.add(key)
        classification = classify_detection_point(
            point,
            polygon,
            min_reception_radius=min_reception_radius,
            max_reception_radius=max_reception_radius,
        )
        if classification.possible:
            possible.append(point)
        if classification.guaranteed:
            robust.append(point)

    empty = np.empty((0, 2), dtype=float)
    return (
        np.asarray(possible, dtype=float) if possible else empty.copy(),
        np.asarray(robust, dtype=float) if robust else empty.copy(),
    )


def _sample_first_region(first_region, count):
    """从 Ω1 的边界和内部确定性抽取代表性真实源位置。"""
    polygon = np.asarray(first_region, dtype=float)
    if count < 3:
        raise ValueError("source_sample_count must be at least 3")
    boundary_count = max(2, count - 1)
    indices = np.linspace(0, len(polygon), boundary_count, endpoint=False, dtype=int)
    boundary = polygon[np.unique(indices)]
    centroid = np.mean(polygon, axis=0)

    samples = [centroid]
    samples.extend(boundary)
    if len(samples) < count:
        samples.extend((centroid + point) / 2.0 for point in boundary)

    unique = []
    seen = set()
    for point in samples:
        key = tuple(np.round(point, 8))
        if key not in seen:
            seen.add(key)
            unique.append(point)
        if len(unique) == count:
            break
    return np.asarray(unique, dtype=float)


def _bearing_deg(origin, target):
    vector = target - origin
    if np.linalg.norm(vector) <= 1e-9:
        raise ValueError("bearing is undefined when origin equals target")
    return math.degrees(math.atan2(vector[1], vector[0])) % 360.0


def _crossing_angle_deg(S1, S2, source):
    first = source - S1
    second = source - S2
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-9:
        return 90.0  # near 情形比普通测向提供更强的信息。
    cosine = float(np.dot(first, second) / denominator)
    angle = math.degrees(math.acos(np.clip(abs(cosine), 0.0, 1.0)))
    return angle


def _evaluate_candidate(
    S1,
    theta1,
    S2,
    source_samples,
    *,
    target_center,
    target_radius,
    max_reception_radius,
    angle_error_deg,
    circle_segments,
):
    diameters = []
    crossing_angles = []
    for source in source_samples:
        # 避免离散样本与候选点重合而人为制造一个“完美”候选点。
        if np.linalg.norm(source - S2) <= 5.0:
            continue
        theta2 = _bearing_deg(S2, source)
        region = localize_region(
            [S1, S2],
            [theta1, theta2],
            R_max=max_reception_radius,
            R_target=target_radius,
            angle_error_deg=angle_error_deg,
            target_center=target_center,
            circle_segments=circle_segments,
        )
        if len(region) == 0:
            # 数值离散可能在边界样本处形成空集；这种候选不应被虚假奖励。
            return None
        diameter, _ = convex_diameter(region)
        diameters.append(diameter)
        crossing_angles.append(_crossing_angle_deg(S1, S2, source))

    if not diameters:
        return None
    return CandidateEvaluation(
        point=S2.copy(),
        worst_case_diameter=float(max(diameters)),
        typical_crossing_angle_deg=float(np.median(crossing_angles)),
        movement_distance=float(np.linalg.norm(S2 - S1)),
    )


def optimize_second_point(
    S1,
    theta1,
    *,
    target_center=(0.0, 0.0),
    target_radius=TARGET_RADIUS,
    min_reception_radius=MIN_RECEPTION_RADIUS,
    max_reception_radius=MAX_RECEPTION_RADIUS,
    angle_error_deg=1.0,
    candidate_spacing=150.0,
    source_sample_count=13,
    circle_segments=180,
):
    """离散求解 ``argmin_S2 max_Gi D(S2,Gi)``。

    主目标严格采用两次测向后定位区域直径的最坏值。只在最坏直径相同
    时，才依次偏好更短移动距离和更大的典型交会角。默认仅评价稳健
    候选点，从而在未知接收半径属于 [1000,1500] 时保证第二次可检测。
    """
    S1 = _as_point(S1, "S1")
    target_center = _as_point(target_center, "target_center")
    omega1 = first_possible_region(
        S1,
        theta1,
        target_center=target_center,
        target_radius=target_radius,
        max_reception_radius=max_reception_radius,
        angle_error_deg=angle_error_deg,
        circle_segments=circle_segments,
    )
    if len(omega1) < 3:
        raise ValueError("the first measurement produces an empty possible region")

    possible_points, robust_points = generate_candidate_points(
        omega1,
        spacing=candidate_spacing,
        min_reception_radius=min_reception_radius,
        max_reception_radius=max_reception_radius,
    )
    source_samples = _sample_first_region(omega1, source_sample_count)

    evaluations = []
    for S2 in robust_points:
        evaluation = _evaluate_candidate(
            S1,
            theta1,
            S2,
            source_samples,
            target_center=target_center,
            target_radius=target_radius,
            max_reception_radius=max_reception_radius,
            angle_error_deg=angle_error_deg,
            circle_segments=circle_segments,
        )
        if evaluation is not None:
            evaluations.append(evaluation)

    evaluations.sort(
        key=lambda item: (
            item.worst_case_diameter,
            item.movement_distance,
            -item.typical_crossing_angle_deg,
        )
    )
    if not evaluations:
        return SecondPointOptimizationResult(
            first_region=omega1,
            possible_candidate_points=possible_points,
            robust_candidate_points=robust_points,
            source_samples=source_samples,
            evaluations=tuple(),
            recommended_point=None,
            worst_case_diameter=math.inf,
            typical_crossing_angle_deg=math.nan,
        )

    best = evaluations[0]
    return SecondPointOptimizationResult(
        first_region=omega1,
        possible_candidate_points=possible_points,
        robust_candidate_points=robust_points,
        source_samples=source_samples,
        evaluations=tuple(evaluations),
        recommended_point=best.point.copy(),
        worst_case_diameter=best.worst_case_diameter,
        typical_crossing_angle_deg=best.typical_crossing_angle_deg,
    )


def get_problem2_point_dynamic(
    S1,
    theta1,
    attempt=0,
    *,
    estimated_distance=1200.0,
    preferred_robot_radius=1700.0,
    fallback_robot_radius=1800.0,
):
    """供 Q3 实时使用的快速启发式，不代表问题二严格数学模型。

    ``estimated_distance`` 及两个机器狗活动半径均为工程参数，不是题目
    对源距或机器狗位置的硬约束。两侧先统一比较：优先保留位于 1700 m
    内的点；若均不满足，再保留仍位于 1800 m 内的点。
    """
    S1 = _as_point(S1, "S1")
    scales = (1.0, 0.5, 0.25, 0.15, 0.1)
    if attempt < 0 or attempt >= len(scales):
        return None, None

    direction = math.radians(theta1)
    estimated_source = S1 + estimated_distance * np.array(
        [math.cos(direction), math.sin(direction)]
    )
    offset = 800.0 * scales[attempt]

    candidates = []
    for sign in (1, -1):
        perpendicular = math.radians(theta1 + sign * 90.0)
        point = estimated_source + offset * np.array(
            [math.cos(perpendicular), math.sin(perpendicular)]
        )
        candidates.append((sign, point, float(np.linalg.norm(point))))

    preferred = [item for item in candidates if item[2] <= preferred_robot_radius]
    valid = preferred or [item for item in candidates if item[2] <= fallback_robot_radius]
    if not valid:
        return None, None

    # +90° 仅作稳定的平局规则，不再声称偏移越大就一定越优。
    sign, point, _ = sorted(valid, key=lambda item: item[0] != 1)[0]
    return point, (float(sign * 90.0), offset)


def generate_candidate_region(S1, theta1, **kwargs):
    """旧名称兼容层；新代码应使用 generate_candidate_points。"""
    warnings.warn(
        "generate_candidate_region is deprecated; use first_possible_region "
        "and generate_candidate_points instead",
        DeprecationWarning,
        stacklevel=2,
    )
    omega1 = first_possible_region(S1, theta1, **kwargs)
    return generate_candidate_points(omega1)


def plot_problem2_result(result, S1, theta1, *, target_center=(0.0, 0.0), output_path=None):
    """绘制目标区、Ω1、两层候选区域点云及推荐 S2。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    S1 = _as_point(S1, "S1")
    target_center = _as_point(target_center, "target_center")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.35)

    ax.add_patch(
        Circle(
            target_center,
            TARGET_RADIUS,
            fill=False,
            linestyle="--",
            color="black",
            label="目标区域 B(O,1800)",
        )
    )
    omega = np.vstack((result.first_region, result.first_region[0]))
    ax.fill(omega[:, 0], omega[:, 1], color="tab:red", alpha=0.28, label="第一次可能区域 Ω1")
    ax.plot(omega[:, 0], omega[:, 1], color="tab:red", linewidth=1.5)

    possible = result.possible_candidate_points
    robust = result.robust_candidate_points
    if len(possible):
        ax.scatter(
            possible[:, 0], possible[:, 1], s=8, color="lightgreen", alpha=0.35,
            label="可能检测区域（离散）",
        )
    if len(robust):
        ax.scatter(
            robust[:, 0], robust[:, 1], s=16, color="tab:green", alpha=0.8,
            label="稳健候选区域（离散）",
        )

    ax.scatter(*S1, marker="^", s=90, color="tab:blue", label="第一次检测点 S1")
    for error in (-1.0, 1.0):
        angle = math.radians(theta1 + error)
        endpoint = S1 + MAX_RECEPTION_RADIUS * np.array([math.cos(angle), math.sin(angle)])
        ax.plot([S1[0], endpoint[0]], [S1[1], endpoint[1]], color="tab:blue", linewidth=1.0)

    if result.recommended_point is not None:
        ax.scatter(
            *result.recommended_point,
            marker="*",
            s=180,
            color="purple",
            label="推荐第二检测点 S2*",
            zorder=5,
        )

    all_points = [result.first_region, possible, robust, np.asarray([S1])]
    if result.recommended_point is not None:
        all_points.append(np.asarray([result.recommended_point]))
    visible = np.vstack([points for points in all_points if len(points)])
    margin = 250.0
    ax.set_xlim(np.min(visible[:, 0]) - margin, np.max(visible[:, 0]) + margin)
    ax.set_ylim(np.min(visible[:, 1]) - margin, np.max(visible[:, 1]) + margin)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("问题二：第二检测点候选区域与最坏直径优化")
    ax.legend(loc="best")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=180)
    return fig, ax


if __name__ == "__main__":
    example = optimize_second_point(
        (0.0, 0.0), 30.0, candidate_spacing=200.0, source_sample_count=11
    )
    print("推荐 S2:", example.recommended_point)
    print("最坏情况直径:", example.worst_case_diameter)
    print("典型交会角:", example.typical_crossing_angle_deg)
