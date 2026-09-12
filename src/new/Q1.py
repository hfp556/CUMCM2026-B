"""问题一：带示向误差与接收范围约束的交会定位。"""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


DEFAULT_ANGLE_ERROR_DEG = 1.0
DEFAULT_RECEPTION_RADIUS = 1500.0
DEFAULT_TARGET_RADIUS = 1800.0
GEOMETRY_TOLERANCE = 1e-9


def _cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _normalise_polygon(poly: Iterable[Sequence[float]]) -> np.ndarray:
    points = np.asarray(poly, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    points = points.reshape((-1, 2))

    cleaned = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - cleaned[-1]) > GEOMETRY_TOLERANCE:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[-1] - cleaned[0]) <= GEOMETRY_TOLERANCE:
        cleaned.pop()
    return np.asarray(cleaned, dtype=float)


def _signed_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    return 0.5 * float(
        np.sum(poly[:, 0] * np.roll(poly[:, 1], -1))
        - np.sum(poly[:, 1] * np.roll(poly[:, 0], -1))
    )


def generate_circle_polygon(center, radius, num_segments=360):
    """用逆时针凸多边形近似圆盘边界。"""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if num_segments < 12:
        raise ValueError("num_segments must be at least 12")

    center = np.asarray(center, dtype=float)
    if center.shape != (2,):
        raise ValueError("center must contain exactly two coordinates")
    angles = np.linspace(0.0, 2.0 * np.pi, num_segments, endpoint=False)
    return center + radius * np.column_stack((np.cos(angles), np.sin(angles)))


def clip_polygon_by_halfplane(
    poly, point, normal, tolerance=GEOMETRY_TOLERANCE
):
    """用 ``(P-point)·normal >= 0`` 半平面裁剪凸多边形。"""
    polygon = _normalise_polygon(poly)
    if len(polygon) == 0:
        return polygon

    point = np.asarray(point, dtype=float)
    normal = np.asarray(normal, dtype=float)
    if point.shape != (2,) or normal.shape != (2,):
        raise ValueError("point and normal must be two-dimensional")
    if np.linalg.norm(normal) <= tolerance:
        raise ValueError("normal must be non-zero")

    result = []
    previous = polygon[-1]
    previous_value = float(np.dot(previous - point, normal))
    previous_inside = previous_value >= -tolerance

    for current in polygon:
        current_value = float(np.dot(current - point, normal))
        current_inside = current_value >= -tolerance

        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > tolerance:
                fraction = previous_value / denominator
                result.append(previous + fraction * (current - previous))
        if current_inside:
            result.append(current)

        previous = current
        previous_value = current_value
        previous_inside = current_inside

    return _normalise_polygon(result)


def clip_polygon_by_convex_polygon(subject, clipper):
    """求两个凸多边形的交集，clipper 的每条边只使用一次。"""
    result = _normalise_polygon(subject)
    boundary = _normalise_polygon(clipper)
    if len(result) == 0 or len(boundary) < 3:
        return np.empty((0, 2), dtype=float)

    if _signed_area(boundary) < 0:
        boundary = boundary[::-1]

    for index, start in enumerate(boundary):
        end = boundary[(index + 1) % len(boundary)]
        edge = end - start
        # 逆时针凸多边形内部永远位于每条有向边的左侧。
        inward_normal = np.array([-edge[1], edge[0]], dtype=float)
        result = clip_polygon_by_halfplane(result, start, inward_normal)
        if len(result) == 0:
            break
    return result


def _wedge_normals(theta_deg, angle_error_deg):
    lower = math.radians(theta_deg - angle_error_deg)
    upper = math.radians(theta_deg + angle_error_deg)
    lower_direction = np.array([math.cos(lower), math.sin(lower)])
    upper_direction = np.array([math.cos(upper), math.sin(upper)])

    # 扇区位于下边界射线左侧、上边界射线右侧。
    lower_inward = np.array([-lower_direction[1], lower_direction[0]])
    upper_inward = np.array([upper_direction[1], -upper_direction[0]])
    return lower_inward, upper_inward


def localize_region(
    S_list,
    theta_list,
    R_max=DEFAULT_RECEPTION_RADIUS,
    R_target=DEFAULT_TARGET_RADIUS,
    angle_error_deg=DEFAULT_ANGLE_ERROR_DEG,
    target_center=(0.0, 0.0),
    circle_segments=360,
):
    """计算所有观测共同允许的干扰源位置区域。

    每次 ``direction`` 观测加入三个约束：目标圆盘、检测点处最大
    1500 m 接收圆盘，以及示向度 ``±angle_error_deg`` 的角扇区。
    这里没有 1000 m 内边界；1000 m 是接收半径下界，不是源距下界。
    """
    if len(S_list) != len(theta_list):
        raise ValueError("S_list and theta_list must have the same length")
    if R_max <= 0 or R_target <= 0:
        raise ValueError("radii must be positive")
    if not 0 < angle_error_deg < 90:
        raise ValueError("angle_error_deg must be between 0 and 90 degrees")

    region = generate_circle_polygon(target_center, R_target, circle_segments)

    for sensor, theta_deg in zip(S_list, theta_list):
        sensor = np.asarray(sensor, dtype=float)
        if sensor.shape != (2,) or not np.all(np.isfinite(sensor)):
            raise ValueError("each sensor position must contain two finite coordinates")
        if not math.isfinite(float(theta_deg)):
            raise ValueError("each bearing must be finite")

        lower_normal, upper_normal = _wedge_normals(theta_deg, angle_error_deg)
        region = clip_polygon_by_halfplane(region, sensor, lower_normal)
        region = clip_polygon_by_halfplane(region, sensor, upper_normal)
        if len(region) == 0:
            return region

        reception_disk = generate_circle_polygon(sensor, R_max, circle_segments)
        region = clip_polygon_by_convex_polygon(region, reception_disk)
        if len(region) == 0:
            return region

    return region


def point_in_convex_polygon(point, poly, tolerance=GEOMETRY_TOLERANCE):
    """判断点是否位于凸多边形内部或边界上。"""
    polygon = _normalise_polygon(poly)
    point = np.asarray(point, dtype=float)
    if len(polygon) == 0:
        return False
    if len(polygon) == 1:
        return np.linalg.norm(point - polygon[0]) <= tolerance
    if len(polygon) == 2:
        edge = polygon[1] - polygon[0]
        return (
            abs(_cross_2d(edge, point - polygon[0])) <= tolerance
            and np.dot(point - polygon[0], point - polygon[1]) <= tolerance
        )

    signs = np.array(
        [
            _cross_2d(polygon[(i + 1) % len(polygon)] - polygon[i], point - polygon[i])
            for i in range(len(polygon))
        ]
    )
    return bool(np.all(signs >= -tolerance) or np.all(signs <= tolerance))


def convex_diameter(poly):
    """用旋转卡壳计算凸多边形直径及一对直径端点。"""
    polygon = _normalise_polygon(poly)
    n = len(polygon)
    if n == 0:
        raise ValueError("poly must not be empty")
    if n == 1:
        return 0.0, (polygon[0].copy(), polygon[0].copy())
    if n == 2:
        return float(np.linalg.norm(polygon[0] - polygon[1])), (
            polygon[0].copy(),
            polygon[1].copy(),
        )
    if _signed_area(polygon) < 0:
        polygon = polygon[::-1]

    def area2(i, j, k):
        return abs(_cross_2d(polygon[j] - polygon[i], polygon[k] - polygon[i]))

    best_squared = -1.0
    best_pair = (polygon[0], polygon[1])

    def consider(a, b):
        nonlocal best_squared, best_pair
        squared = float(np.dot(polygon[a] - polygon[b], polygon[a] - polygon[b]))
        if squared > best_squared:
            best_squared = squared
            best_pair = (polygon[a].copy(), polygon[b].copy())

    j = 1
    for i in range(n):
        next_i = (i + 1) % n
        while area2(i, next_i, (j + 1) % n) > area2(i, next_i, j) + GEOMETRY_TOLERANCE:
            j = (j + 1) % n
        consider(i, j)
        consider(next_i, j)
        if abs(area2(i, next_i, (j + 1) % n) - area2(i, next_i, j)) <= GEOMETRY_TOLERANCE:
            consider(i, (j + 1) % n)
            consider(next_i, (j + 1) % n)

    return math.sqrt(max(best_squared, 0.0)), best_pair


@dataclass(frozen=True)
class DiameterCircleCoverage:
    diameter: float
    diameter_pair: tuple
    center: np.ndarray
    radius: float
    max_center_distance: float
    farthest_point: np.ndarray
    covers: bool


def diameter_circle_coverage(poly, tolerance=1e-7):
    """判断以一对区域直径端点为直径的圆能否覆盖整个凸区域。

    凸区域到固定圆心的最远点必为顶点，因此检查所有多边形顶点即可。
    该性质并非对任意凸区域恒成立；锐角三角形就是反例。
    """
    polygon = _normalise_polygon(poly)
    if len(polygon) == 0:
        raise ValueError("poly must not be empty")

    diameter, pair = convex_diameter(polygon)
    center = (pair[0] + pair[1]) / 2.0
    radius = diameter / 2.0
    distances = np.linalg.norm(polygon - center, axis=1)
    farthest_index = int(np.argmax(distances))
    max_distance = float(distances[farthest_index])
    scaled_tolerance = tolerance * max(1.0, diameter)
    return DiameterCircleCoverage(
        diameter=diameter,
        diameter_pair=(pair[0].copy(), pair[1].copy()),
        center=center,
        radius=radius,
        max_center_distance=max_distance,
        farthest_point=polygon[farthest_index].copy(),
        covers=max_distance <= radius + scaled_tolerance,
    )
