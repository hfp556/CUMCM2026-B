# -*- coding: utf-8 -*-
"""
问题二：第二个检测点选择策略与候选区域

已知：
    第一个检测点 S1 = (x1, y1)
    在该点测得全向干扰源示向度 theta1（单位：度）

目标：
    选择第二个检测点 S2，使两条测向线交会角接近 90°，
    从而获得较好的定位效果，即较小 GDOP。

本文件只包含问题二内容，不包含问题三的扫描、清除、API 调用等逻辑。
"""

import math
import numpy as np


def get_problem2_point_dynamic(S1, theta1, attempt=0):
    """
    问题二：第二个检测点动态选择策略。

    参数：
        S1      : 第一个检测点，形如 [x1, y1] 或 np.array([x1, y1])
        theta1  : 第一个检测点处测得的目标示向度，单位：度
        attempt : 候选尝试序号，0~4，对应偏移距离逐步缩小

    返回：
        S2      : 第二个检测点坐标 np.array([x2, y2])，若无效则返回 None
        params  : (偏转角, 偏移距离)，若无效则返回 None

    策略：
        1. 按第一测向线估计目标大概位置：
           G_est = S1 + 1200 * [cos(theta1), sin(theta1)]
        2. 在 G_est 处作第一测向线的垂线；
        3. 第二个检测点放在该垂线上，偏移距离 d 从 800m 逐步缩小：
           800, 400, 200, 120, 80；
        4. 优先取 theta1 + 90° 方向；
        5. 若超出安全半径 1700m，则改取 theta1 - 90° 方向；
        6. 若仍超过 1800m，则该候选无效，尝试下一档 d。
    """
    est_distance = 1200.0

    S1 = np.asarray(S1, dtype=float)

    # 估计目标位置
    G_est = S1 + est_distance * np.array([
        math.cos(math.radians(theta1)),
        math.sin(math.radians(theta1))
    ])

    # 偏移距离候选：从大到小
    scale = [1.0, 0.5, 0.25, 0.15, 0.1]
    if attempt >= len(scale):
        return None, None

    d = 800.0 * scale[attempt]

    # 优先取 theta1 + 90°，即垂直于第一测向线方向
    perp_angle = math.radians(theta1 + 90)
    S2 = G_est + d * np.array([
        math.cos(perp_angle),
        math.sin(perp_angle)
    ])

    # 如果超出安全半径，换另一侧
    if np.linalg.norm(S2) > 1700:
        perp_angle = math.radians(theta1 - 90)
        S2 = G_est + d * np.array([
            math.cos(perp_angle),
            math.sin(perp_angle)
        ])

    # 仍超出允许半径，则放弃该候选
    if np.linalg.norm(S2) > 1800:
        return None, None

    return S2, (90.0, d)


def generate_candidate_region(S1, theta1,
                              est_distance=1200.0,
                              max_radius=1800.0,
                              safe_radius=1700.0,
                              base_d=800.0,
                              scales=None):
    """
    生成第二个检测点的候选区域。

    候选区域本质：
        以估计目标点 G_est 为中心，
        沿第一测向线垂直方向的两侧线段。

    返回：
        按优先级排序的候选点列表，每个元素为 dict。
    """
    if scales is None:
        scales = [1.0, 0.5, 0.25, 0.15, 0.1]

    S1 = np.asarray(S1, dtype=float)
    theta = math.radians(theta1)

    G_est = S1 + est_distance * np.array([
        math.cos(theta),
        math.sin(theta)
    ])

    candidates = []

    for attempt, scale in enumerate(scales):
        d = base_d * scale

        # 两侧都作为候选：theta1 + 90° 和 theta1 - 90°
        for sign, name in [(+1, "+90"), (-1, "-90")]:
            perp = math.radians(theta1 + sign * 90)

            S2 = G_est + d * np.array([
                math.cos(perp),
                math.sin(perp)
            ])

            r = float(np.linalg.norm(S2))

            if r > max_radius:
                continue

            candidates.append({
                "attempt": attempt,
                "sign": name,
                "d": d,
                "S2": S2,
                "radius": r,
                "inside_safe": r <= safe_radius,
                "perp_angle_deg": (theta1 + sign * 90) % 360,
            })

    # 排序优先级：
    # 1. 在安全半径内优先；
    # 2. 偏移距离 d 越大越好，交会更强；
    # 3. theta1 + 90° 优先。
    candidates.sort(
        key=lambda c: (
            not c["inside_safe"],
            -c["d"],
            c["sign"] != "+90"
        )
    )

    return candidates


if __name__ == "__main__":
    # 简单测试
    S1 = np.array([0.0, 0.0])
    theta1 = 30.0

    print("动态选点结果：")
    for k in range(5):
        S2, params = get_problem2_point_dynamic(S1, theta1, attempt=k)
        print(f"attempt={k}, S2={S2}, params={params}")

    print("\n候选区域：")
    for c in generate_candidate_region(S1, theta1):
        print(c)