"""
Q3 方向A：三点交会（L 形布局）
- S1 测 θ1
- S2 = S1 + 500m 垂直 θ1，测 θ2
- S3 = S2 + 500m 沿 θ1，测 θ3
- 三点交会，D≤50 直接清除，否则迭代兜底
"""
import math
import time
import itertools
import numpy as np
from Q1 import localize_region, convex_diameter
from api_utils import post, base, measure

_req_counter = itertools.count()


def _uid(prefix):
    return f"{prefix}-{next(_req_counter)}-{int(time.time()*1000)}"


def try_clear_at(channel, pos, tag):
    if np.linalg.norm(pos) > 1800:
        return False
    payload = base(_uid(f"clear-{tag}-{channel}"))
    payload["position"] = {"x": float(pos[0]), "y": float(pos[1])}
    payload["channel"] = channel
    rc = post("/clear", payload)
    return rc and rc.get("clear_result") == "success"


def clear_channel(channel, S_list, theta_list):
    """问题一稳健逼近（迭代 6 + 停滞检测 + 中期提前清除）"""
    D = float('inf')
    prev_D = float('inf')
    max_iterations = 6
    stagnation_count = 0
    tried_early_clear = False
    for iteration in range(max_iterations):
        poly = localize_region(S_list, theta_list)
        if len(poly) == 0:
            return False, S_list[-1]
        D, _ = convex_diameter(poly)
        print(f"  [{channel}] 迭代 {iteration+1}: D = {D:.2f} 米")

        cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])

        should_try = (D <= 20.0) or (D <= 35.0 and iteration >= 1 and not tried_early_clear)
        if should_try:
            if D <= 20.0:
                print(f"  [{channel}] 🎯 D <= 20 必清")
            else:
                print(f"  [{channel}] 🎯 中期尝试清除 D={D:.1f}")
                tried_early_clear = True
            payload = base(_uid(f"clear-ch{channel}"))
            payload["position"] = {"x": float(cx), "y": float(cy)}
            payload["channel"] = channel
            resp = post("/clear", payload)
            if resp and resp.get("clear_result") == "success":
                print(f"  [{channel}] ✅ 清除成功！")
                return True, np.array([cx, cy])

        if iteration >= 2 and abs(prev_D - D) < 3.0:
            stagnation_count += 1
            if stagnation_count >= 2:
                print(f"  [{channel}] ⚠️ D 已停滞，强制清除...")
                payload = base(_uid(f"clear-force-{channel}"))
                payload["position"] = {"x": float(cx), "y": float(cy)}
                payload["channel"] = channel
                resp = post("/clear", payload)
                if resp and resp.get("clear_result") == "success":
                    print(f"  [{channel}] ✅ 清除成功！")
                    return True, np.array([cx, cy])
                return False, np.array([cx, cy])
        else:
            stagnation_count = 0
        prev_D = D

        resp = measure(cx, cy, channel, _uid(f"iter-{channel}"))
        if resp is None:
            return False, np.array([cx, cy])
        mr = resp.get("measure_result")
        if mr == "near":
            payload = base(_uid(f"clear-near-{channel}"))
            payload["position"] = {"x": float(cx), "y": float(cy)}
            payload["channel"] = channel
            rc = post("/clear", payload)
            if rc and rc.get("clear_result") == "success":
                print(f"  [{channel}] ✅ 清除成功（near）！")
                return True, np.array([cx, cy])
            return False, np.array([cx, cy])
        elif mr == "direction":
            S_list.append(np.array([cx, cy]))
            theta_list.append(resp["svd_deg"])
        else:
            return False, np.array([cx, cy])
    return False, S_list[-1]


def three_point_clear(channel, S1, theta1):
    """
    ⚠️ 方向A核心：L 形三点交会
    S2 = S1 + 500m 垂直（±90°）
    S3 = S2 + 500m 沿 θ1
    """
    S_list = [S1]
    theta_list = [theta1]

    # === S2：先试 +90°，失败试 -90° ===
    S2 = None
    S2_theta = None
    for sign in (1, -1):
        perp = math.radians(theta1 + sign * 90)
        S2_try = S1 + 500 * np.array([math.cos(perp), math.sin(perp)])
        if np.linalg.norm(S2_try) > 1700:
            continue
        print(f"  [{channel}] 三点L: S2(±{sign*90}°) -> ({S2_try[0]:.0f}, {S2_try[1]:.0f})")
        r2 = measure(S2_try[0], S2_try[1], channel, _uid(f"3p-S2-{sign}-{channel}"))
        if not r2:
            continue
        mr2 = r2.get("measure_result")
        if mr2 == "near":
            if try_clear_at(channel, S2_try, f"3p-S2near-{sign}"):
                print(f"  [{channel}] ✅ S2 near 清除")
                return True, S2_try
            continue
        if mr2 == "direction":
            S2 = S2_try
            S2_theta = r2["svd_deg"]
            break

    if S2 is None:
        # S2 完全失败，退化为单点
        print(f"  [{channel}] 三点L: S2 全失败 → 单点迭代")
        return clear_channel(channel, [S1], [theta1])

    S_list.append(S2)
    theta_list.append(S2_theta)

    # === S3 = S2 + 500m 沿 θ1 方向 ===
    dir1 = math.radians(theta1)
    S3 = S2 + 500 * np.array([math.cos(dir1), math.sin(dir1)])
    if np.linalg.norm(S3) <= 1700:
        print(f"  [{channel}] 三点L: S3 -> ({S3[0]:.0f}, {S3[1]:.0f})")
        r3 = measure(S3[0], S3[1], channel, _uid(f"3p-S3-{channel}"))
        if r3:
            mr3 = r3.get("measure_result")
            if mr3 == "near":
                if try_clear_at(channel, S3, "3p-S3near"):
                    print(f"  [{channel}] ✅ S3 near 清除")
                    return True, S3
            elif mr3 == "direction":
                S_list.append(S3)
                theta_list.append(r3["svd_deg"])

    # === 三点交会 ===
    poly = localize_region(S_list, theta_list)
    if len(poly) == 0:
        print(f"  [{channel}] 三点L: 交会为空 → 迭代")
        return clear_channel(channel, S_list, theta_list)
    D, _ = convex_diameter(poly)
    print(f"  [{channel}] 三点L 交会 D = {D:.2f} 米 ({len(S_list)} 个点)")

    # 若 D ≤ 50 直接尝试清除
    if D <= 50.0:
        cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
        print(f"  [{channel}] 🎯 三点L 直接清除 D={D:.1f}")
        if try_clear_at(channel, (cx, cy), "3p-direct"):
            print(f"  [{channel}] ✅ 三点L 直接清除成功")
            return True, np.array([cx, cy])

    # D 太大或直接清除失败 → 交给 clear_channel 迭代（已有 2-3 个点，收敛更快）
    return clear_channel(channel, S_list, theta_list)


def scan_and_clear(current_pos, start_time, cleared_channels):
    scan_x, scan_y = current_pos[0], current_pos[1]
    found_channels = []

    print(f"\n--- 扫描 ({scan_x:.0f}, {scan_y:.0f}) ---")
    for ch in range(1, 21):
        if ch in cleared_channels:
            continue
        resp = measure(scan_x, scan_y, ch, _uid(f"scan-{ch}-{int(scan_x)}-{int(scan_y)}"))
        if resp and resp.get("measure_result") == "direction":
            print(f"✅ 频道 {ch} 有信号，{resp['svd_deg']}°")
            found_channels.append((ch, resp['svd_deg']))
        elif resp and resp.get("measure_result") == "near":
            print(f"🔥 频道 {ch} 距离过近，直接清除！")
            payload = base(_uid(f"clear-near-scan-{ch}"))
            payload["position"] = {"x": scan_x, "y": scan_y}
            payload["channel"] = ch
            post("/clear", payload)
            cleared_channels.add(ch)

    cleared_count = 0
    for ch, theta1 in found_channels:
        if time.time() - start_time > 15 * 60:
            break
        if ch in cleared_channels:
            continue
        print(f"\n=== 处理频道 {ch} ===")
        S1 = np.array([scan_x, scan_y])
        ok, _ = three_point_clear(ch, S1, theta1)
        if ok:
            cleared_channels.add(ch)
            cleared_count += 1

    return cleared_count


def solve_tsp_nearest_neighbor(points, start_idx=0):
    n = len(points)
    visited = [False] * n
    order = [start_idx]
    visited[start_idx] = True
    current = start_idx
    for _ in range(n - 1):
        best, bj = float('inf'), -1
        for j in range(n):
            if not visited[j]:
                d = np.linalg.norm(points[current] - points[j])
                if d < best:
                    best, bj = d, j
        if bj == -1:
            break
        order.append(bj)
        visited[bj] = True
        current = bj
    return [points[i] for i in order]


def main():
    start_time = time.time()
    res = post("/enter", base(_uid("enter")))
    if not res or res.get("accepted") is not True:
        print("❌ 进入失败")
        return
    print("✅ 进入成功！【方向A：三点L形交会】...\n")

    cleared_channels = set()
    raw_points = [
        np.array([0.0, 0.0]),
        np.array([1200.0, 0.0]), np.array([-1200.0, 0.0]),
        np.array([0.0, 1200.0]), np.array([0.0, -1200.0]),
        np.array([850.0, 850.0]), np.array([-850.0, 850.0]),
        np.array([-850.0, -850.0]), np.array([850.0, -850.0])
    ]
    scan_points = solve_tsp_nearest_neighbor(raw_points, start_idx=0)

    for i, scan_pos in enumerate(scan_points):
        if time.time() - start_time > 16 * 60:
            print("⏰ 时间到，停止扫描")
            break
        print(f"\n{'='*50}\n📍 扫描点 {i+1}/{len(scan_points)}: ({scan_pos[0]:.0f}, {scan_pos[1]:.0f})\n{'='*50}")
        scan_and_clear(scan_pos, start_time, cleared_channels)
        if len(cleared_channels) >= 20:
            break

    print(f"\n🎉 结束！共清除 {len(cleared_channels)} 个干扰源。")
    print(f"成功频道：{sorted(cleared_channels)}")
    post("/exit", base(_uid("exit")))


if __name__ == "__main__":
    main()