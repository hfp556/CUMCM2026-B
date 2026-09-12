"""
Q4.py — 问题四主程序（方案一：8 个外围补充点）
保留 A/B/D/E 全部优化，仅扩展扫描点覆盖外围盲区。
"""

import math
import time
import numpy as np

from Q1 import localize_region, convex_diameter
from Q2_FOR4 import get_problem4_point_dynamic_alternating
from api_utils import post, base, measure


# ============================================================
# 全局状态（方向B）
# ============================================================
channel_state = {}


def init_channel_state(channel):
    if channel not in channel_state:
        channel_state[channel] = {
            "tried_offsets": set(),
            "last_good_pos": None,
            "fail_count": 0,
            "cleared": False,
        }


def mark_cleared(channel):
    init_channel_state(channel)
    channel_state[channel]["cleared"] = True


# ============================================================
# 方案一：8 个外围补充点（23 个点）
# ============================================================
def generate_grid_points(step=800.0, max_radius=1800.0):
    """
    生成网格点：
    - 15 个内层 800m 网格点
    - 8 个半径 1700m 的外围点（每 45° 一个）
    """
    points = []
    rng = np.arange(-max_radius, max_radius + step, step)
    for x in rng:
        for y in rng:
            if math.sqrt(x * x + y * y) <= max_radius:
                points.append(np.array([float(x), float(y)]))

    # ⚠️ 外围补充点：半径 1700m，每 45°
    outer_radius = 1700.0
    for angle_deg in range(0, 360, 45):
        ang = math.radians(angle_deg)
        points.append(np.array([
            outer_radius * math.cos(ang),
            outer_radius * math.sin(ang)
        ]))

    return points


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


# ============================================================
# clear_channel（方向 A + D）
# ============================================================
def clear_channel(channel, S_list, theta_list):
    init_channel_state(channel)

    for iteration in range(7):
        poly = localize_region(S_list, theta_list)
        if len(poly) == 0:
            print(f"  [{channel}] ⚠️ 交会区域为空。")
            return False, S_list[-1] if S_list else np.array([0.0, 0.0])

        D, _ = convex_diameter(poly)
        print(f"  [{channel}] 迭代 {iteration+1}: D = {D:.2f} 米")

        should_try_clear = (D <= 20.0) or (D <= 50.0 and iteration >= 2)
        if should_try_clear:
            cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
            print(f"  [{channel}] 🎯 尝试清除 @ ({cx:.0f}, {cy:.0f}) [D={D:.1f}]")
            payload = base(f"clear-{channel}-{iteration}-{int(time.time()*1000)}")
            payload["position"] = {"x": float(cx), "y": float(cy)}
            payload["channel"] = channel
            resp = post("/clear", payload)
            if resp and resp.get("clear_result") == "success":
                print(f"  [{channel}] ✅ 清除成功！")
                mark_cleared(channel)
                channel_state[channel]["last_good_pos"] = np.array([cx, cy])
                return True, np.array([cx, cy])

        cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
        print(f"  [{channel}] 移动至 ({cx:.0f}, {cy:.0f}) 追加测量...")
        resp = measure(cx, cy, channel, f"it-{channel}-{iteration}-{int(time.time()*1000)}")

        if resp is None:
            return False, np.array([cx, cy])

        mr = resp.get("measure_result")

        if mr == "near":
            payload = base(f"clear-near-{channel}-{iteration}-{int(time.time()*1000)}")
            payload["position"] = {"x": float(cx), "y": float(cy)}
            payload["channel"] = channel
            rc = post("/clear", payload)
            if rc and rc.get("clear_result") == "success":
                print(f"  [{channel}] ✅ 清除成功（near）！")
                mark_cleared(channel)
                return True, np.array([cx, cy])
            return False, np.array([cx, cy])

        elif mr == "direction":
            S_list.append(np.array([cx, cy]))
            theta_list.append(resp["svd_deg"])

        else:
            print(f"  [{channel}] ⚠️ 中心点盲区。尝试反向逃逸...")
            last_theta = theta_list[-1] if theta_list else 0.0
            back_ang = math.radians(last_theta + 180)
            back_x = cx + 200 * math.cos(back_ang)
            back_y = cy + 200 * math.sin(back_ang)

            if np.linalg.norm([back_x, back_y]) > 1780:
                return False, np.array([cx, cy])

            print(f"  [{channel}] 逃逸至 ({back_x:.0f}, {back_y:.0f})...")
            resp_b = measure(back_x, back_y, channel, f"bk-{channel}-{iteration}-{int(time.time()*1000)}")
            if resp_b is None:
                return False, np.array([cx, cy])

            mr_b = resp_b.get("measure_result")
            if mr_b == "near":
                payload = base(f"clear-bk-{channel}-{iteration}-{int(time.time()*1000)}")
                payload["position"] = {"x": float(back_x), "y": float(back_y)}
                payload["channel"] = channel
                rc = post("/clear", payload)
                if rc and rc.get("clear_result") == "success":
                    print(f"  [{channel}] ✅ 逃逸清除成功！")
                    mark_cleared(channel)
                    return True, np.array([back_x, back_y])
                return False, np.array([back_x, back_y])
            elif mr_b == "direction":
                print(f"  [{channel}] 逃逸后重新捕获 {resp_b['svd_deg']}°")
                S_list.append(np.array([back_x, back_y]))
                theta_list.append(resp_b["svd_deg"])
            else:
                print(f"  [{channel}] 逃逸也失败，放弃。")
                return False, np.array([cx, cy])

    print(f"  [{channel}] ⚠️ 迭代耗尽，放弃。")
    return False, S_list[-1] if S_list else np.array([0.0, 0.0])


# ============================================================
# 阶梯逼近（方向E）
# ============================================================
def ladder_advance(channel, S0, theta):
    step = 150.0
    cur_pos = np.array(S0, dtype=float)
    for k in range(1, 11):
        ang = math.radians(theta)
        px = cur_pos[0] + step * math.cos(ang)
        py = cur_pos[1] + step * math.sin(ang)
        if np.linalg.norm([px, py]) > 1780:
            return False, cur_pos

        print(f"  [{channel}] 阶梯 {k}: 推进至 ({px:.0f}, {py:.0f})...")
        r = measure(px, py, channel, f"ld-{channel}-{k}-{int(time.time()*1000)}")
        if r is None:
            return False, np.array([px, py])
        cur_pos = np.array([px, py])

        mr = r.get("measure_result")
        if mr == "near":
            payload = base(f"clear-ld-{channel}-{k}-{int(time.time()*1000)}")
            payload["position"] = {"x": float(px), "y": float(py)}
            payload["channel"] = channel
            rc = post("/clear", payload)
            if rc and rc.get("clear_result") == "success":
                print(f"  [{channel}] ✅ 阶梯清除成功！")
                mark_cleared(channel)
                return True, cur_pos
        elif mr == "direction":
            S_list = [np.array(S0), cur_pos]
            theta_list = [theta, r["svd_deg"]]
            ok, _ = clear_channel(channel, S_list, theta_list)
            if ok:
                return True, cur_pos
        else:
            perp = math.radians(theta + 90)
            for sign in (1, -1):
                sx = px + sign * 150 * math.cos(perp)
                sy = py + sign * 150 * math.sin(perp)
                if np.linalg.norm([sx, sy]) > 1780:
                    continue
                print(f"  [{channel}] 横向摆动 ({sx:.0f}, {sy:.0f})...")
                rs = measure(sx, sy, channel, f"sw-{channel}-{k}-{sign}-{int(time.time()*1000)}")
                if rs and rs.get("measure_result") == "near":
                    payload = base(f"clear-sw-{channel}-{k}-{sign}-{int(time.time()*1000)}")
                    payload["position"] = {"x": float(sx), "y": float(sy)}
                    payload["channel"] = channel
                    rc = post("/clear", payload)
                    if rc and rc.get("clear_result") == "success":
                        mark_cleared(channel)
                        return True, np.array([sx, sy])
                elif rs and rs.get("measure_result") == "direction":
                    S_list = [np.array(S0), np.array([sx, sy])]
                    theta_list = [theta, rs["svd_deg"]]
                    ok, _ = clear_channel(channel, S_list, theta_list)
                    if ok:
                        return True, np.array([sx, sy])
    return False, cur_pos


# ============================================================
# 扫描 + 处理
# ============================================================
def scan_and_clear(current_pos, start_time, cleared_set):
    scan_x, scan_y = current_pos[0], current_pos[1]
    found = []
    print(f"\n--- 在 ({scan_x:.0f}, {scan_y:.0f}) 扫描频道 ---")
    for ch in range(1, 21):
        if ch in cleared_set:
            continue
        r = measure(scan_x, scan_y, ch, f"sc-{ch}-{int(scan_x)}-{int(scan_y)}-{int(time.time()*1000)}")
        if r and r.get("measure_result") == "direction":
            print(f"✅ 频道 {ch} 有信号, {r['svd_deg']}°")
            found.append((ch, r['svd_deg']))
        elif r and r.get("measure_result") == "near":
            payload = base(f"clear-near-sc-{ch}-{int(time.time()*1000)}")
            payload["position"] = {"x": scan_x, "y": scan_y}
            payload["channel"] = ch
            post("/clear", payload)
            mark_cleared(ch)
            cleared_set.add(ch)

    cnt = 0
    for ch, th in found:
        if time.time() - start_time > 16 * 60:
            break
        if ch in cleared_set:
            continue

        init_channel_state(ch)
        state = channel_state[ch]
        print(f"\n=== 处理频道 {ch} (历史失败 {state['fail_count']} 次) ===")

        S1 = np.array([scan_x, scan_y])

        if state["fail_count"] >= 2:
            print(f"  [{ch}] ⚡ 历史失败≥2，直接启用阶梯逼近...")
            ok, _ = ladder_advance(ch, S1, th)
            if ok:
                cleared_set.add(ch)
                cnt += 1
            else:
                state["fail_count"] += 1
            continue

        S_list, theta_list = [S1], [th]
        p2_ok = False
        for att in range(10):
            S2, params = get_problem4_point_dynamic_alternating(S1, th, attempt=att)
            if S2 is None:
                break
            delta, d = params
            if delta in state["tried_offsets"]:
                continue
            state["tried_offsets"].add(delta)

            print(f"  [{ch}] Q2试探 {att+1}: 偏转{delta}°, {d:.0f}m -> ({S2[0]:.0f},{S2[1]:.0f})")
            r2 = measure(S2[0], S2[1], ch, f"p2-{ch}-{att}-{int(time.time()*1000)}")
            if r2 and r2.get("measure_result") == "direction":
                S_list.append(S2)
                theta_list.append(r2["svd_deg"])
                p2_ok = True
                break
            elif r2 and r2.get("measure_result") == "near":
                payload = base(f"clear-p2near-{ch}-{att}-{int(time.time()*1000)}")
                payload["position"] = {"x": float(S2[0]), "y": float(S2[1])}
                payload["channel"] = ch
                rc = post("/clear", payload)
                if rc and rc.get("clear_result") == "success":
                    mark_cleared(ch)
                    cleared_set.add(ch)
                    cnt += 1
                    p2_ok = True
                    state["fail_count"] = 0
                break

        if p2_ok and len(S_list) > 1:
            ok, last_pos = clear_channel(ch, S_list, theta_list)
            if ok:
                cleared_set.add(ch)
                cnt += 1
                state["fail_count"] = 0
                state["last_good_pos"] = last_pos
            else:
                state["fail_count"] += 1
                print(f"  [{ch}] 🔄 clear_channel 失败，切换阶梯逼近...")
                ok2, _ = ladder_advance(ch, S1, th)
                if ok2:
                    cleared_set.add(ch)
                    cnt += 1
                    state["fail_count"] = 0
                else:
                    state["fail_count"] += 1
        elif not p2_ok:
            print(f"  [{ch}] 🚀 Q2 试探不足，启用阶梯逼近...")
            ok, _ = ladder_advance(ch, S1, th)
            if ok:
                cleared_set.add(ch)
                cnt += 1
                state["fail_count"] = 0
            else:
                state["fail_count"] += 1

    return cnt, np.array([scan_x, scan_y])


# ============================================================
# 主程序
# ============================================================
def main():
    start = time.time()
    r = post("/enter", base(f"enter-{int(time.time()*1000)}"))
    if not r or r.get("accepted") is not True:
        print("❌ 进入失败")
        return
    print("✅ 进入成功！【方案一：8 个外围补充点 + A/B/D/E】\n")

    cleared = set()

    grid_pts = generate_grid_points(step=800.0, max_radius=1800.0)
    print(f"📐 生成网格点：{len(grid_pts)} 个")
    for i, p in enumerate(grid_pts):
        print(f"   {i+1}. ({p[0]:.0f}, {p[1]:.0f})")

    start_idx = 0
    min_d = float('inf')
    for i, p in enumerate(grid_pts):
        if np.linalg.norm(p) < min_d:
            min_d = np.linalg.norm(p)
            start_idx = i

    pts = solve_tsp_nearest_neighbor(grid_pts, start_idx=start_idx)
    total_dist = sum(np.linalg.norm(pts[i] - pts[i-1]) for i in range(1, len(pts)))
    print(f"\n📏 TSP 排序后总移动距离：{total_dist:.0f}m\n")

    for i, p in enumerate(pts):
        if time.time() - start > 16 * 60:
            print("⏰ 时间到，停止搜索")
            break
        print(f"\n{'='*50}\n📍 扫描点 {i+1}/{len(pts)}: ({p[0]:.0f}, {p[1]:.0f})\n{'='*50}")
        scan_and_clear(p, start, cleared)
        if len(cleared) >= 20:
            break

    print(f"\n🎉 结束！共清除 {len(cleared)} 个目标。")
    print("\n📊 频道处理状态汇总：")
    for ch in sorted(channel_state.keys()):
        s = channel_state[ch]
        print(f"  频道 {ch}: cleared={s['cleared']}, fails={s['fail_count']}, tried_offsets={len(s['tried_offsets'])}")

    post("/exit", base(f"exit-{int(time.time()*1000)}"))


if __name__ == "__main__":
    main()