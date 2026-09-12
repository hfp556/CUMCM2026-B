"""
Q3_fast2.py — 在 Q3_fast.py 基础上进一步压缩虚拟时间
====================================================
改动点（策略框架不变：全覆盖扫描 + 交会定位/追迹清除）：

【扫描】8 点正八边形环（半径 945 m，弦 723 m，总路程 6008 m，原 7200 m）
  - 覆盖保证：圆盘(1800) 内任意点到最近扫描点 ≤ 994.8 m < 1000 m ≤ R_i，
    所有干扰源必定被发现；圆心距各环点 945 m ≤ 1000 m，无需原点测量。
  - 跳频：区域直径 D ≤ 60 m（已定位）或区域距扫描点 > 1500 m（必无信号）
    的频道不再重复测量，省测量与换频道时间。
  - 方向：根据首个环点的示向度估计源群方位，选择扫描终点靠近源群一侧。
  - 顺路清除：区域已建立（D ≤ 90 m）且绕路 ≤ 250 m 的频道，在扫描途中
    顺手清掉，省去之后专门跑一趟。

【处理】Held-Karp 精确 TSP（位掩码 DP，≤16 点）替代最近邻 + 2-opt；
  每清一个源后，从机器人实际位置重新规划剩余巡回。
其余（D≤90 质心清除、追迹 200 m 步进 + 夹逼收尾）与 Q3_fast.py 一致。

原文件 Q3_fast.py / Q3_dzz.py 均保持不变。
"""

import math
import sys
import time
import itertools

import numpy as np

# Windows 控制台编码兼容（避免 emoji 在 GBK 终端下报错）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from geo_common import localize_region, convex_diameter
from api_utils import post, base

_req_counter = itertools.count()
LAST_VT = 0.0          # 最近一次有效虚拟时刻（秒）
CURRENT_POS = np.array([0.0, 0.0])   # 机器人当前位置（由本程序发出的指令推算）
SCAN_R = 945.0         # 八边形扫描环半径（覆盖最坏点 994.8 m < 1000 m）
SCAN_N = 8             # 环上扫描点数
HOMING_STEP = 200.0    # 追迹大步长
BRACKET_END = 36.0     # 夹逼收尾阈值（bracket/2 ≤ 18 m + 侧向误差 < 20 m）
R_MAX = 1500.0         # 干扰源接收半径上界
ENROUTE_DETOUR = 250.0 # 扫描途中顺路清除的绕路阈值（米）


def _uid(prefix):
    return f"{prefix}-{next(_req_counter)}-{int(time.time()*1000)}"


def _post(path, payload, retries=2):
    """发送请求；网络异常时按接口规范复用原 request_id 重试。"""
    for attempt in range(retries + 1):
        resp = post(path, payload)
        if resp is not None:
            return resp
        if attempt < retries:
            time.sleep(0.3)
    return None


def _track(resp):
    """记录虚拟时刻（单调更新；accepted=false 时 vt=0 会被自然忽略）。"""
    global LAST_VT
    if resp is not None:
        vt = resp.get("virtual_time_s")
        if vt is not None and float(vt) > LAST_VT:
            LAST_VT = float(vt)
    return resp


def measure(x, y, channel):
    global CURRENT_POS
    payload = base(_uid(f"m-{channel}"))
    payload["position"] = {"x": float(x), "y": float(y)}
    payload["channel"] = channel
    resp = _track(_post("/measure", payload))
    if resp is not None and resp.get("accepted", True) is not False:
        CURRENT_POS = np.array([float(x), float(y)])
    return resp


def try_clear(channel, pos, tag=""):
    global CURRENT_POS
    if np.linalg.norm(pos) > 1900.0:
        return False
    payload = base(_uid(f"clear-{tag}-{channel}"))
    payload["position"] = {"x": float(pos[0]), "y": float(pos[1])}
    payload["channel"] = channel
    resp = _track(_post("/clear", payload))
    if resp is not None and resp.get("accepted", True) is not False:
        CURRENT_POS = np.array([float(pos[0]), float(pos[1])])
    return bool(resp and resp.get("clear_result") == "success")


def uvec(deg):
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)])


def angdiff(a, b):
    """a - b 的带符号角度差，范围 (-180, 180]。"""
    d = (a - b + 180.0) % 360.0 - 180.0
    if d == -180.0:
        d = 180.0
    return d


def region_of(S_list, theta_list):
    """交会区域；观测矛盾时依次丢弃最旧观测直到非空。"""
    pts = [np.asarray(p, dtype=float) for p in S_list]
    ths = list(theta_list)
    while len(pts) >= 1:
        poly = localize_region(pts, ths)
        if len(poly) > 0:
            return poly
        pts = pts[1:]
        ths = ths[1:]
    return np.empty((0, 2))


def should_skip_scan(ch, detections, cleared, P, cache):
    """扫描跳频判定：已定位 / 区域距扫描点必无信号 → 跳过测量。"""
    if ch in cleared:
        return True
    if ch not in detections:
        return False                       # 未检测到的频道必须测（覆盖保证）
    key = (ch, len(detections[ch][0]))
    if key in cache:
        D, poly = cache[key]
    else:
        poly = region_of(detections[ch][0], detections[ch][1])
        if len(poly) == 0:
            cache[key] = (float("inf"), poly)
            return False
        D, _ = convex_diameter(poly)
        cache[key] = (D, poly)
    if D <= 60.0:
        return True                        # 区域已足够小，无需更多观测
    if D <= 90.0:
        # 区域所有顶点距扫描点均 > 1500 m（接收半径上界）→ 必然无信号
        if min(float(np.linalg.norm(v - P)) for v in poly) > R_MAX:
            return True
    return False


def iterative_clear(channel, S_list, theta_list, max_iter=8):
    """以交会区域质心迭代逼近并清除（兜底/收尾）。"""
    for i in range(max_iter):
        poly = region_of(S_list, theta_list)
        if len(poly) == 0:
            return False
        D, _ = convex_diameter(poly)
        cx = float(np.mean(poly[:, 0]))
        cy = float(np.mean(poly[:, 1]))
        print(f"  [{channel}] 迭代 {i+1}: D = {D:.1f} m")
        if D <= 30.0:
            if try_clear(channel, (cx, cy), "iter"):
                print(f"  [{channel}] ✅ 清除成功！")
                return True
            # 清除失败：原地补测，收紧后再清
            r = measure(cx, cy, channel)
            if r and r.get("measure_result") == "near":
                return try_clear(channel, (cx, cy), "iter-near")
            if r and r.get("measure_result") == "direction":
                S_list.append(np.array([cx, cy]))
                theta_list.append(r["svd_deg"])
                continue
        r = measure(cx, cy, channel)
        if r is None:
            return False
        mr = r.get("measure_result")
        if mr == "near":
            ok = try_clear(channel, (cx, cy), "iter-near")
            print(f"  [{channel}] {'✅' if ok else '❌'} near 清除")
            return ok
        if mr == "direction":
            S_list.append(np.array([cx, cy]))
            theta_list.append(r["svd_deg"])
        else:
            # no_signal（异常）：沿最新示向度再走 100 m 探测一次
            th = theta_list[-1]
            nxt = np.array([cx, cy]) + 100.0 * uvec(th)
            r2 = measure(nxt[0], nxt[1], channel)
            if r2 and r2.get("measure_result") == "near":
                return try_clear(channel, nxt, "iter-adv")
            return False
    return False


def bracket_clear(channel, pos, theta, bracket, S_list, theta_list):
    """源在当前位置沿示向度 ±bracket 米内：对半夹逼 → 收尾清除。"""
    cur = np.asarray(pos, dtype=float)
    cur_th = theta
    for _ in range(10):
        if bracket <= BRACKET_END:
            # 收尾：向源走 bracket/2（≤18 m，侧向误差 ≤0.7 m）→ 必在 20 m 内
            fin = cur + (bracket / 2.0) * uvec(cur_th)
            if try_clear(channel, fin, "bracket-end"):
                print(f"  [{channel}] ✅ 夹逼收尾清除成功！")
                return True
            # 保险：原地补测后再清
            r = measure(fin[0], fin[1], channel)
            if r and r.get("measure_result") == "near":
                return try_clear(channel, fin, "bracket-end2")
            if r and r.get("measure_result") == "direction":
                th = r["svd_deg"]
                S_list.append(fin)
                theta_list.append(th)
                fin2 = fin + (bracket / 4.0) * uvec(th)
                return try_clear(channel, fin2, "bracket-end3")
            return iterative_clear(channel, S_list, theta_list)
        step = min(bracket, max(BRACKET_END, bracket / 2.0))
        nxt = cur + step * uvec(cur_th)
        r = measure(nxt[0], nxt[1], channel)
        if r is None:
            return iterative_clear(channel, S_list, theta_list)
        mr = r.get("measure_result")
        if mr == "near":
            ok = try_clear(channel, nxt, "bracket-near")
            return ok or iterative_clear(channel, S_list, theta_list)
        if mr == "direction":
            th = r["svd_deg"]
            S_list.append(nxt)
            theta_list.append(th)
            if abs(angdiff(th, cur_th)) > 90.0:
                bracket = step                 # 又越过了：源在本步内
            else:
                bracket = max(bracket - step, 0.0)   # 源仍在前方
            cur, cur_th = nxt, th
        else:
            # 距源 ≤ 400 m < 接收半径，no_signal 不应发生；保守继续前进
            cur = nxt
            bracket = max(bracket - step, 0.0)
    return iterative_clear(channel, S_list, theta_list)


def homing_clear(channel, start_pos, theta, bracket0, S_list, theta_list):
    """追迹逼近：从 start_pos 沿示向度前进，越过源后夹逼收尾。
    bracket0：从 start_pos 出发时源在示向度后方的深度上界。"""
    cur = np.asarray(start_pos, dtype=float)
    cur_th = theta

    # ---- 先原地检测一次（确认源在前方还是后方）----
    r = measure(cur[0], cur[1], channel)
    if r is None:
        return False
    mr = r.get("measure_result")
    if mr == "near":
        return try_clear(channel, cur, "home-near")
    if mr == "direction":
        th = r["svd_deg"]
        if abs(angdiff(th, theta)) > 90.0:
            # 源在后方 → 直接进入夹逼
            return bracket_clear(channel, cur, th, bracket0, S_list, theta_list)
        cur_th = th
    # no_signal：源在前方更远处（仍在接收半径之外），继续前进

    # ---- 阶段1：沿方位前进，步长 200 m ----
    for _ in range(8):
        nxt = cur + HOMING_STEP * uvec(cur_th)
        if np.linalg.norm(nxt) > 2100.0:
            nxt = cur + 100.0 * uvec(cur_th)
            if np.linalg.norm(nxt) > 2100.0:
                break
        r = measure(nxt[0], nxt[1], channel)
        if r is None:
            return iterative_clear(channel, S_list, theta_list)
        mr = r.get("measure_result")
        if mr == "near":
            ok = try_clear(channel, nxt, "home-near")
            if ok:
                print(f"  [{channel}] ✅ 追迹 near 清除成功！")
            return ok or iterative_clear(channel, S_list, theta_list)
        if mr == "direction":
            th = r["svd_deg"]
            S_list.append(nxt)
            theta_list.append(th)
            if abs(angdiff(th, cur_th)) > 90.0:
                # 越过了源：源在最后 200 m 步内
                return bracket_clear(channel, nxt, th, HOMING_STEP, S_list, theta_list)
            cur, cur_th = nxt, th
        else:
            # no_signal：源仍在接收半径外，继续同方向前进
            cur = nxt
    # 8 步仍未翻转（理论上不可能，源距 ≤ 1500 m）→ 兜底
    return iterative_clear(channel, S_list, theta_list)


def join_depth(lo, hi):
    """追迹入射线深度：略偏向浅端（越过源的代价高于前进）。"""
    return lo + 0.35 * (hi - lo)


def process_channel(ch, S_list, theta_list):
    """处理一个频道：交会直径决定 直接清除 / 补测迭代 / 追迹。"""
    poly = region_of(S_list, theta_list)
    if len(poly) == 0:
        print(f"  [{ch}] ⚠️ 交会为空（异常），单点追迹兜底")
        return homing_clear(ch, S_list[0], theta_list[0], 1500.0, S_list, theta_list)
    D, _ = convex_diameter(poly)
    cx = float(np.mean(poly[:, 0]))
    cy = float(np.mean(poly[:, 1]))
    print(f"  [{ch}] 交会 D = {D:.1f} m（{len(S_list)} 个观测）")
    if D <= 90.0:
        # 直接去质心：D ≤ 30 大概率一次成功；否则补测后迭代
        if try_clear(ch, (cx, cy), "direct"):
            print(f"  [{ch}] ✅ 直接清除成功！")
            return True
        r = measure(cx, cy, ch)
        if r is not None:
            if r.get("measure_result") == "near":
                return try_clear(ch, (cx, cy), "direct-near")
            if r.get("measure_result") == "direction":
                S_list.append(np.array([cx, cy]))
                theta_list.append(r["svd_deg"])
        return iterative_clear(ch, S_list, theta_list)
    # 追迹：选离质心最近的观测点 P*，沿其射线前进
    best = 0
    for i in range(1, len(S_list)):
        if np.linalg.norm(S_list[i] - np.array([cx, cy])) < \
           np.linalg.norm(S_list[best] - np.array([cx, cy])):
            best = i
    P = np.asarray(S_list[best], dtype=float)
    th = theta_list[best]
    u = uvec(th)
    depth = (poly - P) @ u
    lo = float(np.min(depth))
    hi = float(np.max(depth))
    jd = join_depth(lo, hi)
    join = P + jd * u
    print(f"  [{ch}] 追迹：源深 {lo:.0f}~{hi:.0f} m，从 ({join[0]:.0f}, {join[1]:.0f}) 出发")
    return homing_clear(ch, join, th, jd - lo, S_list, theta_list)


def tsp_exact(points, start_point):
    """Held-Karp 精确 TSP：开放路径，虚拟起点 start_point（不在访问集内）。"""
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0]
    pts = [np.asarray(p, dtype=float) for p in points]
    d = np.array([[float(np.linalg.norm(pts[i] - pts[j])) for j in range(n)]
                  for i in range(n)])
    size = 1 << n
    dp = np.full((size, n), np.inf)
    par = np.full((size, n), -1, dtype=np.int32)
    for j in range(n):
        dp[1 << j][j] = float(np.linalg.norm(np.asarray(start_point) - pts[j]))
    for mask in range(1, size):
        base = dp[mask]
        if not np.isfinite(base).any():
            continue
        for k in range(n):
            if (mask >> k) & 1:
                continue
            cand = base + d[:, k]
            m = int(np.argmin(cand))
            nm = mask | (1 << k)
            if cand[m] < dp[nm][k]:
                dp[nm][k] = cand[m]
                par[nm][k] = m
    full = size - 1
    j = int(np.argmin(dp[full]))
    order = []
    mask = full
    while j != -1:
        order.append(j)
        nj = int(par[mask][j])
        mask ^= (1 << j)
        j = nj
    order.reverse()
    return order


def main():
    start_time = time.time()
    res = _track(_post("/enter", base(_uid("enter"))))
    if not res or res.get("accepted") is not True:
        print("❌ 进入失败")
        return
    print("✅ 进入成功！【快速版2：8点覆盖扫描 + 交会定位/追迹清除 + 精确TSP】\n")

    cleared = set()
    detections = {}   # channel -> ([points], [thetas])
    cache = {}        # (ch, 观测数) -> (D, poly) 跳频缓存

    # ===== 阶段1：发现扫描（八边形环，无原点测量）=====
    # 首个环点任意取（问题旋转对称）
    P0 = SCAN_R * np.array([1.0, 0.0])

    def scan_point(P, i):
        print(f"\n{'='*50}\n📍 扫描点 {i+1}/{SCAN_N}: "
              f"({P[0]:.0f}, {P[1]:.0f})")
        for ch in range(1, 21):
            if should_skip_scan(ch, detections, cleared, P, cache):
                continue
            r = measure(P[0], P[1], ch)
            if r is None:
                continue
            mr = r.get("measure_result")
            if mr == "direction":
                detections.setdefault(ch, ([], []))
                detections[ch][0].append(P.copy())
                detections[ch][1].append(r["svd_deg"])
                cache.pop((ch, len(detections[ch][0]) - 1), None)
                print(f"  ✅ 频道 {ch} 有信号，{r['svd_deg']}°")
            elif mr == "near":
                print(f"  🔥 频道 {ch} 距离过近，直接清除！")
                if try_clear(ch, P, "scan-near"):
                    cleared.add(ch)

    scan_point(P0, 0)

    # 根据首个环点示向度估计源群方位，选择扫描方向（终点靠近源群）
    sx = sy = 0.0
    for ch in detections:
        for th in detections[ch][1]:
            v = uvec(th)
            sx += v[0]
            sy += v[1]
    theta_c = math.degrees(math.atan2(sy, sx)) if (sx or sy) else 0.0
    # 两种方向的终点：+45° 或 -45°（首个环点在 0°）
    d_plus = abs(angdiff(45.0, theta_c))
    d_minus = abs(angdiff(-45.0, theta_c))
    sdir = 1 if d_plus <= d_minus else -1
    print(f"\n源群方位估计 {theta_c:.0f}°，扫描终点取 "
          f"{'+45°' if sdir == 1 else '-45°'}")

    tried_enroute = set()   # 途中清除已尝试过的频道（避免重复兜圈）
    for k in range(1, SCAN_N):
        if time.time() - start_time > 17 * 60:
            print("⏰ 程序运行时间不足，停止扫描")
            break
        a = math.radians(45.0 * k * sdir)
        P = SCAN_R * np.array([math.cos(a), math.sin(a)])
        scan_point(P, k)

        # ---- 顺路清除：区域已建立（D ≤ 90）且绕路 ≤ 阈值的频道 ----
        if k >= 2:
            nk = k + 1
            if nk < SCAN_N:
                Q = SCAN_R * np.array([math.cos(math.radians(45.0 * nk * sdir)),
                                       math.sin(math.radians(45.0 * nk * sdir))])
                while True:
                    best_ch, best_det = None, float("inf")
                    best_S, best_T = None, None
                    for ch in detections:
                        if ch in cleared or ch in tried_enroute:
                            continue
                        key = (ch, len(detections[ch][0]))
                        if key in cache:
                            D, poly = cache[key]
                        else:
                            poly = region_of(detections[ch][0], detections[ch][1])
                            if len(poly) == 0:
                                cache[key] = (float("inf"), poly)
                                continue
                            D, _ = convex_diameter(poly)
                            cache[key] = (D, poly)
                        if D > 90.0:
                            continue
                        C = np.array([float(np.mean(poly[:, 0])),
                                      float(np.mean(poly[:, 1]))])
                        det = (float(np.linalg.norm(CURRENT_POS - C))
                               + float(np.linalg.norm(C - Q))
                               - float(np.linalg.norm(CURRENT_POS - Q)))
                        if det < best_det:
                            best_det = det
                            best_ch = ch
                            best_S = list(detections[ch][0])
                            best_T = list(detections[ch][1])
                    if best_ch is None or best_det > ENROUTE_DETOUR:
                        break
                    tried_enroute.add(best_ch)
                    print(f"\n🚗 顺路清除频道 {best_ch}（绕路 {best_det:.0f} m）")
                    if process_channel(best_ch, best_S, best_T):
                        cleared.add(best_ch)

    print(f"\n发现阶段结束：检测到 {len(detections)} 个频道，"
          f"就地清除 {len(cleared)} 个；虚拟时间 {LAST_VT:.0f} s")

    # ===== 阶段2：规划并处理 =====
    targets = []
    for ch in sorted(detections):
        pts, ths = detections[ch]
        S_list = list(pts)
        theta_list = list(ths)
        poly = region_of(S_list, theta_list)
        point = S_list[0]
        if len(poly) > 0:
            D, _ = convex_diameter(poly)
            cx = float(np.mean(poly[:, 0]))
            cy = float(np.mean(poly[:, 1]))
            if D <= 90.0:
                point = np.array([cx, cy])
            else:
                # 追迹代理点：区域沿最近观测射线的中深位置
                best = 0
                for i in range(1, len(S_list)):
                    if np.linalg.norm(S_list[i] - np.array([cx, cy])) < \
                       np.linalg.norm(S_list[best] - np.array([cx, cy])):
                        best = i
                P = np.asarray(S_list[best], dtype=float)
                u = uvec(theta_list[best])
                depth = (poly - P) @ u
                lo = float(np.min(depth))
                hi = float(np.max(depth))
                point = P + join_depth(lo, hi) * u   # 与 process_channel 的入射线一致
        targets.append({"ch": ch, "point": point, "S": S_list, "T": theta_list})

    remaining = [t for t in targets if t["ch"] not in cleared]
    order = tsp_exact([t["point"] for t in remaining], CURRENT_POS)
    print(f"\n处理顺序：{[remaining[i]['ch'] for i in order]}")
    while remaining:
        if time.time() - start_time > 17 * 60:
            print("⏰ 时间到，停止处理")
            break
        # 每清一个源后，从机器人实际位置重新规划剩余巡回（精确 TSP）
        order = tsp_exact([t["point"] for t in remaining], CURRENT_POS)
        t = remaining.pop(order[0])
        ch = t["ch"]
        print(f"\n=== 处理频道 {ch} ===")
        if process_channel(ch, t["S"], t["T"]):
            cleared.add(ch)

    print(f"\n🎉 结束！共清除 {len(cleared)} 个干扰源。")
    print(f"成功频道：{sorted(cleared)}")
    failed = sorted(set(detections) - cleared)
    if failed:
        print(f"⚠️ 未清除频道：{failed}")
    print(f"总虚拟时间：{LAST_VT:.1f} 秒（{LAST_VT/60.0:.1f} 分钟）")
    _track(_post("/exit", base(_uid("exit"))))


if __name__ == "__main__":
    main()
