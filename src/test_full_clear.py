import time
import numpy as np
from Q1 import localize_region, convex_diameter
from Q2 import get_problem2_point_dynamic
from api_utils import post, base, measure

def clear_channel(channel, S_list, theta_list):
    """问题一稳健逼近逻辑（保留原样）"""
    D = float('inf')
    max_iterations = 6
    for iteration in range(max_iterations):
        poly = localize_region(S_list, theta_list)
        if len(poly) == 0:
            print(f"  [{channel}] ⚠️ 交会区域为空，信号丢失。跳过。")
            return False, S_list[-1]
        D, _ = convex_diameter(poly)
        print(f"  [{channel}] 迭代 {iteration+1}: 直径 D = {D:.2f} 米")
        if D <= 20.0:
            center_x = np.mean(poly[:, 0])
            center_y = np.mean(poly[:, 1])
            print(f"  [{channel}] 🎉 D <= 20米，前往中心 ({center_x:.1f}, {center_y:.1f}) 清除...")
            payload = base(f"clear-ch{channel}")
            payload["position"] = {"x": float(center_x), "y": float(center_y)}
            payload["channel"] = channel
            resp = post("/clear", payload)
            if resp and resp.get("clear_result") == "success":
                print(f"  [{channel}] ✅ 清除成功！")
                return True, np.array([center_x, center_y])
            else:
                print(f"  [{channel}] ❌ 未命中，继续迭代...")
        center_x = np.mean(poly[:, 0])
        center_y = np.mean(poly[:, 1])
        print(f"  [{channel}] 移动到中心 ({center_x:.1f}, {center_y:.1f}) 追加测量...")
        resp = measure(center_x, center_y, channel, f"iter-{channel}-{iteration}")
        if resp is None:
            return False, np.array([center_x, center_y])
        if resp.get("measure_result") == "near":
            print(f"  [{channel}] 🔥 距离过近！直接清除...")
            payload = base(f"clear-near-{channel}")
            payload["position"] = {"x": float(center_x), "y": float(center_y)}
            payload["channel"] = channel
            resp_clear = post("/clear", payload)
            if resp_clear and resp_clear.get("clear_result") == "success":
                print(f"  [{channel}] ✅ 清除成功！")
                return True, np.array([center_x, center_y])
            return False, np.array([center_x, center_y])
        elif resp.get("measure_result") == "direction":
            new_theta = resp["svd_deg"]
            S_list.append(np.array([center_x, center_y]))
            theta_list.append(new_theta)
        else:
            print(f"  [{channel}] ⚠️ 追加测量无信号，丢失目标。")
            return False, np.array([center_x, center_y])
    print(f"  [{channel}] ⚠️ 迭代耗尽，放弃该目标。")
    return False, S_list[-1]

def scan_and_clear(current_pos, start_time, cleared_channels):
    """
    【优化6：多区域扫描】在指定位置扫描所有频道，并清除未清除的目标。
    返回：成功清除的数量，机器狗最终位置
    """
    scan_x, scan_y = current_pos[0], current_pos[1]
    found_channels = []
    
    print(f"\n--- 在 ({scan_x:.0f}, {scan_y:.0f}) 扫描频道 1~20 ---")
    for ch in range(1, 21):
        if ch in cleared_channels:
            continue # 已经清除的跳过
        resp = measure(scan_x, scan_y, ch, f"scan-{ch}-{int(scan_x)}-{int(scan_y)}")
        if resp and resp.get("measure_result") == "direction":
            print(f"✅ 频道 {ch} 有信号，示向度 {resp['svd_deg']}°")
            found_channels.append((ch, resp['svd_deg']))
        elif resp and resp.get("measure_result") == "near":
            print(f"🔥 频道 {ch} 距离过近，直接原地清除！")
            payload = base(f"clear-near-scan-{ch}")
            payload["position"] = {"x": scan_x, "y": scan_y}
            payload["channel"] = ch
            post("/clear", payload)
            cleared_channels.add(ch)

    cleared_count = 0
    for ch, theta1 in found_channels:
        if time.time() - start_time > 15 * 60:
            print("⏰ 现实时间不足，停止本区域搜索。")
            break
        if ch in cleared_channels:
            continue
            
        print(f"\n=== 开始处理频道 {ch} ===")
        S1 = np.array([scan_x, scan_y])
        S_list = [S1]
        theta_list = [theta1]
        p2_attempt = 0
        p2_success = False
        
        while p2_attempt < 5:
            # Q3 使用快速启发式接口；严格问题二模型见 Q2.optimize_second_point。
            S2, params = get_problem2_point_dynamic(S1, theta1, attempt=p2_attempt)
            if S2 is None:
                break
            delta, dist = params
            print(f"  [{ch}] 问题二尝试 {p2_attempt+1}: 偏转 {delta}°, 距离 {dist}m -> ({S2[0]:.0f}, {S2[1]:.0f})")
            resp2 = measure(S2[0], S2[1], ch, f"p2-{ch}-{p2_attempt}")
            if resp2 and resp2.get("measure_result") == "direction":
                theta2 = resp2["svd_deg"]
                S_list.append(S2)
                theta_list.append(theta2)
                p2_success = True
                break
            elif resp2 and resp2.get("measure_result") == "near":
                payload = base(f"clear-near-S2-{ch}-{p2_attempt}")
                payload["position"] = {"x": float(S2[0]), "y": float(S2[1])}
                payload["channel"] = ch
                resp_clear = post("/clear", payload)
                if resp_clear and resp_clear.get("clear_result") == "success":
                    cleared_channels.add(ch)
                    cleared_count += 1
                    p2_success = True
                break
            else:
                p2_attempt += 1
                
        if p2_success and len(S_list) > 1:
            success, _ = clear_channel(ch, S_list, theta_list)
            if success:
                cleared_channels.add(ch)
                cleared_count += 1
        elif not p2_success:
            success, _ = clear_channel(ch, [S1], [theta1])
            if success:
                cleared_channels.add(ch)
                cleared_count += 1
                
    return cleared_count, np.array([scan_x, scan_y])

def main():
    start_time = time.time()
    res = post("/enter", base("enter-multi-scan"))
    if not res or res.get("accepted") is not True:
        print("❌ 进入失败")
        return
    print("✅ 进入成功！开始【全局多区域扫描 + GDOP + 稳健逼近】模式...\n")

    cleared_channels = set()
    total_cleared = 0
    current_pos = np.array([0.0, 0.0])
    
    # 定义扫描战略点（覆盖整个 1800m 区域）
    # 中心、东、西、南、北、以及四个象限中距离原点 1000-1200m 的位置
    scan_points = [
        np.array([0.0, 0.0]),
        np.array([1200.0, 0.0]),
        np.array([-1200.0, 0.0]),
        np.array([0.0, 1200.0]),
        np.array([0.0, -1200.0]),
        np.array([850.0, 850.0]),
        np.array([-850.0, 850.0]),
        np.array([-850.0, -850.0]),
        np.array([850.0, -850.0])
    ]

    for i, scan_pos in enumerate(scan_points):
        if time.time() - start_time > 16 * 60:
            print("⏰ 现实时间已超过16分钟，停止搜索，准备退出！")
            break
            
        print(f"\n{'='*50}")
        print(f"📍 战略扫描点 {i+1}/{len(scan_points)}: ({scan_pos[0]:.0f}, {scan_pos[1]:.0f})")
        print(f"{'='*50}")
        
        # 移动到扫描点（利用 measure 的移动机制，空扫一次频道即可移到该点）
        # 为了移动过去，我们发送一个无效频道的 measure，或者直接利用第一次真实 measure 移动
        # 这里直接传入 scan_pos，第一次 measure 就会自动扣移动时间
        
        cleared, final_pos = scan_and_clear(scan_pos, start_time, cleared_channels)
        total_cleared += cleared
        current_pos = final_pos
        
        # 检查是否所有频道都已清除（20个频道最多20个干扰源）
        if len(cleared_channels) >= 20:
            print("🎉 所有频道均已清除完毕！")
            break

    print(f"\n🎉 本轮测试结束！共成功清除 {len(cleared_channels)} 个干扰源 (总清除次数 {total_cleared})。")
    post("/exit", base("exit-multi-scan"))

if __name__ == "__main__":
    main()
