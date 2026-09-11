import numpy as np
import math

def generate_circle_polygon(center, radius, num_segments=360):
    """生成以center为圆心，radius为半径的凸多边形（用于近似圆盘）"""
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

def clip_polygon_by_halfplane(poly, point, normal):
    """
    用半平面 (P - point) dot normal >= 0 裁剪凸多边形
    normal 指向保留的一侧
    """
    if len(poly) == 0:
        return np.empty((0, 2))
    
    new_poly = []
    n = len(poly)
    for i in range(n):
        P = poly[i]
        Q = poly[(i + 1) % n]
        
        val_P = np.dot(P - point, normal)
        val_Q = np.dot(Q - point, normal)
        
        if val_P >= -1e-9:  # P在内部或边界上
            new_poly.append(P)
        if (val_P > 1e-9 and val_Q < -1e-9) or (val_P < -1e-9 and val_Q > 1e-9):
            # 发生跨越，计算交点
            t = val_P / (val_P - val_Q)
            intersection = P + t * (Q - P)
            new_poly.append(intersection)
            
    # 去重
    if not new_poly:
        return np.empty((0, 2))
    unique_poly = [new_poly[0]]
    for p in new_poly[1:]:
        if np.linalg.norm(p - unique_poly[-1]) > 1e-6:
            unique_poly.append(p)
            
    return np.array(unique_poly)

def localize_region(S_list, theta_list, R_max=1500.0, R_target=1800.0):
    """
    核心定位算法：带有效接收半径和目标区域约束的交会定位
    """
    # 1. 初始化区域为目标区域（半径1800m的圆盘多边形）
    region = generate_circle_polygon((0, 0), R_target, num_segments=720)
    
    # 2. 逐个检测点进行约束裁剪
    for S, theta_deg in zip(S_list, theta_list):
        # 生成该检测点的有效接收圆盘（半径1500m）
        reception_disk = generate_circle_polygon(S, R_max, num_segments=360)
        
        # 计算两条误差边界射线的法向量（指向扇形内部）
        # 误差为 ±1度，射线方向角为 theta ± 1
        for sign in [-1.0, 1.0]:
            angle_rad = math.radians(theta_deg + sign * 1.0)
            # 射线方向向量 u = [cos, sin]
            u = np.array([math.cos(angle_rad), math.sin(angle_rad)])
            # 法向量 n 逆时针旋转90度，确保指向左侧（即扇形内部）
            n = np.array([-u[1], u[0]])
            
            # 用半平面裁剪接收圆盘
            # 注意：如果是 -1 度边界，内部在左侧；+1度边界，内部也在左侧
            # 但我们需要的是两个半平面的交集。
            # 这里通过调整法向量方向来实现
            
            # 为了确保保留两个边界之间的区域，我们分两次裁剪：
            # 裁剪出 theta - 1 左侧的区域
            # 裁剪出 theta + 1 右侧的区域（即法向量反向）
            if sign == -1.0:
                # 保留 (P - S) dot n >= 0
                reception_disk = clip_polygon_by_halfplane(reception_disk, S, n)
            else:
                # 保留 (P - S) dot (-n) >= 0
                reception_disk = clip_polygon_by_halfplane(reception_disk, S, -n)
                
        if len(reception_disk) == 0:
            return np.empty((0, 2)) # 无交会区域
            
        # 用当前检测点的有效扇形区域去裁剪全局区域
        # 由于两个凸多边形求交较复杂，这里采用Sutherland-Hodgman多边形裁剪算法
        # 为简化代码篇幅，这里用半平面依次裁剪全局区域（等价于凸多边形求交）
        # 实际应用中，直接用裁剪后的 reception_disk 的每条边去裁剪 region
        # 此处为了清晰，我们直接对 region 用边界射线进行裁剪，再与圆盘求交
        # 更严谨的方法：把 reception_disk 的每条边作为半平面，裁剪 region
        
        # 重新初始化该点的扇形区域
        sector = generate_circle_polygon(S, R_max, num_segments=360)
        for sign in [-1.0, 1.0]:
            angle_rad = math.radians(theta_deg + sign * 1.0)
            u = np.array([math.cos(angle_rad), math.sin(angle_rad)])
            n = np.array([-u[1], u[0]])
            if sign == -1.0:
                sector = clip_polygon_by_halfplane(sector, S, n)
            else:
                sector = clip_polygon_by_halfplane(sector, S, -n)
        
        # 将 sector 与 region 求交（通过用 sector 的每条边裁剪 region）
        for i in range(len(sector)):
            P1 = sector[i]
            P2 = sector[(i + 1) % len(sector)]
            # 边 P1->P2 的法向量指向内部
            edge = P2 - P1
            normal = np.array([-edge[1], edge[0]]) # 逆时针旋转90度
            # 确保法向量指向 sector 内部
            if np.dot(sector[0] - P1, normal) < 0:
                normal = -normal
            region = clip_polygon_by_halfplane(region, P1, normal)
            if len(region) == 0:
                return np.empty((0, 2))
                
    return region

def convex_diameter(poly):
    """旋转卡壳求凸多边形直径"""
    n = len(poly)
    if n == 1: return 0.0, (poly[0], poly[0])
    if n == 2: return np.linalg.norm(poly[0] - poly[1]), (poly[0], poly[1])
    
    j = 1
    best_d = 0.0
    best_pair = (poly[0], poly[1])
    
    for i in range(n):
        ni = (i + 1) % n
        while True:
            nj = (j + 1) % n
            cur = abs(np.cross(poly[ni] - poly[i], poly[j] - poly[i]))
            nxt = abs(np.cross(poly[ni] - poly[i], poly[nj] - poly[i]))
            if nxt > cur:
                j = nj
            else:
                break
        for k in [j, (j + 1) % n]:
            d = np.linalg.norm(poly[i] - poly[k])
            if d > best_d:
                best_d = d
                best_pair = (poly[i], poly[k])
            d = np.linalg.norm(poly[ni] - poly[k])
            if d > best_d:
                best_d = d
                best_pair = (poly[ni], poly[k])
    return best_d, best_pair