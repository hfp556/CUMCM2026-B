import numpy as np
import matplotlib.pyplot as plt
from 问题一 import localize_region, convex_diameter, generate_circle_polygon

# 设置中文字体，防止图中中文显示为方块
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 准备测试数据 (对应图片中的附录2图2)
S_list = [(0, 0), (1000, 0)]
theta_list = [60.0, 120.0]

# 2. 调用你的定位函数
poly = localize_region(S_list, theta_list)

if len(poly) == 0:
    print("❌ 错误：定位区域为空！两个传感器的视线没有交会。")
else:
    # 3. 计算定位区域的直径
    D, pair = convex_diameter(poly)
    print(f"✅ 定位区域顶点数: {len(poly)}")
    print(f"✅ 定位区域直径 D: {D:.2f} 米")
    print(f"✅ 直径对应的顶点: {pair[0]}, {pair[1]}")

    # 4. 可视化验证（原 add.py 的绘图部分）
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 8))

    # (1) 画最终定位区域（红色填充）
    poly_closed = np.vstack([poly, poly[0]]) # 闭合多边形
    ax.plot(poly_closed[:, 0], poly_closed[:, 1], 'r-', linewidth=2, label='定位区域')
    ax.fill(poly_closed[:, 0], poly_closed[:, 1], color='red', alpha=0.3)

    # 画出直径
    ax.plot([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]], 'g--', linewidth=2, label=f'直径 D={D:.1f}m')

    # (2) 画目标区域外圈 (半径1800m)
    target_circle = generate_circle_polygon((0,0), 1800.0, num_segments=200)
    ax.plot(target_circle[:, 0], target_circle[:, 1], 'k:', label='目标区域边界(1800m)')

    # (3) 画传感器、接收圆盘、射线
    colors = ['blue', 'orange']
    for i, (S, theta) in enumerate(zip(S_list, theta_list)):
        c = colors[i % len(colors)]

        # 画检测点
        ax.scatter(S[0], S[1], color=c, s=100, marker='^', zorder=5, label=f'传感器{i+1} ({S})')

        # 画接收圆盘 (半径1500m)
        recv_circle = generate_circle_polygon(S, 1500.0, num_segments=200)
        ax.plot(recv_circle[:, 0], recv_circle[:, 1], color=c, linestyle='--', alpha=0.5,
                label=f'接收盘半径1500m' if i == 0 else None)

        # 画射线 (误差范围 ±1度)
        for sign in [-1.0, 1.0]:
            angle_rad = np.radians(theta + sign * 1.0)
            # 射线终点设定为 2000m 长
            end_x = S[0] + 2000 * np.cos(angle_rad)
            end_y = S[1] + 2000 * np.sin(angle_rad)
            ax.plot([S[0], end_x], [S[1], end_y], color=c, linewidth=1, alpha=0.7)
            ax.text(end_x, end_y, f'{theta+sign}°', color=c)

    # 设置坐标轴比例相等，防止图形被拉伸变形
    ax.set_aspect('equal')
    ax.set_xlim(-1000, 2500)
    ax.set_ylim(-1000, 2000)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    ax.set_title("问题一：交会定位区域可视化验证")
    plt.savefig("问题一检测.png", dpi=150)
    plt.show()
