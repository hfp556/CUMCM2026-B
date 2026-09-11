import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon

# 设置中文字体，防止图中中文显示为方块
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 参数设定 =================
theta1_deg = 0.0          # 第一个检测点测得的示向度（设为0度，即正东，方便计算）
theta1_rad = np.radians(theta1_deg)
S1 = np.array([0.0, 0.0]) # 第一个检测点位置

R_max = 1500.0            # 干扰源最大有效接收半径（最坏情况）
R_target = 1800.0         # 目标区域半径
d_min = 0.0               # 机器狗移动的最小距离
d_max = 1800.0            # 机器狗移动的最大距离（受目标区域限制）

# ================= 候选区域边界推导 =================
# 根据之前推导：d <= 3000 * cos(delta_theta)
# 其中 delta_theta 是第二个检测点相对 theta1 的偏转角
# 为了让交会角接近90度，delta_theta 应该接近 90度，但受限于 d > 0
# 我们计算不同 delta_theta 下，允许的最大移动距离 d

delta_thetas = np.linspace(0, np.pi/2, 100) # 0 到 90度
d_max_allowed = 3000 * np.cos(delta_thetas) # d <= 3000 * cos(delta_theta)
d_max_allowed = np.minimum(d_max_allowed, d_max) # 同时受限于目标区域1800米

# ================= 绘制候选区域 =================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.6)

# 1. 画出目标区域（半径1800m的大圆）
target_circle = Circle(S1, R_target, color='gray', fill=False, linestyle='--', label='目标区域 (R=1800m)')
ax.add_patch(target_circle)

# 2. 画出干扰源的可能位置（极坐标下的扇环，距离1000-1500m，方向 theta1）
# 干扰源在正东方向，距离 1000 到 1500 之间
source_min = S1 + 1000 * np.array([np.cos(theta1_rad), np.sin(theta1_rad)])
source_max = S1 + 1500 * np.array([np.cos(theta1_rad), np.sin(theta1_rad)])
ax.plot([source_min[0], source_max[0]], [source_min[1], source_max[1]], 
        color='red', linewidth=4, label='干扰源可能位置 (距离1000~1500m)')
ax.scatter(source_min[0], source_min[1], color='red', marker='o')
ax.scatter(source_max[0], source_max[1], color='red', marker='s')

# 3. 画出候选区域（满足 d <= 3000*cos(delta_theta) 的扇形环）
# 构造候选区域的边界点
# 边界1: d = 3000 * cos(delta_theta)
boundary_x = []
boundary_y = []
for dt, d in zip(delta_thetas, d_max_allowed):
    if d > 0:
        # 对称画在两侧（上方和下方）
        angle = theta1_rad + dt
        boundary_x.append(S1[0] + d * np.cos(angle))
        boundary_y.append(S1[1] + d * np.sin(angle))
        
# 下半部分对称
boundary_x_lower = []
boundary_y_lower = []
for dt, d in zip(delta_thetas, d_max_allowed):
    if d > 0:
        angle = theta1_rad - dt
        boundary_x_lower.append(S1[0] + d * np.cos(angle))
        boundary_y_lower.append(S1[1] + d * np.sin(angle))

# 合并形成闭合多边形
candidate_x = boundary_x + boundary_x_lower[::-1]
candidate_y = boundary_y + boundary_y_lower[::-1]
candidate_poly = Polygon(np.column_stack((candidate_x, candidate_y)), 
                         closed=True, color='green', alpha=0.3, label='候选区域')
ax.add_patch(candidate_poly)

# 4. 画出第一个检测点 S1 和推荐位置 S2
ax.scatter(S1[0], S1[1], color='blue', s=100, zorder=5, label='第一个检测点 S1')
# 推荐一个较好的位置，比如 delta_theta = 60度，d = 1500m
dt_recommend = np.radians(60)
d_recommend = 1500
S2_rec = S1 + d_recommend * np.array([np.cos(theta1_rad + dt_recommend), np.sin(theta1_rad + dt_recommend)])
ax.scatter(S2_rec[0], S2_rec[1], color='purple', s=100, marker='*', zorder=5, label='推荐的第二个检测点 S2 (偏转60°, 距离1500m)')

# 画出连线
ax.plot([S1[0], S2_rec[0]], [S1[1], S2_rec[1]], 'purple', linestyle='--', alpha=0.7)
ax.plot([S1[0], source_max[0]], [S1[1], source_max[1]], 'red', linestyle=':', alpha=0.7)

# 添加交会角示意
# 在 G_max (1500, 0) 处，指向 S1 和 S2 的向量
G_max = np.array([1500.0, 0.0])
vec_G_S1 = S1 - G_max
vec_G_S2 = S2_rec - G_max
# 计算交会角
cos_alpha = np.dot(vec_G_S1, vec_G_S2) / (np.linalg.norm(vec_G_S1) * np.linalg.norm(vec_G_S2))
alpha_deg = np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
ax.text(G_max[0] - 100, G_max[1] + 50, f'交会角 α ≈ {alpha_deg:.1f}°', color='black', fontsize=12)

# ================= 设置图形属性 =================
ax.set_xlim(-500, 2000)
ax.set_ylim(-1000, 1000)
ax.set_xlabel('X 坐标 (米)', fontsize=12)
ax.set_ylabel('Y 坐标 (米)', fontsize=12)
ax.set_title('问题2：第二个检测点候选区域与选择策略示意图', fontsize=14)
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig("问题二检测.png", dpi=150)
plt.show()