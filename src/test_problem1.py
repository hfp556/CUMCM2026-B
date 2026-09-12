"""问题一的可视化检查脚本；自动化断言见 test_q1_q2.py。"""

import numpy as np

from Q1 import (
    diameter_circle_coverage,
    generate_circle_polygon,
    localize_region,
)


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    sensors = [(0.0, 0.0), (1000.0, 0.0)]
    bearings = [60.0, 120.0]
    region = localize_region(sensors, bearings)
    if len(region) == 0:
        raise RuntimeError("定位区域为空，请检查输入观测")

    coverage = diameter_circle_coverage(region)
    print(f"定位区域顶点数：{len(region)}")
    print(f"定位区域直径 D：{coverage.diameter:.3f} m")
    print(f"直径圆半径：{coverage.radius:.3f} m")
    print(f"区域顶点到圆心的最大距离：{coverage.max_center_distance:.3f} m")
    print("直径圆能覆盖定位区域：", "是" if coverage.covers else "否")

    fig, ax = plt.subplots(figsize=(9, 8))
    closed = np.vstack((region, region[0]))
    ax.fill(closed[:, 0], closed[:, 1], color="tab:red", alpha=0.3, label="定位区域")
    ax.plot(closed[:, 0], closed[:, 1], color="tab:red")
    ax.plot(
        [coverage.diameter_pair[0][0], coverage.diameter_pair[1][0]],
        [coverage.diameter_pair[0][1], coverage.diameter_pair[1][1]],
        "g--",
        label=f"直径 D={coverage.diameter:.2f} m",
    )
    ax.add_patch(
        Circle(
            coverage.center,
            coverage.radius,
            fill=False,
            color="purple",
            linestyle="-.",
            label="以区域直径为直径的圆",
        )
    )

    target = generate_circle_polygon((0.0, 0.0), 1800.0, 240)
    ax.plot(target[:, 0], target[:, 1], "k:", label="目标区域边界")
    for index, sensor in enumerate(sensors, start=1):
        ax.scatter(*sensor, marker="^", s=80, label=f"检测点 S{index}")

    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("问题一：定位区域、直径及直径圆覆盖判断")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig("问题一检测.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
