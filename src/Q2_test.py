"""问题二严格模型的示例输出与可视化。"""

import numpy as np

from Q2 import optimize_second_point, plot_problem2_result


def main():
    S1 = np.array([0.0, 0.0])
    theta1 = 0.0
    result = optimize_second_point(
        S1,
        theta1,
        candidate_spacing=150.0,
        source_sample_count=13,
        circle_segments=180,
    )

    print(f"第一次可能区域 Ω1 顶点数：{len(result.first_region)}")
    print(f"可能检测候选点数：{len(result.possible_candidate_points)}")
    print(f"稳健候选点数：{len(result.robust_candidate_points)}")
    print(f"推荐第二检测点 S2：{result.recommended_point}")
    print(f"推荐点最坏定位直径：{result.worst_case_diameter:.3f} m")
    print(f"推荐点典型交会角：{result.typical_crossing_angle_deg:.3f}°")

    figure, _ = plot_problem2_result(
        result,
        S1,
        theta1,
        output_path="问题二检测.png",
    )
    figure.show()


if __name__ == "__main__":
    main()
