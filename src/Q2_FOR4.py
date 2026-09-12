"""Q2.py — 问题四：近距多方向盲区试探"""
import math
import numpy as np

def _as_point(point, name="point"):
    result = np.asarray(point, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain two finite coordinates")
    return result

def get_problem4_point_dynamic_alternating(S1, theta1, attempt=0):
    """
    近距多方向盲区试探：
    - 距离优先短（100~700m），保证第二点大概率在覆盖区内
    - 方向以 ±30° 为步长，从正负两侧交替尝试
    """
    S1 = _as_point(S1, "S1")
    
    # (偏转角, 距离) 按"小偏角 + 近距离"优先
    strategies = [
        (30, 200), (-30, 200),
        (45, 250), (-45, 250),
        (60, 300), (-60, 300),
        (30, 350), (-30, 350),
        (90, 400), (-90, 400),
        (45, 450), (-45, 450),
        (60, 500), (-60, 500),
        (120, 500), (-120, 500),
        (30, 600), (-30, 600),
        (90, 600), (-90, 600),
        (150, 500), (-150, 500),
    ]
    if attempt >= len(strategies):
        return None, None
    
    delta, d = strategies[attempt]
    ang = math.radians(theta1 + delta)
    S2 = S1 + d * np.array([math.cos(ang), math.sin(ang)])
    
    if np.linalg.norm(S2) > 1780:
        return None, None
    return S2, (float(delta), float(d))