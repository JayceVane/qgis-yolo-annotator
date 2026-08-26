"""验证 OBB 角点手柄旋转模型（纯数值，复刻 _apply_edit_drag vertex 分支数学）。"""

import math


def rot_edit(orig, drag_idx, mouse):
    o = orig[(drag_idx + 2) % 4]
    d0 = (orig[drag_idx][0] - o[0], orig[drag_idx][1] - o[1])
    d1 = (mouse[0] - o[0], mouse[1] - o[1])
    len0 = math.hypot(*d0)
    len1 = math.hypot(*d1)
    if len0 < 1e-9 or len1 < 1e-9:
        return orig
    ang = math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])
    scale = len1 / len0
    ca, sa = math.cos(ang), math.sin(ang)

    def rot(p):
        rx, ry = p[0] - o[0], p[1] - o[1]
        return (o[0] + (rx * ca - ry * sa) * scale, o[1] + (rx * sa + ry * ca) * scale)

    return [rot(p) if i != (drag_idx + 2) % 4 else p for i, p in enumerate(orig)]


def is_rectangle(pts, tol=1e-9):
    """对边平行且相等（e01 = -e23）、邻边垂直。"""
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1])
    def add(a, b):
        return (a[0] + b[0], a[1] + b[1])
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]
    e01 = sub(pts[1], pts[0])
    e12 = sub(pts[2], pts[1])
    e23 = sub(pts[3], pts[2])
    e30 = sub(pts[0], pts[3])
    return (
        abs(dot(e01, e12)) < tol
        and abs(add(e01, e23)[0]) < tol and abs(add(e01, e23)[1]) < tol
        and abs(add(e12, e30)[0]) < tol and abs(add(e12, e30)[1]) < tol
    )


orig = [(0, 0), (10, 0), (10, 5), (0, 5)]  # 10×5 矩形，拖角点 0，对角 (10,5) 固定
# 原对角方向 (−10,−5) 角度 θ0；把鼠标放到 θ0+30° 方向 → 期望整体旋转 30°
theta0 = math.atan2(orig[0][1] - 5, orig[0][0] - 10)
mouse = (
    10 + 8 * math.cos(theta0 + math.radians(30)),
    5 + 8 * math.sin(theta0 + math.radians(30)),
)
result = rot_edit(orig, 0, mouse)
print("rotated:", [(round(x, 3), round(y, 3)) for x, y in result])
print("仍是矩形:", is_rectangle(result))
print("对角未动:", result[2] == (10, 5))
# 等比缩放：边长比保持
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
r = dist(result[1], result[0]) / dist(result[3], result[0])
print(f"长宽比 {r:.6f} == 2.0: {abs(r - 2.0) < 1e-9}")
# 旋转角：p0→p1 边相对原 0° 应为 30°
ang = math.degrees(math.atan2(result[1][1] - result[0][1], result[1][0] - result[0][0]))
print(f"旋转角 {ang:.4f}°（期望 30）")
# 拖到鼠标与对角重合（退化）→ 不变形
same = rot_edit(orig, 0, (10, 5))
print("退化输入保持原状:", same == orig)
