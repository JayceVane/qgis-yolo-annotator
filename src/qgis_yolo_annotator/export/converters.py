"""标注 shapes → DOTA / VOC / YOLO 系标签格式转换（纯函数）。

输入统一约定：shapes 为 X-AnyLabeling dict 列表，points 已位于**输出图像**的
像素坐标系（切片导出前已完成平移/缩放）。类别 id 由 class_names 行序决定。

边界策略：
- clip: 越界角点裁剪到图像边界（裁剪后退化面积过小的目标丢弃）
- skip: 任一角点越界则丢弃该目标
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405: 仅序列化写出 VOC XML，从不解析外部输入
from typing import Sequence

BOUNDARY_CLIP = "clip"
BOUNDARY_SKIP = "skip"

# 裁剪后多边形最小面积（像素²），低于此视为退化丢弃
_MIN_CLIP_AREA = 4.0


def _shape_points(shape: dict) -> list[list[float]] | None:
    """提取 shape 的 (4, 2) 角点；非四点 shape 返回 None。"""
    points = shape.get("points") or []
    if len(points) != 4:
        return None
    return [[float(p[0]), float(p[1])] for p in points]


def _clip_points(
    points: list[list[float]], width: int, height: int
) -> list[list[float]] | None:
    """角点裁剪到图像边界；退化（面积过小或共线）返回 None。"""
    clipped = [[min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))] for x, y in points]
    xs = [p[0] for p in clipped]
    ys = [p[1] for p in clipped]
    area = 0.5 * abs(
        sum(xs[i] * ys[(i + 1) % 4] - xs[(i + 1) % 4] * ys[i] for i in range(4))
    )
    if area < _MIN_CLIP_AREA:
        return None
    return clipped


def _filter_points(
    shape: dict, width: int, height: int, policy: str
) -> list[list[float]] | None:
    """按边界策略返回最终角点；丢弃返回 None。"""
    points = _shape_points(shape)
    if points is None:
        return None
    out_of_bounds = any(
        x < 0 or x > width or y < 0 or y > height for x, y in points
    )
    if not out_of_bounds:
        return points
    if policy == BOUNDARY_CLIP:
        return _clip_points(points, width, height)
    return None


def obb_to_hbb(points: list[list[float]]) -> tuple[float, float, float, float]:
    """旋转框四点 → 轴对齐外接框 (xmin, ymin, xmax, ymax)。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# DOTA: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
# ---------------------------------------------------------------------------

def dota_lines(
    shapes: Sequence[dict],
    class_names: Sequence[str],
    width: int,
    height: int,
    policy: str = BOUNDARY_CLIP,
) -> list[str]:
    """转换 shapes 为 DOTA label 行。

    DOTA 以空格分隔字段，label 内空格替换为下划线（DOTA-devkit 惯例，
    导入端同样按下划线类名解析）。

    Args:
        shapes: X-AnyLabeling shape 列表（坐标为输出图像像素）。
        class_names: 类别表（行序=id）；shape.label 必须在其中。
        width, height: 输出图像尺寸（像素）。
        policy: 边界策略 clip / skip。

    Returns:
        DOTA txt 行列表（不带换行符）。
    """
    lines: list[str] = []
    for shape in shapes:
        points = _filter_points(shape, width, height, policy)
        if points is None:
            continue
        label = str(shape.get("label", ""))
        if label not in class_names:
            continue
        difficult = 1 if shape.get("difficult") else 0
        coords = " ".join(f"{p[0]:.2f} {p[1]:.2f}" for p in points)
        lines.append(f"{coords} {label.replace(' ', '_')} {difficult}")
    return lines


# ---------------------------------------------------------------------------
# YOLO-OBB: cid x1 y1 x2 y2 x3 y3 x4 y4（归一化角点）
# ---------------------------------------------------------------------------

def yolo_obb_lines(
    shapes: Sequence[dict],
    class_names: Sequence[str],
    width: int,
    height: int,
    policy: str = BOUNDARY_CLIP,
) -> list[str]:
    """转换 shapes 为 YOLO-OBB label 行（归一化角点，.6f）。"""
    lines: list[str] = []
    for shape in shapes:
        points = _filter_points(shape, width, height, policy)
        if points is None:
            continue
        cid = _class_id(shape, class_names)
        if cid is None:
            continue
        normed = " ".join(
            f"{min(max(x / width, 0.0), 1.0):.6f} {min(max(y / height, 0.0), 1.0):.6f}"
            for x, y in points
        )
        lines.append(f"{cid} {normed}")
    return lines


# ---------------------------------------------------------------------------
# YOLO-det: cid cx cy w h（归一化 HBB）
# ---------------------------------------------------------------------------

def yolo_det_lines(
    shapes: Sequence[dict],
    class_names: Sequence[str],
    width: int,
    height: int,
    policy: str = BOUNDARY_CLIP,
) -> list[str]:
    """转换 shapes 为 YOLO det label 行（OBB 取外接 HBB，归一化）。"""
    lines: list[str] = []
    for shape in shapes:
        points = _filter_points(shape, width, height, policy)
        if points is None:
            continue
        cid = _class_id(shape, class_names)
        if cid is None:
            continue
        xmin, ymin, xmax, ymax = obb_to_hbb(points)
        cx = (xmin + xmax) / 2.0 / width
        cy = (ymin + ymax) / 2.0 / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


# ---------------------------------------------------------------------------
# VOC XML（HBB bndbox 默认 / 可选四点 polygon）
# ---------------------------------------------------------------------------

def voc_xml(
    shapes: Sequence[dict],
    class_names: Sequence[str],
    folder: str,
    filename: str,
    width: int,
    height: int,
    depth: int = 3,
    policy: str = BOUNDARY_CLIP,
    obb_mode: str = "hbb",
) -> ET.Element:
    """转换 shapes 为 VOC annotation XML 根元素。

    Args:
        obb_mode: "hbb"（外接水平框 bndbox）或 "polygon"（四点 polygon 节点）。

    Returns:
        <annotation> Element（调用方自行写盘）。
    """
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    for shape in shapes:
        points = _filter_points(shape, width, height, policy)
        if points is None:
            continue
        if _class_id(shape, class_names) is None:
            continue
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = str(shape.get("label", ""))
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "occluded").text = "0"
        ET.SubElement(obj, "difficult").text = "1" if shape.get("difficult") else "0"
        if obb_mode == "polygon":
            poly = ET.SubElement(obj, "polygon")
            for i, (x, y) in enumerate(points, start=1):
                ET.SubElement(poly, f"x{i}").text = f"{x:.2f}"
                ET.SubElement(poly, f"y{i}").text = f"{y:.2f}"
        else:
            xmin, ymin, xmax, ymax = obb_to_hbb(points)
            bnd = ET.SubElement(obj, "bndbox")
            ET.SubElement(bnd, "xmin").text = f"{xmin:.2f}"
            ET.SubElement(bnd, "ymin").text = f"{ymin:.2f}"
            ET.SubElement(bnd, "xmax").text = f"{xmax:.2f}"
            ET.SubElement(bnd, "ymax").text = f"{ymax:.2f}"
    return root


def _class_id(shape: dict, class_names: Sequence[str]) -> int | None:
    """shape.label → 类别 id；不在类别表返回 None。"""
    try:
        return class_names.index(str(shape.get("label", "")))
    except ValueError:
        return None
