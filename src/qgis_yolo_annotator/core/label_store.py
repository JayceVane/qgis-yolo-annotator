"""标注存储：X-AnyLabeling JSON 读写（原子写）与 DOTA txt 导入。

坐标 SSOT：像素坐标（影像左上原点，x 向右 y 向下），与 t0/t1 生态互通。
rotation shape（OBB）字段约定：
- points: 4 个 [x, y] 角点（标注顺序）
- direction: atan2(p1.y-p0.y, p1.x-p0.x) 弧度（p0→p1 边与 x 轴夹角）
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

XLABEL_VERSION = "4.0.2"
SUPPORTED_SHAPE_TYPES = (
    "polygon",
    "rectangle",
    "rotation",
    "quadrilateral",
    "point",
    "line",
    "circle",
    "linestrip",
)


def rotation_direction(points: list[list[float]]) -> float:
    """计算 rotation shape 的 direction（p0→p1 边与 x 轴夹角，弧度）。

    Args:
        points: 4 个 [x, y] 角点。

    Returns:
        弧度角（[-pi, pi]，与 X-AnyLabeling geometry.obbAngle 一致）。
    """
    return math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])


def make_shape(
    label: str,
    points: list[list[float]],
    shape_type: str,
    *,
    score: float | None = None,
    difficult: bool = False,
    description: str = "",
    extra: dict | None = None,
) -> dict:
    """构造规范化 shape（兼容 X-AnyLabeling 字段集）。

    Args:
        label: 类别名。
        points: 顶点序列 [[x, y], ...]。
        shape_type: 见 SUPPORTED_SHAPE_TYPES。
        score: 模型置信度（手工标注为 None）。
        difficult: DOTA difficult 标记。
        description: 备注。
        extra: 附加扩展键。

    Returns:
        shape dict。

    Raises:
        ValueError: shape_type 非法或 rotation 顶点数不是 4。
    """
    if shape_type not in SUPPORTED_SHAPE_TYPES:
        raise ValueError(f"不支持的 shape_type: {shape_type}")
    normalized_points = [[float(x), float(y)] for x, y in points]
    shape = {
        "label": label,
        "score": float(score) if score is not None else None,
        "points": normalized_points,
        "group_id": None,
        "description": description,
        "difficult": bool(difficult),
        "shape_type": shape_type,
        "flags": {},
        "attributes": {},
        "kie_linking": [],
    }
    if shape_type == "rotation":
        if len(normalized_points) != 4:
            raise ValueError("rotation(OBB) 需要 4 个点")
        shape["direction"] = rotation_direction(normalized_points)
    if extra:
        shape.update(extra)
    return shape


def make_label_doc(
    image_path: str | Path,
    image_width: int,
    image_height: int,
    shapes: list[dict],
) -> dict:
    """构造完整 X-AnyLabeling 标注文档。"""
    return {
        "version": XLABEL_VERSION,
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": Path(image_path).name,
        "imageData": None,
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
        "description": "",
    }


def load_label(path: str | Path) -> dict | None:
    """读取标注 JSON。

    Args:
        path: JSON 文件路径。

    Returns:
        标注文档 dict；文件不存在返回 None。

    Raises:
        ValueError: JSON 解析失败。
    """
    path = Path(path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"标注 JSON 解析失败: {path}: {exc}") from exc
    if not isinstance(doc, dict) or "shapes" not in doc:
        raise ValueError(f"标注 JSON 缺少 shapes 字段: {path}")
    doc.setdefault("shapes", [])
    return doc


def save_label(path: str | Path, doc: dict) -> None:
    """原子写标注 JSON（临时文件 + replace，避免中断损坏）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)


def import_dota(path: str | Path) -> list[dict]:
    """导入 DOTA txt 为 rotation shapes。

    行格式：x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult

    Args:
        path: DOTA label txt 路径。

    Returns:
        rotation shape 列表（difficult 保留）。

    Raises:
        ValueError: 行格式非法。
    """
    shapes: list[dict] = []
    path = Path(path)
    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 9:
            raise ValueError(f"DOTA 行字段不足（需≥9）: {path}:{lineno}")
        try:
            coords = [float(v) for v in parts[:8]]
        except ValueError as exc:
            raise ValueError(f"DOTA 坐标解析失败: {path}:{lineno}") from exc
        label = parts[8]
        difficult = False
        if len(parts) >= 10:
            try:
                difficult = int(parts[9]) == 1
            except ValueError:
                difficult = False
        points = [
            [coords[0], coords[1]],
            [coords[2], coords[3]],
            [coords[4], coords[5]],
            [coords[6], coords[7]],
        ]
        shapes.append(
            make_shape(label, points, "rotation", difficult=difficult)
        )
    return shapes
