"""场景（AOI）内滑窗推理：按目标地理空间分辨率重采样后推理，坐标映射回原始像素。

几何链（各向异性 rescale）：
    目标网格像素 (i, j) ←→ 原始像素 (scene_x0 + i/rescale_x, scene_y0 + j/rescale_y)
其中 rescale = 影像原始分辨率(以 options.unit 计) / 目标分辨率。

示例：影像 0.5 m/px、目标 0.2 m/px → rescale=2.5（目标网格放大 2.5 倍，
推理窗口 1024px@0.2m 对应原始 409.6px@0.5m）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .inference import Detection, YoloOrtModel, nms_polygons
from .geometry import slide_starts
from .model_registry import ModelConfig
from .raster_io import RasterRef, meters_per_degree
from .project import SceneDef

RES_UNIT_METER = "m"
RES_UNIT_DEGREE = "degree"
VALID_RES_UNITS = (RES_UNIT_METER, RES_UNIT_DEGREE)


@dataclass
class SceneInferOptions:
    """场景推理配置。"""

    target_res: float = 0.2      # 目标分辨率（单位由 unit 决定）
    unit: str = RES_UNIT_METER   # m / degree
    chip_size: int = 1024        # 目标网格上的推理窗口边长（像素）
    overlap: int = 200           # 目标网格上相邻窗口重叠（像素）
    merge_iou: float = 0.5       # 跨窗合并 NMS IoU 阈值

    def __post_init__(self):
        if self.unit not in VALID_RES_UNITS:
            raise ValueError(f"不支持的单位体系: {self.unit}（可选 {VALID_RES_UNITS}）")
        if self.target_res <= 0:
            raise ValueError(f"target_res 非法: {self.target_res}")
        if self.chip_size < 32:
            raise ValueError(f"chip_size 过小: {self.chip_size}")
        if not 0 <= self.overlap < self.chip_size:
            raise ValueError(f"overlap 非法: {self.overlap}")
        if not 0.0 < self.merge_iou < 1.0:
            raise ValueError(f"merge_iou 非法: {self.merge_iou}")


def scene_rescale(
    raster: RasterRef, options: SceneInferOptions
) -> tuple[float, float]:
    """计算场景重采样系数 (rescale_x, rescale_y)（原始像素/目标像素）。

    Args:
        raster: 影像地理参考。
        options: 推理配置。

    Returns:
        (rescale_x, rescale_y)。

    Raises:
        ValueError: 影像无地理参考；或 unit=degree 但影像为投影坐标系
            （投影影像分辨率为米单位，度无意义）。
    """
    if not raster.has_georeference:
        raise ValueError(f"影像无地理参考，无法按目标分辨率推理: {raster.path.name}")
    gt = raster.geotransform
    if options.unit == RES_UNIT_DEGREE:
        if not raster.is_geographic:
            raise ValueError(
                f"影像为投影坐标系（米单位），度/px 单位不适用: {raster.path.name}"
            )
        return abs(gt[1]) / options.target_res, abs(gt[5]) / options.target_res
    # unit == meter
    if raster.is_geographic:
        lat = raster.center_latitude() or 0.0
        lon_m, lat_m = meters_per_degree(lat)
        res_x_m = abs(gt[1]) * lon_m
        res_y_m = abs(gt[5]) * lat_m
    else:
        res_x_m, res_y_m = abs(gt[1]), abs(gt[5])
    return res_x_m / options.target_res, res_y_m / options.target_res


@dataclass
class SceneWindow:
    """单个推理窗口（目标网格 + 对应原始像素窗口）。"""

    target_xywh: tuple[int, int, int, int]  # 目标网格窗口（scene 相对，xywh）
    orig_xywh: tuple[int, int, int, int]    # 原始像素窗口（影像绝对，xywh）


def scene_windows(
    scene: SceneDef,
    image_width: int,
    image_height: int,
    rescale: tuple[float, float],
    options: SceneInferOptions,
) -> list[SceneWindow]:
    """生成场景内的滑窗推理窗口（目标网格生成 → 映射回原始像素，clip 到影像）。

    Args:
        scene: 场景定义。
        image_width, image_height: 影像尺寸。
        rescale: (rescale_x, rescale_y)。
        options: 推理配置。

    Returns:
        SceneWindow 列表。
    """
    rescale_x, rescale_y = rescale
    sx0 = max(0.0, min(scene.bbox[0], image_width))
    sy0 = max(0.0, min(scene.bbox[1], image_height))
    sx1 = max(0.0, min(scene.bbox[2], image_width))
    sy1 = max(0.0, min(scene.bbox[3], image_height))
    scene_w = max(1.0, sx1 - sx0)
    scene_h = max(1.0, sy1 - sy0)

    target_w = max(1, round(scene_w * rescale_x))
    target_h = max(1, round(scene_h * rescale_y))
    step = options.chip_size - options.overlap
    starts_x = slide_starts(target_w, options.chip_size, step)
    starts_y = slide_starts(target_h, options.chip_size, step)

    windows: list[SceneWindow] = []
    for ty in starts_y:
        th = min(options.chip_size, target_h - ty)
        for tx in starts_x:
            tw = min(options.chip_size, target_w - tx)
            # 目标网格窗口 → 原始像素窗口
            ox = int(round(sx0 + tx / rescale_x))
            oy = int(round(sy0 + ty / rescale_y))
            ox2 = int(round(sx0 + (tx + tw) / rescale_x))
            oy2 = int(round(sy0 + (ty + th) / rescale_y))
            ox = max(0, min(ox, image_width - 1))
            oy = max(0, min(oy, image_height - 1))
            ox2 = max(ox + 1, min(ox2, image_width))
            oy2 = max(oy + 1, min(oy2, image_height))
            windows.append(
                SceneWindow(
                    target_xywh=(tx, ty, tw, th),
                    orig_xywh=(ox, oy, ox2 - ox, oy2 - oy),
                )
            )
    return windows


def window_points_to_original(
    points: list[list[float]],
    window: SceneWindow,
    scene: SceneDef,
    rescale: tuple[float, float],
) -> list[list[float]]:
    """窗口内检测点（目标网格 scene 相对坐标）→ 原始像素绝对坐标。"""
    rescale_x, rescale_y = rescale
    tx, ty = window.target_xywh[:2]
    return [
        [scene.bbox[0] + (tx + px) / rescale_x, scene.bbox[1] + (ty + py) / rescale_y]
        for px, py in points
    ]


def infer_scene(
    raster: RasterRef,
    scene: SceneDef,
    model: YoloOrtModel,
    model_cfg: ModelConfig,
    options: SceneInferOptions,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    """在单个场景内滑窗推理，返回 X-AnyLabeling shape 列表（原始像素坐标）。

    Args:
        raster: 影像（已打开）。
        scene: 场景定义。
        model: 推理会话。
        model_cfg: 模型配置（labels/imgsz/conf/iou）。
        options: 推理配置（目标分辨率等）。
        progress_cb: 可选进度回调 fn(done, total, message)。

    Returns:
        shape dict 列表（label/score/points/shape_type/direction）。

    Raises:
        ValueError: 分辨率单位与影像不匹配、或影像无地理参考。
    """
    rescale = scene_rescale(raster, options)
    rescale_x, rescale_y = rescale
    windows = scene_windows(scene, raster.width, raster.height, rescale, options)

    shape_type = {"obb": "rotation", "seg": "polygon"}.get(model.task, "rectangle")
    detections: list[Detection] = []

    for idx, window in enumerate(windows):
        tx, ty, tw, th = window.target_xywh
        ox, oy, ow, oh = window.orig_xywh
        block = raster.read_window_bgr(ox, oy, ow, oh)
        if (block.shape[1], block.shape[0]) != (tw, th):
            interp = cv2.INTER_AREA if (tw < ow) else cv2.INTER_LINEAR
            block = cv2.resize(block, (tw, th), interpolation=interp)
        results = model.infer(block, model_cfg.imgsz, model_cfg.conf, model_cfg.iou)
        for det in results:
            det.points = window_points_to_original(det.points, window, scene, rescale)
            detections.append(det)
        if progress_cb is not None:
            progress_cb(idx + 1, len(windows), f"{scene.name} 窗口 {idx + 1}/{len(windows)}")

    # 跨窗合并：按类别分组做凸多边形 NMS，保留高分检测
    merged: list[Detection] = []
    by_class: dict[int, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.class_index, []).append(det)
    for group in by_class.values():
        polygons = [np.asarray(d.points, dtype=np.float64) for d in group]
        scores = [d.score for d in group]
        keep = nms_polygons(polygons, scores, options.merge_iou)
        merged.extend(group[i] for i in keep)

    shapes: list[dict] = []
    for det in merged:
        label = (
            model_cfg.labels[det.class_index]
            if det.class_index < len(model_cfg.labels)
            else f"class_{det.class_index}"
        )
        shapes.append(
            {
                "label": label,
                "score": round(float(det.score), 4),
                "points": [[float(p[0]), float(p[1])] for p in det.points],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": shape_type,
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        )
        if shape_type == "rotation" and len(det.points) == 4:
            pts = det.points
            shapes[-1]["direction"] = math.atan2(
                pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]
            )
    return shapes
