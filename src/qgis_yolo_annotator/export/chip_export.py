"""GeoTIFF 切片导出：滑窗切片 + 分辨率重采样 + 标注裁剪 + train/val 划分。

坐标换算链（各向异性 rescale）：
    输出像素 (i, j) ←→ 原始像素 (xoff + i*rescale_x, yoff + j*rescale_y)
其中 rescale = 原始分辨率(m/px) / 目标分辨率(m/px)；未指定目标分辨率时为 1。
输出 GeoTIFF 携带子 geotransform（起点 = 窗口左上角原地图坐标，分辨率按 rescale 换算）。

目标保留规则（DOTA-devkit 惯例）：目标中心点落在原始窗口内即保留；
被窗口裁剪的目标可选标记 difficult=1。
"""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET  # nosec B405: 仅序列化写出 VOC XML，从不解析外部输入
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from osgeo import gdal, osr

from ..core.raster_io import RasterRef, meters_per_degree
from . import converters

gdal.UseExceptions()

EXPORT_FORMATS = ("dota", "yolo_obb", "yolo_det", "voc")
_LABEL_SUFFIX = {"voc": ".xml"}


@dataclass
class ExportOptions:
    """数据集导出配置。"""

    format: str = "dota"            # dota / yolo_obb / yolo_det / voc
    chip_size: int | None = 1024    # 输出网格 chip 边长（像素）；None=全图
    overlap: int = 200              # 输出网格相邻 chip 重叠（像素）
    target_res_m: float | None = None  # 目标分辨率 m/px；None=跟随原始
    boundary_policy: str = converters.BOUNDARY_CLIP
    geo_tiff: bool = True           # True=GeoTIFF（带地理信息）/ False=PNG
    voc_obb_mode: str = "hbb"       # voc 格式 OBB 表示：hbb / polygon
    val_ratio: float = 0.2          # 影像级 train/val 划分比例
    seed: int = 42
    mark_clipped_difficult: bool = True  # 被窗口裁剪的目标标记 difficult

    def __post_init__(self):
        if self.format not in EXPORT_FORMATS:
            raise ValueError(f"不支持导出格式: {self.format}（可选 {EXPORT_FORMATS}）")
        if self.chip_size is not None and self.chip_size < 32:
            raise ValueError(f"chip_size 过小: {self.chip_size}")
        if self.overlap < 0 or (self.chip_size and self.overlap >= self.chip_size):
            raise ValueError(f"overlap 非法: {self.overlap}")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError(f"val_ratio 非法: {self.val_ratio}")
        if self.target_res_m is not None and self.target_res_m <= 0:
            raise ValueError(f"target_res_m 非法: {self.target_res_m}")


@dataclass
class ExportStats:
    """单次导出的统计结果。"""

    image_count: int = 0
    chip_count: int = 0
    label_count: int = 0
    skipped_images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image_count": self.image_count,
            "chip_count": self.chip_count,
            "label_count": self.label_count,
            "skipped_images": self.skipped_images,
        }


def compute_rescale(
    raster: RasterRef, target_res_m: float | None
) -> tuple[float, float] | None:
    """计算各向异性重采样系数 (rescale_x, rescale_y)。

    Args:
        raster: 影像地理参考。
        target_res_m: 目标分辨率 m/px；None 或影像无参考返回 None。

    Returns:
        (rescale_x, rescale_y)：原像素/输出像素；None 表示不重采样。
    """
    if target_res_m is None or not raster.has_georeference:
        return None
    gt = raster.geotransform
    res_x_m = abs(gt[1])
    res_y_m = abs(gt[5])
    if raster.is_geographic:
        lat = raster.center_latitude() or 0.0
        lon_m, lat_m = meters_per_degree(lat)
        res_x_m = abs(gt[1]) * lon_m
        res_y_m = abs(gt[5]) * lat_m
    return res_x_m / target_res_m, res_y_m / target_res_m


def generate_chip_windows(
    img_width: int,
    img_height: int,
    chip_w: int,
    chip_h: int,
    step_x: int,
    step_y: int,
) -> list[tuple[int, int, int, int]]:
    """生成滑窗切片窗口（原始像素网格，末窗对齐影像右/下边界）。

    Args:
        img_width, img_height: 影像尺寸。
        chip_w, chip_h: 窗口尺寸。
        step_x, step_y: 步长。

    Returns:
        (xoff, yoff, xsize, ysize) 列表（xsize/ysize 可能小于 chip，仅当影像本身更小）。
    """
    from ..core.geometry import slide_starts

    xs = [(x, min(chip_w, img_width - x)) for x in slide_starts(img_width, chip_w, step_x)]
    ys = [(y, min(chip_h, img_height - y)) for y in slide_starts(img_height, chip_h, step_y)]
    return [(x, y, w, h) for (x, w) in xs for (y, h) in ys]


def _center_in_window(points: list[list[float]], xoff: int, yoff: int, xsize: int, ysize: int) -> bool:
    """目标四点中心是否落在窗口内（DOTA-devkit 惯例）。"""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return xoff <= cx < xoff + xsize and yoff <= cy < yoff + ysize


def export_image(
    raster: RasterRef,
    shapes: list[dict],
    class_names: list[str],
    split_dir: Path,
    options: ExportOptions,
    progress_cb=None,
) -> tuple[int, int]:
    """导出单幅影像为切片数据集。

    Args:
        raster: 影像（已打开）。
        shapes: X-AnyLabeling shapes（原始像素坐标）。
        class_names: 类别表（行序=id）。
        split_dir: 该影像所属 split 目录（含 images/ labels/）。
        options: 导出配置。
        progress_cb: 可选回调 fn(done, total, message)。

    Returns:
        (chip_count, label_count)。

    Raises:
        ValueError: 目标分辨率指定但影像无地理参考。
    """
    rescale = compute_rescale(raster, options.target_res_m)
    if options.target_res_m is not None and rescale is None:
        raise ValueError(
            f"影像无地理参考，无法按目标分辨率重采样: {raster.path.name}"
        )
    if rescale is None:
        rescale_x = rescale_y = 1.0
    else:
        rescale_x, rescale_y = rescale

    if options.chip_size is None:  # 全图模式
        out_w = max(1, round(raster.width * rescale_x))
        out_h = max(1, round(raster.height * rescale_y))
        windows = [(0, 0, raster.width, raster.height)]
        out_sizes = [(out_w, out_h)]
    else:
        # 输出网格 chip → 原始像素窗口：orig = target / rescale
        chip_w_orig = max(1, round(options.chip_size / rescale_x))
        chip_h_orig = max(1, round(options.chip_size / rescale_y))
        step_x = max(1, round((options.chip_size - options.overlap) / rescale_x))
        step_y = max(1, round((options.chip_size - options.overlap) / rescale_y))
        windows = generate_chip_windows(
            raster.width, raster.height, chip_w_orig, chip_h_orig, step_x, step_y
        )
        out_sizes = [
            (max(1, round(w * rescale_x)), max(1, round(h * rescale_y)))
            for _, _, w, h in windows
        ]

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".tif" if options.geo_tiff else ".png"
    stem = raster.path.stem
    chip_count = 0
    label_count = 0

    for idx, ((xoff, yoff, xsize, ysize), (o_w, o_h)) in enumerate(zip(windows, out_sizes)):
        block = raster.read_window_bgr(xoff, yoff, xsize, ysize)
        if (o_w, o_h) != (xsize, ysize):
            block = cv2.resize(block, (o_w, o_h), interpolation=cv2.INTER_AREA)
        chip_name = f"{stem}__{xoff:06d}_{yoff:06d}"
        if options.geo_tiff:
            _write_geotiff(block, raster, xoff, yoff, rescale_x, rescale_y, images_dir / f"{chip_name}.tif")
        else:
            if not cv2.imwrite(str(images_dir / f"{chip_name}.png"), block):
                raise RuntimeError(f"PNG 写入失败: {chip_name}")
        chip_count += 1

        chip_shapes: list[dict] = []
        for shape in shapes:
            points = shape.get("points") or []
            if len(points) != 4 or not _center_in_window(points, xoff, yoff, xsize, ysize):
                continue
            local = [
                [(px - xoff) * rescale_x, (py - yoff) * rescale_y] for px, py in points
            ]
            clipped = any(
                lx < 0 or lx > o_w or ly < 0 or ly > o_h for lx, ly in local
            )
            chip_shape = dict(shape)
            chip_shape["points"] = local
            if clipped and options.mark_clipped_difficult:
                chip_shape["difficult"] = True
            chip_shapes.append(chip_shape)

        if options.format == "voc":
            root = converters.voc_xml(
                chip_shapes, class_names, "labels", f"{chip_name}{suffix}",
                o_w, o_h, policy=options.boundary_policy, obb_mode=options.voc_obb_mode,
            )
            ET.indent(root, space="  ")
            (labels_dir / f"{chip_name}.xml").write_text(
                ET.tostring(root, encoding="unicode"), encoding="utf-8"
            )
            label_count += len(chip_shapes)
        else:
            if options.format == "dota":
                lines = converters.dota_lines(
                    chip_shapes, class_names, o_w, o_h, options.boundary_policy
                )
            elif options.format == "yolo_obb":
                lines = converters.yolo_obb_lines(
                    chip_shapes, class_names, o_w, o_h, options.boundary_policy
                )
            else:
                lines = converters.yolo_det_lines(
                    chip_shapes, class_names, o_w, o_h, options.boundary_policy
                )
            (labels_dir / f"{chip_name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            label_count += len(lines)

        if progress_cb is not None:
            progress_cb(idx + 1, len(windows), f"{stem} {idx + 1}/{len(windows)}")

    return chip_count, label_count


def _write_geotiff(
    block_bgr: np.ndarray,
    raster: RasterRef,
    xoff: int,
    yoff: int,
    rescale_x: float,
    rescale_y: float,
    out_path: Path,
) -> None:
    """写切片 GeoTIFF（BGR→RGB，携带子 geotransform 与投影）。"""
    rgb = block_bgr[..., ::-1]
    h, w = rgb.shape[:2]
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(out_path), w, h, 3, gdal.GDT_Byte, options=["COMPRESS=LZW"])
    try:
        if raster.has_georeference:
            gt = raster.geotransform
            # 1 输出像素 = 1/rescale 原始像素 → 地图步长 = 原步长 / rescale
            sub_gt = (
                gt[0] + xoff * gt[1] + yoff * gt[2],
                gt[1] / rescale_x,
                0.0,
                gt[3] + xoff * gt[4] + yoff * gt[5],
                0.0,
                gt[5] / rescale_y,
            )
            ds.SetGeoTransform(sub_gt)
            if raster.crs_wkt:
                srs = osr.SpatialReference()
                srs.ImportFromWkt(raster.crs_wkt)
                ds.SetSpatialRef(srs)
        for b in range(3):
            ds.GetRasterBand(b + 1).WriteArray(rgb[:, :, b])
        ds.FlushCache()
    finally:
        ds = None


def export_dataset(
    image_jobs: list[tuple[RasterRef, list[dict]]],
    class_names: list[str],
    out_dir: str | Path,
    options: ExportOptions,
    progress_cb=None,
) -> ExportStats:
    """导出整个数据集（多影像 + train/val 划分 + 元数据文件）。

    Args:
        image_jobs: (影像, shapes) 列表（shapes 为原始像素坐标）。
        class_names: 类别表。
        out_dir: 输出根目录。
        options: 导出配置。
        progress_cb: 可选回调 fn(done, total, message)（done 为影像计数）。

    Returns:
        ExportStats。

    Raises:
        ValueError: 配置非法。
    """
    out_dir = Path(out_dir)
    rng = random.Random(options.seed)  # nosec B311: 固定种子保证 train/val 划分可复现，非安全用途
    stats = ExportStats(image_count=len(image_jobs))

    done = 0
    for raster, shapes in image_jobs:
        split = "val" if rng.random() < options.val_ratio else "train"
        try:
            chips, labels = export_image(
                raster, shapes, class_names, out_dir / split, options
            )
        except ValueError as exc:
            stats.skipped_images.append(f"{raster.path.name}: {exc}")
            continue
        stats.chip_count += chips
        stats.label_count += labels
        done += 1
        if progress_cb is not None:
            progress_cb(done, len(image_jobs), f"{raster.path.name} -> {split}")

    (out_dir / "classes.txt").write_text(
        "\n".join(class_names) + "\n", encoding="utf-8"
    )
    if options.format in ("yolo_obb", "yolo_det"):
        task = "obb" if options.format == "yolo_obb" else "detect"
        yaml = (
            f"path: {out_dir.as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(class_names)}\n"
            f"names:\n"
            + "".join(f"  {i}: {n}\n" for i, n in enumerate(class_names))
            + f"task: {task}\n"
        )
        (out_dir / "data.yaml").write_text(yaml, encoding="utf-8")
    (out_dir / "export_report.json").write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats
