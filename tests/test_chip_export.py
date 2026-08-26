"""chip_export 单测：窗口生成、重采样系数、切片端到端（含分辨率重采样）。"""

import numpy as np
import pytest

from qgis_yolo_annotator.core.raster_io import RasterRef
from qgis_yolo_annotator.export.chip_export import (
    ExportOptions,
    compute_rescale,
    export_dataset,
    export_image,
    generate_chip_windows,
)

from .conftest import WGS84_WKT


def test_generate_chip_windows_full_cover():
    wins = generate_chip_windows(100, 80, 50, 40, 30, 20)
    # 覆盖全图：每个像素至少属于一个窗口
    cover = np.zeros((80, 100), dtype=bool)
    for x, y, w, h in wins:
        cover[y : y + h, x : x + w] = True
    assert cover.all()
    # 尺寸合法
    for x, y, w, h in wins:
        assert w <= 50 and h <= 40
        assert x + w <= 100 and y + h <= 80


def test_generate_chip_windows_image_smaller_than_chip():
    wins = generate_chip_windows(30, 20, 50, 40, 10, 10)
    assert wins == [(0, 0, 30, 20)]


def test_compute_rescale_utm(make_geotiff):
    gt = (500000.0, 0.5, 0.0, 4400000.0, 0.0, -0.5)
    ref = RasterRef.open(make_geotiff("u.tif", 32, 32, gt=gt))
    assert compute_rescale(ref, None) is None
    assert compute_rescale(ref, 0.25) == (2.0, 2.0)  # 0.5 -> 0.25 m/px 放大2倍视野
    assert compute_rescale(ref, 1.0) == (0.5, 0.5)


def test_compute_rescale_unreferenced(make_geotiff):
    ref = RasterRef.open(make_geotiff("plain.tif", 32, 32))
    assert compute_rescale(ref, 0.5) is None


def test_compute_rescale_geographic(make_geotiff):
    import math

    gt = (100.0, 2.682209e-06, 0.0, 10.0, 0.0, -2.682209e-06)
    ref = RasterRef.open(make_geotiff("g.tif", 32, 32, gt=gt, wkt=WGS84_WKT))
    orig_res = ref.resolution_m_per_px()
    rx, ry = compute_rescale(ref, orig_res / 2)  # 目标分辨率 = 原始一半
    # 各向异性（经/纬向系数不同）→ 分量围绕 2 微偏，几何平均精确为 2
    assert math.sqrt(rx * ry) == pytest.approx(2.0, rel=1e-9)
    assert 0.98 < rx < 2.02 and 0.98 < ry < 2.02


def _shapes_grid_cells():
    """3 个不重叠 OBB，跨窗口边界一个。"""
    from qgis_yolo_annotator.core.label_store import make_shape

    def box(cx, cy, w, h):
        return make_shape(
            "Car",
            [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
             [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]],
            "rotation",
        )

    return [box(20, 20, 8, 8), box(70, 50, 8, 8), box(50, 24, 8, 8)]  # 最后一个跨 y 边界


def test_export_image_dota_chips(make_geotiff, tmp_path):
    gt = (500000.0, 1.0, 0.0, 4400000.0, 0.0, -1.0)
    path = make_geotiff("scene.tif", 128, 96, gt=gt)
    ref = RasterRef.open(path)
    opts = ExportOptions(format="dota", chip_size=64, overlap=0, geo_tiff=True)
    chips, labels = export_image(ref, _shapes_grid_cells(), ["Car"], tmp_path, opts)
    # 128 宽 → 2 列；96 高 → y∈{0(64), 64(32)} 2 行（末窗对齐下边界）
    assert chips == 4
    files = sorted((tmp_path / "images").glob("*.tif"))
    assert len(files) == chips
    # GeoTIFF 切片携带地理信息：第一个窗口 geotransform 对上
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(str(files[0]))
    sub_gt = ds.GetGeoTransform()
    assert sub_gt[0] == pytest.approx(500000.0)
    assert sub_gt[1] == pytest.approx(1.0)
    assert sub_gt[3] == pytest.approx(4400000.0)
    assert labels >= 2


def test_export_with_resampling(make_geotiff, tmp_path):
    """0.5 m/px 影像 → 0.25 m/px 输出：chip 数量按视野翻倍、geotransform 分辨率减半。"""
    gt = (500000.0, 0.5, 0.0, 4400000.0, 0.0, -0.5)
    path = make_geotiff("res.tif", 128, 128, gt=gt)
    ref = RasterRef.open(path)
    opts = ExportOptions(
        format="dota", chip_size=64, overlap=0, target_res_m=0.25, geo_tiff=True
    )
    chips, _ = export_image(ref, [], ["Car"], tmp_path, opts)
    # 原图 128px @0.5m = 64m 视野；chip 64px @0.25m = 16m 视野 → 4x4=16 chips
    assert chips == 16
    from osgeo import gdal

    gdal.UseExceptions()
    first = sorted((tmp_path / "images").glob("*.tif"))[0]
    ds = gdal.Open(str(first))
    assert ds.RasterXSize == 64
    assert ds.GetGeoTransform()[1] == pytest.approx(0.25)
    assert ds.GetGeoTransform()[5] == pytest.approx(-0.25)


def test_export_dataset_metadata_and_split(make_geotiff, tmp_path):
    gt = (500000.0, 1.0, 0.0, 4400000.0, 0.0, -1.0)
    jobs = []
    for i in range(4):
        ref = RasterRef.open(make_geotiff(f"s{i}.tif", 64, 64, gt=gt))
        jobs.append((ref, []))
    out = tmp_path / "ds"
    opts = ExportOptions(format="yolo_obb", chip_size=None, val_ratio=0.5, seed=7)
    stats = export_dataset(jobs, ["Car"], out, opts)
    assert stats.image_count == 4
    assert stats.chip_count == 4
    assert (out / "classes.txt").read_text(encoding="utf-8").strip() == "Car"
    yaml_text = (out / "data.yaml").read_text(encoding="utf-8")
    assert "task: obb" in yaml_text
    assert "nc: 1" in yaml_text
    # seed 固定 → split 可复现
    train_chips = len(list((out / "train" / "images").glob("*")))
    val_chips = len(list((out / "val" / "images").glob("*")))
    assert train_chips + val_chips == 4


def test_export_voc_full_image(make_geotiff, tmp_path):
    gt = (500000.0, 1.0, 0.0, 4400000.0, 0.0, -1.0)
    ref = RasterRef.open(make_geotiff("v.tif", 128, 96, gt=gt))
    shapes = _shapes_grid_cells()
    out = tmp_path
    chips, labels = export_image(
        ref, shapes, ["Car"], out, ExportOptions(format="voc", chip_size=None)
    )
    assert chips == 1
    assert labels == 3
    xml_files = list((out / "labels").glob("*.xml"))
    assert len(xml_files) == 1


def test_export_options_validation():
    with pytest.raises(ValueError):
        ExportOptions(format="coco")
    with pytest.raises(ValueError):
        ExportOptions(chip_size=8)
    with pytest.raises(ValueError):
        ExportOptions(chip_size=64, overlap=64)
    with pytest.raises(ValueError):
        ExportOptions(val_ratio=1.5)
    with pytest.raises(ValueError):
        ExportOptions(target_res_m=-1)
