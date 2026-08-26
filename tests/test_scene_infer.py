"""scene_infer 单测：重采样系数、窗口几何、坐标往返（不需要模型）。"""

import pytest

from qgis_yolo_annotator.core.project import SceneDef
from qgis_yolo_annotator.core.raster_io import RasterRef
from qgis_yolo_annotator.core.scene_infer import (
    RES_UNIT_DEGREE,
    RES_UNIT_METER,
    SceneInferOptions,
    scene_rescale,
    scene_windows,
    window_points_to_original,
)

from .conftest import WGS84_WKT


def test_options_validation():
    with pytest.raises(ValueError, match="单位"):
        SceneInferOptions(unit="ft")
    with pytest.raises(ValueError):
        SceneInferOptions(target_res=0)
    with pytest.raises(ValueError):
        SceneInferOptions(chip_size=8)
    with pytest.raises(ValueError):
        SceneInferOptions(chip_size=64, overlap=64)


def test_scene_rescale_meter_utm(make_geotiff):
    gt = (500000.0, 0.5, 0.0, 4400000.0, 0.0, -0.5)
    ref = RasterRef.open(make_geotiff("u.tif", 32, 32, gt=gt))
    opts = SceneInferOptions(target_res=0.2, unit=RES_UNIT_METER)
    rx, ry = scene_rescale(ref, opts)
    assert rx == pytest.approx(2.5)  # 0.5/0.2
    assert ry == pytest.approx(2.5)


def test_scene_rescale_degree_geographic(make_geotiff):
    gt = (100.0, 2.682209e-06, 0.0, 10.0, 0.0, -2.682209e-06)
    ref = RasterRef.open(make_geotiff("g.tif", 32, 32, gt=gt, wkt=WGS84_WKT))
    opts = SceneInferOptions(target_res=1.34e-06, unit=RES_UNIT_DEGREE)
    rx, ry = scene_rescale(ref, opts)
    assert rx == pytest.approx(2.0, rel=1e-3)  # 2.682e-6 / 1.34e-6


def test_scene_rescale_degree_on_projected_errors(make_geotiff):
    gt = (500000.0, 0.5, 0.0, 4400000.0, 0.0, -0.5)
    ref = RasterRef.open(make_geotiff("p.tif", 32, 32, gt=gt))
    with pytest.raises(ValueError, match="度/px"):
        scene_rescale(ref, SceneInferOptions(unit=RES_UNIT_DEGREE))


def test_scene_rescale_unreferenced_errors(make_geotiff):
    ref = RasterRef.open(make_geotiff("plain.tif", 32, 32))
    with pytest.raises(ValueError, match="无地理参考"):
        scene_rescale(ref, SceneInferOptions())


def test_scene_windows_coverage_and_clipping(make_geotiff):
    """场景 200×150 原始像素 @0.5m，目标 0.2m → 目标网格 500×375，chip 128/step 64。"""
    gt = (500000.0, 0.5, 0.0, 4400000.0, 0.0, -0.5)
    ref = RasterRef.open(make_geotiff("w.tif", 256, 256, gt=gt))
    scene = SceneDef(name="s", bbox=[20, 30, 220, 180])
    opts = SceneInferOptions(target_res=0.2, chip_size=128, overlap=64)
    rescale = scene_rescale(ref, opts)
    windows = scene_windows(scene, ref.width, ref.height, rescale, opts)
    assert len(windows) > 1
    # 目标网格全覆盖
    target_cover = set()
    for w in windows:
        tx, ty, tw, th = w.target_xywh
        assert tw <= 128 and th <= 128
        for x in range(tx, tx + tw, 8):
            for y in range(ty, ty + th, 8):
                target_cover.add((x, y))
    target_w = round(200 * 2.5)
    target_h = round(150 * 2.5)
    corners = [(0, 0), (target_w - 8, 0), (0, target_h - 8), (target_w - 8, target_h - 8)]
    assert all(c in target_cover for c in corners)
    # 原始窗口在影像内且非空
    for w in windows:
        ox, oy, ow, oh = w.orig_xywh
        assert ow >= 1 and oh >= 1
        assert ox >= 0 and oy >= 0 and ox + ow <= 256 and oy + oh <= 256


def test_scene_windows_single_when_small():
    scene = SceneDef(name="s", bbox=[0, 0, 40, 40])
    opts = SceneInferOptions(chip_size=128, overlap=32)
    windows = scene_windows(scene, 64, 64, (1.0, 1.0), opts)
    assert len(windows) == 1
    assert windows[0].target_xywh == (0, 0, 40, 40)
    assert windows[0].orig_xywh == (0, 0, 40, 40)


def test_window_points_roundtrip():
    """窗口内坐标 → 原始像素坐标的往返一致性。"""
    scene = SceneDef(name="s", bbox=[100.0, 200.0, 300.0, 350.0])
    rescale = (2.5, 2.5)
    window_pts = [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]]

    class _W:  # 最小窗口桩
        target_xywh = (16, 24, 128, 128)

    original = window_points_to_original(window_pts, _W(), scene, rescale)
    # 逆变换：scene_x0 + (tx+px)/rescale
    expected_x = 100.0 + (16 + 10.0) / 2.5
    expected_y = 200.0 + (24 + 20.0) / 2.5
    assert original[0] == pytest.approx([expected_x, expected_y])
    assert original[1] == pytest.approx([100.0 + (16 + 50) / 2.5, expected_y])
