"""raster_io 单测：分辨率换算、像素↔地图变换、窗口读取。"""

import numpy as np
import pytest

from qgis_yolo_annotator.core.raster_io import RasterRef, meters_per_degree

from .conftest import WGS84_WKT


def test_meters_per_degree_equator_and_midlat():
    lon_m, lat_m = meters_per_degree(0.0)
    assert lon_m == pytest.approx(111312.0, rel=1e-3)
    assert lat_m == pytest.approx(110575.0, rel=1e-3)  # 用户示例系数 ~110574
    # 中纬度经度方向收缩
    lon_m45, _ = meters_per_degree(45.0)
    assert lon_m45 < lon_m * 0.75


def test_open_utm_geotiff_pixel_map_roundtrip(make_geotiff):
    gt = (500000.0, 0.5, 0.0, 4400000.0, 0.0, -0.5)
    path = make_geotiff("utm.tif", 100, 80, gt=gt)
    ref = RasterRef.open(path)
    assert (ref.width, ref.height) == (100, 80)
    assert ref.has_georeference
    assert ref.is_geographic is False
    assert ref.resolution_m_per_px() == pytest.approx(0.5)
    # 左上角像素 (0,0) 中心应映射到 gt 原点
    mx, my = ref.pixel_to_map(0.0, 0.0)
    assert (mx, my) == (500000.0, 4400000.0)
    col, row = ref.map_to_pixel(mx, my)
    assert (col, row) == (0.0, 0.0)
    # 任意点往返
    mx, my = ref.pixel_to_map(37.2, 51.9)
    col, row = ref.map_to_pixel(mx, my)
    assert (col, row) == pytest.approx((37.2, 51.9))


def test_geographic_resolution_m_per_px(make_geotiff):
    # 用户示例场景：2.682209e-06 度/px ≈ 0.3 m/px（低纬度）
    gt = (100.0, 2.682209e-06, 0.0, 10.0, 0.0, -2.682209e-06)
    path = make_geotiff("geo.tif", 64, 64, gt=gt, wkt=WGS84_WKT)
    ref = RasterRef.open(path)
    assert ref.is_geographic is True
    res = ref.resolution_m_per_px()
    # 与公式复算一致（影像中心纬度处的经/纬向系数几何平均）
    import math

    center_lat = 10.0 - (64 / 2) * 2.682209e-06
    lon_m, lat_m = meters_per_degree(center_lat)
    expected = 2.682209e-06 * math.sqrt(lon_m * lat_m)
    assert res == pytest.approx(expected, rel=1e-9)
    assert 0.29 < res < 0.31


def test_no_georeference(make_geotiff):
    path = make_geotiff("plain.tif", 32, 32)
    ref = RasterRef.open(path)
    assert not ref.has_georeference
    assert ref.resolution_m_per_px() is None


def test_read_window_bgr_shape_and_out_of_range(make_geotiff):
    path = make_geotiff("rgb.tif", 64, 48)
    ref = RasterRef.open(path)
    block = ref.read_window_bgr(10, 5, 32, 32)
    assert block.shape == (32, 32, 3)
    assert block.dtype == np.uint8
    with pytest.raises(ValueError, match="越界"):
        ref.read_window_bgr(40, 40, 64, 64)


def test_read_window_single_band(tmp_path):
    from osgeo import gdal

    path = tmp_path / "gray.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(path), 16, 16, 1, gdal.GDT_Byte)
    ds.GetRasterBand(1).WriteArray(np.full((16, 16), 200, dtype=np.uint8))
    ds = None
    ref = RasterRef.open(path)
    block = ref.read_window_bgr(0, 0, 16, 16)
    assert block.shape == (16, 16, 3)
    assert (block == 200).all()
