"""xyz_source 单测：几何换算、zoom 选择、虚拟影像网格与瓦片拼接（mock 瓦片）。"""

import math

import numpy as np
import pytest

from qgis_yolo_annotator.core.xyz_source import (
    EPSG3857_WKT,
    TileCache,
    XyzRaster,
    XyzSourceConfig,
    choose_zoom,
    map_to_world_pixel,
    meters_per_pixel,
    world_pixels,
)

URL = "https://example.com/vt/lyrs=s&x={x}&y={y}&z={z}"


def test_config_url_validation():
    with pytest.raises(ValueError, match="URL 模板"):
        XyzSourceConfig(url_template="https://example.com/no-placeholder")
    cfg = XyzSourceConfig(URL, "test", 3, 19)
    assert cfg.to_dict()["url_template"] == URL
    assert XyzSourceConfig.from_dict(cfg.to_dict()).max_zoom == 19


def test_world_pixels_and_resolution():
    assert world_pixels(0) == 256
    assert world_pixels(3) == 2048
    # 赤道 z0：整世界一瓦片 ≈ 156543 m/px
    assert meters_per_pixel(0, 0.0) == pytest.approx(156543.03392, rel=1e-6)
    # 纬度 60° 收缩一半
    assert meters_per_pixel(10, 60.0) == pytest.approx(
        meters_per_pixel(10, 0.0) * 0.5, rel=1e-6
    )


def test_choose_zoom_matches_target():
    # 洛杉矶纬度 0.2 m/px：z19 ≈ 0.248（最近）
    assert choose_zoom(0.2, 33.9, 0, 20) == 19
    # 上限约束
    assert choose_zoom(0.001, 0.0, 0, 18) == 18
    # 粗分辨率
    assert choose_zoom(60.0, 0.0, 0, 20) == 11


def test_map_to_world_pixel_roundtrip():
    x_m, y_m = -13124000.0, 4030000.0
    px, py = map_to_world_pixel(x_m, y_m, 18)
    world = 256 * 2**18
    x_back = px / world * 40075016.68557849 - 20037508.342789246
    y_back = 20037508.342789246 - py / world * 40075016.68557849
    assert x_back == pytest.approx(x_m, abs=1e-6)
    assert y_back == pytest.approx(y_m, abs=1e-6)


@pytest.fixture
def xyz_raster(tmp_path):
    """洛杉矶附近 1km×1km 场景 @z19（无网络：瓦片以合成数据 mock）。"""
    lat, lon = 33.9138, -118.0788
    res = meters_per_pixel(19, lat)
    half = 500.0  # 米
    x0 = lon / 180.0 * 20037508.342789246 - half
    x1 = x0 + 2 * half
    y0 = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * 6378137.0
    bbox = [x0, y0 - half, x1, y0 + half]
    raster = XyzRaster(XyzSourceConfig(URL, "TestSat", 0, 20), bbox, 19, tmp_path)
    # mock：每个瓦片返回按 (tx,ty) 染色的 256 块
    def _fake_tile(tx, ty):
        tile = np.zeros((256, 256, 3), dtype=np.uint8)
        tile[..., 0] = tx % 256
        tile[..., 1] = ty % 256
        tile[..., 2] = 60
        return tile
    raster._get_tile = _fake_tile
    return raster


def test_xyz_raster_grid(xyz_raster):
    r = xyz_raster
    # 地面分辨率：z19 网格 0.2986 m/px × cos(33.9°) ≈ 0.2477
    assert r.resolution_m_per_px() == pytest.approx(0.2477, rel=1e-2)
    # 像素网格：1000m / 0.2986（3857 网格分辨率）≈ 3350 px
    assert 3200 < r.width < 3500
    assert 3200 < r.height < 3500
    assert r.has_georeference
    assert r.is_geographic is False
    assert r.crs_wkt == EPSG3857_WKT
    # geotransform 起点为场景左上
    assert r.geotransform[0] == pytest.approx(r.bbox_map[0])
    assert r.geotransform[3] == pytest.approx(r.bbox_map[3])


def test_xyz_raster_pixel_map_roundtrip(xyz_raster):
    r = xyz_raster
    mx, my = r.pixel_to_map(100.5, 200.5)
    col, row = r.map_to_pixel(mx, my)
    assert (col, row) == pytest.approx((100.5, 200.5))


def test_xyz_raster_read_window_tiles_stitched(xyz_raster):
    """窗口跨多个瓦片：拼接后像素内容与 mock 瓦片坐标一致。"""
    r = xyz_raster
    block = r.read_window_bgr(0, 0, min(600, r.width), min(600, r.height))
    assert block.shape == (min(600, r.height), min(600, r.width), 3)
    # 场景左上角对应的全球瓦片坐标
    gx, gy = map_to_world_pixel(r.bbox_map[0], r.bbox_map[3], 19)
    tx0, ty0 = int(gx // 256), int(gy // 256)
    # 左上像素属于 (tx0, ty0) 瓦片
    assert block[0, 0, 0] == tx0 % 256
    assert block[0, 0, 1] == ty0 % 256


def test_xyz_raster_read_window_out_of_range(xyz_raster):
    with pytest.raises(ValueError, match="越界"):
        xyz_raster.read_window_bgr(0, 0, xyz_raster.width + 1, 10)


def test_tile_cache_roundtrip(tmp_path):
    cache = TileCache(tmp_path)
    url = "https://example.com/tile.png"
    assert cache.get(url) is None  # 无网络（或失败）返回 None 不抛异常
