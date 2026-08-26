"""pytest 配置：src 布局路径注入 + 通用 fixture（合成 GeoTIFF）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def make_geotiff(tmp_path):
    """生成合成 GeoTIFF 的工厂。

    返回 fn(name, w, h, gt=None, wkt=None) -> Path：
    gt/wkt 缺省为无参考影像。
    """
    from osgeo import gdal, osr

    gdal.UseExceptions()

    def _make(name: str, w: int, h: int, gt=None, wkt: str | None = None) -> Path:
        path = tmp_path / name
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(str(path), w, h, 3, gdal.GDT_Byte)
        try:
            rng = np.random.default_rng(0)
            for b in range(3):
                band = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
                ds.GetRasterBand(b + 1).WriteArray(band)
            if gt is not None:
                ds.SetGeoTransform(gt)
            if wkt is not None:
                srs = osr.SpatialReference()
                srs.ImportFromWkt(wkt)
                ds.SetSpatialRef(srs)
            ds.FlushCache()
        finally:
            ds = None
        return path

    return _make


WGS84_WKT = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",1.0]]'
