"""栅格影像地理参考 IO：GDAL 分块读取、像素↔地图坐标变换、分辨率换算（米↔度）。

约定：
- 像素坐标（col=x 向右, row=y 向下）为标注存储 SSOT（X-AnyLabeling 兼容）。
- 地图坐标由 GDAL GeoTransform 仿射变换定义；无参考影像使用 identity 变换。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

# 地理坐标系（度）下每度对应的米数（WGS84 椭球近似，Sjöberg 公式截断）
_EQUATOR_M_PER_DEG_LON = 111412.84
_POLE_M_PER_DEG_LAT = 111132.92
_LAT_CORRECTION = 559.82
_LAT_CORRECTION_HARMONIC = 1.175


def meters_per_degree(latitude_deg: float) -> tuple[float, float]:
    """计算指定纬度处经度/纬度方向的每度米数（WGS84 近似）。

    Args:
        latitude_deg: 纬度（度）。

    Returns:
        (lon_m_per_deg, lat_m_per_deg)：经度方向、纬度方向每度对应的米数。
    """
    lat = math.radians(latitude_deg)
    lon_m = _EQUATOR_M_PER_DEG_LON * math.cos(lat) - 93.5 * math.cos(3 * lat)
    lat_m = (
        _POLE_M_PER_DEG_LAT
        - _LAT_CORRECTION * math.cos(2 * lat)
        + _LAT_CORRECTION_HARMONIC * math.cos(4 * lat)
    )
    return lon_m, lat_m


@dataclass
class RasterRef:
    """单幅栅格影像的地理参考封装（只读、惰性打开数据集）。"""

    path: Path
    width: int
    height: int
    band_count: int
    dtype: int
    geotransform: tuple[float, float, float, float, float, float]
    crs_wkt: str | None
    # 地理坐标系（度单位）时为 True；无参考时 None
    is_geographic: bool | None

    _dataset: gdal.Dataset | None = None

    @classmethod
    def open(cls, path: str | Path) -> "RasterRef":
        """打开栅格并读取地理参考元数据。

        Args:
            path: 影像文件路径（tif/png/jpg 等 GDAL 可读格式）。

        Returns:
            RasterRef 实例。

        Raises:
            RuntimeError: GDAL 无法打开文件时。
        """
        path = Path(path)
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"GDAL 打开失败: {path}")
        gt = ds.GetGeoTransform()
        has_geo = not (gt == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        srs = ds.GetSpatialRef()
        if not has_geo:
            # 无 geotransform：视为纯像素影像
            crs_wkt = None
            is_geo = None
        elif srs is not None:
            crs_wkt = srs.ExportToWkt()
            is_geo = bool(srs.IsGeographic())
        else:
            # 有 geotransform 无 CRS：按投影坐标（线性单位≈米）处理
            crs_wkt = None
            is_geo = False
        return cls(
            path=path,
            width=ds.RasterXSize,
            height=ds.RasterYSize,
            band_count=ds.RasterCount,
            dtype=ds.GetRasterBand(1).DataType,
            geotransform=tuple(gt),
            crs_wkt=crs_wkt,
            is_geographic=is_geo,
            _dataset=ds,
        )

    # ------------------------------------------------------------------ 坐标变换

    def pixel_to_map(
        self, col: np.ndarray | float, row: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """像素坐标 → 地图坐标（仿射变换，支持 numpy 向量化）。

        Args:
            col: 列坐标（x，向右）。
            row: 行坐标（y，向下）。

        Returns:
            (map_x, map_y) 地图坐标。
        """
        gt0, gt1, gt2, gt3, gt4, gt5 = self.geotransform
        map_x = gt0 + col * gt1 + row * gt2
        map_y = gt3 + col * gt4 + row * gt5
        return map_x, map_y

    def map_to_pixel(
        self, map_x: np.ndarray | float, map_y: np.ndarray | float
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """地图坐标 → 像素坐标（逆仿射变换，当前实现要求 gt2==gt4==0 即无旋转）。

        Args:
            map_x: 地图 X。
            map_y: 地图 Y。

        Returns:
            (col, row) 像素坐标。

        Raises:
            ValueError: 影像带旋转（gt2/gt4 非 0）时不支持。
        """
        gt0, gt1, gt2, gt3, gt4, gt5 = self.geotransform
        if gt2 != 0.0 or gt4 != 0.0:
            raise ValueError(
                f"带旋转的 GeoTransform 不支持逆变换: {self.geotransform}"
            )
        col = (map_x - gt0) / gt1
        row = (map_y - gt3) / gt5
        return col, row

    # ------------------------------------------------------------------ 分辨率

    def center_latitude(self) -> float | None:
        """影像中心纬度（度）。仅地理坐标系返回有效值。"""
        if not self.is_geographic:
            return None
        cx = (self.geotransform[0] + self.geotransform[0]
              + self.width * self.geotransform[1]) / 2.0
        cy = (self.geotransform[3] + self.geotransform[3]
              + self.height * self.geotransform[5]) / 2.0
        _ = cx
        return float(cy)

    def resolution_m_per_px(self) -> float | None:
        """影像地面分辨率（米/像素）。

        投影坐标系：直接取 |gt1|（假设米单位）；
        地理坐标系（度/像素）：按中心纬度换算为米，取 x/y 方向几何平均；
        无参考：返回 None。

        Returns:
            分辨率（米/像素）；无法确定时 None。
        """
        if self.is_geographic is None:
            return None
        res_x = abs(self.geotransform[1])
        res_y = abs(self.geotransform[5])
        if self.is_geographic:
            lat = self.center_latitude()
            lon_m, lat_m = meters_per_degree(lat if lat is not None else 0.0)
            res_x_m = res_x * lon_m
            res_y_m = res_y * lat_m
        else:
            res_x_m, res_y_m = res_x, res_y
        return math.sqrt(res_x_m * res_y_m)

    # ------------------------------------------------------------------ 读取

    def read_window_bgr(
        self, xoff: int, yoff: int, xsize: int, ysize: int
    ) -> np.ndarray:
        """分块读取指定窗口为 BGR uint8 数组（供 cv2 推理管线）。

        波段映射：3 波段按 RGB→BGR 反转；1 波段灰度复制三通道；
        ≥4 波段取前三波段（RGB）→ BGR。非 uint8 数据线性拉伸到 0-255。

        Args:
            xoff: 窗口左上角列（像素）。
            yoff: 窗口左上角行（像素）。
            xsize: 窗口宽（像素）。
            ysize: 窗口高（像素）。

        Returns:
            shape=(ysize, xsize, 3) 的 BGR uint8 数组。

        Raises:
            ValueError: 窗口超出影像范围。
        """
        if xoff < 0 or yoff < 0 or xoff + xsize > self.width or yoff + ysize > self.height:
            raise ValueError(
                f"窗口越界: off=({xoff},{yoff}) size=({xsize},{ysize}) "
                f"raster=({self.width}x{self.height})"
            )
        ds = self._ensure_dataset()
        arr = ds.ReadAsArray(xoff, yoff, xsize, ysize)
        if arr is None:
            raise RuntimeError(f"GDAL ReadAsArray 失败: {self.path} window=({xoff},{yoff},{xsize},{ysize})")
        return _to_bgr_u8(arr)

    def _ensure_dataset(self) -> gdal.Dataset:
        if self._dataset is None:
            self._dataset = gdal.Open(str(self.path), gdal.GA_ReadOnly)
            if self._dataset is None:
                raise RuntimeError(f"GDAL 重新打开失败: {self.path}")
        return self._dataset

    def close(self) -> None:
        """释放 GDAL 数据集句柄。"""
        self._dataset = None

    @property
    def has_georeference(self) -> bool:
        """影像是否携带地理参考（有 geotransform 即视为有）。"""
        return self.is_geographic is not None


def _to_bgr_u8(arr: np.ndarray) -> np.ndarray:
    """GDAL ReadAsArray 结果 → BGR uint8 (h, w, 3)。"""
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.shape[0] == 1:
        gray = _stretch_u8(arr[0])
        return np.stack([gray] * 3, axis=-1)
    rgb = arr[:3]
    rgb = np.stack([_stretch_u8(b) for b in rgb], axis=-1)
    return np.ascontiguousarray(rgb[..., ::-1])


def _stretch_u8(band: np.ndarray) -> np.ndarray:
    """单波段归一化到 uint8（已是 uint8 直接返回，否则 min-max 拉伸）。"""
    if band.dtype == np.uint8:
        return band
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        return np.zeros(band.shape, dtype=np.uint8)
    scaled = (band.astype(np.float32) - vmin) / (vmax - vmin) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)
