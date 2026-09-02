"""在线 XYZ 瓦片虚拟影像：把「场景矩形 + zoom」包装成 RasterRef 兼容对象。

设计要点：
- Web Mercator (EPSG:3857) 全球像素坐标系：zoom z 下世界宽 256*2^z 像素
- 场景虚拟影像的像素网格 = 场景矩形本身（起点=左上角地图坐标），
  因此场景自身像素 bbox 恒为 [0, 0, width, height]，推理/导出管线零改动
- read_window_bgr 按需下载瓦片（磁盘缓存，跨会话复用）
- duck-type 兼容 RasterRef 接口：width/height/geotransform/pixel_to_map/
  map_to_pixel/read_window_bgr/resolution_m_per_px/has_georeference/close
"""

from __future__ import annotations

import hashlib
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

# Web Mercator 常量
_WORLD_M = 40075016.68557849  # 赤道周长（米）
_TILE_PX = 256

_HEADERS = {"User-Agent": "Mozilla/5.0 (QGIS YoloAnnotator)"}
_DOWNLOAD_RETRIES = 3
_EMPTY_TILE = None  # 下载失败以占位灰块代替，保证管线不中断

_URL_PATTERN_MISSING = ("{x}", "{y}", "{z}")


@dataclass
class XyzSourceConfig:
    """XYZ 瓦片源定义。"""

    url_template: str  # 含 {z}{x}{y}
    title: str = "XYZ"
    min_zoom: int = 0
    max_zoom: int = 20

    def __post_init__(self):
        if any(token not in self.url_template for token in _URL_PATTERN_MISSING):
            raise ValueError(f"URL 模板缺少 {{z}}/{{x}}/{{y}}: {self.url_template}")

    def to_dict(self) -> dict:
        return {
            "url_template": self.url_template,
            "title": self.title,
            "min_zoom": self.min_zoom,
            "max_zoom": self.max_zoom,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "XyzSourceConfig":
        return cls(
            url_template=str(data["url_template"]),
            title=str(data.get("title", "XYZ")),
            min_zoom=int(data.get("min_zoom", 0)),
            max_zoom=int(data.get("max_zoom", 20)),
        )


def world_pixels(zoom: int) -> int:
    """zoom 级别下的全球边长（像素）。"""
    return _TILE_PX * (2 ** int(zoom))


def meters_per_pixel(zoom: int, latitude_deg: float = 0.0) -> float:
    """Web Mercator 地面分辨率（米/像素，随纬度收缩）。"""
    return _WORLD_M / world_pixels(zoom) * math.cos(math.radians(latitude_deg))


def choose_zoom(target_res_m: float, latitude_deg: float, min_zoom: int = 0, max_zoom: int = 20) -> int:
    """按目标地面分辨率选最接近的 zoom（不超源上限）。"""
    best, best_diff = min_zoom, float("inf")
    for z in range(min_zoom, max_zoom + 1):
        diff = abs(meters_per_pixel(z, latitude_deg) - target_res_m)
        if diff < best_diff:
            best, best_diff = z, diff
    return best


def map_to_world_pixel(x_m: float, y_m: float, zoom: int) -> tuple[float, float]:
    """EPSG:3857 坐标 → 全球像素坐标（浮点）。"""
    world_px = world_pixels(zoom)
    px = (x_m + _WORLD_M / 2) / _WORLD_M * world_px
    py = (_WORLD_M / 2 - y_m) / _WORLD_M * world_px
    return px, py


class TileCache:
    """瓦片磁盘缓存（url hash 命名，跨会话复用）。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, url: str) -> Path:
        # 仅作本地缓存文件名（非安全用途）；sha1 已够且最短
        digest = hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest()[:24]
        return self.root / f"{digest}.png"

    def get(self, url: str) -> np.ndarray | None:
        """取瓦片（缓存命中直接读；未命中下载后写缓存）。"""
        cached = self._path_for(url)
        if cached.is_file():
            data = cv2.imread(str(cached), cv2.IMREAD_COLOR)
            if data is not None:
                return data
        data = self._download(url)
        if data is not None:
            cv2.imwrite(str(cached), data)
        return data

    def _download(self, url: str) -> np.ndarray | None:
        # 瓦片源只允许 http(s)，杜绝 file:/自定义 scheme 被 urlopen 展开
        if urlparse(url).scheme not in ("http", "https"):
            return _EMPTY_TILE
        for _attempt in range(_DOWNLOAD_RETRIES):
            try:
                request = urllib.request.Request(url, headers=_HEADERS)
                with urllib.request.urlopen(request, timeout=30) as resp:  # nosec B310: 入口已限定 http(s)
                    payload = resp.read()
                tile = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if tile is not None:
                    return tile
            except (urllib.error.URLError, OSError):
                continue
        return _EMPTY_TILE


# EPSG:3857 WKT（导出 GeoTIFF 时写入投影用；定义在类前供 __init__ 引用）
EPSG3857_WKT = (
    'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",'
    'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],'
    'PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],'
    'AXIS["Northing",NORTH],AUTHORITY["EPSG","3857"]]'
)


class XyzRaster:
    """场景级在线影像：给定地图矩形 + zoom 构建像素网格，按需读瓦片。

    与 RasterRef duck-type 兼容（像素网格起点=场景左上角）。
    """

    def __init__(
        self,
        source: XyzSourceConfig,
        bbox_map: list[float],
        zoom: int,
        cache_dir: str | Path,
        name: str | None = None,
    ):
        """初始化虚拟影像。

        Args:
            source: XYZ 瓦片源。
            bbox_map: [xmin, ymin, xmax, ymax] EPSG:3857 地图坐标。
            zoom: 瓦片级别。
            cache_dir: 瓦片缓存目录。
            name: 影像名（导出文件命名用）。同一图层的多个场景必须传各自
                场景名，否则导出文件同名互相覆盖；缺省用“图层_z”。
        """
        if len(bbox_map) != 4 or bbox_map[0] >= bbox_map[2] or bbox_map[1] >= bbox_map[3]:
            raise ValueError(f"bbox_map 非法: {bbox_map}")
        zoom = int(zoom)
        if not source.min_zoom <= zoom <= source.max_zoom:
            raise ValueError(
                f"zoom {zoom} 超出源范围 [{source.min_zoom}, {source.max_zoom}]"
            )
        xmin, ymin, xmax, ymax = bbox_map
        px0, py0 = map_to_world_pixel(xmin, ymax, zoom)
        px1, py1 = map_to_world_pixel(xmax, ymin, zoom)
        self.source = source
        self.bbox_map = [float(v) for v in bbox_map]
        self.zoom = zoom
        self.width = max(1, int(math.ceil(px1 - px0)))
        self.height = max(1, int(math.ceil(py1 - py0)))
        res = _WORLD_M / world_pixels(zoom)
        self.geotransform = (xmin, res, 0.0, ymax, 0.0, -res)
        self.crs_wkt = EPSG3857_WKT  # 导出 GeoTIFF/建图层时携带投影
        self.is_geographic = False
        self.path = Path(name) if name else Path(f"{source.title}_z{zoom}")  # 兼容 RasterRef.path 命名语义
        self.band_count = 3
        self._cache = TileCache(cache_dir)
        self._tile_store: dict[tuple[int, int, int], np.ndarray | None] = {}

    # ------------------------------------------------------------- RasterRef 接口

    @property
    def has_georeference(self) -> bool:
        return True

    def pixel_to_map(self, col, row):
        gt0, gt1, _gt2, gt3, _gt4, gt5 = self.geotransform
        return gt0 + col * gt1, gt3 + row * gt5

    def map_to_pixel(self, map_x, map_y):
        gt0, gt1, _gt2, gt3, _gt4, gt5 = self.geotransform
        return (map_x - gt0) / gt1, (map_y - gt3) / gt5

    def center_latitude(self) -> float:
        """场景中心纬度（度，用于地面分辨率显示）。"""
        _xmin, ymin, _xmax, ymax = self.bbox_map
        y_mid = (ymin + ymax) / 2
        return math.degrees(math.atan(math.sinh(y_mid / 6378137.0)))

    def resolution_m_per_px(self) -> float:
        return meters_per_pixel(self.zoom, self.center_latitude())

    def close(self) -> None:
        """兼容接口：释放瓦片内存缓存。"""
        self._tile_store.clear()

    def read_window_bgr(self, xoff: int, yoff: int, xsize: int, ysize: int) -> np.ndarray:
        """读取窗口（场景像素坐标）→ 下载瓦片拼接裁剪为 BGR。

        Args:
            xoff, yoff: 窗口左上（场景像素）。
            xsize, ysize: 窗口尺寸。

        Returns:
            (ysize, xsize, 3) BGR uint8。

        Raises:
            ValueError: 窗口越界。
        """
        if xoff < 0 or yoff < 0 or xoff + xsize > self.width or yoff + ysize > self.height:
            raise ValueError(
                f"窗口越界: off=({xoff},{yoff}) size=({xsize},{ysize}) "
                f"raster=({self.width}x{self.height})"
            )
        xmin_m, _ymin_m, _xmax_m, ymax_m = self.bbox_map
        res = self.geotransform[1]
        # 场景像素 → 全球像素
        gx0 = (xmin_m + xoff * res + _WORLD_M / 2) / _WORLD_M * world_pixels(self.zoom)
        gy0 = (_WORLD_M / 2 - (ymax_m - yoff * res)) / _WORLD_M * world_pixels(self.zoom)
        gx1 = gx0 + xsize
        gy1 = gy0 + ysize
        tx0, ty0 = int(gx0 // _TILE_PX), int(gy0 // _TILE_PX)
        tx1, ty1 = int(gx1 // _TILE_PX), int(gy1 // _TILE_PX)

        canvas_w = (tx1 - tx0 + 1) * _TILE_PX
        canvas_h = (ty1 - ty0 + 1) * _TILE_PX
        canvas = np.full((canvas_h, canvas_w, 3), 128, dtype=np.uint8)
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile = self._get_tile(tx, ty)
                if tile is None:
                    continue
                y_dst = (ty - ty0) * _TILE_PX
                x_dst = (tx - tx0) * _TILE_PX
                canvas[y_dst : y_dst + _TILE_PX, x_dst : x_dst + _TILE_PX] = tile
        # 全球像素 → canvas 内偏移裁剪
        cx0 = int(gx0) - tx0 * _TILE_PX
        cy0 = int(gy0) - ty0 * _TILE_PX
        block = canvas[cy0 : cy0 + ysize, cx0 : cx0 + xsize]
        if block.shape[0] != ysize or block.shape[1] != xsize:
            padded = np.full((ysize, xsize, 3), 128, dtype=np.uint8)
            padded[: block.shape[0], : block.shape[1]] = block
            block = padded
        return np.ascontiguousarray(block)

    # ------------------------------------------------------------- 内部

    def _get_tile(self, tx: int, ty: int) -> np.ndarray | None:
        """取瓦片（内存 → 磁盘缓存 → 网络）。"""
        key = (self.zoom, tx, ty)
        if key in self._tile_store:
            return self._tile_store[key]
        url = (
            self.source.url_template.replace("{z}", str(self.zoom))
            .replace("{x}", str(tx))
            .replace("{y}", str(ty))
        )
        tile = self._cache.get(url)
        self._tile_store[key] = tile
        return tile
