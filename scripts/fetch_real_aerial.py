"""下载 Google Satellite 瓦片拼 GeoTIFF（EPSG:3857）作为真实测试影像。"""

import math
import os
import urllib.request

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
Z = 19
LAT, LON = 33.9138, -118.0788  # 洛杉矶居民区+停车场（车辆/泳池密集）
GRID = 5  # 5x5 tiles -> 1280x1280 px


def latlon_to_tile(lat: float, lon: float, z: int):
    n = 2**z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return int(x), int(y)


def tile_origin_meter(x: int, y: int, z: int):
    """瓦片左上角的 Web Mercator 坐标。"""
    n = 2**z
    return -180.0 + x / n * 360.0, math.atanh(math.cos(math.pi * (2 * y / n - 1)))


x0, y0 = latlon_to_tile(LAT, LON, Z)
print(f"tile origin: x={x0} y={y0} z={Z}")

canvas = np.zeros((256 * GRID, 256 * GRID, 3), dtype=np.uint8)
for dx in range(GRID):
    for dy in range(GRID):
        x, y = x0 + dx, y0 + dy
        url = URL.format(x=x, y=y, z=Z)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        arr = gdal.GetDriverByName("Memory").Create("", 0, 0, 0, 0)
        _ = arr
        import cv2

        tile = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if tile is None:
            raise RuntimeError(f"tile decode failed: {x},{y}")
        canvas[dy * 256 : (dy + 1) * 256, dx * 256 : (dx + 1) * 256] = tile

lon0, lat0_m = tile_origin_meter(x0, y0, Z)
res = 156543.03392 * math.cos(math.radians(LAT)) / (2 ** (Z - 1)) if False else (40075016.686 / 256 / 2**Z)
gt = (lon0, res, 0.0, lat0_m, 0.0, -res)
print(f"geotransform: {gt} (res≈{res:.3f} m/px)")

out = r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\real_aerial.tif"
driver = gdal.GetDriverByName("GTiff")
ds = driver.Create(out, canvas.shape[1], canvas.shape[0], 3, gdal.GDT_Byte, options=["COMPRESS=LZW"])
for b in range(3):
    ds.GetRasterBand(b + 1).WriteArray(canvas[:, :, b])
ds.SetGeoTransform(gt)
srs = osr.SpatialReference()
srs.ImportFromEPSG(3857)
ds.SetSpatialRef(srs)
ds.FlushCache()
ds = None
print("saved:", out, os.path.getsize(out) // 1024, "KB")

# 验证推理
import sys

sys.path.insert(0, r"D:\Workspace\SarDetection\t2_qgis_intelligent\src")
from qgis_yolo_annotator.core.inference import YoloOrtModel
from qgis_yolo_annotator.core.raster_io import RasterRef

ref = RasterRef.open(out)
print("raster res:", round(ref.resolution_m_per_px(), 4), "m/px")
session = YoloOrtModel(r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\yolo11n-obb.onnx", "obb")
block = ref.read_window_bgr(0, 0, 640, 640)
dets = session.infer(block, 640, conf=0.25, iou=0.45)
full = ref.read_window_bgr(0, 0, ref.width, ref.height)
dets_full = session.infer(full, 1280, conf=0.15, iou=0.45)
print("full-image detections:", len(dets_full))
for d in dets_full[:12]:
    pts = np.asarray(d.points)
    w = float(np.linalg.norm(pts[1] - pts[0])); h = float(np.linalg.norm(pts[3] - pts[0]))
    ang = math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
    print(f"  cls={d.class_index} score={d.score:.3f} {w:.0f}x{h:.0f}px angle={ang:.1f}00b0 center=({pts[:,0].mean():.0f},{pts[:,1].mean():.0f})")
print("detections (0-640 tile):", len(dets))
for d in dets[:10]:
    pts = np.asarray(d.points)
    w = float(np.linalg.norm(pts[1] - pts[0]))
    h = float(np.linalg.norm(pts[3] - pts[0]))
    ang = math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
    print(f"  cls={d.class_index} score={d.score:.3f} {w:.0f}x{h:.0f}px angle={ang:.1f}° center=({pts[:,0].mean():.0f},{pts[:,1].mean():.0f})")
