"""验证 pt→onnx 转换桥（用 p2gsd.pt 实测）。"""

import sys
import time

sys.path.insert(0, r"D:\Workspace\SarDetection\t2_qgis_intelligent\src")
from pathlib import Path

from qgis_yolo_annotator.core.pt_converter import (
    convert_pt_to_onnx,
    find_ai_env_python,
    python_has_ultralytics,
    read_pt_metadata,
)

py = find_ai_env_python()
print("detected ai_env python:", py, "| has ultralytics:", python_has_ultralytics(py))

t0 = time.time()
meta = read_pt_metadata(r"D:\JayceVane\Downloads\p2gsd.pt", py)
print(f"pt meta ({time.time()-t0:.1f}s):", meta)

cache = Path(r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\models_cache")
t0 = time.time()
onnx = convert_pt_to_onnx(
    r"D:\JayceVane\Downloads\p2gsd.pt", py, meta["imgsz"] or 1024, cache
)
print(f"converted ({time.time()-t0:.1f}s):", onnx, onnx.stat().st_size // 1024 // 1024, "MB")

# 用推理管线验证转换产物
from qgis_yolo_annotator.core.inference import YoloOrtModel
from qgis_yolo_annotator.core.model_registry import read_onnx_metadata

om = read_onnx_metadata(onnx)
print("onnx meta:", {k: (v[:3] + ["..."] if k == "labels" and v else v) for k, v in om.items()})
session = YoloOrtModel(onnx, om["task"] or "obb")
from qgis_yolo_annotator.core.raster_io import RasterRef

raster = RasterRef.open(r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\real_aerial.tif")
block = raster.read_window_bgr(0, 0, raster.width, raster.height)
dets = session.infer(block, om["imgsz"] or 1024, conf=0.2, iou=0.45)
import math

import numpy as np

print("p2gsd detections:", len(dets))
for d in dets[:8]:
    pts = np.asarray(d.points)
    ang = math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
    lbl = om["labels"][d.class_index] if d.class_index < len(om["labels"]) else d.class_index
    print(f"  {lbl} score={d.score:.3f} angle={ang:.1f} center=({pts[:,0].mean():.0f},{pts[:,1].mean():.0f})")
