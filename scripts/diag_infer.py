"""推理管线诊断：直接调 infer_scene 打印中间统计（独立进程，无超时限制）。"""

import sys
import time

sys.path.insert(0, r"D:\Workspace\SarDetection\t2_qgis_intelligent\src")
import numpy as np

from qgis_yolo_annotator.core.inference import YoloOrtModel, rotate_box_points
from qgis_yolo_annotator.core.model_registry import ModelConfig
from qgis_yolo_annotator.core.raster_io import RasterRef
from qgis_yolo_annotator.core.scene_infer import SceneInferOptions, scene_rescale, scene_windows
from qgis_yolo_annotator.core.project import SceneDef

raster = RasterRef.open(r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\test_scene.tif")
cfg = ModelConfig(
    name="t", task="obb",
    file_path=r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\yolo11n-obb.onnx",
    labels=["x"], imgsz=640, conf=0.15, iou=0.45,
)
session = YoloOrtModel(cfg.file_path, "obb")
options = SceneInferOptions(target_res=0.2, unit="m", chip_size=1024, overlap=200)

rescale = scene_rescale(raster, options)
print("rescale:", rescale)
scene = SceneDef(name="diag", bbox=[0, 0, 1000, 1000])
windows = scene_windows(scene, raster.width, raster.height, rescale, options)
print("windows:", len(windows))

t0 = time.time()
total_raw = 0
for i, w in enumerate(windows):
    tx, ty, tw, th = w.target_xywh
    ox, oy, ow, oh = w.orig_xywh
    block = raster.read_window_bgr(ox, oy, ow, oh)
    import cv2
    if (block.shape[1], block.shape[0]) != (tw, th):
        block = cv2.resize(block, (tw, th), interpolation=cv2.INTER_AREA if tw < ow else cv2.INTER_LINEAR)
    dets = session.infer(block, cfg.imgsz, cfg.conf, cfg.iou)
    total_raw += len(dets)
    print(f"win{i}: target=({tx},{ty},{tw},{th}) orig=({ox},{oy},{ow},{oh}) dets={len(dets)} t={time.time()-t0:.1f}s")
    if dets:
        d = dets[0]
        print("   sample det: score=%.3f pts0=%s" % (d.score, [(round(p[0],1), round(p[1],1)) for p in d.points[:2]]))
print("total raw dets:", total_raw, f"in {time.time()-t0:.1f}s")
