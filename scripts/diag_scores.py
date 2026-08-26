"""检查类别分数通道值域（是否需要 sigmoid 适配）。"""

import sys

sys.path.insert(0, r"D:\Workspace\SarDetection\t2_qgis_intelligent\src")
import numpy as np
import onnxruntime as ort
import cv2

path = r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\yolo11n-obb.onnx"
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

# 用真实图像块（而非随机噪声）驱动
from qgis_yolo_annotator.core.raster_io import RasterRef

raster = RasterRef.open(r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\test_scene.tif")
block = raster.read_window_bgr(0, 0, 640, 640)
rgb = cv2.cvtColor(cv2.resize(block, (640, 640)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
tensor = rgb.transpose(2, 0, 1)[None]

outs = session.run(None, {"images": tensor})
pred = outs[0][0]  # (20, 8400)
scores = pred[5:20, :]
print("class scores: min=%.4f max=%.4f mean=%.4f" % (scores.min(), scores.max(), scores.mean()))
print("top5 raw:", np.sort(scores.reshape(-1))[-5:])
sig = 1.0 / (1.0 + np.exp(-scores))
print("after sigmoid top5:", np.sort(sig.reshape(-1))[-5:])
print("angle channel: min=%.4f max=%.4f" % (pred[4].min(), pred[4].max()))
xywh = pred[:4]
print("xywh x-range: %.1f~%.1f y-range: %.1f~%.1f" % (xywh[0].min(), xywh[0].max(), xywh[1].min(), xywh[1].max()))
