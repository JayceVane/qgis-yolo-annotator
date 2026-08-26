"""对比 .pt 与导出 onnx 的 OBB 预测（验证 angle 通道是否在导出中丢失）。"""

import numpy as np
import cv2
from ultralytics import YOLO

# 用真实图块驱动（与 onnx 诊断同一块）；cv2 可直接读 tiff
bgr = cv2.imread(r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\test_scene.tif")
block = bgr[0:640, 0:640]
cv2.imwrite(r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\diag_block.png", block)

model = YOLO(r"D:\Workspace\SarDetection\t2_qgis_intelligent\yolo11n-obb.pt")
results = model.predict(
    r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\diag_block.png",
    imgsz=640, conf=0.25, verbose=False,
)
for r in results:
    print("pt det count:", len(r.obb) if r.obb is not None else 0)
    if r.obb is not None and len(r.obb):
        print("pt xywhr[:3]:", r.obb.xywhr[:3].numpy())
        print("pt conf[:3]:", r.obb.conf[:3].numpy())
        print("pt cls[:3]:", r.obb.cls[:3].numpy())

onnx_model = YOLO(r"D:\Workspace\SarDetection\t2_qgis_intelligent\yolo11n-obb.onnx")
results2 = onnx_model.predict(
    r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\diag_block.png",
    imgsz=640, conf=0.25, verbose=False,
)
for r in results2:
    print("onnx det count:", len(r.obb) if r.obb is not None else 0)
    if r.obb is not None and len(r.obb):
        print("onnx xywhr[:3]:", r.obb.xywhr[:3].numpy())
        print("onnx conf[:3]:", r.obb.conf[:3].numpy())
