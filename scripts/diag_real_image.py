"""用真实遥感影像（car_det 样本）验证插件推理管线的 OBB 输出（角度/分数）。"""

import sys

sys.path.insert(0, r"D:\Workspace\SarDetection\t2_qgis_intelligent\src")
import cv2
import numpy as np

from qgis_yolo_annotator.core.inference import YoloOrtModel

img_path = r"D:\Workspace\SarDetection\d0_dataset\cardet\car_det\images\train\7431.png"
model_path = r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\yolo11n-obb.onnx"

image = cv2.imread(img_path)
print("image:", image.shape)
session = YoloOrtModel(model_path, "obb")
dets = session.infer(image, imgsz=640, conf=0.25, iou=0.45)
print("detections:", len(dets))
for d in dets[:8]:
    pts = np.asarray(d.points)
    w = np.linalg.norm(pts[1] - pts[0])
    h = np.linalg.norm(pts[3] - pts[0])
    ang = np.degrees(np.arctan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]))
    print(
        f"  score={d.score:.3f} cls={d.class_index:2d} "
        f"size={w:.0f}x{h:.0f} angle={ang:6.1f}° center=({pts[:,0].mean():.0f},{pts[:,1].mean():.0f})"
    )
