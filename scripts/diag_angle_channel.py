"""验证 OBB onnx 角度通道位置：第 4 行 vs 最后一行。"""

import sys

sys.path.insert(0, r"D:\Workspace\SarDetection\t2_qgis_intelligent\src")
import numpy as np
import onnxruntime as ort
import cv2

path = r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\yolo11n-obb.onnx"
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
image = cv2.imread(r"D:\Workspace\SarDetection\d0_dataset\cardet\car_det\images\train\7431.png")
resized = cv2.resize(image, (640, 640))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
outs = session.run(None, {"images": tensor})
pred = outs[0][0]  # (20, 8400)

print("shape:", pred.shape)
print("row4  (assumed angle): min=%.4f max=%.4f nonzero=%d" % (pred[4].min(), pred[4].max(), np.count_nonzero(pred[4])))
print("row19 (last):          min=%.4f max=%.4f nonzero=%d" % (pred[19].min(), pred[19].max(), np.count_nonzero(pred[19])))
print("row19 sample:", np.round(pred[19, :10], 4))
# 各行值域速览
for i in [4, 5, 15, 18, 19]:
    print(f"row{i}: min={pred[i].min():.3f} max={pred[i].max():.3f} mean={pred[i].mean():.4f}")
