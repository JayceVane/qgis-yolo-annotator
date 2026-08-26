"""检查 ultralytics 8.4 导出 OBB onnx 的输出布局与值域。"""

import sys

sys.path.insert(0, r"D:\Workspace\SarDetection\t2_qgis_intelligent\src")
import numpy as np
import onnxruntime as ort

path = r"D:\Workspace\SarDetection\t2_qgis_intelligent\.test_data\yolo11n-obb.onnx"
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
print("inputs:", [(i.name, i.shape) for i in session.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in session.get_outputs()])
meta = session.get_modelmeta().custom_metadata_map
print("meta keys:", list(meta.keys()))
print("names:", meta.get("names"))
print("task:", meta.get("task"), "| imgsz:", meta.get("imgsz"))

x = np.random.rand(1, 3, 640, 640).astype(np.float32)
outs = session.run(None, {session.get_inputs()[0].name: x})
for o, out in zip(session.get_outputs(), outs):
    flat = out.reshape(-1)
    print(f"output {o.name}: shape={out.shape} min={flat.min():.3f} max={flat.max():.3f}")
pred = outs[0]
print("pred[0, :12, 0] =", np.round(pred[0, :12, 0], 3))
print("pred[0, 4, :5] (angle?) =", np.round(pred[0, 4, :5], 3))
