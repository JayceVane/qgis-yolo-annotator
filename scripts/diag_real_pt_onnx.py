"""真实影像上对比 pt/onnx 的 OBB 预测输出（xywhr 角度）。"""

from ultralytics import YOLO

img = r"D:\Workspace\SarDetection\d0_dataset\cardet\car_det\images\train\7431.png"

pt = YOLO(r"D:\Workspace\SarDetection\t2_qgis_intelligent\yolo11n-obb.pt")
r1 = pt.predict(img, imgsz=640, conf=0.3, verbose=False)[0]
print("PT dets:", len(r1.obb) if r1.obb is not None else 0)
if r1.obb is not None and len(r1.obb):
    print("PT xywhr[:3]:\n", r1.obb.xywhr[:3].numpy())
    print("PT conf[:3]:", r1.obb.conf[:3].numpy())
    print("PT cls[:3]:", r1.obb.cls[:3].numpy())

ox = YOLO(r"D:\Workspace\SarDetection\t2_qgis_intelligent\yolo11n-obb.onnx")
r2 = ox.predict(img, imgsz=640, conf=0.3, verbose=False)[0]
print("ONNX dets:", len(r2.obb) if r2.obb is not None else 0)
if r2.obb is not None and len(r2.obb):
    print("ONNX xywhr[:3]:\n", r2.obb.xywhr[:3].numpy())
    print("ONNX conf[:3]:", r2.obb.conf[:3].numpy())
    print("ONNX cls[:3]:", r2.obb.cls[:3].numpy())
