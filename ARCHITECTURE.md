# ARCHITECTURE — qgis_yolo_annotator

## 系统概览

```
QGIS 4 (Qt6 / Python 3.12)
└── qgis_yolo_annotator（插件包）
    ├── plugin.py           入口：菜单/工具栏/QDockWidget 生命周期
    ├── gui/
    │   ├── main_dock.py    主面板（工程/标注推理/导出 三页签），信号接线中枢
    │   ├── controller.py   控制器：工程↔图层↔影像状态（唯一拥有者）
    │   ├── obb_edit_tool.py OBB 绘制/编辑 QgsMapTool
    │   ├── scene_tool.py   场景矩形绘制 QgsMapTool
    │   ├── annotation_layer.py shapes↔QgsFeature 双向转换 + 分类渲染
    │   ├── scene_layer.py  场景图层 + 状态渲染
    │   ├── model_dialog.py / export_dialog.py / class_dialog.py
    ├── core/               纯逻辑（不依赖 qgis，可 pytest）
    │   ├── raster_io.py    RasterRef：GDAL 分块读、像素↔地图仿射、分辨率换算（米↔度）
    │   ├── inference.py    YoloOrtModel：ONNX det/obb/seg + letterbox + 自实现旋转 NMS
    │   ├── scene_infer.py  场景滑窗推理几何（目标分辨率重采样 + 跨窗合并）
    │   ├── model_registry.py 模型注册表（profile 目录 JSON 持久化）
    │   ├── label_store.py  X-AnyLabeling JSON 读写 + DOTA 导入
    │   ├── project.py      标注工程（project.json + labels/；场景三态挂影像）
    │   └── geometry.py     slide_starts 滑窗起点（切片/推理共用）
    ├── export/
    │   ├── converters.py   DOTA/VOC/YOLO 格式转换（纯函数 + 边界策略）
    │   └── chip_export.py  切片导出（GeoTIFF 子 geotransform + 分辨率重采样）
    └── tasks/
        ├── infer_task.py   场景推理 QgsTask（后台）
        └── export_task.py  数据集导出 QgsTask（后台）
```

## 核心决策（SSOT 与数据流）

1. **标注磁盘格式 = X-AnyLabeling JSON（像素坐标）**；画布显示 = 内存 QgsVectorLayer（地图坐标）。
   两者由 `RasterRef.pixel_to_map / map_to_pixel`（GDAL GeoTransform 仿射）双向换算；
   磁盘是唯一持久层，图层随影像加载重建，保存时整层导出。
2. **OBB SSOT = 4 角点像素坐标 + direction 弧度**（atan2(p0→p1)，与 X-AnyLabeling 一致）；
   导出时按目标格式转换（DOTA 像素角点 / YOLO 归一化 / VOC HBB 外接）。
3. **状态挂在场景（Scene）上**：unannotated → annotated → verified；
   影像不持有状态（由场景聚合展示）。场景 bbox 为像素坐标（跟影像走）。
4. **分辨率语义**：`rescale = 原始分辨率(以选定单位) / 目标分辨率`
   - 推理（scene_infer）：目标网格像素 (i,j) ↔ 原始像素 (scene_x0 + i/rescale_x, ...)
   - 导出（chip_export）：输出 chip = 目标网格窗口；读原始窗口（target/rescale）重采样写出；
     子 geotransform 分辨率 = 原 gt / rescale
   - 地理坐标系（度）按影像中心纬度的米/度系数换算（WGS84 椭球近似）
5. **推理在 QGIS 进程内**（onnxruntime CPU），会话缓存 + 线程锁串行；
   OBB NMS 自实现凸多边形 IoU（Sutherland-Hodgman + shoelace），
   不依赖 cv2.dnn.NMSBoxesRotated（OpenCV 4/5 行为不一致）。

## 依赖边界

- QGIS Python（自带）：qgis.core/gui、osgeo(gdal)、numpy
- pip 补装：onnxruntime、opencv-python-headless（letterbox/seg 轮廓用）
- core/ 与 export/ **禁止 import qgis**（pytest 可直接跑）；GUI 依赖注入 iface

## 已知边界与迭代方向

- 编辑写层绕过 QGIS undo 栈（dataProvider 直写）；后续可接 editing session
- 边中点手柄为单轴伸缩；角点手柄可旋转（覆盖主要精修场景）
- 导入：X-AnyLabeling JSON（工程 labels/ 直接读）与 DOTA txt（label_store.import_dota，未接 UI）
- seg 任务推理可用；编辑手柄面向 OBB（polygon 可编辑但无专门优化）
