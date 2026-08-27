# qgis_yolo_annotator v0.2.0

面向遥感影像的 QGIS 4 智能标注插件，首个发布版。

## 推理

- YOLO 系列 ONNX 模型（det / obb / seg），进程内 onnxruntime，支持 ultralytics 8.3/8.4 导出布局（OBB 角度通道位置自适应）
- `.pt` 权重自动转 ONNX（可配置外部 ai_env Python 环境）
- 当前视图推理 + 场景滑窗推理（后台 QgsTask，跨窗旋转 NMS 合并），输入地面分辨率可配置（m/px 或 °/px 单位体系）

## 标注（OBB 特别优化）

- 两点式绘制旋转框，角点/边中点手柄精修（旋转 + 等比缩放），中心平移，方向键微调
- 数字快捷键改类别、右键类别菜单、多选批量改、A→B 全局替换、类别改名联动
- Ctrl+Z / Ctrl+Y 撤销重做，Space 平移，出场景自动切换平移
- 类别管理仿 X-AnyLabeling：31 色自动配色、顺序调整、自定义快捷键

## 在线影像工作流

- 无需本地 tiff：QuickMapServices 在线图层（Google Satellite 等）上直接画场景
- 场景虚拟像素网格（EPSG:3857 + zoom，瓦片磁盘缓存），导出在线影像 GeoTIFF + 标签
- 场景范围可拖拽调整，已有标注自动随网格迁移（原点平移精确换算）

## 工程与导出

- 标注工程（project.json + labels/*.json，X-AnyLabeling JSON 兼容，像素坐标互通）
- 场景三态（未标注/已标注/已审核），工程树导航，删除场景联动标注
- DOTA / YOLO-OBB / YOLO-det / VOC 导出；GeoTIFF 切片携带地理信息，目标分辨率重采样（如 0.2966 m/px），train/val 划分
- DOTA / YOLO-OBB 标签导入
- 每步操作即时原子落盘

## 质量

- 78 项单元测试（core/export 纯逻辑）+ 无头集成实测脚本（scripts/diag_*.py）
- 语法 + pyflakes 未定义名检查（scripts/check_syntax.py）
