# CONTEXT.md — 会话交接

更新：2026-08-26（本轮：场景范围调整）

## 本轮：场景范围调整（调场景工具）

- `gui/scene_edit_tool.py`：SceneEditTool——悬停命中 4 角点 + 4 边中点手柄（10px 容差）拖拽改大小、矩形内部拖拽整体移动；黄色橡皮带预览；Esc/右键取消
- `controller.update_scene_extent(scene_name, rect)`：
  - 文件场景：像素 bbox clip 到影像范围，标注（影像级坐标）不受影响
  - xyz 场景：更新 map_bbox（zoom 不变）；网格原点变更时标注按像素平移迁移（`label_store.shift_shapes`），越出新范围标注保留并计数提示；当前场景即时重开虚拟网格 + 重建标注图层
  - 迁移数学：shift_x=(old_x0-new_x0)/res，shift_y=(new_y1-old_y1)/res（res=meters_per_pixel(zoom) 纬度 0）
- `controller.scene_map_rect(scene)`：场景→画布矩形（zoom_to_scene 重构共用）
- main_dock 标注工具行新增「调场景」按钮
- 无头实测 scripts/diag_scene_edit.py 四用例全过（西/北扩迁移数值精确、收缩保留、过小拒绝、文件场景 clip）；77 单测全过
- 注意：shape points 格式为 [[x,y],...] 对列表（非 {"x","y"} 字典）

## 四轮：改类别交互（v0.1.3）

修复「预测类别改起来麻烦」：obb_edit_tool 新增——
- **右键菜单改类别**（主入口）：选中后右键或直接右键点击目标 → QMenu 全类别列表（颜色图标 + 快捷键标注 + 当前类勾选）→ 点击即改；菜单含删除项
- 数字键改类别修复：keyPressEvent 处理后 e.accept()（不再被 QGIS 抢键）
- 选中/改类时状态栏提示操作方式（「右键改类别 / 数字键快速改 / Del 删除」）
- 批量改：QGIS 属性表选中多要素改 label 字段（原生能力）
- 实测：Small Car→Van→Cargo Truck 连改 + JSON 落盘 ✓；68 单测全过

## 三轮：在线 XYZ 影像直接标注（v0.1.2）摘要

**无需本地 tiff：QuickMapServices 在线图层直接画场景→标注→导出 GeoTIFF**，实测全通过：

- core/xyz_source.py：XyzRaster 虚拟影像（场景矩形+zoom=像素网格，Web Mercator 全球像素坐标系；瓦片磁盘缓存 profile/tiles_cache；duck-type RasterRef → 推理/导出管线零改动）；choose_zoom 按目标分辨率自动选级
- project.py：SceneDef 扩展 kind=xyz/map_bbox/zoom/source；ImageEntry kind=xyz（"xyz://图层名" 工作集）；标注按场景存 labels/<scene>.json
- controller：detect_xyz_layer（解析 type=xyz 图层 source 的 url 模板/zmin/zmax）、attach_xyz_workset、load_scene（构造 XyzRaster+重建标注层）、raster_for_scene/scene_pixel_view
- main_dock：场景列表双击切换；xyz 推理逐场景独立 QgsTask（scene_done 信号带 raster 引用防网格错位）；导出 xyz jobs
- 实测：Google Satellite 600m×600m 场景 → z19（0.248 m/px）→ p2gsd 703 检出（score 0.91）→ 导出 3001×3001 GeoTIFF @0.2m/px EPSG:3857 + 635 DOTA 行
- 68 单测全过（新增 test_xyz_source.py 9 个）

## 二轮：pt 权重支持（v0.1.1）摘要

**qgis_yolo_annotator 插件 v0.1.1：新增 .pt 权重直接支持**，端到端验收通过：

- ✅ core/pt_converter.py：外部 AI 环境子进程转换桥（纯逻辑无 qgis 依赖）
  - pt 元数据读取（task/类别/imgsz，3-4 秒）
  - pt→onnx 导出（copy-pt-to-cache 策略：产物落缓存、不污染源目录、规避跨盘 rename）
  - **子进程必须剥离 PYTHONHOME/PYTHONPATH**（python-qgis.bat 的环境变量会劫持外部 Python）
- ✅ tasks/pt_convert_task.py：后台 QgsTask（进度/成功/失败信号）
- ✅ gui/model_dialog.py：文件选择支持 *.onnx+*.pt；**AI 环境界面可配置**（路径输入+浏览+检测+自动探测，QSettings `qgis_yolo_annotator/ai_env_python` 持久化，配置优先于探测）；类别表改 QPlainTextEdit（37+ 类多行编辑）
- ✅ 实测 p2gsd.pt（用户自己的 OBB 模型：37 类 fair1m、imgsz=1024）：转换 9 秒（缓存后秒回）→ 登记 → 场景推理 334 检出（141 Small Car/131 Van，角度正常）
- ✅ 59 单测全过；缓存目录 profile/qgis_yolo_annotator/models_cache/

## 一轮交付（v0.1.0）摘要

**qgis_yolo_annotator 插件 v0.1.0 全功能交付**，端到端验收通过：

- ✅ 插件骨架：QGIS 4.2.1 加载正常（菜单/工具栏/三页签 Dock）
- ✅ core 层（纯逻辑，59 pytest 全过）：raster_io / inference / scene_infer / model_registry / label_store / project / geometry
- ✅ export 层：converters（DOTA/VOC/YOLO-OBB/YOLO-det）+ chip_export（切片+分辨率重采样+GeoTIFF 子 geotransform）
- ✅ GUI：Controller / 标注图层 / 场景图层 / OBB 编辑工具 / 场景绘制工具 / 主 Dock / 模型・导出・类别对话框
- ✅ tasks：SceneInferTask / ExportTask（QgsTask 后台，GDAL 线程内重开句柄）
- ✅ 端到端实测：真实航拍（Google Satellite 瓦片拼接 EPSG:3857 @0.3m/px）→ 场景 → 0.15 m/px 滑窗推理 → 22 个 small vehicle OBB（角度~30°正确）→ 图层渲染 → JSON 落盘 → DOTA 导出（25 chips @0.15m/px，格式正确）
- ✅ 部署脚本 scripts/deploy.bat、package.bat；文档 README/ARCHITECTURE/ENVIRONMENT

## 关键修复记录（本次会话踩坑）

1. QGIS 4 API：`Qt.DockWidgetArea.RightDockWidgetArea`、无 `addPluginMenuButton`（用 `pluginToolBar()`）
2. PyQt6 不接受 tuple→QgsPointXY 隐式转换（PyQt5 行为），必须显式构造
3. GDAL Dataset 跨线程不安全：QgsTask.run 开头必须 `raster.close()` 让读取线程惰性重开
4. **ultralytics 8.4 OBB onnx 输出布局 `[xywh, cls(nc), angle]`（角度在最后）**，旧版是 `[xywh, angle, cls]`；inference.py 用「sigmoid 类别恒非负 vs 角度可负」自适应判别
5. OBB NMS 自实现（cv2.dnn.NMSBoxesRotated 在 OpenCV 4/5 行为不一致）+ bbox 预筛 + topk（2000 候选 0.02s）
6. 静态输入模型 letterbox 用 `static_imgsz`（导出固定 640 时传大会报维度错误）
7. **QGIS 子进程调用外部 Python 必须剥离 PYTHONHOME/PYTHONPATH**（python-qgis.bat 的变量被继承后 ai_env 被劫持为 QGIS Python，import ultralytics 失败）；`Path.replace` 不能跨盘移动（pt 导出用 copy-to-cache + shutil.move）
8. XYZ 图层 source() 格式：`type=xyz&zmin=0&zmax=20&url=<URL-encoded>`；URL 模板 {x}{y}{z} 顺序不固定（Google 是 x&y&z）。3857 网格分辨率恒定（z19=0.2986），地面分辨率随纬度 cos 收缩（显示用）

## 测试资产（.test_data/，未 gitignore 提交价值）

- `test_scene.tif`：合成 2000×2000 UTM @0.5m/px（无 DOTA 检出，测管线用）
- `real_aerial.tif`：真实航拍 1280×1280 EPSG:3857 @0.3m/px（22 车辆检出）
- `yolo11n-obb.onnx`：DOTA 预训练导出模型
- `proj_real/`：验收工程（15 类别 + 1 场景 + 22 标注）
- `export_dota_015/`：导出产物样例
- scripts/diag_*.py、fetch_real_aerial.py：诊断/采样工具

## 未尽事项 / 下一步

1. OBB 手工绘制/编辑工具已实现但未做鼠标交互实测（MCP 无法模拟完整拖拽；建议用户在 QGIS 里实测两点式绘制+手柄编辑）
2. DOTA txt 导入（label_store.import_dota）未接 UI
3. 编辑操作未接 QGIS undo 栈（dataProvider 直写）
4. seg 任务推理可用未实测
5. 若正式启用建议 git init + 提交基线

## 环境

- QGIS 4.2.1 运行中（MCP 连接正常），插件已部署至 profile python/plugins/
- 详见 ENVIRONMENT.md
