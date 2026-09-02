# qgis_yolo_annotator — QGIS 遥感影像 YOLO 智能标注插件

[![QGIS 插件仓库](https://img.shields.io/badge/QGIS_Plugin_Repository-YOLO_Annotator-589632)](https://plugins.qgis.org/plugins/qgis_yolo_annotator/)
[![GitHub Release](https://img.shields.io/github/v/release/JayceVane/qgis-yolo-annotator)](https://github.com/JayceVane/qgis-yolo-annotator/releases)
[![License](https://img.shields.io/badge/license-GPL--2.0--or--later-blue)](LICENSE)

面向 QGIS 4.x 的遥感影像智能标注工具：加载 YOLO 系列 ONNX 模型推理生成预标注，
提供 OBB（旋转框）绘制与精修交互，按场景（AOI）管理标注进度，导出
DOTA / VOC / YOLO-OBB 数据集（GeoTIFF 切片携带地理信息，支持指定输出地面分辨率重采样）。

- **官方插件页**：<https://plugins.qgis.org/plugins/qgis_yolo_annotator/>（插件管理器内搜索 "YOLO Annotator" 即可安装）
- **代码仓库**：<https://github.com/JayceVane/qgis-yolo-annotator>

## 功能总览

| 模块 | 能力 |
|------|------|
| 模型管理 | **.pt 权重与 .onnx 双支持**：pt 自动经外部 AI 环境（ultralytics）后台转 onnx 并缓存（复用免重转）；AI 环境解释器可在界面上配置（浏览/检测/自动探测，QSettings 持久化）；onnx metadata 自动读类别/输入尺寸/任务；conf/iou 可调 |
| 场景（AOI） | 画布拖矩形添加场景；三态状态（未标注/已标注/已审核）；状态按颜色渲染；**支持在线 XYZ 图层（QuickMapServices）直接画场景——无需先下载 tiff**，按目标分辨率自动选 zoom，瓦片按需下载缓存 |
| 推理 | 选中场景滑窗推理；**目标地理分辨率可配置（默认 0.2 m/px，单位 m/px 或 °/px）**；跨窗按类别 NMS 合并；后台 QgsTask 不阻塞 UI |
| OBB 编辑 | 两点式绘制（Shift=15° 吸附）；角点手柄（平行四边形约束，支持旋转变形）、边中点手柄（法向伸缩）、内部拖动（平移）；**改类别：右键菜单（全类别列表）或数字快捷键**；Del 删除、Ctrl+C/V 复制粘贴、方向键微调 |
| 标注存储 | X-AnyLabeling JSON（像素坐标），与 t0_dataset_utils / t1_xanylabeling_web / X-AnyLabeling 桌面版全生态互通；地理参考由 GeoTIFF geotransform 双向换算 |
| 导出 | DOTA（像素角点+difficult）/ VOC（HBB 或四点 polygon）/ YOLO-OBB（归一化角点）/ YOLO-det；切片尺寸/重叠可配；**输出分辨率重采样（m/px 或 °/px，度值按影像中心纬度换算）**；GeoTIFF 切片携带子 geotransform；train/val 划分 + classes.txt + data.yaml |
| 工程管理 | project.json + labels/ 目录；影像清单（引用不拷贝）；批量导入；进度统计 |

## 安装

**方式一（推荐）：QGIS 插件管理器在线安装**

菜单 Plugins → 管理并安装插件 → 搜索 "**YOLO Annotator**" → 安装。
（或从[官方插件页](https://plugins.qgis.org/plugins/qgis_yolo_annotator/)下载 zip 后「从 ZIP 安装」）

**方式二：从源码部署（开发/抢先体验）**

```bat
:: 1. 安装依赖到 QGIS Python（仅需一次）
cmd /c "D:\Tools\QGIS\bin\python-qgis.bat -m pip install -r requirements-qgis.txt"

:: 2. 部署插件（拷贝到 QGIS profile 并自动检查依赖）
scripts\deploy.bat
```

重启 QGIS → 插件菜单启用「YOLO Annotator」→ 工具栏按钮打开主面板。

## 快速上手

### 在线影像工作流（无需本地 tiff）

1. QGIS 菜单 Web → QuickMapServices → Google → Satellite 打开在线图层，缩放到目标区域
2. 插件面板**新建工程** → 添加类别
3. 「**画场景**」工具直接在在线影像上拖矩形——每一块即一个场景（按目标分辨率自动选 zoom，瓦片自动下载缓存到 profile）
4. 登记模型（.onnx 或 .pt）→ 选中场景 → **推理选中场景**（或「画 OBB」手工标注）
5. 双击场景列表可切换当前编辑场景
6. **导出数据集**：在线场景自动落成 GeoTIFF（含 EPSG:3857 地理信息，可按目标分辨率重采样如 0.2 m/px）+ DOTA/VOC/YOLO 标签

### 本地文件工作流
1. **新建工程**（工程页 → 选择空目录），添加类别（类别管理）
2. **添加影像**（单个文件或文件夹递归导入，tif/png/jpg）
3. 双击影像加载 → 画布叠加标注图层与场景图层
4. **画场景**（标注页 → 画场景工具，拖矩形圈出关注区）
5. **模型管理**登记模型：直接选 `.onnx`（metadata 自动预填）或选 `.pt`（首次自动转 onnx，需在对话框上方配置 AI 环境 Python，如 `D:/DevKit/anaconda3/envs/ai_env/python.exe`）→ 模型下拉选择
6. 设置目标分辨率（默认 0.2 m/px）→ 场景列表选中 → **推理选中场景**
7. **画 OBB** 工具手工补充/修正；点击选中后拖角点（旋转）、边中点（伸缩）、内部（平移）；数字键改类别
8. 场景状态按钮标记进度（未标注 ○ / 已标注 ◐ / 已审核 ●）
9. **导出数据集**：选格式/切片/分辨率 → 输出 images+labels 目录

## 开发

```bash
# 单元测试（QGIS Python，纯逻辑不依赖 qgis）
cmd /c "D:\Tools\QGIS\bin\python-qgis.bat -m pytest tests -q --basetemp=.pytest_tmp"

# 开发迭代：改 src/ 后
scripts\deploy.bat   # 或直接 cp -r src/qgis_yolo_annotator %APPDATA%/QGIS/QGIS4/profiles/default/python/plugins/
```

详见 [ARCHITECTURE.md](ARCHITECTURE.md)（模块结构）与 [ENVIRONMENT.md](ENVIRONMENT.md)（环境事实）。
