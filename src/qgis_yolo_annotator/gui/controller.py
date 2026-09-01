"""控制器：衔接工程数据、QGIS 图层与标注工具的中央协调器。"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from qgis.core import (
    QgsApplication,
    QgsCoordinateTransform,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QObject, QSettings, pyqtSignal

from ..core.inference import get_session
from ..core.label_store import count_shapes_outside, shift_shapes
from ..core.model_registry import ModelConfig, ModelRegistry
from ..core.project import (
    SCENE_STATUS_ANNOTATED,
    SCENE_STATUS_UNANNOTATED,
    AnnotationProject,
    SceneDef,
)
from ..core.raster_io import RasterRef
from ..core.xyz_source import (
    XyzRaster,
    XyzSourceConfig,
    choose_zoom,
    meters_per_pixel,
)
from .annotation_layer import (
    LAYER_NAME as ANN_LAYER_NAME,
    apply_class_renderer,
    create_annotation_layer,
    feature_to_shape,
    shape_to_feature,
)
from .scene_layer import (
    LAYER_NAME as SCENE_LAYER_NAME,
    create_scene_layer,
    rebuild_scene_features,
)


def registry_store_path() -> Path:
    """模型注册表持久化路径（QGIS profile 下，跨工程共享）。"""
    base = Path(QgsApplication.qgisSettingsDirPath())
    return base / "qgis_yolo_annotator" / "models.json"


def models_cache_dir_hosted() -> Path:
    """pt 转换产物（onnx）缓存目录（QGIS profile 下）。"""
    base = Path(QgsApplication.qgisSettingsDirPath())
    path = base / "qgis_yolo_annotator" / "models_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tiles_cache_dir() -> Path:
    """在线瓦片下载缓存目录（QGIS profile 下，跨会话复用）。"""
    base = Path(QgsApplication.qgisSettingsDirPath())
    path = base / "qgis_yolo_annotator" / "tiles_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


# 会话恢复（QSettings 键）
_SETTING_LAST_PROJECT = "qgis_yolo_annotator/last_project"
_SETTING_LAST_IMAGE = "qgis_yolo_annotator/last_image"
_SETTING_LAST_SCENE = "qgis_yolo_annotator/last_scene"

# 场景最小边长（像素）：小于此值视为误画/误拖，拒绝创建或调整
_MIN_SCENE_PX = 2.0


class Controller(QObject):
    """插件状态中枢：当前工程 / 当前影像 / 标注与场景图层 / 模型注册表。

    Signals:
        image_loaded(str): 当前影像切换。
        project_changed(): 工程打开/创建/影像场景变动。
        labels_changed(): 标注图层内容变化（供保存提示）。
        status_message(str): 状态栏消息。
    """

    image_loaded = pyqtSignal(str)
    project_changed = pyqtSignal()
    labels_changed = pyqtSignal()
    status_message = pyqtSignal(str)

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.project: AnnotationProject | None = None
        self.registry = ModelRegistry(registry_store_path())
        self.raster: RasterRef | XyzRaster | None = None
        self.raster_layer: QgsRasterLayer | None = None
        self.ann_layer: QgsVectorLayer | None = None
        self.scene_layer: QgsVectorLayer | None = None
        self.current_image: str | None = None
        self.current_scene: SceneDef | None = None  # xyz 激活场景
        self.target_res_m = 0.2  # 画场景时的 zoom 选择基准（Dock 推理参数联动）

    # ------------------------------------------------------------------ 工程

    def _reset_context(self) -> None:
        """清空当前影像/场景上下文并移除画布上的旧图层（切换工程前调用）。"""
        self._teardown_layers()
        self.current_image = None
        self.current_scene = None

    def create_project(self, root: str, name: str) -> None:
        """新建并打开工程。"""
        self._reset_context()  # 旧工程的场景框/标注框不留在画布上
        self.project = AnnotationProject.create(root, name)
        QSettings().setValue(_SETTING_LAST_PROJECT, str(Path(root).resolve()))
        self.project_changed.emit()
        self.status_message.emit(f"已创建工程: {name}")

    def open_project(self, root: str) -> None:
        """打开工程。"""
        self._reset_context()
        self.project = AnnotationProject.open(root)
        QSettings().setValue(_SETTING_LAST_PROJECT, str(Path(root).resolve()))
        self.project_changed.emit()
        self.status_message.emit(f"已打开工程: {self.project.name}")

    def _remember_location(self, scene_name: str = "") -> None:
        """记录当前影像/场景，供插件重载或重启后自动恢复。"""
        settings = QSettings()
        settings.setValue(_SETTING_LAST_IMAGE, self.current_image or "")
        settings.setValue(_SETTING_LAST_SCENE, scene_name)

    def restore_last_session(self) -> bool:
        """恢复上次工程与影像/场景（插件重载、QGIS 重启后调用）。

        Returns:
            是否成功恢复了工程。
        """
        settings = QSettings()
        project_root = settings.value(_SETTING_LAST_PROJECT, "", type=str)
        last_image = settings.value(_SETTING_LAST_IMAGE, "", type=str)
        last_scene = settings.value(_SETTING_LAST_SCENE, "", type=str)
        # 注意：先读位置记录再开工程——open_project 等操作会重写位置记录
        if not project_root or not Path(project_root, "project.json").is_file():
            return False
        try:
            self.open_project(project_root)
        except (FileNotFoundError, ValueError, OSError):
            return False
        if not last_image:
            return True
        try:
            if last_image.startswith("xyz://"):
                entry = self.project.find_image(last_image)
                if entry is None:
                    return True
                scene = next(
                    (s for s in entry.scenes if s.name == last_scene), None
                )
                if scene is not None and scene.source:
                    self.load_scene(scene)  # 内含 attach + 画布定位
                return True
            if Path(last_image).is_file():
                self.load_image(last_image)
                if last_scene:
                    for scene in self.scenes_of_current_image():
                        if scene.name == last_scene:
                            self.zoom_to_scene(scene)
                            break
        except (RuntimeError, ValueError) as exc:
            self.status_message.emit(f"恢复上次位置失败: {exc}")
        return True

    def save_project(self) -> None:
        """保存工程与当前影像标注。"""
        if self.project is None:
            return
        self.save_current_labels()
        self.project.save()
        self.status_message.emit("工程已保存")

    # ------------------------------------------------------------------ 影像

    def load_image(self, path: str) -> None:
        """加载影像到画布并重建标注/场景图层。

        Args:
            path: 影像文件路径（须已在工程中，否则自动添加）。
        """
        if self.project is None:
            raise RuntimeError("尚未打开工程")
        self.project.add_image(path)  # 幂等
        self.project.save()
        self._teardown_layers()

        raster_layer = QgsRasterLayer(path, Path(path).stem)
        if not raster_layer.isValid():
            raise RuntimeError(f"影像加载失败: {path}")
        QgsProject.instance().addMapLayer(raster_layer)
        self.raster_layer = raster_layer
        self.iface.setActiveLayer(raster_layer)

        self.raster = RasterRef.open(path)
        self.current_image = str(Path(path).resolve())

        crs_wkt = self.raster.crs_wkt
        self.ann_layer = create_annotation_layer(crs_wkt)
        self.scene_layer = create_scene_layer(crs_wkt)
        apply_class_renderer(self.ann_layer, self.project.classes)
        QgsProject.instance().addMapLayer(self.ann_layer)
        QgsProject.instance().addMapLayer(self.scene_layer)
        self._rebuild_annotation_features()
        self._rebuild_scene_features()

        self.iface.mapCanvas().setExtent(raster_layer.extent())
        self.iface.mapCanvas().refresh()
        self._remember_location()  # 文件影像无活动场景
        self.image_loaded.emit(self.current_image)
        self.status_message.emit(
            f"已加载 {Path(path).name}"
            f"（{self.raster.width}×{self.raster.height}"
            f"{f'，{self.raster.resolution_m_per_px():.4g} m/px' if self.raster.resolution_m_per_px() else ''}）"
        )

    def _teardown_layers(self, remove_raster_layer: bool = True) -> None:
        """移除旧图层并释放栅格句柄（attach xyz 工作集时可保留文件栅格图层）。"""
        qgs = QgsProject.instance()
        self._purge_stale_plugin_layers()
        for layer in (self.ann_layer, self.scene_layer):
            if layer is not None:
                if layer.isEditable():
                    layer.rollBack()  # JSON 已随操作落盘，缓冲直接弃置避免移除时弹窗
                qgs.removeMapLayer(layer.id())
        if remove_raster_layer and self.raster_layer is not None:
            qgs.removeMapLayer(self.raster_layer.id())
        if self.raster is not None:
            self.raster.close()
        self.ann_layer = None
        self.scene_layer = None
        if remove_raster_layer:
            self.raster_layer = None
        self.raster = None

    def _purge_stale_plugin_layers(self) -> None:
        """清理插件重载后遗留的孤儿标注/场景图层（按图层名匹配，跳过本实例在用的）。"""
        qgs = QgsProject.instance()
        keep_ids = {
            layer.id()
            for layer in (self.ann_layer, self.scene_layer)
            if layer is not None
        }
        for layer in list(qgs.mapLayers().values()):
            if (
                layer.name() in (ANN_LAYER_NAME, SCENE_LAYER_NAME)
                and layer.id() not in keep_ids
            ):
                qgs.removeMapLayer(layer.id())

    def _rebuild_annotation_features(self) -> None:
        """从工程标注 JSON 重建标注图层要素（xyz 场景按场景名取 JSON）。

        重建 = 更换编辑上下文：先回滚编辑缓冲（JSON 已随每次操作落盘），
        再全量替换要素并清空撤销栈（跨场景撤销无意义）。
        """
        assert self.project and self.raster and self.ann_layer and self.current_image
        layer = self.ann_layer
        if layer.isEditable():
            layer.rollBack()
        shapes = self.project.load_image_labels(
            self.current_image, scene_name=self.scene_label_scope()
        )
        provider = layer.dataProvider()
        provider.truncate()
        features = [shape_to_feature(s, self.raster) for s in shapes]
        provider.addFeatures(features)
        layer.updateFields()
        if not layer.isEditable():
            layer.startEditing()  # 常驻编辑会话：所有操作进 QGIS undo 栈
        layer.undoStack().clear()
        layer.triggerRepaint()

    def _rebuild_scene_features(self) -> None:
        """从工程场景重建场景图层要素（raster 允许 None：xyz 场景不依赖）。"""
        assert self.project and self.scene_layer and self.current_image
        entry = self.project.find_image(self.current_image)
        if entry is not None:
            rebuild_scene_features(self.scene_layer, entry, self.raster)

    # ------------------------------------------------------------------ 标注同步

    def save_current_labels(self) -> bool:
        """标注图层 → 工程 JSON（像素坐标；xyz 场景按场景名存）。

        Returns:
            是否执行了保存（有当前影像时 True）。
        """
        if (
            self.project is None
            or self.ann_layer is None
            or self.raster is None
            or self.current_image is None
        ):
            return False
        shapes = []
        for feature in self.ann_layer.getFeatures():
            shape = feature_to_shape(feature, self.raster)
            if shape is not None:
                shapes.append(shape)
        self.project.save_image_labels(
            self.current_image,
            shapes,
            self.raster.width,
            self.raster.height,
            scene_name=self.scene_label_scope(),
        )
        # 手工绘制/编辑也推进状态：未标注场景一旦有标注即视为已标注
        # （只升级不打折：已标注/已审核不被自动改回）
        scope = self.scene_label_scope()
        if (
            shapes
            and scope is not None
            and self.current_scene is not None
            and self.current_scene.name == scope
            and self.current_scene.status == SCENE_STATUS_UNANNOTATED
        ):
            self.set_scene_status(scope, SCENE_STATUS_ANNOTATED)
        return True

    # ------------------------------------------------------------------ 场景

    def detect_xyz_layer(self) -> tuple[XyzSourceConfig, str] | None:
        """在当前工程图层中查找 XYZ 瓦片图层（如 QuickMapServices 的 Google Satellite）。

        Returns:
            (XyzSourceConfig, 图层标题)；找不到返回 None。
        """
        from qgis.core import Qgis

        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() != Qgis.LayerType.Raster:
                continue
            source = layer.source()
            # 参数顺序不保证 type=xyz 在最前（如 crs=...&type=xyz&url=...）
            if "type=xyz" not in source:
                continue
            params = {}
            for segment in source.split("&"):
                if "=" in segment:
                    key, value = segment.split("=", 1)
                    params[key] = value
            url = unquote(params.get("url", ""))
            if "{z}" not in url or "{x}" not in url or "{y}" not in url:
                continue
            try:
                config = XyzSourceConfig(
                    url_template=url,
                    title=layer.name(),
                    min_zoom=int(params.get("zmin", 0)),
                    max_zoom=int(params.get("zmax", 20)),
                )
            except ValueError:
                continue
            return config, layer.name()
        return None

    def attach_xyz_workset(self, source: XyzSourceConfig, layer_title: str):
        """确保 xyz 工作集影像条目存在并设为当前影像（不加本地栅格图层）。"""
        if self.project is None:
            raise RuntimeError("尚未打开工程")
        workset_id = f"xyz://{layer_title}"
        entry = self.project.add_image(workset_id, kind="xyz")
        if self.current_image != workset_id:
            self._teardown_layers(remove_raster_layer=False)
            self.current_image = workset_id
            self.current_scene = None
            self.raster = None
            crs_wkt = self._web_mercator_wkt()
            self.ann_layer = create_annotation_layer(crs_wkt)
            self.scene_layer = create_scene_layer(crs_wkt)
            apply_class_renderer(self.ann_layer, self.project.classes)
            self._purge_stale_plugin_layers()
            QgsProject.instance().addMapLayer(self.ann_layer)
            QgsProject.instance().addMapLayer(self.scene_layer)
            self._rebuild_scene_features()
            self.iface.mapCanvas().refresh()
            self.image_loaded.emit(workset_id)
        return entry

    @staticmethod
    def _web_mercator_wkt() -> str:
        from ..core.xyz_source import EPSG3857_WKT

        return EPSG3857_WKT

    def _rect_to_pixel_bbox(self, rect: QgsRectangle) -> list[float] | None:
        """画布矩形 → 当前影像像素 bbox（clip 到影像范围）。

        Returns:
            [col0, row0, col1, row1]；与影像无交集时 None。
        """
        col0, row0 = self.raster.map_to_pixel(rect.xMinimum(), rect.yMaximum())
        col1, row1 = self.raster.map_to_pixel(rect.xMaximum(), rect.yMinimum())
        bbox = [
            max(0.0, min(col0, col1)),
            max(0.0, min(row0, row1)),
            min(float(self.raster.width), max(col0, col1)),
            min(float(self.raster.height), max(row0, row1)),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        return bbox

    def add_scene_from_map(
        self, rect: QgsRectangle, target_res_m: float | None = None
    ) -> SceneDef | None:
        """画布矩形（地图坐标）→ 新场景。

        文件影像：矩形换算为像素 bbox（clip 到影像范围）；
        无文件影像（或当前为 xyz 工作集）且画布存在 XYZ 图层：
        创建在线瓦片场景（EPSG:3857 矩形 + 按目标分辨率自动选 zoom）。

        Args:
            rect: 地图坐标矩形。
            target_res_m: zoom 选择基准分辨率（缺省用 self.target_res_m）。

        Returns:
            新建 SceneDef；无可用数据源或范围过小时 None。
        """
        if self.project is None:
            return None
        rect = QgsRectangle(rect)
        rect.normalize()
        if rect.isEmpty():
            return None
        res = target_res_m if target_res_m is not None else self.target_res_m

        xyz_mode = self.current_image is None or self.current_image.startswith("xyz://")
        if xyz_mode:
            detected = self.detect_xyz_layer()
            if detected is not None:
                return self._add_xyz_scene(rect, res, *detected)
            # 不落入文件分支：xyz 工作集下 raster 是虚拟网格，
            # 拿它当影像换算会把场外矩形裁成 0 像素（误报「范围过小」）
            self.status_message.emit(
                "画布中未识别到在线 XYZ 图层，请先在 QuickMapServices 打开"
            )
            return None
        # ---- 文件影像分支
        if self.raster is None or self.current_image is None:
            self.status_message.emit("请先加载影像")
            return None
        bbox = self._rect_to_pixel_bbox(rect)
        if bbox is None:
            self.status_message.emit("场景范围与影像无交集，请画在影像范围内")
            return None
        if bbox[2] - bbox[0] < _MIN_SCENE_PX or bbox[3] - bbox[1] < _MIN_SCENE_PX:
            self.status_message.emit("场景范围过小，已忽略")
            return None
        scene = self.project.add_scene(self.current_image, bbox)
        self._rebuild_scene_features()
        self.project.save()
        self.project_changed.emit()
        self.status_message.emit(f"已添加场景 {scene.name}（{scene.width:.0f}×{scene.height:.0f} px）")
        return scene

    def _add_xyz_scene(
        self,
        rect: QgsRectangle,
        target_res_m: float,
        source: XyzSourceConfig,
        layer_title: str,
    ) -> SceneDef | None:
        """创建在线瓦片场景（画布 CRS → EPSG:3857，自动选 zoom）。"""
        import math

        entry = self.attach_xyz_workset(source, layer_title)
        rect_3857 = self._to_web_mercator(rect)
        if rect_3857 is None or rect_3857.isEmpty():
            self.status_message.emit("场景范围转换失败，已忽略")
            return None
        y_mid = (rect_3857.yMinimum() + rect_3857.yMaximum()) / 2
        latitude = math.degrees(math.atan(math.sinh(y_mid / 6378137.0)))
        zoom = choose_zoom(target_res_m, latitude, source.min_zoom, source.max_zoom)
        scene = self.project.add_scene(
            entry.path,
            [0, 0, 0, 0],
            kind="xyz",
            map_bbox=[
                rect_3857.xMinimum(),
                rect_3857.yMinimum(),
                rect_3857.xMaximum(),
                rect_3857.yMaximum(),
            ],
            zoom=zoom,
            source=source.to_dict(),
        )
        if self.current_scene is None:
            self.load_scene(scene)
        else:
            self._rebuild_scene_features()
        self.project.save()
        self.project_changed.emit()
        self.status_message.emit(
            f"已添加在线场景 {scene.name}"
            f"（z{zoom}，≈{meters_per_pixel(zoom, latitude):.3f} m/px，来自 {layer_title}）"
        )
        return scene

    def _to_web_mercator(self, rect: QgsRectangle) -> QgsRectangle | None:
        """画布 CRS 矩形 → EPSG:3857。"""
        from qgis.core import QgsCoordinateReferenceSystem

        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        target = QgsCoordinateReferenceSystem("EPSG:3857")
        if canvas_crs == target:
            return QgsRectangle(rect)
        transform = QgsCoordinateTransform(canvas_crs, target, QgsProject.instance())
        try:
            return transform.transformBoundingBox(rect)
        except Exception:  # noqa: BLE001  变换失败按不可用处理
            return None

    def load_scene(self, scene: SceneDef) -> None:
        """加载 xyz 场景：构造虚拟影像网格（XyzRaster）并重建标注图层，画布定位到场景。"""
        if self.project is None or scene.kind != "xyz" or not scene.source:
            return
        entry = self._entry_of_scene(scene)
        if entry is None:
            return
        source = XyzSourceConfig.from_dict(scene.source)
        self.attach_xyz_workset(source, entry.path.removeprefix("xyz://"))
        if self.raster is not None:
            self.raster.close()
        raster = XyzRaster(source, scene.map_bbox, scene.zoom, tiles_cache_dir())
        self.raster = raster
        self.current_scene = scene
        self._remember_location(scene.name)
        self.raster_layer = None  # xyz 模式无本地栅格图层（在线图层由用户自开）
        self._rebuild_annotation_features()
        self._rebuild_scene_features()
        self.zoom_to_scene(scene)
        self.image_loaded.emit(self.current_image)
        self.status_message.emit(
            f"已加载场景 {scene.name}"
            f"（{raster.width}×{raster.height} px @z{scene.zoom}，"
            f"≈{raster.resolution_m_per_px():.3f} m/px）"
        )

    def scene_map_rect(self, scene: SceneDef) -> QgsRectangle | None:
        """场景范围的画布坐标矩形（xyz 用 map_bbox，文件场景像素→地图换算）。

        Returns:
            画布 CRS 矩形；场景无可用范围或坐标变换失败时 None。
        """
        from qgis.core import QgsRectangle

        if scene.kind == "xyz" and scene.map_bbox:
            return self._from_web_mercator(QgsRectangle(*scene.map_bbox))
        if self.raster is None:
            return None
        x0, y0 = self.raster.pixel_to_map(scene.bbox[0], scene.bbox[1])
        x1, y1 = self.raster.pixel_to_map(scene.bbox[2], scene.bbox[3])
        return QgsRectangle(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def zoom_to_scene(self, scene: SceneDef) -> None:
        """画布定位到场景范围。"""
        rect = self.scene_map_rect(scene)
        if rect is None or rect.isEmpty():
            return
        rect = rect.buffered(max(rect.width(), rect.height()) * 0.08 + 1.0)
        canvas = self.iface.mapCanvas()
        canvas.setExtent(rect)
        canvas.refresh()

    def _from_web_mercator(self, rect: QgsRectangle) -> QgsRectangle | None:
        """EPSG:3857 矩形 → 画布 CRS。"""
        from qgis.core import QgsCoordinateReferenceSystem

        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        source = QgsCoordinateReferenceSystem("EPSG:3857")
        if canvas_crs == source:
            return QgsRectangle(rect)
        transform = QgsCoordinateTransform(source, canvas_crs, QgsProject.instance())
        try:
            return transform.transformBoundingBox(rect)
        except Exception:  # noqa: BLE001  变换失败按不可用处理
            return None

    def _entry_of_scene(self, scene: SceneDef):
        """场景所属的影像条目（按对象或同名匹配）。"""
        if self.project is None:
            return None
        for entry in self.project.images:
            if any(s is scene for s in entry.scenes) or any(
                s.name == scene.name for s in entry.scenes
            ):
                return entry
        return None

    def scene_pixel_view(self, scene: SceneDef) -> SceneDef:
        """推理/导出用场景视图：xyz 场景的像素 bbox = 虚拟影像全网格。"""
        if scene.kind != "xyz" or self.raster is None:
            return scene
        return SceneDef(
            name=scene.name,
            bbox=[0.0, 0.0, float(self.raster.width), float(self.raster.height)],
            status=scene.status,
        )

    def raster_for_scene(self, scene: SceneDef):
        """获取场景的栅格对象：xyz 场景按需构造（或复用已加载网格）。"""
        if scene.kind != "xyz":
            return self.raster
        if (
            self.raster is not None
            and self.current_scene is not None
            and self.current_scene.name == scene.name
        ):
            return self.raster
        self.load_scene(scene)
        return self.raster

    def scene_label_scope(self) -> str | None:
        """当前标注的存储作用域（xyz 场景名；文件影像为 None）。"""
        if self.current_image and self.current_image.startswith("xyz://"):
            return self.current_scene.name if self.current_scene else None
        return None

    def ensure_scene_context(self) -> bool:
        """确保存在可绘制的标注上下文（栅格网格已加载）。

        xyz 场景未加载时自动加载上次（或首个）场景——绘制/标注均依赖
        虚拟网格的像素↔地图换算，仅挂工作集不足以画框。

        Returns:
            是否已就绪（原本就绪或自动加载成功）。
        """
        if self.raster is not None:
            return True
        if self.project is None:
            return False
        entry = (
            self.project.find_image(self.current_image)
            if self.current_image
            else None
        )
        if entry is None or entry.kind != "xyz":
            entry = next((e for e in self.project.images if e.kind == "xyz"), None)
        if entry is None or not entry.scenes:
            return False
        settings = QSettings()
        last_scene = settings.value(_SETTING_LAST_SCENE, "", type=str)
        scene = next(
            (s for s in entry.scenes if s.name == last_scene), None
        ) or entry.scenes[0]
        self.load_scene(scene)
        self.status_message.emit(f"已自动加载场景 {scene.name}（未推理也可直接画框）")
        return True

    def set_scene_status(self, scene_name: str, status: str) -> None:
        """更新当前影像某场景状态并刷新。"""
        if self.project is None or self.current_image is None:
            return
        self.project.set_scene_status(self.current_image, scene_name, status)
        self._rebuild_scene_features()
        self.project.save()
        self.project_changed.emit()

    def remove_scene(self, scene_name: str, delete_labels: bool = False) -> int:
        """删除当前影像的某场景。

        Args:
            scene_name: 场景名。
            delete_labels: xyz 场景为 True 时同步删除其标注 JSON
                （文件影像场景的 JSON 为整影像共享，不做删除）。

        Returns:
            删除的标注 JSON 文件数（0 或 1）。
        """
        if self.project is None or self.current_image is None:
            return 0
        entry = self.project.find_image(self.current_image)
        if entry is None:
            return 0
        target = next((s for s in entry.scenes if s.name == scene_name), None)
        if target is None:
            return 0
        entry.scenes = [s for s in entry.scenes if s.name != scene_name]
        removed = 0
        clear_visual = False
        if target.kind == "xyz":
            # 只要删的是当前加载的场景就先解除引用，与是否连删 JSON 无关：
            # 引用悬空会让后续保存把旧要素写回失效作用域
            if self.current_scene is not None and self.current_scene.name == scene_name:
                self.current_scene = None
                clear_visual = True
                if self.ann_layer is not None and self.ann_layer.isEditable():
                    self.ann_layer.rollBack()
                if self.raster is not None:
                    self.raster.close()
                self.raster = None
            if delete_labels:
                label_path = self.project.label_path(self.current_image, scene_name=scene_name)
                if label_path.is_file():
                    label_path.unlink()
                    removed = 1
        self.ensure_annotation_editable()  # rollBack 会退出编辑会话，恢复之
        if clear_visual and self.ann_layer is not None:
            # 标注图层只承载当前场景的要素；rollBack 只丢弃未提交编辑，
            # 已入库要素须显式清空，画布才与「场景已删」一致
            layer = self.ann_layer
            layer.dataProvider().truncate()
            layer.updateFields()
            layer.undoStack().clear()
            layer.triggerRepaint()
        self._rebuild_scene_features()
        self.project.save()
        self.project_changed.emit()
        return removed

    def scenes_of_current_image(self) -> list[SceneDef]:
        """当前影像的场景列表。"""
        if self.project is None or self.current_image is None:
            return []
        entry = self.project.find_image(self.current_image)
        return list(entry.scenes) if entry else []

    def update_scene_extent(self, scene_name: str, rect: QgsRectangle) -> bool:
        """调整当前影像某场景的范围（画布坐标矩形）。

        文件场景：换算为像素 bbox 并 clip 到影像范围，标注为影像级坐标不受影响。
        xyz 场景：更新 map_bbox（zoom 不变）；网格原点变更时已有标注按像素
        平移迁移保持对齐，越出新范围的标注保留（仅计数提示）。

        Args:
            scene_name: 场景名。
            rect: 新范围（画布 CRS）。

        Returns:
            是否成功调整（范围过小/坐标变换失败时 False）。
        """
        if self.project is None or self.current_image is None:
            return False
        entry = self.project.find_image(self.current_image)
        if entry is None:
            return False
        target = next((s for s in entry.scenes if s.name == scene_name), None)
        if target is None:
            return False
        rect = QgsRectangle(rect)
        rect.normalize()
        if rect.isEmpty():
            return False

        if target.kind == "xyz":
            rect_m = self._to_web_mercator(rect)
            if rect_m is None:
                self.status_message.emit("坐标变换失败，无法调整场景")
                return False
            # 网格分辨率：纬度 0 的 meters_per_pixel 即 3857 单位/像素
            grid_res = meters_per_pixel(target.zoom)
            new_w = rect_m.width() / grid_res
            new_h = rect_m.height() / grid_res
            if new_w < _MIN_SCENE_PX or new_h < _MIN_SCENE_PX:
                self.status_message.emit("场景范围过小，已忽略")
                return False
            is_current = (
                self.current_scene is not None
                and self.current_scene.name == scene_name
            )
            if is_current:
                self.save_current_labels()  # 旧网格标注先落盘，防迁移后被覆写
            shapes = self.project.load_image_labels(
                self.current_image, scene_name=scene_name
            )
            if shapes and target.map_bbox:
                old_x0, _, _, old_y1 = target.map_bbox
                shift_x = (old_x0 - rect_m.xMinimum()) / grid_res
                shift_y = (rect_m.yMaximum() - old_y1) / grid_res
                if abs(shift_x) > 1e-9 or abs(shift_y) > 1e-9:
                    shapes = shift_shapes(shapes, shift_x, shift_y)
            outside = count_shapes_outside(shapes, new_w, new_h) if shapes else 0
            target.map_bbox = [
                rect_m.xMinimum(),
                rect_m.yMinimum(),
                rect_m.xMaximum(),
                rect_m.yMaximum(),
            ]
            target.bbox = [0.0, 0.0, new_w, new_h]
            if shapes:
                self.project.save_image_labels(
                    self.current_image,
                    shapes,
                    int(new_w),
                    int(new_h),
                    scene_name=scene_name,
                )
            if is_current:
                # 网格已变：重开虚拟影像并按迁移后的 JSON 重建标注图层
                if self.raster is not None:
                    self.raster.close()
                self.raster = None
                self.current_scene = None  # 强制 load_scene 重走加载路径
                self.load_scene(target)
            self._rebuild_scene_features()
            self.project.save()
            self.project_changed.emit()
            msg = f"场景 {scene_name} 已调整为 {new_w:.0f}×{new_h:.0f} px"
            if shapes:
                msg += f"；{len(shapes)} 个标注已随网格迁移"
            if outside:
                msg += f"，{outside} 个越出新范围（已保留）"
            self.status_message.emit(msg)
            return True

        # ---- 文件影像分支
        if self.raster is None or self.current_image is None:
            self.status_message.emit("请先加载影像")
            return False
        bbox = self._rect_to_pixel_bbox(rect)
        if bbox is None:
            self.status_message.emit("场景范围与影像无交集，请画在影像范围内")
            return False
        if bbox[2] - bbox[0] < _MIN_SCENE_PX or bbox[3] - bbox[1] < _MIN_SCENE_PX:
            self.status_message.emit("场景范围过小，已忽略")
            return False
        target.bbox = bbox
        self._rebuild_scene_features()
        self.project.save()
        self.project_changed.emit()
        self.status_message.emit(
            f"场景 {scene_name} 已调整为 {target.width:.0f}×{target.height:.0f} px"
        )
        return True

    # ------------------------------------------------------------------ 模型

    def get_model_session(self, config: ModelConfig):
        """获取模型推理会话（进程内缓存）。"""
        return get_session(config.file_path, config.task)

    # ------------------------------------------------------------------ 批量编辑

    def batch_replace_label(self, old_name: str, new_name: str, project_wide: bool) -> int:
        """批量替换类别：当前影像（图层+JSON）或整个工程（labels/*.json）。

        当前图层经编辑命令替换（单步可撤销）；其余 JSON 直接改写。

        Args:
            old_name: 原类别名。
            new_name: 目标类别名。
            project_wide: True=整个工程；False=仅当前影像/场景。

        Returns:
            替换的标注数量。

        Raises:
            RuntimeError: 未打开工程。
        """
        if self.project is None:
            raise RuntimeError("尚未打开工程")
        self.save_current_labels()  # 当前图层先落盘，保证 JSON 为最新
        count = 0
        if self.ann_layer is not None and self.raster is not None:
            # 当前图层：编辑命令分组（单步撤销）
            self.ensure_annotation_editable()
            attr_idx = self.ann_layer.fields().indexOf("label")
            matching = [
                f.id()
                for f in self.ann_layer.getFeatures()
                if f.attribute("label") == old_name
            ]
            if matching:
                self.ann_layer.beginEditCommand(f"批量替换 {old_name}→{new_name}")
                for fid in matching:
                    self.ann_layer.changeAttributeValue(fid, attr_idx, new_name)
                self.ann_layer.endEditCommand()
                count += len(matching)
        # JSON 侧：当前场景 JSON 已由图层落盘步骤之外单独处理——
        # 图层替换后再落盘即可；其余场景/影像的 JSON 工程级替换（排除当前）
        if project_wide:
            current_path = None
            if self.current_image is not None:
                current_path = self.project.label_path(
                    self.current_image, scene_name=self.scene_label_scope()
                )
            count += self.project.replace_label(
                old_name, new_name, exclude_path=current_path
            )
        elif self.current_image is not None:
            # 非图层可见的直标文件影像（ann_layer 为空时兜底直接改 JSON）
            label_path = self.project.label_path(
                self.current_image, scene_name=self.scene_label_scope()
            )
            from ..core.label_store import load_label, save_label

            try:
                doc = load_label(label_path)
            except ValueError:
                doc = None
            if doc is not None and self.ann_layer is None:
                changed = 0
                for shape in doc.get("shapes", []):
                    if shape.get("label") == old_name:
                        shape["label"] = new_name
                        changed += 1
                if changed:
                    save_label(label_path, doc)
                    count += changed
        self.save_current_labels()  # 图层替换结果落盘
        if self.ann_layer is not None and self.raster is not None:
            self.project_changed.emit()
        return count

    def rename_class_sync(self, old_name: str, new_name: str) -> int:
        """类别表改名并同步全部标注（当前图层走可撤销编辑，其余改 JSON）。"""
        return self.batch_replace_label(old_name, new_name, project_wide=True)

    # ------------------------------------------------------------------ 撤销/重做

    def ensure_annotation_editable(self) -> None:
        """确保标注图层处于编辑会话（rollBack/场景删除等会退出编辑模式）。"""
        layer = self.ann_layer
        if layer is not None and not layer.isEditable():
            layer.startEditing()

    def undo_annotation(self) -> bool:
        """撤销当前标注图层上一步操作（QGIS undo 栈）。

        Returns:
            是否执行了撤销。
        """
        layer = self.ann_layer
        if layer is None or not layer.undoStack().canUndo():
            return False
        layer.undoStack().undo()
        # 撤销可能删掉/改掉选中目标，选择状态按现状重建
        if self._selected_missing():
            self.on_selection_invalidated()
        self.labels_changed.emit()
        return True

    def redo_annotation(self) -> bool:
        """重做当前标注图层上一步被撤销的操作。

        Returns:
            是否执行了重做。
        """
        layer = self.ann_layer
        if layer is None or not layer.undoStack().canRedo():
            return False
        layer.undoStack().redo()
        if self._selected_missing():
            self.on_selection_invalidated()
        self.labels_changed.emit()
        return True

    def _selected_missing(self) -> bool:
        """当前主选中要素是否已不存在。"""
        if self.ann_layer is None:
            return True
        tool = self.iface.mapCanvas().mapTool()
        fid = getattr(tool, "_selected_fid", None)
        if fid is None:
            return False
        return not self.ann_layer.getFeature(fid).isValid()

    def on_selection_invalidated(self) -> None:
        """撤销/重做导致选中目标失效时回调（由工具注册清理）。"""
        tool = self.iface.mapCanvas().mapTool()
        clear = getattr(tool, "_clear_selection", None)
        if callable(clear):
            clear()
