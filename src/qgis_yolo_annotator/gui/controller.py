"""控制器：衔接工程数据、QGIS 图层与标注工具的中央协调器。"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

from qgis.core import (
    QgsApplication,
    QgsCoordinateTransform,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QObject, pyqtSignal

from ..core.inference import get_session
from ..core.model_registry import ModelConfig, ModelRegistry
from ..core.project import AnnotationProject, SceneDef
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

    def create_project(self, root: str, name: str) -> None:
        """新建并打开工程。"""
        self.project = AnnotationProject.create(root, name)
        self.project_changed.emit()
        self.status_message.emit(f"已创建工程: {name}")

    def open_project(self, root: str) -> None:
        """打开工程。"""
        self.project = AnnotationProject.open(root)
        self.project_changed.emit()
        self.status_message.emit(f"已打开工程: {self.project.name}")

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
        self.image_loaded.emit(self.current_image)
        self.status_message.emit(
            f"已加载 {Path(path).name}"
            f"（{self.raster.width}×{self.raster.height}"
            f"{f'，{self.raster.resolution_m_per_px():.4g} m/px' if self.raster.resolution_m_per_px() else ''}）"
        )

    def _teardown_layers(self, remove_raster_layer: bool = True) -> None:
        """移除旧图层并释放栅格句柄（attach xyz 工作集时可保留文件栅格图层）。"""
        qgs = QgsProject.instance()
        for layer in (self.ann_layer, self.scene_layer):
            if layer is not None:
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

    def _rebuild_annotation_features(self) -> None:
        """从工程标注 JSON 重建标注图层要素（xyz 场景按场景名取 JSON）。"""
        assert self.project and self.raster and self.ann_layer and self.current_image
        shapes = self.project.load_image_labels(
            self.current_image, scene_name=self.scene_label_scope()
        )
        provider = self.ann_layer.dataProvider()
        provider.truncate()
        features = [shape_to_feature(s, self.raster) for s in shapes]
        provider.addFeatures(features)
        self.ann_layer.triggerRepaint()

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
            if not source.startswith("type=xyz"):
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
            QgsProject.instance().addMapLayer(self.ann_layer)
            QgsProject.instance().addMapLayer(self.scene_layer)
            self._rebuild_scene_features()
            self.image_loaded.emit(workset_id)
        return entry

    @staticmethod
    def _web_mercator_wkt() -> str:
        from ..core.xyz_source import EPSG3857_WKT

        return EPSG3857_WKT

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
            if self.raster is None:
                self.status_message.emit(
                    "请先加载影像，或打开 QuickMapServices 在线图层后画场景"
                )
                return None
        # ---- 文件影像分支
        if self.raster is None or self.current_image is None:
            self.status_message.emit("请先加载影像")
            return None
        col0, row0 = self.raster.map_to_pixel(rect.xMinimum(), rect.yMaximum())
        col1, row1 = self.raster.map_to_pixel(rect.xMaximum(), rect.yMinimum())
        bbox = [
            max(0.0, min(col0, col1)),
            max(0.0, min(row0, row1)),
            min(float(self.raster.width), max(col0, col1)),
            min(float(self.raster.height), max(row0, row1)),
        ]
        if bbox[2] - bbox[0] < 2.0 or bbox[3] - bbox[1] < 2.0:
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
        """加载 xyz 场景：构造虚拟影像网格（XyzRaster）并重建标注图层。"""
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
        self.raster_layer = None  # xyz 模式无本地栅格图层（在线图层由用户自开）
        self._rebuild_annotation_features()
        self._rebuild_scene_features()
        self.image_loaded.emit(self.current_image)
        self.status_message.emit(
            f"已加载场景 {scene.name}"
            f"（{raster.width}×{raster.height} px @z{scene.zoom}，"
            f"≈{raster.resolution_m_per_px():.3f} m/px）"
        )

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

    def set_scene_status(self, scene_name: str, status: str) -> None:
        """更新当前影像某场景状态并刷新。"""
        if self.project is None or self.current_image is None:
            return
        self.project.set_scene_status(self.current_image, scene_name, status)
        self._rebuild_scene_features()
        self.project.save()
        self.project_changed.emit()

    def remove_scene(self, scene_name: str) -> None:
        """删除当前影像的某场景。"""
        if self.project is None or self.current_image is None:
            return
        entry = self.project.find_image(self.current_image)
        if entry is None:
            return
        entry.scenes = [s for s in entry.scenes if s.name != scene_name]
        self._rebuild_scene_features()
        self.project.save()
        self.project_changed.emit()

    def scenes_of_current_image(self) -> list[SceneDef]:
        """当前影像的场景列表。"""
        if self.project is None or self.current_image is None:
            return []
        entry = self.project.find_image(self.current_image)
        return list(entry.scenes) if entry else []

    # ------------------------------------------------------------------ 模型

    def get_model_session(self, config: ModelConfig):
        """获取模型推理会话（进程内缓存）。"""
        return get_session(config.file_path, config.task)
