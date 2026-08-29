"""场景（AOI）图层：工程场景的地图可视化与状态渲染。"""

from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCategorizedSymbolRenderer,
    QgsField,
    QgsFeature,
    QgsGeometry,
    QgsPalLayerSettings,
    QgsRectangle,
    QgsRendererCategory,
    QgsSymbol,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor

from ..core.project import (
    SCENE_STATUS_ANNOTATED,
    SCENE_STATUS_LABELS_ORDER,
    SCENE_STATUS_UNANNOTATED,
    SCENE_STATUS_VERIFIED,
    ImageEntry,
)

LAYER_NAME = "yolo_annotator_scenes"

_STATUS_COLORS = {
    SCENE_STATUS_UNANNOTATED: "#999999",
    SCENE_STATUS_ANNOTATED: "#3498db",
    SCENE_STATUS_VERIFIED: "#2ecc71",
}


def create_scene_layer(crs: str | None) -> QgsVectorLayer:
    """创建场景内存图层（Polygon，字段 name/status/image_path）。

    crs 传 authid 或 WKT（setCrs 方式，见 create_annotation_layer 说明）。
    """
    layer = QgsVectorLayer("Polygon?memory", LAYER_NAME, "memory")
    if crs:
        from qgis.core import QgsCoordinateReferenceSystem

        layer.setCrs(QgsCoordinateReferenceSystem(crs))
    provider = layer.dataProvider()
    provider.addAttributes(
        [
            QgsField("name", QVariant.String),
            QgsField("status", QVariant.String),
            QgsField("image_path", QVariant.String),
        ]
    )
    layer.updateFields()
    _apply_status_renderer(layer)
    _apply_name_labeling(layer)
    return layer


def _apply_name_labeling(layer: QgsVectorLayer) -> None:
    """场景名标注：矩形中心显示 name（白字黑边，卫星影像上可读）。"""
    settings = QgsPalLayerSettings()
    settings.fieldName = "name"
    settings.isExpression = False
    settings.placement = Qgis.LabelPlacement.Horizontal
    text_format = QgsTextFormat()
    text_format.setSize(11)
    text_format.setSizeUnit(Qgis.RenderUnit.Points)
    text_format.setColor(QColor(255, 255, 255))
    buffer = QgsTextBufferSettings()
    buffer.setEnabled(True)
    buffer.setSize(1.5)
    buffer.setColor(QColor(0, 0, 0, 220))
    text_format.setBuffer(buffer)
    settings.setFormat(text_format)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    layer.setLabelsEnabled(True)


def rebuild_scene_features(
    layer: QgsVectorLayer, entry: ImageEntry, raster
) -> None:
    """用影像条目的场景重建图层要素（全量替换）。

    xyz 场景直接使用 map_bbox（EPSG:3857，与图层 CRS 一致）；
    文件场景经 raster.pixel_to_map 换算。
    """
    provider = layer.dataProvider()
    provider.truncate()
    features = []
    for scene in entry.scenes:
        feature = QgsFeature(layer.fields())
        feature.setAttribute("name", scene.name)
        feature.setAttribute("status", scene.status)
        feature.setAttribute("image_path", entry.path)
        if scene.kind == "xyz" and scene.map_bbox:
            x0, y0, x1, y1 = scene.map_bbox
            rect = QgsRectangle(x0, y0, x1, y1)
        else:
            x0, y0 = raster.pixel_to_map(scene.bbox[0], scene.bbox[1])
            x1, y1 = raster.pixel_to_map(scene.bbox[2], scene.bbox[3])
            rect = QgsRectangle(float(x0), float(y0), float(x1), float(y1))
        rect.normalize()
        feature.setGeometry(QgsGeometry.fromRect(rect))
        features.append(feature)
    provider.addFeatures(features)
    layer.triggerRepaint()


def _apply_status_renderer(layer: QgsVectorLayer) -> None:
    """场景按状态分类渲染（无填充 + 状态色粗边框）。"""
    categories = []
    for status, color in _STATUS_COLORS.items():
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol_layer = symbol.symbolLayer(0)
        symbol_layer.setFillColor(QColor(0, 0, 0, 0))  # 透明填充
        symbol_layer.setStrokeColor(QColor(color))
        symbol_layer.setStrokeWidth(1.6)
        categories.append(
            QgsRendererCategory(status, symbol, SCENE_STATUS_LABELS_ORDER[status])
        )
    layer.setRenderer(QgsCategorizedSymbolRenderer("status", categories))
