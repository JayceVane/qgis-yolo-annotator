"""标注图层：X-AnyLabeling shapes ↔ QgsVectorLayer（地图坐标）双向同步。

图层 schema（Polygon，4 点矩形环）：
    label(str) score(double, NULL=手工) difficult(int) source(str) direction(double)
内存图层随影像加载重建；保存时整层导出为 shapes（像素坐标）。
"""

from __future__ import annotations

from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsField,
    QgsFields,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsRendererCategory,
    QgsSymbol,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor

from ..core.label_store import rotation_direction
from ..core.raster_io import RasterRef

LAYER_NAME = "yolo_annotator_labels"
_FIELDS_SPEC = [
    ("label", QVariant.String, "类别名"),
    ("score", QVariant.Double, "模型置信度（手工标注为 NULL）"),
    ("difficult", QVariant.Int, "DOTA difficult"),
    ("source", QVariant.String, "manual / pred"),
    ("direction", QVariant.Double, "OBB 方向（弧度，p0→p1）"),
]

_FALLBACK_COLOR = "#ff77ff"


def create_annotation_layer(crs_wkt: str | None) -> QgsVectorLayer:
    """创建标注内存图层（含字段与分类渲染占位）。

    Args:
        crs_wkt: 影像 CRS WKT；None 时图层无 CRS（像素坐标直存）。

    Returns:
        QgsVectorLayer（Polygon memory）。
    """
    uri = "Polygon?memory"
    if crs_wkt:
        uri += f"&crs={crs_wkt.replace('"', "'")}"
    layer = QgsVectorLayer(uri, LAYER_NAME, "memory")
    provider = layer.dataProvider()
    fields = QgsFields()
    for name, qtype, _doc in _FIELDS_SPEC:
        fields.append(QgsField(name, qtype))
    provider.addAttributes(fields)
    layer.updateFields()
    return layer


def shape_to_feature(shape: dict, raster: RasterRef, fid: int = 0) -> QgsFeature:
    """shape（像素坐标）→ 地图坐标 QgsFeature。"""
    feature = QgsFeature(fid)
    feature.setFields(_fields(shape))
    pts_px = shape.get("points") or []
    map_pts = []
    for x, y in pts_px:
        mx, my = raster.pixel_to_map(float(x), float(y))
        map_pts.append(QgsPointXY(float(mx), float(my)))
    geometry = QgsGeometry.fromPolygonXY([map_pts])
    feature.setGeometry(geometry)
    fill = {
        "label": str(shape.get("label", "")),
        "difficult": 1 if shape.get("difficult") else 0,
        "source": "pred" if shape.get("score") is not None else "manual",
    }
    score = shape.get("score")
    fill["score"] = float(score) if score is not None else None
    direction = shape.get("direction")
    if direction is None and len(pts_px) == 4:
        direction = rotation_direction(pts_px)
    fill["direction"] = float(direction) if direction is not None else None
    feature.setAttribute("label", fill["label"])
    feature.setAttribute("score", fill["score"])
    feature.setAttribute("difficult", fill["difficult"])
    feature.setAttribute("source", fill["source"])
    feature.setAttribute("direction", fill["direction"])
    return feature


def feature_to_shape(feature: QgsFeature, raster: RasterRef) -> dict | None:
    """地图坐标 QgsFeature → shape（像素坐标）；几何非 4 点环返回 None。"""
    geometry = feature.geometry()
    if geometry is None or geometry.isEmpty():
        return None
    polygon = geometry.asPolygon()
    if not polygon or len(polygon[0]) not in (4, 5):  # 5 = 闭环首尾点重复
        return None
    ring = polygon[0][:4]
    pts_px = []
    for vertex in ring:
        col, row = raster.map_to_pixel(vertex.x(), vertex.y())
        pts_px.append([float(col), float(row)])
    label = feature.attribute("label") or ""
    score = feature.attribute("score")
    shape = {
        "label": str(label),
        "score": float(score) if score is not None else None,
        "points": pts_px,
        "group_id": None,
        "description": "",
        "difficult": bool(feature.attribute("difficult") or 0),
        "shape_type": "rotation",
        "flags": {},
        "attributes": {},
        "kie_linking": [],
        "direction": rotation_direction(pts_px),
    }
    return shape


def apply_class_renderer(layer: QgsVectorLayer, classes: list) -> None:
    """按类别分类渲染（类别色半透明填充 + 类别色描边）。"""

    def _make_symbol(color: QColor) -> QgsSymbol:
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol.setColor(QColor(color.red(), color.green(), color.blue(), 90))
        symbol_layer = symbol.symbolLayer(0)
        symbol_layer.setStrokeColor(color)
        symbol_layer.setStrokeWidth(0.8)
        return symbol

    categories = [
        QgsRendererCategory(class_def.name, _make_symbol(QColor(class_def.color or _FALLBACK_COLOR)), class_def.name)
        for class_def in classes
    ]
    # 未匹配类别兜底（与正常类别一致的半透明填充）
    fallback_color = QColor(_FALLBACK_COLOR)
    categories.append(
        QgsRendererCategory("", _make_symbol(fallback_color), "(其他)")
    )
    layer.setRenderer(QgsCategorizedSymbolRenderer("label", categories))
    layer.triggerRepaint()


def _fields(shape: dict) -> QgsFields:
    """构造与图层一致的字段集（shape_to_feature 用）。"""
    fields = QgsFields()
    for name, qtype, _doc in _FIELDS_SPEC:
        fields.append(QgsField(name, qtype))
    return fields
