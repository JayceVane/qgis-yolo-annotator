"""画矩形添加场景（AOI）的地图工具：拖拽矩形 → controller.add_scene_from_map。"""

from __future__ import annotations

from qgis.core import Qgis, QgsPointXY, QgsRectangle
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtCore import Qt


class SceneDrawTool(QgsMapTool):
    """场景绘制工具（左键拖拽出矩形，松开即添加场景）。"""

    def __init__(self, canvas, controller):
        super().__init__(canvas)
        self.controller = controller
        self._start: QgsPointXY | None = None
        self._rubber = QgsRubberBand(canvas, Qgis.GeometryType.Polygon)
        self._rubber.setStrokeColor(QColor(52, 152, 219, 230))
        self._rubber.setFillColor(QColor(52, 152, 219, 40))
        self._rubber.setWidth(2)

    def deactivate(self):
        self._reset()
        super().deactivate()

    def canvasPressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._start = self.toMapCoordinates(e.pos())

    def canvasMoveEvent(self, e):
        if self._start is None:
            return
        rect = QgsRectangle(self._start, self.toMapCoordinates(e.pos()))
        rect.normalize()
        self._rubber.reset(Qgis.GeometryType.Polygon)
        self._rubber.addPoint(QgsPointXY(rect.xMinimum(), rect.yMinimum()))
        self._rubber.addPoint(QgsPointXY(rect.xMaximum(), rect.yMinimum()))
        self._rubber.addPoint(QgsPointXY(rect.xMaximum(), rect.yMaximum()))
        self._rubber.addPoint(QgsPointXY(rect.xMinimum(), rect.yMaximum()))

    def canvasReleaseEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton or self._start is None:
            return
        rect = QgsRectangle(self._start, self.toMapCoordinates(e.pos()))
        self._reset()
        rect.normalize()
        if not rect.isEmpty():
            self.controller.add_scene_from_map(rect)

    def _reset(self):
        self._start = None
        self._rubber.reset(Qgis.GeometryType.Polygon)
