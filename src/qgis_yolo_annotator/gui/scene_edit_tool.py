"""调整场景范围的地图工具：拖角点/边改大小，拖内部整体移动。"""

from __future__ import annotations

from qgis.core import Qgis, QgsPointXY, QgsRectangle
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QCursor

# 手柄命中半径（屏幕像素）
_HANDLE_HIT_PX = 10.0


class SceneEditTool(QgsMapTool):
    """场景范围编辑工具。

    悬停反馈：角点/边中点手柄十字光标，矩形内部移动光标，外部箭头；
    左键拖拽实时预览（黄色矩形），松开提交 controller.update_scene_extent；
    Esc/右键取消当前拖拽。命中与几何换算每次事件实时计算，场景增删即时生效。
    """

    def __init__(self, canvas, controller):
        super().__init__(canvas)
        self.controller = controller
        self._drag: dict | None = None  # {scene_name, mode, orig, press}
        self._hover_mode: str | None = None
        self._rubber = QgsRubberBand(canvas, Qgis.GeometryType.Polygon)
        self._rubber.setStrokeColor(QColor(241, 196, 15, 235))
        self._rubber.setFillColor(QColor(241, 196, 15, 36))
        self._rubber.setWidth(2)

    def deactivate(self):
        self._cancel_drag()
        self.canvas().setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().deactivate()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape and self._drag is not None:
            self._cancel_drag()
        else:
            super().keyPressEvent(e)

    def canvasPressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self._cancel_drag()
            return
        if e.button() != Qt.MouseButton.LeftButton or self._drag is not None:
            return
        hit = self._hit_test(self.toMapCoordinates(e.pos()))
        if hit is None:
            return
        scene, rect, mode = hit
        self._drag = {
            "scene_name": scene.name,
            "mode": mode,
            "orig": QgsRectangle(rect),
            "press": self.toMapCoordinates(e.pos()),
        }

    def canvasMoveEvent(self, e):
        point = self.toMapCoordinates(e.pos())
        if self._drag is not None:
            self._set_rubber(self._preview_rect(point, self._drag))
            return
        hit = self._hit_test(point)
        mode = hit[2] if hit is not None else None
        if mode == self._hover_mode:
            return
        self._hover_mode = mode
        if mode is None:
            shape = Qt.CursorShape.ArrowCursor
        elif mode == "move":
            shape = Qt.CursorShape.SizeAllCursor
        else:
            shape = Qt.CursorShape.CrossCursor
        self.canvas().setCursor(QCursor(shape))

    def canvasReleaseEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton or self._drag is None:
            return
        drag = self._drag
        rect = self._preview_rect(self.toMapCoordinates(e.pos()), drag)
        self._cancel_drag()
        if rect is not None:
            self.controller.update_scene_extent(drag["scene_name"], rect)

    # ------------------------------------------------------------------ 命中/几何

    def _scene_rects(self) -> list[tuple[object, QgsRectangle]]:
        """当前影像各场景的 (scene, 画布坐标矩形)。"""
        result = []
        for scene in self.controller.scenes_of_current_image():
            rect = self.controller.scene_map_rect(scene)
            if rect is not None and not rect.isEmpty():
                result.append((scene, rect))
        return result

    def _hit_test(self, point: QgsPointXY):
        """命中检测：手柄（4 角 + 4 边中点，取最近）→ 矩形内部。

        Returns:
            (scene, rect, mode)；mode 为 corner:i / edge:i / move；未命中 None。
        """
        scenes = self._scene_rects()
        tol = self.canvas().mapSettings().mapUnitsPerPixel() * _HANDLE_HIT_PX
        best = None  # (dist2, scene, rect, mode)
        for scene, rect in scenes:
            cx = (rect.xMinimum() + rect.xMaximum()) / 2
            cy = (rect.yMinimum() + rect.yMaximum()) / 2
            corners = [
                (rect.xMinimum(), rect.yMaximum()),
                (rect.xMaximum(), rect.yMaximum()),
                (rect.xMaximum(), rect.yMinimum()),
                (rect.xMinimum(), rect.yMinimum()),
            ]
            edges = [
                (cx, rect.yMaximum()),
                (rect.xMaximum(), cy),
                (cx, rect.yMinimum()),
                (rect.xMinimum(), cy),
            ]
            for i, (hx, hy) in enumerate(corners):
                d2 = (point.x() - hx) ** 2 + (point.y() - hy) ** 2
                if d2 <= tol * tol and (best is None or d2 < best[0]):
                    best = (d2, scene, rect, f"corner:{i}")
            for i, (hx, hy) in enumerate(edges):
                d2 = (point.x() - hx) ** 2 + (point.y() - hy) ** 2
                if d2 <= tol * tol and (best is None or d2 < best[0]):
                    best = (d2, scene, rect, f"edge:{i}")
        if best is not None:
            return best[1], best[2], best[3]
        for scene, rect in scenes:
            if (
                rect.xMinimum() <= point.x() <= rect.xMaximum()
                and rect.yMinimum() <= point.y() <= rect.yMaximum()
            ):
                return scene, rect, "move"
        return None

    def _preview_rect(self, point: QgsPointXY, drag: dict) -> QgsRectangle:
        """拖拽当前点 → 新矩形（corner 两轴跟随，edge 单轴，move 平移）。"""
        orig = drag["orig"]
        xmin, xmax = orig.xMinimum(), orig.xMaximum()
        ymin, ymax = orig.yMinimum(), orig.yMaximum()
        if drag["mode"] == "move":
            dx = point.x() - drag["press"].x()
            dy = point.y() - drag["press"].y()
            rect = QgsRectangle(xmin + dx, ymin + dy, xmax + dx, ymax + dy)
        else:
            kind, idx = drag["mode"].split(":")
            idx = int(idx)
            if kind == "corner":
                if idx in (0, 3):  # 左侧角
                    xmin = point.x()
                else:
                    xmax = point.x()
                if idx in (0, 1):  # 上侧角（地图 y 大）
                    ymax = point.y()
                else:
                    ymin = point.y()
            else:  # edge: 0 上 1 右 2 下 3 左
                if idx == 0:
                    ymax = point.y()
                elif idx == 1:
                    xmax = point.x()
                elif idx == 2:
                    ymin = point.y()
                else:
                    xmin = point.x()
            rect = QgsRectangle(xmin, ymin, xmax, ymax)
        rect.normalize()
        return rect

    def _set_rubber(self, rect: QgsRectangle):
        self._rubber.reset(Qgis.GeometryType.Polygon)
        for x, y in (
            (rect.xMinimum(), rect.yMinimum()),
            (rect.xMaximum(), rect.yMinimum()),
            (rect.xMaximum(), rect.yMaximum()),
            (rect.xMinimum(), rect.yMaximum()),
        ):
            self._rubber.addPoint(QgsPointXY(x, y))

    def _cancel_drag(self):
        self._drag = None
        self._rubber.reset(Qgis.GeometryType.Polygon)
