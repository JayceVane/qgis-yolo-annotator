"""OBB（旋转框）绘制与编辑地图工具。

绘制（两点式，X-AnyLabeling 手感）：
    左键定首角点 → 拖拽定长边（Shift=15° 吸附）→ 左键 → 拖拽定宽度 → 左键完成

编辑（点击命中已选 OBB 后拖拽）：
    角点手柄：平行四边形约束拖动（对角固定，邻点沿边滑动 → 天然支持旋转/变长短）
    边中点手柄：沿法向单轴伸缩（角度不变）
    内部：整体平移

快捷键：
    数字/字母（类别 hotkey）改选中类别；Del 删除；Esc 取消；方向键微调（像素步长）
"""

from __future__ import annotations

import math

from qgis.core import Qgis, QgsGeometry, QgsPointXY
from qgis.gui import QgsMapTool, QgsRubberBand, QgsVertexMarker
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QCursor

_SNAP_ANGLE_DEG = 15.0
_NUDGE_PX = 1.0
_NUDGE_PX_FAST = 10.0
_HIT_RADIUS_PX = 7.0


def _vsub(a: QgsPointXY, b: QgsPointXY) -> QgsPointXY:
    return QgsPointXY(a.x() - b.x(), a.y() - b.y())


def _vadd(a: QgsPointXY, b: QgsPointXY) -> QgsPointXY:
    return QgsPointXY(a.x() + b.x(), a.y() + b.y())


def _vmul(a: QgsPointXY, k: float) -> QgsPointXY:
    return QgsPointXY(a.x() * k, a.y() * k)


def _vdot(a: QgsPointXY, b: QgsPointXY) -> float:
    return a.x() * b.x() + a.y() * b.y()


def _vlen(a: QgsPointXY) -> float:
    return math.hypot(a.x(), a.y())


def _vunit(a: QgsPointXY) -> QgsPointXY:
    length = _vlen(a)
    return QgsPointXY(a.x() / length, a.y() / length) if length > 1e-12 else QgsPointXY(0.0, 0.0)


class ObbEditTool(QgsMapTool):
    """OBB 绘制/编辑工具（绑定标注图层与控制器）。"""

    def __init__(self, canvas, controller, default_label: str = ""):
        super().__init__(canvas)
        self.controller = controller
        self.default_label = default_label
        self._state = "idle"  # idle / drawing_edge / drawing_width
        self._p0: QgsPointXY | None = None
        self._p1: QgsPointXY | None = None
        self._preview = QgsRubberBand(canvas, Qgis.GeometryType.Polygon)
        self._preview.setStrokeColor(QColor(255, 200, 0, 220))
        self._preview.setFillColor(QColor(255, 200, 0, 60))
        self._preview.setWidth(2)
        self._line_hint = QgsRubberBand(canvas, Qgis.GeometryType.Line)
        self._line_hint.setStrokeColor(QColor(255, 200, 0, 220))
        self._line_hint.setWidth(1)
        # 编辑状态
        self._selected_fid: int | None = None
        self._edit_mode: str | None = None  # vertex / edge / move
        self._edit_index: int = -1
        self._drag_anchor: QgsPointXY | None = None
        self._drag_orig_pts: list[QgsPointXY] = []
        self._handles: list[QgsVertexMarker] = []
        self._copied_pts: list[QgsPointXY] | None = None
        self._last_cursor_pos: QgsPointXY | None = None

    # ------------------------------------------------------------------ 工具公共

    def deactivate(self):
        self._reset_interaction()
        super().deactivate()

    def canvasMoveEvent(self, e):
        point = self.toMapCoordinates(e.pos())
        self._last_cursor_pos = point
        if self._state == "drawing_edge" and self._p0 is not None:
            self._p1 = self._apply_snap(self._p0, point, e.modifiers())
            self._line_hint.reset(Qgis.GeometryType.Line)
            self._line_hint.addPoint(self._p0)
            self._line_hint.addPoint(self._p1)
        elif self._state == "drawing_width" and self._p0 is not None and self._p1 is not None:
            pts = self._obb_from_width(self._p0, self._p1, point)
            self._set_preview(pts)
        elif self._state == "idle":
            if self._edit_mode is not None:
                self._apply_edit_drag(point)
            elif self._selected_fid is not None:
                self._update_hover_cursor(point)

    def canvasPressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            if e.button() == Qt.MouseButton.RightButton:
                self._handle_context_menu(e)
            return
        point = self.toMapCoordinates(e.pos())
        if self._state == "idle":
            self._handle_idle_click(point, e.modifiers())
        elif self._state == "drawing_edge":
            self._p1 = self._apply_snap(self._p0, point, e.modifiers())
            self._state = "drawing_width"
            self._line_hint.reset(Qgis.GeometryType.Line)
        elif self._state == "drawing_width":
            self._finish_draw(point)

    def canvasReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._edit_mode is not None:
            self._commit_edit()

    def _handle_context_menu(self, e):
        """右键：绘制中取消；否则命中/沿用选中目标并弹出类别修改菜单。"""
        if self._state != "idle":
            self._cancel_action()
            return
        point = self.toMapCoordinates(e.pos())
        # 未选中或点击处命中另一目标 → 先切换选中
        if self._selected_fid is None or not self._hit_selected(point):
            hit_fid = self._hit_feature(point)
            if hit_fid is None:
                self._cancel_action()  # 空白处右键：取消选中
                return
            self._select(hit_fid)
        # QgsMapMouseEvent 无 globalPos()：画布局部坐标 → 全局屏幕坐标
        self._show_class_menu(self.canvas().mapToGlobal(e.pos()))

    def _hit_selected(self, point: QgsPointXY) -> bool:
        """点击位置是否落在当前选中目标内（右键不误换目标）。"""
        layer = self.controller.ann_layer
        if layer is None or self._selected_fid is None:
            return False
        feature = layer.getFeature(self._selected_fid)
        geometry = feature.geometry()
        if geometry is None or geometry.isEmpty():
            return False
        tolerance = self.canvas().mapSettings().mapUnitsPerPixel() * _HIT_RADIUS_PX
        in_geometry = (
            geometry.boundingBox().contains(point) and geometry.contains(point)
        )
        return in_geometry or (self._hit_handle(point, tolerance) is not None)

    def _hit_feature(self, point: QgsPointXY) -> int | None:
        """点击位置命中的要素 fid。"""
        layer = self.controller.ann_layer
        if layer is None:
            return None
        tolerance = self.canvas().mapSettings().mapUnitsPerPixel() * _HIT_RADIUS_PX
        hit = self._hit_handle(point, tolerance)
        if hit is not None and self._selected_fid is not None:
            return self._selected_fid
        for feature in layer.getFeatures():
            geometry = feature.geometry()
            if geometry is None or geometry.isEmpty():
                continue
            if geometry.boundingBox().contains(point) and geometry.contains(point):
                return feature.id()
        return None

    def _show_class_menu(self, global_pos):
        """弹出类别修改菜单（全部类别，带颜色与快捷键标注）。"""
        from qgis.PyQt.QtGui import QColor, QIcon, QPixmap
        from qgis.PyQt.QtWidgets import QMenu

        project = self.controller.project
        layer = self.controller.ann_layer
        if project is None or layer is None or self._selected_fid is None:
            return
        current_label = ""
        feature = layer.getFeature(self._selected_fid)
        if feature.isValid():
            current_label = str(feature.attribute("label") or "")

        menu = QMenu(self.canvas())
        menu.setWindowTitle("修改类别")
        current_row = menu.addAction(f"当前：{current_label or '(无类别)'}")
        current_row.setEnabled(False)
        menu.addSeparator()
        for index, class_def in enumerate(project.classes, start=1):
            pixmap = QPixmap(12, 12)
            pixmap.fill(QColor(class_def.color or "#ff77ff"))
            action = menu.addAction(QIcon(pixmap), class_def.name)
            if class_def.hotkey:
                action.setText(f"{class_def.name}    [{class_def.hotkey}]")
            if class_def.name == current_label:
                action.setCheckable(True)
                action.setChecked(True)
            action.triggered.connect(
                lambda _checked, name=class_def.name: self._set_selected_label(name)
            )
        menu.addSeparator()
        delete_action = menu.addAction("删除目标 (Del)")
        delete_action.triggered.connect(self._delete_selected)
        menu.exec(global_pos)

    def keyPressEvent(self, e):
        key = e.key()
        text = e.text()
        if key == Qt.Key.Key_Escape:
            self._cancel_action()
            e.accept()
            return
        if self._selected_fid is not None:
            if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self._delete_selected()
                e.accept()
                return
            if key == Qt.Key.Key_C and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._copy_selected()
                e.accept()
                return
            if key == Qt.Key.Key_V and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._paste_copied()
                e.accept()
                return
            if key in (
                Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down,
            ):
                step = _NUDGE_PX_FAST if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else _NUDGE_PX
                self._nudge_selected(key, step)
                e.accept()
                return
            if text:
                class_def = (
                    self.controller.project.class_by_hotkey(text)
                    if self.controller.project
                    else None
                )
                if class_def is not None:
                    self._set_selected_label(class_def.name)
                    e.accept()
                    return
        super().keyPressEvent(e)

    # ------------------------------------------------------------------ 绘制流程

    def _handle_idle_click(self, point: QgsPointXY, modifiers):
        # 先尝试进入编辑（命中手柄/要素），否则开始绘制
        if self._try_begin_edit(point):
            return
        self._clear_selection()
        self._p0 = point
        self._state = "drawing_edge"
        self.canvas().setCursor(self._make_cross_cursor())

    def _finish_draw(self, point: QgsPointXY):
        assert self._p0 is not None and self._p1 is not None
        pts = self._obb_from_width(self._p0, self._p1, point)
        if _vlen(_vsub(pts[1], pts[0])) < 1e-6 or _vlen(_vsub(pts[3], pts[0])) < 1e-6:
            self._reset_interaction()
            return
        self._add_feature(pts, self.default_label)
        self._reset_interaction()

    def _obb_from_width(
        self, p0: QgsPointXY, p1: QgsPointXY, cursor: QgsPointXY
    ) -> list[QgsPointXY]:
        """p0→p1 为一边，cursor 到该边垂距定宽（带符号）。"""
        edge = _vsub(p1, p0)
        unit = _vunit(edge)
        normal = QgsPointXY(-unit.y(), unit.x())
        offset = _vdot(_vsub(cursor, p0), normal)
        p3 = _vadd(p0, _vmul(normal, offset))
        p2 = _vadd(p1, _vmul(normal, offset))
        return [p0, p1, p2, p3]

    def _apply_snap(
        self, origin: QgsPointXY, point: QgsPointXY, modifiers
    ) -> QgsPointXY:
        if not (modifiers & Qt.KeyboardModifier.ShiftModifier):
            return point
        angle = math.atan2(point.y() - origin.y(), point.x() - origin.x())
        snapped = round(angle / math.radians(_SNAP_ANGLE_DEG)) * math.radians(_SNAP_ANGLE_DEG)
        radius = _vlen(_vsub(point, origin))
        return QgsPointXY(
            origin.x() + radius * math.cos(snapped),
            origin.y() + radius * math.sin(snapped),
        )

    def _add_feature(self, pts: list[QgsPointXY], label: str) -> None:
        layer = self.controller.ann_layer
        if layer is None:
            return
        from .annotation_layer import shape_to_feature

        raster = self.controller.raster
        pts_px = []
        for p in pts:
            col, row = raster.map_to_pixel(p.x(), p.y())
            pts_px.append([float(col), float(row)])
        from ..core.label_store import make_shape

        shape = make_shape(label, pts_px, "rotation")
        feature = shape_to_feature(shape, raster)
        layer.dataProvider().addFeature(feature)
        layer.triggerRepaint()
        self.controller.labels_changed.emit()

    # ------------------------------------------------------------------ 编辑流程

    def _try_begin_edit(self, point: QgsPointXY) -> bool:
        layer = self.controller.ann_layer
        if layer is None:
            return False
        tolerance = self.canvas().mapSettings().mapUnitsPerPixel() * _HIT_RADIUS_PX
        if self._selected_fid is not None:
            hit = self._hit_handle(point, tolerance)
            if hit is not None:
                mode, index, pts = hit
                self._edit_mode = mode
                self._edit_index = index
                self._drag_orig_pts = pts
                self._drag_anchor = point
                return True
        # 未命中手柄 → 查找要素内部
        for feature in layer.getFeatures():
            geometry = feature.geometry()
            if geometry is None or geometry.isEmpty():
                continue
            if geometry.boundingBox().contains(point) and geometry.contains(point):
                self._select(feature.id())
                self._edit_mode = "move"
                self._drag_anchor = point
                self._drag_orig_pts = self._feature_ring(feature)
                return True
        return False

    def _hit_handle(self, point: QgsPointXY, tolerance: float):
        """命中选中要素的角点/边中点手柄 → (mode, index, 当前四点)。"""
        layer = self.controller.ann_layer
        if layer is None or self._selected_fid is None:
            return None
        feature = layer.getFeature(self._selected_fid)
        pts = self._feature_ring(feature)
        if len(pts) != 4:
            return None
        for i, vertex in enumerate(pts):
            if _vlen(_vsub(vertex, point)) <= tolerance:
                return ("vertex", i, pts)
        for i in range(4):
            mid = _vmul(_vadd(pts[i], pts[(i + 1) % 4]), 0.5)
            if _vlen(_vsub(mid, point)) <= tolerance:
                return ("edge", i, pts)
        return None

    def _commit_edit(self):
        self._edit_mode = None
        self._edit_index = -1
        self._drag_anchor = None
        self._show_handles()

    def _apply_edit_drag(self, point: QgsPointXY):
        """拖拽中实时更新几何（编辑模式由 _edit_mode 决定）。"""
        if self._edit_mode is None or not self._drag_orig_pts:
            return
        pts = list(self._drag_orig_pts)
        if self._edit_mode == "move":
            delta = _vsub(point, self._drag_anchor)
            pts = [_vadd(p, delta) for p in pts]
        elif self._edit_mode == "vertex":
            fixed_opposite = (self._edit_index + 2) % 4
            prev_i = (self._edit_index + 3) % 4
            next_i = (self._edit_index + 1) % 4
            o = pts[fixed_opposite]
            axis_prev = _vunit(_vsub(pts[prev_i], o))
            axis_next = _vunit(_vsub(pts[next_i], o))
            d = _vsub(point, o)
            new_prev = _vadd(o, _vmul(axis_prev, _vdot(d, axis_prev)))
            new_next = _vadd(o, _vmul(axis_next, _vdot(d, axis_next)))
            pts[prev_i] = new_prev
            pts[next_i] = new_next
            pts[self._edit_index] = _vadd(new_prev, _vsub(new_next, o))
        elif self._edit_mode == "edge":
            i = self._edit_index
            center = _vmul(
                _vadd(_vadd(pts[0], pts[1]), _vadd(pts[2], pts[3])), 0.25
            )
            axis = _vunit(_vsub(pts[(i + 1) % 4], pts[i]))
            normal = QgsPointXY(-axis.y(), axis.x())
            half = _vdot(_vsub(point, center), normal)
            sign = 1.0 if _vdot(_vsub(pts[i], center), normal) >= 0 else -1.0
            offset = _vmul(normal, half * sign - _vdot(_vsub(pts[i], center), normal))
            pts[i] = _vadd(pts[i], offset)
            pts[(i + 1) % 4] = _vadd(pts[(i + 1) % 4], offset)
        self._write_geometry(pts)
        self._set_preview(pts)
        self._show_handles()

    def _write_geometry(self, pts: list[QgsPointXY]):
        layer = self.controller.ann_layer
        if layer is None or self._selected_fid is None:
            return
        ring = pts + [pts[0]]
        geometry = QgsGeometry.fromPolygonXY([ring])
        layer.dataProvider().changeGeometryValues({self._selected_fid: geometry})
        layer.triggerRepaint()
        self.controller.labels_changed.emit()

    # ------------------------------------------------------------------ 选择/快捷操作

    def _select(self, fid: int):
        self._selected_fid = fid
        self._show_handles()
        # 状态栏操作提示（改类别的主入口告知）
        label = ""
        layer = self.controller.ann_layer
        if layer is not None:
            feature = layer.getFeature(fid)
            if feature.isValid():
                label = str(feature.attribute("label") or "")
        try:
            self.controller.iface.statusBarIface().showMessage(
                f"已选中「{label}」：右键改类别 / 数字键快速改 / Del 删除 / 方向键微调"
            )
        except RuntimeError:
            pass

    def _set_selected_label(self, label: str):
        layer = self.controller.ann_layer
        if layer is None or self._selected_fid is None:
            return
        layer.dataProvider().changeAttributeValues(
            {self._selected_fid: {layer.fields().indexOf("label"): label}}
        )
        layer.triggerRepaint()
        self.controller.labels_changed.emit()
        self._show_handles()
        try:
            self.controller.iface.statusBarIface().showMessage(
                f"类别已改为「{label}」（自动保存）"
            )
        except RuntimeError:
            pass

    def _clear_selection(self):
        self._selected_fid = None
        self._edit_mode = None
        for marker in self._handles:
            self.canvas().scene().removeItem(marker)
        self._handles = []

    def _show_handles(self):
        self._clear_handles()
        layer = self.controller.ann_layer
        if layer is None or self._selected_fid is None:
            return
        feature = layer.getFeature(self._selected_fid)
        pts = self._feature_ring(feature)
        if len(pts) != 4:
            return
        for vertex in pts:  # 角点：黄
            marker = QgsVertexMarker(self.canvas())
            marker.setCenter(vertex)
            marker.setColor(QColor(255, 255, 0, 230))
            marker.setIconSize(8)
            marker.setIconType(QgsVertexMarker.IconType.ICON_CIRCLE)
            marker.setPenWidth(2)
            self._handles.append(marker)
        for i in range(4):  # 边中点：青
            mid = _vmul(_vadd(pts[i], pts[(i + 1) % 4]), 0.5)
            marker = QgsVertexMarker(self.canvas())
            marker.setCenter(mid)
            marker.setColor(QColor(0, 255, 255, 230))
            marker.setIconSize(7)
            marker.setIconType(QgsVertexMarker.IconType.ICON_CIRCLE)
            marker.setPenWidth(2)
            self._handles.append(marker)

    def _clear_handles(self):
        for marker in self._handles:
            self.canvas().scene().removeItem(marker)
        self._handles = []

    def _delete_selected(self):
        layer = self.controller.ann_layer
        if layer is None or self._selected_fid is None:
            return
        layer.dataProvider().deleteFeatures([self._selected_fid])
        layer.triggerRepaint()
        self._clear_selection()
        self.controller.labels_changed.emit()

    def _set_selected_label(self, label: str):
        layer = self.controller.ann_layer
        if layer is None or self._selected_fid is None:
            return
        layer.dataProvider().changeAttributeValues(
            {self._selected_fid: {layer.fields().indexOf("label"): label}}
        )
        layer.triggerRepaint()
        self.controller.labels_changed.emit()

    def _copy_selected(self):
        layer = self.controller.ann_layer
        if layer is None or self._selected_fid is None:
            return
        self._copied_pts = self._feature_ring(layer.getFeature(self._selected_fid))

    def _paste_copied(self):
        if not self._copied_pts:
            return
        feature_label = ""
        layer = self.controller.ann_layer
        if layer is not None and self._selected_fid is not None:
            feature_label = layer.getFeature(self._selected_fid).attribute("label") or ""
        anchor = self._last_cursor_pos or self._copied_pts[0]
        offset = _vsub(anchor, self._copied_pts[0])
        pts = [_vadd(p, offset) for p in self._copied_pts]
        self._add_feature(pts, feature_label or self.default_label)

    def _nudge_selected(self, key, step_px: float):
        step = self.canvas().mapSettings().mapUnitsPerPixel() * step_px
        dx, dy = 0.0, 0.0
        if key == Qt.Key.Key_Left:
            dx = -step
        elif key == Qt.Key.Key_Right:
            dx = step
        elif key == Qt.Key.Key_Up:
            dy = step  # 地图 y 向上为正（与像素 y 相反由 map_to_pixel 吸收）
        else:
            dy = -step
        layer = self.controller.ann_layer
        if layer is None or self._selected_fid is None:
            return
        pts = self._feature_ring(layer.getFeature(self._selected_fid))
        pts = [_vadd(p, QgsPointXY(dx, dy)) for p in pts]
        self._write_geometry(pts)
        self._show_handles()

    # ------------------------------------------------------------------ 杂项

    def _feature_ring(self, feature) -> list[QgsPointXY]:
        """要素四点环（闭环重复点去除）。"""
        geometry = feature.geometry()
        if geometry is None or geometry.isEmpty():
            return []
        polygon = geometry.asPolygon()
        if not polygon:
            return []
        ring = [QgsPointXY(p.x(), p.y()) for p in polygon[0]]
        if len(ring) >= 5 and ring[0].distance(ring[-1]) < 1e-12:
            ring = ring[:-1]
        return ring

    def _set_preview(self, pts: list[QgsPointXY]):
        self._preview.reset(Qgis.GeometryType.Polygon)
        for p in pts:
            self._preview.addPoint(p)

    def _reset_interaction(self):
        self._state = "idle"
        self._p0 = self._p1 = None
        self._preview.reset(Qgis.GeometryType.Polygon)
        self._line_hint.reset(Qgis.GeometryType.Line)
        self._clear_handles()
        self.canvas().setCursor(self._make_cross_cursor())

    def _cancel_action(self):
        if self._state != "idle":
            self._reset_interaction()
        else:
            self._clear_selection()

    def _update_hover_cursor(self, point: QgsPointXY):
        tolerance = self.canvas().mapSettings().mapUnitsPerPixel() * _HIT_RADIUS_PX
        hit = self._hit_handle(point, tolerance)
        self.canvas().setCursor(
            self._make_cross_cursor() if hit is None else self._make_hand_cursor()
        )

    def _make_cross_cursor(self):
        return QCursor(Qt.CursorShape.CrossCursor)

    def _make_hand_cursor(self):
        return QCursor(Qt.CursorShape.PointingHandCursor)
