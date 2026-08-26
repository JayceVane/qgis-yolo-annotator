"""OBB（旋转框）绘制与编辑地图工具。

绘制（两点式，X-AnyLabeling 手感）：
    左键定首角点 → 拖拽定长边（Shift=15° 吸附）→ 左键 → 拖拽定宽度 → 左键完成

编辑（点击命中已选 OBB 后拖拽）：
    角点手柄：旋转 + 等比缩放（对角固定，对角线跟随鼠标，长宽比锁定）
    边中点手柄：沿该边法向单轴伸缩（对边固定，角度不变）
    内部：整体平移

场景感知：鼠标移出当前场景边界自动切换为平移模式（抓手，左键拖拽平移），
回到场景内恢复绘制/编辑；无需手动切工具。

快捷键：
    Space 按住 + 左键拖拽：平移画布（场景内也可用）
    数字/字母（类别 hotkey）改选中类别；Del 删除；Esc 取消；
    方向键微调（Shift=快速）；Ctrl+C/V 复制粘贴；Ctrl+Z/Y 撤销重做
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
        # 多选（Ctrl+点击批量改类别/删除）
        self._selected_fids: set[int] = set()
        # Space 平移状态
        self._space_held: bool = False
        self._panning: bool = False
        self._pan_last_pos = None  # QPoint（屏幕像素锚点）
        # 场景感知：出场景自动转平移
        self._outside: bool = False

    # ------------------------------------------------------------------ 工具公共

    def deactivate(self):
        self._reset_interaction()
        super().deactivate()

    def canvasPressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            if not self._outside:  # 场景外右键不弹菜单（此时是平移模式）
                self._handle_context_menu(e)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        # Space 按住 / 场景外：左键拖拽 = 平移画布
        if self._space_held or self._outside:
            self._panning = True
            self._pan_last_pos = e.pos()
            self.canvas().setCursor(self._make_closed_hand_cursor())
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

    def canvasMoveEvent(self, e):
        point = self.toMapCoordinates(e.pos())
        self._last_cursor_pos = point
        # 场景感知：进出场景边界切换 平移/标注 模式
        boundary = self._scene_boundary()
        outside = boundary is not None and not boundary.contains(point)
        if outside != self._outside:
            self._set_outside(outside)
        # 平移（Space 或 场景外拖拽）：跟随鼠标增量移动画布中心
        if self._panning and self._pan_last_pos is not None:
            pos = e.pos()
            dx = pos.x() - self._pan_last_pos.x()
            dy = pos.y() - self._pan_last_pos.y()
            self._pan_last_pos = pos
            mupp = self.canvas().mapSettings().mapUnitsPerPixel()
            center = self.canvas().center()
            self.canvas().setCenter(
                QgsPointXY(center.x() - dx * mupp, center.y() + dy * mupp)
            )
            self.canvas().refresh()
            return
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
            elif self._selected_fid is not None and not self._space_held and not self._outside:
                self._update_hover_cursor(point)

    def canvasReleaseEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self._panning:
            self._panning = False
            self._pan_last_pos = None
            if self._space_held or self._outside:
                self.canvas().setCursor(self._make_open_hand_cursor())
            else:
                self.canvas().setCursor(self._make_cross_cursor())
            return
        if self._edit_mode is not None:
            self._commit_edit()

    # ------------------------------------------------------------------ 场景感知

    def _scene_boundary(self):
        """当前场景（或影像全域）在画布 CRS 下的边界矩形；无上下文返回 None。"""
        ctrl = self.controller
        from qgis.core import (
            QgsCoordinateReferenceSystem,
            QgsCoordinateTransform,
            QgsProject,
            QgsRectangle,
        )

        canvas_crs = self.canvas().mapSettings().destinationCrs()

        def to_canvas(rect: QgsRectangle, wkt: str | None):
            if wkt is None:
                return rect
            src = QgsCoordinateReferenceSystem(wkt)
            if not src.isValid() or src == canvas_crs:
                return rect
            try:
                return QgsCoordinateTransform(
                    src, canvas_crs, QgsProject.instance()
                ).transformBoundingBox(rect)
            except Exception:  # noqa: BLE001  变换失败按原坐标使用
                return rect

        scene = ctrl.current_scene
        if scene is not None and scene.kind == "xyz" and scene.map_bbox:
            return to_canvas(QgsRectangle(*scene.map_bbox), ctrl._web_mercator_wkt())
        if scene is not None and ctrl.raster is not None:
            x0, y0 = ctrl.raster.pixel_to_map(scene.bbox[0], scene.bbox[1])
            x1, y1 = ctrl.raster.pixel_to_map(scene.bbox[2], scene.bbox[3])
            rect = QgsRectangle(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            return to_canvas(rect, ctrl.raster.crs_wkt)
        if ctrl.raster is not None:
            # 无活动场景：以整幅影像为界（影像外 = 平移区）
            x0, y0 = ctrl.raster.pixel_to_map(0, 0)
            x1, y1 = ctrl.raster.pixel_to_map(ctrl.raster.width, ctrl.raster.height)
            rect = QgsRectangle(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            return to_canvas(rect, ctrl.raster.crs_wkt)
        return None

    def _set_outside(self, outside: bool):
        """进出场景边界时的模式切换：外=平移（并中断进行中的绘制/编辑）。"""
        self._outside = outside
        if outside:
            if self._state != "idle":
                self._reset_interaction()
            if self._edit_mode is not None:
                self._commit_edit()
            if not self._space_held:
                self.canvas().setCursor(self._make_open_hand_cursor())
        else:
            if not self._space_held:
                self.canvas().setCursor(self._make_cross_cursor())

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
        """点击位置是否落在任一选中目标内（右键不误换目标）。"""
        layer = self.controller.ann_layer
        if layer is None or not self._active_fids():
            return False
        tolerance = self.canvas().mapSettings().mapUnitsPerPixel() * _HIT_RADIUS_PX
        if self._hit_handle(point, tolerance) is not None:
            return True
        for fid in self._active_fids():
            geometry = layer.getFeature(fid).geometry()
            if geometry is None or geometry.isEmpty():
                continue
            if geometry.boundingBox().contains(point) and geometry.contains(point):
                return True
        return False

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
        """弹出类别修改菜单（全部类别，带颜色与快捷键标注；多选时批量应用）。"""
        from qgis.PyQt.QtGui import QColor, QIcon, QPixmap
        from qgis.PyQt.QtWidgets import QMenu

        project = self.controller.project
        layer = self.controller.ann_layer
        fids = self._active_fids()
        if project is None or layer is None or not fids:
            return
        current_label = ""
        feature = layer.getFeature(fids[0])
        if feature.isValid():
            current_label = str(feature.attribute("label") or "")

        menu = QMenu(self.canvas())
        menu.setWindowTitle("修改类别")
        if len(fids) > 1:
            header = menu.addAction(f"批量修改 {len(fids)} 个目标")
        else:
            header = menu.addAction(f"当前：{current_label or '(无类别)'}")
        header.setEnabled(False)
        menu.addSeparator()
        for class_def in project.classes:
            pixmap = QPixmap(12, 12)
            pixmap.fill(QColor(class_def.color or "#ff77ff"))
            action = menu.addAction(QIcon(pixmap), class_def.name)
            if class_def.hotkey:
                action.setText(f"{class_def.name}    [{class_def.hotkey}]")
            if class_def.name == current_label and len(fids) == 1:
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
        if key == Qt.Key.Key_Space and not e.isAutoRepeat():
            # 按住空格进入平移模式（消费事件，阻断画布对空格的默认行为）
            self._space_held = True
            self.canvas().setCursor(self._make_open_hand_cursor())
            e.accept()
            return
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
            if key == Qt.Key.Key_A and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._select_all()
                e.accept()
                return
            if key == Qt.Key.Key_Z and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.controller.redo_annotation()
                else:
                    self.controller.undo_annotation()
                e.accept()
                return
            if key == Qt.Key.Key_Y and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.controller.redo_annotation()
                e.accept()
                return
            if key in (
                Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down,
            ):
                step = _NUDGE_PX_FAST if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else _NUDGE_PX
                self._nudge_selected(key, step)
                e.accept()
                return
            if text and text.strip():  # 空格留给平移，不作类别快捷键
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

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space_held = False
            if self._panning:
                self._panning = False
                self._pan_last_pos = None
            self.canvas().setCursor(self._make_cross_cursor())
            e.accept()
            return
        super().keyReleaseEvent(e)

    # ------------------------------------------------------------------ 绘制流程

    def _handle_idle_click(self, point: QgsPointXY, modifiers):
        # Ctrl+点击：多选切换（不进入绘制/编辑）
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            hit_fid = self._hit_feature(point)
            if hit_fid is not None:
                if hit_fid in self._selected_fids:
                    self._selected_fids.discard(hit_fid)
                else:
                    self._selected_fids.add(hit_fid)
                if self._selected_fids:
                    self._selected_fid = sorted(self._selected_fids)[0]
                    self._show_handles()
                else:
                    self._clear_selection()
                self._report_multi_selection()
            else:
                self._clear_selection()
            return
        # 先尝试进入编辑（命中手柄/要素），否则开始绘制
        if self._try_begin_edit(point):
            return
        self._clear_selection()
        self._p0 = point
        self._state = "drawing_edge"
        self.canvas().setCursor(self._make_cross_cursor())

    def _report_multi_selection(self):
        """多选后状态栏提示。"""
        try:
            if len(self._selected_fids) > 1:
                self.controller.iface.statusBarIface().showMessage(
                    f"已多选 {len(self._selected_fids)} 个目标：右键批量改类别 / Del 批量删除"
                )
        except RuntimeError:
            pass

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
        layer.beginEditCommand("绘制 OBB")
        layer.addFeature(feature)
        layer.endEditCommand()
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
                layer.beginEditCommand("编辑 OBB")  # 一次拖拽 = 一步撤销
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
                layer.beginEditCommand("移动 OBB")
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
        layer = self.controller.ann_layer
        if layer is not None and self._edit_mode is not None:
            layer.endEditCommand()
        self._edit_mode = None
        self._edit_index = -1
        self._drag_anchor = None
        self._show_handles()

    def _apply_edit_drag(self, point: QgsPointXY):
        """拖拽中实时更新几何。

        - move：整体平移
        - vertex：对角固定，整个矩形绕其旋转 + 等比缩放（对角线跟随鼠标）
        - edge：对边固定，被拖边沿自身法向平移（单轴伸缩，角度不变）
        """
        if self._edit_mode is None or not self._drag_orig_pts:
            return
        pts = list(self._drag_orig_pts)
        if self._edit_mode == "move":
            delta = _vsub(point, self._drag_anchor)
            pts = [_vadd(p, delta) for p in pts]
        elif self._edit_mode == "vertex":
            i = self._edit_index
            o_i = (i + 2) % 4
            o = pts[o_i]
            d0 = _vsub(self._drag_orig_pts[i], o)  # 原对角向量
            d1 = _vsub(point, o)  # 目标对角向量
            len0, len1 = _vlen(d0), _vlen(d1)
            if len0 < 1e-9 or len1 < 1e-9:
                return
            ang = math.atan2(d1.y(), d1.x()) - math.atan2(d0.y(), d0.x())
            scale = len1 / len0
            cos_a, sin_a = math.cos(ang), math.sin(ang)

            def _rot(p: QgsPointXY) -> QgsPointXY:
                rel = _vsub(p, o)
                return QgsPointXY(
                    o.x() + (rel.x() * cos_a - rel.y() * sin_a) * scale,
                    o.y() + (rel.x() * sin_a + rel.y() * cos_a) * scale,
                )

            pts = [_rot(p) if idx != o_i else p for idx, p in enumerate(pts)]
        elif self._edit_mode == "edge":
            i = self._edit_index
            opp_i = (i + 2) % 4
            edge = _vsub(pts[(i + 1) % 4], pts[i])
            normal = _vunit(QgsPointXY(-edge.y(), edge.x()))
            if _vlen(normal) < 1e-9:
                return
            old_off = _vdot(_vsub(pts[i], pts[opp_i]), normal)
            new_off = _vdot(_vsub(point, pts[opp_i]), normal)
            shift = new_off - old_off
            pts[i] = _vadd(pts[i], _vmul(normal, shift))
            pts[(i + 1) % 4] = _vadd(pts[(i + 1) % 4], _vmul(normal, shift))
        self._write_geometry(pts)
        self._set_preview(pts)
        self._show_handles()

    def _write_geometry(self, pts: list[QgsPointXY]):
        layer = self.controller.ann_layer
        if layer is None or self._selected_fid is None:
            return
        ring = pts + [pts[0]]
        geometry = QgsGeometry.fromPolygonXY([ring])
        layer.changeGeometry(self._selected_fid, geometry)
        layer.triggerRepaint()
        self.controller.labels_changed.emit()

    # ------------------------------------------------------------------ 选择/快捷操作

    def _select(self, fid: int):
        """单选（重置多选集合）。"""
        self._selected_fid = fid
        self._selected_fids = {fid}
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
                f"已选中「{label}」：右键改类别 / 数字键快速改 / Del 删除 / "
                f"Ctrl+点击多选 / 方向键微调"
            )
        except RuntimeError:
            pass

    def _active_fids(self) -> list[int]:
        """当前作用目标（多选集合优先，兼容旧单选字段）。"""
        if self._selected_fids:
            return sorted(self._selected_fids)
        return [self._selected_fid] if self._selected_fid is not None else []

    def _set_selected_label(self, label: str):
        """给全部选中目标改类别（批量，单步撤销）。"""
        layer = self.controller.ann_layer
        fids = self._active_fids()
        if layer is None or not fids:
            return
        attr_idx = layer.fields().indexOf("label")
        layer.beginEditCommand(f"改类别 → {label}")
        for fid in fids:
            layer.changeAttributeValue(fid, attr_idx, label)
        layer.endEditCommand()
        layer.triggerRepaint()
        self.controller.labels_changed.emit()
        self._show_handles()
        try:
            if len(fids) > 1:
                self.controller.iface.statusBarIface().showMessage(
                    f"已批量修改 {len(fids)} 个目标为「{label}」（自动保存，Ctrl+Z 撤销）"
                )
            else:
                self.controller.iface.statusBarIface().showMessage(
                    f"类别已改为「{label}」（自动保存，Ctrl+Z 撤销）"
                )
        except RuntimeError:
            pass

    def _clear_selection(self):
        self._selected_fid = None
        self._selected_fids = set()
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
        """删除全部选中目标（批量，单步撤销）。"""
        layer = self.controller.ann_layer
        fids = self._active_fids()
        if layer is None or not fids:
            return
        layer.beginEditCommand(f"删除 {len(fids)} 个目标")
        for fid in fids:
            layer.deleteFeature(fid)
        layer.endEditCommand()
        layer.triggerRepaint()
        self._clear_selection()
        self.controller.labels_changed.emit()

    def _select_all(self):
        """Ctrl+A：全选当前标注图层要素。"""
        layer = self.controller.ann_layer
        if layer is None:
            return
        self._selected_fids = {f.id() for f in layer.getFeatures()}
        if self._selected_fids:
            self._selected_fid = sorted(self._selected_fids)[0]
            self._clear_handles()  # 多选不画手柄
            self._report_multi_selection()

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
        layer.beginEditCommand("微调 OBB")
        self._write_geometry(pts)
        layer.endEditCommand()
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

    def _make_open_hand_cursor(self):
        return QCursor(Qt.CursorShape.OpenHandCursor)

    def _make_closed_hand_cursor(self):
        return QCursor(Qt.CursorShape.ClosedHandCursor)
