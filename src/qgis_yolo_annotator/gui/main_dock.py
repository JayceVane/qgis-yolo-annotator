"""主 Dock 面板：工程管理 / 标注与推理 / 数据集导出 三个页签。"""

from __future__ import annotations

from pathlib import Path

from qgis.core import QgsApplication
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.project import (
    SCENE_STATUS_ANNOTATED,
    SCENE_STATUS_LABELS_ORDER,
    SCENE_STATUS_UNANNOTATED,
    SCENE_STATUS_VERIFIED,
)
from ..core.raster_io import meters_per_degree
from ..core.scene_infer import RES_UNIT_DEGREE, RES_UNIT_METER, SceneInferOptions
from ..tasks.export_task import ExportTask
from ..tasks.infer_task import SceneInferTask
from .class_dialog import ClassDialog
from .controller import Controller
from .export_dialog import ExportDialog
from .model_dialog import ModelDialog
from .obb_edit_tool import ObbEditTool
from .scene_tool import SceneDrawTool

_STATUS_ICONS = {
    SCENE_STATUS_UNANNOTATED: "○",
    SCENE_STATUS_ANNOTATED: "◐",
    SCENE_STATUS_VERIFIED: "●",
}


class MainDock(QWidget):
    """插件主面板（由 QDockWidget 包装）。"""

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.controller = Controller(iface)
        self._obb_tool: ObbEditTool | None = None
        self._scene_tool: SceneDrawTool | None = None
        self._scene_edit_tool = None
        self._tasks: list = []

        self.tabs = QTabWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        self.tabs.addTab(self._build_project_tab(), "工程")
        self.tabs.addTab(self._build_annotate_tab(), "标注与推理")
        self.tabs.addTab(self._build_export_tab(), "导出")

        self.controller.status_message.connect(
            lambda msg: self.iface.messageBar().pushMessage(
                "YOLO Annotator", msg, duration=4
            )
        )
        self.controller.image_loaded.connect(self._on_image_loaded)
        self.controller.project_changed.connect(self._refresh_project_views)
        self.controller.project_changed.connect(self._refresh_models)
        self.controller.labels_changed.connect(self._mark_pending_save)

        # 插件重载 / QGIS 重启后自动恢复上次工程与影像/场景
        self.controller.restore_last_session()

    # ================================================================== 工程页

    def _build_project_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        row1 = QHBoxLayout()
        btn_new = QPushButton("新建工程")
        btn_open = QPushButton("打开工程")
        btn_save = QPushButton("保存工程")
        btn_new.clicked.connect(self._new_project)
        btn_open.clicked.connect(self._open_project)
        btn_save.clicked.connect(self._save_project)
        for b in (btn_new, btn_open, btn_save):
            row1.addWidget(b)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        btn_add_img = QPushButton("添加影像…")
        btn_add_dir = QPushButton("导入文件夹…")
        btn_import_labels = QPushButton("导入标注(YOLO-OBB/DOTA)…")
        btn_add_img.clicked.connect(self._add_images)
        btn_add_dir.clicked.connect(self._add_image_folder)
        btn_import_labels.clicked.connect(self._import_yolo_obb_labels)
        row2.addWidget(btn_add_img)
        row2.addWidget(btn_add_dir)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(btn_import_labels)
        btn_import_dota = QPushButton("导入 DOTA…")
        btn_import_dota.clicked.connect(self._import_dota_labels)
        row3.addWidget(btn_import_dota)
        layout.addLayout(row3)

        self.tree_project = QTreeWidget()
        self.tree_project.setHeaderLabels(["影像 / 场景", "状态"])
        self.tree_project.itemDoubleClicked.connect(self._on_tree_double_click)
        self.tree_project.itemClicked.connect(self._on_tree_clicked)
        layout.addWidget(self.tree_project, 1)

        self.label_stats = QLabel("未打开工程")
        self.label_stats.setWordWrap(True)
        layout.addWidget(self.label_stats)
        return page

    def _build_annotate_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # 工具行
        box_tools = QGroupBox("标注工具")
        tools_row = QHBoxLayout(box_tools)
        self.btn_tool_obb = QPushButton("画 OBB")
        self.btn_tool_scene = QPushButton("画场景")
        self.btn_tool_edit_scene = QPushButton("调场景")
        self.btn_tool_edit_scene.setToolTip(
            "拖角点/边调整场景大小，拖内部整体移动；在线场景的标注自动随网格迁移"
        )
        self.btn_tool_pan = QPushButton("指针/平移")
        self.combo_default_label = QComboBox()
        self.btn_classes = QPushButton("类别管理")
        self.btn_batch_edit = QPushButton("批量编辑…")
        self.btn_tool_obb.clicked.connect(self._activate_obb_tool)
        self.btn_tool_scene.clicked.connect(self._activate_scene_tool)
        self.btn_tool_edit_scene.clicked.connect(self._activate_scene_edit_tool)
        self.btn_tool_pan.clicked.connect(self._activate_pan)
        self.btn_classes.clicked.connect(self._manage_classes)
        self.btn_batch_edit.clicked.connect(self._batch_edit_labels)
        tools_row.addWidget(self.btn_tool_obb)
        tools_row.addWidget(self.btn_tool_scene)
        tools_row.addWidget(self.btn_tool_edit_scene)
        tools_row.addWidget(self.btn_tool_pan)
        tools_row.addWidget(self.combo_default_label, 1)
        tools_row.addWidget(self.btn_classes)
        tools_row.addWidget(self.btn_batch_edit)
        layout.addWidget(box_tools)

        # 模型行
        box_model = QGroupBox("模型")
        model_row = QHBoxLayout(box_model)
        self.combo_model = QComboBox()
        btn_models = QPushButton("模型管理")
        btn_models.clicked.connect(self._manage_models)
        model_row.addWidget(self.combo_model, 1)
        model_row.addWidget(btn_models)
        layout.addWidget(box_model)

        # 推理参数
        box_infer = QGroupBox("推理参数（选中场景滑窗推理）")
        form_infer = QFormLayout(box_infer)
        self.spin_target_res = QDoubleSpinBox()
        self.spin_target_res.setDecimals(6)
        self.spin_target_res.setRange(0.000001, 1000.0)
        self.spin_target_res.setValue(0.2)
        self.spin_target_res.setSingleStep(0.05)
        self.combo_res_unit = QComboBox()
        self.combo_res_unit.addItem("m/px", RES_UNIT_METER)
        self.combo_res_unit.addItem("°/px", RES_UNIT_DEGREE)
        res_row = QWidget()
        res_layout = QHBoxLayout(res_row)
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.addWidget(self.spin_target_res)
        res_layout.addWidget(self.combo_res_unit)
        self.label_res_hint = QLabel("（未加载影像）")
        self.spin_chip = QSpinBox()
        self.spin_chip.setRange(64, 8192)
        self.spin_chip.setSingleStep(128)
        self.spin_chip.setValue(1024)
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 4096)
        self.spin_overlap.setValue(200)
        self.spin_merge_iou = QDoubleSpinBox()
        self.spin_merge_iou.setRange(0.1, 0.9)
        self.spin_merge_iou.setValue(0.5)
        self.spin_merge_iou.setSingleStep(0.05)
        self.spin_target_res.valueChanged.connect(
            lambda v: setattr(self.controller, "target_res_m", float(v))
        )
        form_infer.addRow("目标分辨率", res_row)
        form_infer.addRow("", self.label_res_hint)
        form_infer.addRow("chip(px)", self.spin_chip)
        form_infer.addRow("重叠(px)", self.spin_overlap)
        form_infer.addRow("跨窗合并 IoU", self.spin_merge_iou)
        layout.addWidget(box_infer)

        # 场景列表
        box_scenes = QGroupBox("当前影像场景")
        scenes_layout = QVBoxLayout(box_scenes)
        self.list_scenes = QListWidget()
        self.list_scenes.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.list_scenes.itemDoubleClicked.connect(self._on_scene_double_clicked)
        scenes_layout.addWidget(self.list_scenes)
        status_row = QHBoxLayout()
        for status in SCENE_STATUS_LABELS_ORDER:
            btn = QPushButton(_STATUS_ICONS[status])
            btn.setToolTip(f"标记选中场景为「{SCENE_STATUS_LABELS_ORDER[status]}」")
            btn.setFixedWidth(36)
            btn.clicked.connect(
                lambda _checked, s=status: self._mark_scenes(s)
            )
            status_row.addWidget(btn)
        status_row.addStretch(1)
        btn_del_scene = QPushButton("删除场景")
        btn_del_scene.clicked.connect(self._delete_scenes)
        status_row.addWidget(btn_del_scene)
        scenes_layout.addLayout(status_row)
        layout.addWidget(box_scenes, 1)

        self.btn_infer = QPushButton("▶ 推理选中场景")
        self.btn_infer.clicked.connect(self._run_inference)
        layout.addWidget(self.btn_infer)
        return page

    def _build_export_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.btn_export = QPushButton("导出数据集…")
        self.btn_export.clicked.connect(self._export_dataset)
        layout.addWidget(self.btn_export)
        self.label_export_stats = QLabel("尚未导出")
        self.label_export_stats.setWordWrap(True)
        layout.addWidget(self.label_export_stats)
        layout.addStretch(1)
        return page

    # ================================================================== 工程

    def _new_project(self):
        directory = QFileDialog.getExistingDirectory(self, "选择工程目录（新建）")
        if not directory:
            return
        try:
            self.controller.create_project(directory, Path(directory).name)
        except (FileExistsError, OSError) as exc:
            QMessageBox.warning(self, "工程", str(exc))
            return
        self._after_project_open()

    def _open_project(self):
        from qgis.PyQt.QtWidgets import QFileDialog

        directory = QFileDialog.getExistingDirectory(self, "选择工程目录（含 project.json）")
        if not directory:
            return
        try:
            self.controller.open_project(directory)
        except (FileNotFoundError, ValueError) as exc:
            QMessageBox.warning(self, "工程", str(exc))
            return
        self._after_project_open()

    def _after_project_open(self):
        self._refresh_project_views()
        self._refresh_models()

    def _save_project(self):
        self.controller.save_project()

    def _add_images(self):
        files, _filter = QFileDialog.getOpenFileNames(
            self,
            "添加影像",
            "",
            "影像 (*.tif *.tiff *.png *.jpg *.jpeg *.bmp)",
        )
        if not files or self.controller.project is None:
            return
        for f in files:
            self.controller.project.add_image(f)
        self.controller.project.save()
        self.controller.project_changed.emit()

    def _add_image_folder(self):
        directory = QFileDialog.getExistingDirectory(self, "导入影像文件夹（递归）")
        if not directory or self.controller.project is None:
            return
        added = self.controller.project.add_image_folder(directory)
        self.controller.project.save()
        self.controller.status_message.emit(f"导入 {len(added)} 幅影像")
        self.controller.project_changed.emit()

    def _import_yolo_obb_labels(self):
        """导入 YOLO-OBB 数据集标注：选 images 目录 + labels 目录（同名配对）。"""
        self._import_label_dataset("yolo_obb")

    def _import_dota_labels(self):
        """导入 DOTA 数据集标注：选 images 目录 + labelTxt 目录（同名配对）。"""
        self._import_label_dataset("dota")

    def _import_label_dataset(self, fmt: str):
        """通用标注导入：images/labels 目录同名配对 → 工程影像+场景/标注。

        项目结构惯例：images/train|val/*.tif + labels(train|val)/*.txt；
        labels 文件按 <stem>.txt 与影像配对，导入为该影像的直标（无场景拆分）。
        """
        from qgis_yolo_annotator.core.label_store import import_dota, import_yolo_obb
        from qgis_yolo_annotator.core.raster_io import RasterRef

        project = self.controller.project
        if project is None:
            self.iface.messageBar().pushMessage(
                "YOLO Annotator", "请先新建/打开工程", duration=3
            )
            return
        images_dir = QFileDialog.getExistingDirectory(
            self, f"选择[{fmt}] images 目录（含影像）"
        )
        if not images_dir:
            return
        labels_dir = QFileDialog.getExistingDirectory(
            self, f"选择[{fmt}] labels 目录（同名 .txt/.xml 标注）"
        )
        if not labels_dir:
            return

        # 类别表：优先工程类别；labels 旁有 classes.txt 则提示核对
        classes_file = Path(labels_dir).parent / "classes.txt"
        external_names: list[str] = []
        if classes_file.is_file():
            external_names = [
                line.strip()
                for line in classes_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for name in external_names:
                project.add_class(name)

        imported_images = 0
        imported_shapes = 0
        label_files = sorted(Path(labels_dir).rglob("*.txt"))
        for label_path in label_files:
            if label_path.name == "classes.txt":
                continue
            stem = label_path.stem
            image_path = self._find_image_by_stem(Path(images_dir), stem)
            if image_path is None:
                continue
            try:
                raster_ref = RasterRef.open(image_path)
            except (RuntimeError, OSError):
                continue
            entry = project.add_image(image_path)
            if fmt == "yolo_obb":
                shapes = import_yolo_obb(
                    label_path, raster_ref.width, raster_ref.height,
                    [c.name for c in project.classes],
                )
            else:
                shapes = import_dota(label_path)
            if shapes:
                project.save_image_labels(
                    image_path, shapes, raster_ref.width, raster_ref.height
                )
                imported_shapes += len(shapes)
            imported_images += 1
            raster_ref.close()
        project.save()
        self.controller.project_changed.emit()
        self.iface.messageBar().pushMessage(
            "YOLO Annotator",
            f"导入完成：{imported_images} 幅影像、{imported_shapes} 个标注"
            + (f"（classes.txt {len(external_names)} 类已并入工程）" if external_names else ""),
            duration=6,
        )

    @staticmethod
    def _find_image_by_stem(images_dir: Path, stem: str) -> str | None:
        """在 images 目录递归查找主名匹配的影像文件。"""
        from ..core.project import IMAGE_EXTENSIONS

        for path in sorted(images_dir.rglob("*")):
            if path.is_file() and path.stem == stem and path.suffix.lower() in IMAGE_EXTENSIONS:
                return str(path)
        return None

    def _on_tree_clicked(self, item: QTreeWidgetItem, _column: int):
        """单击场景行：切换到该场景视图（在线场景加载虚拟影像并定位）。"""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or not str(data).startswith("scene::"):
            return
        _tag, image_path, scene_name = str(data).split("::", 2)
        self.navigate_to_scene(image_path, scene_name)

    def navigate_to_scene(self, image_path: str, scene_name: str):
        """切换到指定场景的视图（必要时先加载归属影像/工作集）。"""
        ctrl = self.controller
        if ctrl.project is None:
            return
        if image_path.startswith("xyz://"):
            if ctrl.current_image != image_path:
                detected = ctrl.detect_xyz_layer()
                if detected is not None:
                    ctrl.attach_xyz_workset(*detected)
                else:
                    # 在线图层不在场时用场景自带的瓦片源配置恢复工作集
                    from ..core.xyz_source import XyzSourceConfig

                    entry = ctrl.project.find_image(image_path)
                    if entry is None or not entry.scenes:
                        return
                    ctrl.attach_xyz_workset(
                        XyzSourceConfig.from_dict(entry.scenes[0].source),
                        image_path.removeprefix("xyz://"),
                    )
            scene = next(
                (s for s in ctrl.scenes_of_current_image() if s.name == scene_name),
                None,
            )
            if scene is None:
                return
            if ctrl.current_scene is None or ctrl.current_scene.name != scene_name:
                ctrl.save_current_labels()
                ctrl.load_scene(scene)  # 内含画布定位
            else:
                ctrl.zoom_to_scene(scene)
        else:
            if ctrl.current_image != image_path or ctrl.raster is None:
                try:
                    ctrl.load_image(image_path)
                except (RuntimeError, ValueError) as exc:
                    QMessageBox.warning(self, "加载影像", str(exc))
                    return
            scene = next(
                (s for s in ctrl.scenes_of_current_image() if s.name == scene_name),
                None,
            )
            if scene is not None:
                ctrl.zoom_to_scene(scene)

    def _on_tree_double_click(self, item: QTreeWidgetItem, _column: int):
        image_path = item.data(0, Qt.ItemDataRole.UserRole)
        # 场景行（scene:: 编码）由单击导航处理，双击忽略
        if not image_path or str(image_path).startswith("scene::"):
            return
        try:
            if image_path.startswith("xyz://"):
                # 在线工作集：探测画布 XYZ 图层后 attach（场景列表随之刷新）
                detected = self.controller.detect_xyz_layer()
                if detected is None:
                    self.iface.messageBar().pushMessage(
                        "YOLO Annotator",
                        "画布中未找到在线 XYZ 图层，请先在 QuickMapServices 打开",
                        duration=5,
                    )
                    return
                self.controller.attach_xyz_workset(*detected)
                self._refresh_scene_list()
                return
            self.controller.load_image(image_path)
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "加载影像", str(exc))

    def _on_scene_double_clicked(self, item: QListWidgetItem):
        """双击场景：在线场景加载其虚拟影像并定位画布；文件场景直接定位到该场景范围。"""
        scene_name = item.data(Qt.ItemDataRole.UserRole)
        if not scene_name:
            return
        for scene in self.controller.scenes_of_current_image():
            if scene.name == scene_name:
                if scene.kind == "xyz":
                    self.controller.save_current_labels()
                    self.controller.load_scene(scene)  # load_scene 内含画布定位
                elif self.controller.raster is not None:
                    self.controller.zoom_to_scene(scene)
                else:
                    self.iface.messageBar().pushMessage(
                        "YOLO Annotator",
                        "请先双击工程列表中的影像加载",
                        duration=4,
                    )
                return

    # ================================================================== 视图刷新

    def _refresh_project_views(self):
        project = self.controller.project
        self.tree_project.clear()
        if project is None:
            self.label_stats.setText("未打开工程")
            return
        for entry in project.images:
            top = QTreeWidgetItem([Path(entry.path).name, f"{len(entry.scenes)} 场景"])
            top.setData(0, Qt.ItemDataRole.UserRole, entry.path)
            for scene in entry.scenes:
                child = QTreeWidgetItem(
                    [
                        f"{_STATUS_ICONS[scene.status]} {scene.name}",
                        SCENE_STATUS_LABELS_ORDER[scene.status],
                    ]
                )
                # 场景行编码归属，单击即切换到该场景视图
                child.setData(
                    0, Qt.ItemDataRole.UserRole, f"scene::{entry.path}::{scene.name}"
                )
                top.addChild(child)
            self.tree_project.addTopLevelItem(top)
        counts = project.scene_status_counts()
        total = project.scene_count()
        self.label_stats.setText(
            f"影像 {len(project.images)} 幅 | 场景 {total}"
            f"（未标注 {counts[SCENE_STATUS_UNANNOTATED]}"
            f" / 已标注 {counts[SCENE_STATUS_ANNOTATED]}"
            f" / 已审核 {counts[SCENE_STATUS_VERIFIED]}）"
        )
        self._refresh_scene_list()
        self._refresh_label_combo()

    def _refresh_scene_list(self):
        self.list_scenes.clear()
        for scene in self.controller.scenes_of_current_image():
            if scene.kind == "xyz":
                extent_text = f"z{scene.zoom}，{scene.width:.0f}×{scene.height:.0f} m"
            else:
                extent_text = f"{scene.width:.0f}×{scene.height:.0f} px"
            item = QListWidgetItem(
                f"{_STATUS_ICONS[scene.status]} {scene.name}（{extent_text}）"
            )
            item.setData(Qt.ItemDataRole.UserRole, scene.name)
            self.list_scenes.addItem(item)

    def _refresh_label_combo(self):
        current = self.combo_default_label.currentText()
        self.combo_default_label.blockSignals(True)
        self.combo_default_label.clear()
        if self.controller.project:
            for class_def in self.controller.project.classes:
                self.combo_default_label.addItem(class_def.name)
        if current:
            index = self.combo_default_label.findText(current)
            if index >= 0:
                self.combo_default_label.setCurrentIndex(index)
        self.combo_default_label.blockSignals(False)

    def _refresh_models(self):
        current = self.combo_model.currentText()
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        for config in self.controller.registry.list_models():
            self.combo_model.addItem(f"{config.name} [{config.task}]", config.name)
        if self.controller.project and self.controller.project.active_model:
            index = self.combo_model.findData(self.controller.project.active_model)
            if index >= 0:
                self.combo_model.setCurrentIndex(index)
        elif current:
            index = self.combo_model.findText(current)
            if index >= 0:
                self.combo_model.setCurrentIndex(index)
        self.combo_model.blockSignals(False)

    def _on_image_loaded(self, image_path: str):
        self._refresh_scene_list()
        self._refresh_label_combo()
        raster = self.controller.raster
        if raster is None:
            return
        if raster.has_georeference:
            res = raster.resolution_m_per_px()
            if raster.is_geographic:
                lat = raster.center_latitude() or 0.0
                lon_m, lat_m = meters_per_degree(lat)
                self.label_res_hint.setText(
                    f"原始分辨率 ≈ {res:.6g} m/px（地理影像，中心纬度 {lat:.2f}°）"
                )
                self._degree_factor = (lon_m + lat_m) / 2.0
            else:
                self.label_res_hint.setText(f"原始分辨率 ≈ {res:.6g} m/px（投影影像）")
                self._degree_factor = None
        else:
            self.label_res_hint.setText("影像无地理参考（仅支持像素坐标流程）")
            self._degree_factor = None

    def _mark_pending_save(self):
        if self.controller.current_image:
            self.controller.save_current_labels()

    # ================================================================== 工具

    def _activate_obb_tool(self):
        if self.controller.ann_layer is None:
            self.iface.messageBar().pushMessage(
                "YOLO Annotator", "请先在工程页双击影像加载", duration=3
            )
            return
        self._obb_tool = ObbEditTool(
            self.iface.mapCanvas(),
            self.controller,
            default_label=self.combo_default_label.currentText(),
        )
        self.iface.mapCanvas().setMapTool(self._obb_tool)

    def _activate_scene_tool(self):
        if self.controller.project is None:
            self.iface.messageBar().pushMessage(
                "YOLO Annotator", "请先新建/打开工程", duration=3
            )
            return
        # 无当前影像时：画布有 XYZ 在线图层即可直接画场景（自动建工作集）
        if self.controller.current_image is None and self.controller.detect_xyz_layer() is None:
            self.iface.messageBar().pushMessage(
                "YOLO Annotator",
                "请先加载影像，或在 QuickMapServices 打开在线图层",
                duration=4,
            )
            return
        self._scene_tool = SceneDrawTool(self.iface.mapCanvas(), self.controller)
        self.iface.mapCanvas().setMapTool(self._scene_tool)

    def _activate_scene_edit_tool(self):
        if self.controller.project is None or self.controller.current_image is None:
            self.iface.messageBar().pushMessage(
                "YOLO Annotator", "请先加载影像或在线工作集", duration=3
            )
            return
        if not self.controller.scenes_of_current_image():
            self.iface.messageBar().pushMessage(
                "YOLO Annotator", "当前影像暂无场景，请先用「画场景」添加", duration=3
            )
            return
        from .scene_edit_tool import SceneEditTool

        self._scene_edit_tool = SceneEditTool(
            self.iface.mapCanvas(), self.controller
        )
        self.iface.mapCanvas().setMapTool(self._scene_edit_tool)

    def _activate_pan(self):
        self.iface.actionPan().trigger()

    def _manage_classes(self):
        project = self.controller.project
        if project is None:
            return
        old_names = [c.name for c in project.classes]
        dialog = ClassDialog(project.classes, self)
        if not dialog.exec():
            return
        project.classes = dialog.classes
        # 检测行序对应的重命名 → 询问是否同步全部标注
        renamed = [
            (old_name, new_def.name)
            for old_name, new_def in zip(old_names, project.classes)
            if new_def.name != old_name
        ]
        if renamed and any(
            self.controller.project.label_counts().get(old, 0) > 0
            for old, _new in renamed
        ):
            detail = "、".join(f"「{o}」→「{n}」" for o, n in renamed)
            answer = QMessageBox.question(
                self,
                "同步标注",
                f"检测到类别改名：{detail}\n是否同步更新工程中所有标注的类别？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                for old_name, new_name in renamed:
                    self.controller.rename_class_sync(old_name, new_name)
        from .annotation_layer import apply_class_renderer

        if self.controller.ann_layer is not None:
            apply_class_renderer(self.controller.ann_layer, project.classes)
        self._refresh_label_combo()
        project.save()

    def _manage_models(self):
        dialog = ModelDialog(self.controller.registry, self)
        dialog.exec()
        self._refresh_models()

    def _batch_edit_labels(self):
        """批量替换类别（当前影像 / 整个工程）。"""
        project = self.controller.project
        if project is None or not project.classes:
            self.iface.messageBar().pushMessage(
                "YOLO Annotator", "请先打开工程并配置类别", duration=3
            )
            return
        from .batch_edit_dialog import BatchEditDialog

        dialog = BatchEditDialog(
            [c.name for c in project.classes],
            project.label_counts(),
            parent=self,
        )
        if not dialog.exec():
            return
        old_name, new_name = dialog.get_old(), dialog.get_new()
        wide = dialog.is_project_wide()
        confirm = QMessageBox.question(
            self,
            "批量编辑",
            f"把{('整个工程' if wide else '当前影像/场景')}中所有「{old_name}」"
            f"改为「{new_name}」？此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            count = self.controller.batch_replace_label(old_name, new_name, wide)
        except RuntimeError as exc:
            QMessageBox.warning(self, "批量编辑", str(exc))
            return
        self.controller.status_message.emit(
            f"批量替换完成：{count} 个标注「{old_name}」→「{new_name}」"
        )

    # ================================================================== 场景操作

    def _selected_scene_names(self) -> list[str]:
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.list_scenes.selectedItems()
        ]

    def _mark_scenes(self, status: str):
        names = self._selected_scene_names()
        if not names:
            self.iface.messageBar().pushMessage(
                "YOLO Annotator", "请先在场景列表中选择场景", duration=3
            )
            return
        for name in names:
            self.controller.set_scene_status(name, status)
        self.controller.status_message.emit(
            f"已标记 {len(names)} 个场景为「{SCENE_STATUS_LABELS_ORDER[status]}」"
        )

    def _delete_scenes(self):
        names = self._selected_scene_names()
        if not names:
            return
        # xyz 场景的标注独立成 JSON：确认是否连带删除
        scenes = {
            s.name: s
            for s in self.controller.scenes_of_current_image()
        }
        with_labels = [
            n for n in names if scenes.get(n) is not None and scenes[n].kind == "xyz"
        ]
        delete_labels = False
        if with_labels:
            answer = QMessageBox.question(
                self,
                "删除场景",
                f"是否同时删除 {len(with_labels)} 个在线场景的标注 JSON？"
                "（文件影像场景的标注为整影像共享，不受影响）",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            delete_labels = answer == QMessageBox.StandardButton.Yes
        removed_files = 0
        for name in names:
            removed_files += self.controller.remove_scene(
                name, delete_labels=delete_labels
            )
        msg = f"已删除 {len(names)} 个场景"
        if removed_files:
            msg += f"（含 {removed_files} 个标注 JSON）"
        self.controller.status_message.emit(msg)

    # ================================================================== 推理

    def _run_inference(self):
        project = self.controller.project
        if project is None or self.controller.current_image is None:
            self.iface.messageBar().pushMessage(
                "YOLO Annotator", "请先加载影像或在线图层场景", duration=3
            )
            return
        if self.combo_model.count() == 0:
            self._refresh_models()
        model_name = self.combo_model.currentData()
        if not model_name:
            self.iface.messageBar().pushMessage(
                "YOLO Annotator", "请先在模型管理中登记模型", duration=3
            )
            return
        config = self.controller.registry.get(model_name)
        if config is None:
            return
        names = self._selected_scene_names()
        entry = project.find_image(self.controller.current_image)
        scenes = [s for s in (entry.scenes if entry else []) if s.name in names]
        if not scenes:
            self.iface.messageBar().pushMessage(
                "YOLO Annotator", "请先在场景列表中选择要推理的场景", duration=3
            )
            return
        options = SceneInferOptions(
            target_res=self.spin_target_res.value(),
            unit=self.combo_res_unit.currentData(),
            chip_size=self.spin_chip.value(),
            overlap=self.spin_overlap.value(),
            merge_iou=self.spin_merge_iou.value(),
        )
        try:
            session = self.controller.get_model_session(config)
        except RuntimeError as exc:
            QMessageBox.warning(self, "推理", str(exc))
            return
        project.active_model = model_name
        is_xyz = entry is not None and entry.kind == "xyz"
        pending = len(scenes)
        self.btn_infer.setEnabled(False)

        def _re_enable():
            nonlocal pending
            pending -= 1
            if pending <= 0:
                self.btn_infer.setEnabled(True)

        for scene in scenes:
            raster = self.controller.raster_for_scene(scene)
            if raster is None:
                _re_enable()
                continue
            view = self.controller.scene_pixel_view(scene)
            task = SceneInferTask(
                f"场景推理 {config.name} {scene.name}",
                raster,
                [view],
                session,
                config,
                options,
            )
            task.scene_done.connect(self._on_scene_inferred)
            task.progress_text.connect(
                lambda msg: self.iface.statusBarIface().showMessage(f"推理: {msg}")
            )
            task.failed.connect(
                lambda reason: QMessageBox.warning(self, "推理", reason)
            )
            task.taskCompleted.connect(_re_enable)
            task.taskTerminated.connect(_re_enable)
            QgsApplication.taskManager().addTask(task)
            self._tasks.append(task)
            if not is_xyz:
                break  # 文件模式单任务多场景；xyz 每场景一个网格独立任务

    def _on_scene_inferred(self, scene_name: str, shapes: list, raster):
        """推理结果（像素 shapes）按其网格追加到标注图层，场景状态 → 已标注。"""
        from .annotation_layer import shape_to_feature

        controller = self.controller
        if controller.ann_layer is None or raster is None:
            return
        # xyz 场景推理结果落层前确保当前网格与结果网格一致
        if (
            controller.current_scene is not None
            and controller.current_scene.name != scene_name
        ):
            controller.save_current_labels()  # 先保存前一场景
            for scene in controller.scenes_of_current_image():
                if scene.name == scene_name:
                    controller.load_scene(scene)
                    break
        features = [shape_to_feature(s, raster) for s in shapes]
        if features:
            layer = controller.ann_layer
            controller.ensure_annotation_editable()
            layer.beginEditCommand(f"推理结果 {scene_name}")  # 整批 = 一步撤销
            for feature in features:
                layer.addFeature(feature)
            layer.endEditCommand()
            layer.triggerRepaint()
        controller.set_scene_status(scene_name, SCENE_STATUS_ANNOTATED)
        controller.save_current_labels()
        controller.status_message.emit(
            f"场景 {scene_name}：新增 {len(features)} 个标注"
        )

    # ================================================================== 导出

    def _export_dataset(self):
        project = self.controller.project
        if project is None or not project.images:
            self.iface.messageBar().pushMessage(
                "YOLO Annotator", "工程中无影像可导出", duration=3
            )
            return
        res_text = self.label_res_hint.text()
        dialog = ExportDialog(
            res_text,
            degree_to_m_factor=getattr(self, "_degree_factor", None),
            parent=self,
        )
        if not dialog.exec():
            return
        options = dialog.get_options()
        out_dir = dialog.get_output_dir()

        # 收集（栅格, shapes）任务列表；标注读取前先落盘当前影像。
        # xyz 场景：raster 本身就是场景网格（全图模式=整场景一张 GeoTIFF）
        self.controller.save_current_labels()
        from ..core.raster_io import RasterRef
        from ..core.xyz_source import XyzRaster, XyzSourceConfig
        from .controller import tiles_cache_dir

        image_jobs = []
        for entry in project.images:
            if entry.kind == "xyz":
                # 在线场景：逐场景虚拟影像（瓦片下载）+ 场景级标注
                for scene in entry.scenes:
                    try:
                        raster_ref = XyzRaster(
                            XyzSourceConfig.from_dict(scene.source),
                            scene.map_bbox,
                            scene.zoom,
                            tiles_cache_dir(),
                        )
                    except (ValueError, KeyError) as exc:
                        self.iface.messageBar().pushMessage(
                            "YOLO Annotator",
                            f"跳过在线场景 {scene.name}: {exc}",
                            duration=6,
                        )
                        continue
                    shapes = project.load_image_labels(entry.path, scene_name=scene.name)
                    image_jobs.append((raster_ref, shapes))
                continue
            try:
                raster_ref = RasterRef.open(entry.path)
            except (RuntimeError, OSError) as exc:
                self.iface.messageBar().pushMessage(
                    "YOLO Annotator", f"跳过影像 {entry.path}: {exc}", duration=6
                )
                continue
            shapes = project.load_image_labels(entry.path)
            image_jobs.append((raster_ref, shapes))

        task = ExportTask(image_jobs, [c.name for c in project.classes], out_dir, options)
        task.progress_text.connect(
            lambda msg: self.iface.statusBarIface().showMessage(f"导出: {msg}")
        )
        task.done.connect(self._on_export_done)
        task.failed.connect(lambda reason: QMessageBox.warning(self, "导出", reason))
        QgsApplication.taskManager().addTask(task)
        self._tasks.append(task)

    def _on_export_done(self, stats):
        skipped = "\n".join(stats.skipped_images)
        self.label_export_stats.setText(
            f"导出完成：影像 {stats.image_count}、切片 {stats.chip_count}、"
            f"标注 {stats.label_count}"
            + (f"\n跳过：{skipped}" if skipped else "")
        )
        self.controller.status_message.emit("数据集导出完成")
