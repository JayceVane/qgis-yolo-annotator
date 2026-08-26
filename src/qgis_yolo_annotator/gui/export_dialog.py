"""数据集导出对话框：格式 / 切片 / 分辨率重采样 / 划分参数。"""

from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..export import converters
from ..export.chip_export import EXPORT_FORMATS, ExportOptions

_FORMAT_LABELS = {
    "dota": "DOTA（像素角点 + difficult）",
    "yolo_obb": "YOLO-OBB（归一化角点）",
    "yolo_det": "YOLO-det（外接 HBB）",
    "voc": "VOC XML（OBB→HBB / polygon）",
}


class ExportDialog(QDialog):
    """导出参数收集器；accept 后通过 get_options()/get_output_dir() 取结果。"""

    def __init__(self, current_res_text: str = "", degree_to_m_factor: float | None = None, parent=None):
        """初始化导出对话框。

        Args:
            current_res_text: 当前影像分辨率提示文本。
            degree_to_m_factor: 地理影像中心纬度处的米/度系数（None 时隐藏度单位）。
        """
        super().__init__(parent)
        self.setWindowTitle("导出数据集")
        self.resize(520, 560)
        self._out_dir = ""
        self._degree_to_m = degree_to_m_factor

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.combo_format = QComboBox()
        for key in EXPORT_FORMATS:
            self.combo_format.addItem(_FORMAT_LABELS[key], key)
        self.combo_format.setCurrentIndex(0)
        self.combo_format.currentIndexChanged.connect(self._sync_visibility)
        form.addRow("标注格式", self.combo_format)
        layout.addLayout(form)

        # ---- 切片参数
        box_chip = QGroupBox("切片")
        grid_chip = QGridLayout(box_chip)
        self.spin_chip = QSpinBox()
        self.spin_chip.setRange(32, 8192)
        self.spin_chip.setSingleStep(128)
        self.spin_chip.setValue(1024)
        self.check_full_image = QCheckBox("全图导出（不切片）")
        self.check_full_image.toggled.connect(self._sync_visibility)
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 4096)
        self.spin_overlap.setSingleStep(50)
        self.spin_overlap.setValue(200)
        grid_chip.addWidget(QLabel("chip 尺寸(px)"), 0, 0)
        grid_chip.addWidget(self.spin_chip, 0, 1)
        grid_chip.addWidget(QLabel("重叠(px)"), 0, 2)
        grid_chip.addWidget(self.spin_overlap, 0, 3)
        grid_chip.addWidget(self.check_full_image, 1, 0, 1, 4)
        layout.addWidget(box_chip)

        # ---- 分辨率
        box_res = QGroupBox("输出地面分辨率（重采样）")
        grid_res = QGridLayout(box_res)
        self.check_resample = QCheckBox("指定目标分辨率")
        self.check_resample.toggled.connect(self._sync_visibility)
        self.spin_res = QDoubleSpinBox()
        self.spin_res.setDecimals(6)
        self.spin_res.setRange(0.000001, 1000.0)
        self.spin_res.setValue(0.2)
        self.spin_res.setSingleStep(0.05)
        self.combo_unit = QComboBox()
        self.combo_unit.addItem("m/px（米/像素）", "m")
        if degree_to_m_factor is not None:
            self.combo_unit.addItem("°/px（度/像素，仅地理坐标影像）", "degree")
        self.label_res_hint = QLabel(current_res_text or "（未加载影像）")
        self.label_res_hint.setWordWrap(True)
        grid_res.addWidget(self.check_resample, 0, 0, 1, 2)
        grid_res.addWidget(self.spin_res, 1, 0)
        grid_res.addWidget(self.combo_unit, 1, 1)
        grid_res.addWidget(self.label_res_hint, 2, 0, 1, 2)
        layout.addWidget(box_res)

        # ---- 其他选项
        box_misc = QGroupBox("输出选项")
        form_misc = QFormLayout(box_misc)
        self.check_geo = QCheckBox("GeoTIFF（切片携带地理信息）")
        self.check_geo.setChecked(True)
        self.check_geo.toggled.connect(self._sync_visibility)
        self.combo_boundary = QComboBox()
        self.combo_boundary.addItem("clip（越界裁剪保留）", converters.BOUNDARY_CLIP)
        self.combo_boundary.addItem("skip（越界丢弃）", converters.BOUNDARY_SKIP)
        self.combo_voc_mode = QComboBox()
        self.combo_voc_mode.addItem("外接水平框 bndbox", "hbb")
        self.combo_voc_mode.addItem("四点 polygon", "polygon")
        self.check_clipped_difficult = QCheckBox("被窗口裁剪的目标标记 difficult")
        self.check_clipped_difficult.setChecked(True)
        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.0, 0.9)
        self.spin_val.setSingleStep(0.05)
        self.spin_val.setValue(0.2)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        form_misc.addRow(self.check_geo)
        form_misc.addRow("越界目标", self.combo_boundary)
        form_misc.addRow("VOC OBB 表示", self.combo_voc_mode)
        form_misc.addRow(self.check_clipped_difficult)
        form_misc.addRow("验证集比例", self.spin_val)
        form_misc.addRow("随机种子", self.spin_seed)
        layout.addWidget(box_misc)

        # ---- 输出目录
        out_row = QHBoxLayout()
        self.edit_out = QLineEdit()
        btn_out = QPushButton("选择目录…")
        btn_out.clicked.connect(self._pick_dir)
        out_row.addWidget(self.edit_out, 1)
        out_row.addWidget(btn_out)
        out_box = QWidget()
        out_box.setLayout(out_row)
        form_out = QFormLayout()
        form_out.addRow("输出目录", out_box)
        layout.addLayout(form_out)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._sync_visibility()

    # ------------------------------------------------------------------ 结果

    def get_options(self) -> ExportOptions:
        """构建导出配置（accept 后有效；度单位自动换算为 m）。"""
        target_res = None
        if self.check_resample.isChecked():
            target_res = self.spin_res.value()
            if self.combo_unit.currentData() == "degree":
                if not self._degree_to_m:
                    raise ValueError("当前影像不支持度/px 单位")
                target_res = target_res * self._degree_to_m
        return ExportOptions(
            format=self.combo_format.currentData(),
            chip_size=None if self.check_full_image.isChecked() else self.spin_chip.value(),
            overlap=self.spin_overlap.value(),
            target_res_m=target_res,
            geo_tiff=self.check_geo.isChecked(),
            boundary_policy=self.combo_boundary.currentData(),
            voc_obb_mode=self.combo_voc_mode.currentData(),
            val_ratio=self.spin_val.value(),
            seed=self.spin_seed.value(),
            mark_clipped_difficult=self.check_clipped_difficult.isChecked(),
        )

    def get_output_dir(self) -> str:
        return self._out_dir

    # ------------------------------------------------------------------ 内部

    def _sync_visibility(self):
        self.spin_overlap.setEnabled(not self.check_full_image.isChecked())
        self.spin_chip.setEnabled(not self.check_full_image.isChecked())
        self.spin_res.setEnabled(self.check_resample.isChecked())
        self.combo_unit.setEnabled(self.check_resample.isChecked())
        is_voc = self.combo_format.currentData() == "voc"
        self.combo_voc_mode.setEnabled(is_voc)

    def _pick_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self.edit_out.setText(directory)

    def _on_accept(self):
        self._out_dir = self.edit_out.text().strip()
        if not self._out_dir:
            self.edit_out.setFocus()
            return
        try:
            self.get_options()
        except ValueError as exc:
            from qgis.PyQt.QtWidgets import QMessageBox

            QMessageBox.warning(self, "导出", f"参数错误: {exc}")
            return
        self.accept()
