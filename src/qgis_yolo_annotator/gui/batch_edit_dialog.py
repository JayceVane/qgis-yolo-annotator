"""批量编辑对话框：类别 A→B 替换（当前影像 / 整个工程），带数量预览。"""

from __future__ import annotations

from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)


class BatchEditDialog(QDialog):
    """收集替换参数；accept 后经 get_old()/get_new()/is_project_wide() 取值。"""

    def __init__(self, class_names: list[str], counts: dict[str, int], parent=None):
        """初始化。

        Args:
            class_names: 工程类别名列表。
            counts: 各类别当前标注数量（预览展示；缺省为 0）。
        """
        super().__init__(parent)
        self.setWindowTitle("批量编辑类别")
        self.resize(400, 240)
        self._counts = counts

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.combo_old = QComboBox()
        self.combo_new = QComboBox()
        for name in class_names:
            self.combo_old.addItem(self._label(name), name)
            self.combo_new.addItem(self._label(name), name)
        if class_names:
            self.combo_new.setCurrentIndex(1 if len(class_names) > 1 else 0)
        self.combo_old.currentIndexChanged.connect(self._sync_hint)
        form.addRow("原类别", self.combo_old)
        form.addRow("改为", self.combo_new)
        layout.addLayout(form)

        box_scope = QGroupBox("作用范围")
        scope_layout = QVBoxLayout(box_scope)
        self.radio_current = QRadioButton("仅当前影像 / 场景")
        self.radio_project = QRadioButton("整个工程（所有标注 JSON）")
        self.radio_current.setChecked(True)
        scope_layout.addWidget(self.radio_current)
        scope_layout.addWidget(self.radio_project)
        layout.addWidget(box_scope)

        self.label_hint = QLabel("")
        self.label_hint.setWordWrap(True)
        layout.addWidget(self.label_hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("替换")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._sync_hint()

    def _label(self, name: str) -> str:
        count = self._counts.get(name, 0)
        return f"{name}（{count} 个标注）"

    def get_old(self) -> str:
        return self.combo_old.currentData()

    def get_new(self) -> str:
        return self.combo_new.currentData()

    def is_project_wide(self) -> bool:
        return self.radio_project.isChecked()

    def _sync_hint(self):
        old = self.get_old()
        self.label_hint.setText(
            f"「{old}」当前共 {self._counts.get(old, 0)} 个标注将被改为"
            f"「{self.get_new()}」。"
        )

    def _on_accept(self):
        if self.get_old() == self.get_new():
            from qgis.PyQt.QtWidgets import QMessageBox

            QMessageBox.warning(self, "批量编辑", "原类别与目标类别相同")
            return
        self.accept()
