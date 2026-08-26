"""类别管理对话框：增删改 / 颜色 / 快捷键。"""

from __future__ import annotations

from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from ..core.project import ClassDef


class ClassDialog(QDialog):
    """工程类别表编辑（保存后由调用方重建渲染）。"""

    def __init__(self, classes: list[ClassDef], parent=None):
        super().__init__(parent)
        self.setWindowTitle("类别管理")
        self.resize(620, 400)
        self.classes = [ClassDef(c.name, c.color, c.hotkey) for c in classes]
        self._selected_color = ""

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.currentItemChanged.connect(self._on_select)
        layout.addWidget(self.list_widget, 1)

        add_row = QHBoxLayout()
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("新类别名称")
        self.btn_color = QPushButton("颜色…")
        self.btn_color.clicked.connect(self._pick_color)
        self.btn_add = QPushButton("添加")
        self.btn_add.clicked.connect(self._add)
        self.btn_rename = QPushButton("重命名选中")
        self.btn_rename.clicked.connect(self._rename)
        self.btn_del = QPushButton("删除选中")
        self.btn_del.clicked.connect(self._delete)
        self.btn_auto = QPushButton("重置自动配色")
        self.btn_auto.setToolTip("按行序用 X-AnyLabeling 调色板重排全部类别颜色")
        self.btn_auto.clicked.connect(self._reassign_colors)
        add_row.addWidget(self.edit_name, 1)
        add_row.addWidget(self.btn_color)
        add_row.addWidget(self.btn_add)
        add_row.addWidget(self.btn_rename)
        add_row.addWidget(self.btn_del)
        add_row.addWidget(self.btn_auto)
        layout.addLayout(add_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._reload()

    def _reload(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for class_def in self.classes:
            item = QListWidgetItem(
                f"{class_def.name}    快捷键: {class_def.hotkey or '-'}"
            )
            if class_def.color:
                item.setForeground(QColor(class_def.color))
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def _on_select(self, current, _prev):
        if current is None:
            return
        row = self.list_widget.row(current)
        self.edit_name.setText(self.classes[row].name)
        self._selected_color = self.classes[row].color

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._selected_color or "#e6194b"), self)
        if color.isValid():
            self._selected_color = color.name()

    def _add(self):
        name = self.edit_name.text().strip()
        if not name:
            return
        for class_def in self.classes:
            if class_def.name == name:
                return
        self.classes.append(ClassDef(name, self._selected_color, ""))
        self._auto_hotkeys()
        self._reload()

    def _rename(self):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        name = self.edit_name.text().strip()
        if name:
            self.classes[row].name = name
            if self._selected_color:
                self.classes[row].color = self._selected_color
            self._reload()

    def _delete(self):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.classes):
            self.classes.pop(row)
            self._reload()

    def _auto_hotkeys(self):
        """为无快捷键类别分配未占用的数字键。"""
        for index, class_def in enumerate(self.classes):
            if class_def.hotkey:
                continue
            for digit in "123456789":
                if all(c.hotkey != digit for c in self.classes):
                    class_def.hotkey = digit
                    break

    def _reassign_colors(self):
        """按行序用 X-AnyLabeling 调色板重排颜色。"""
        from ..core.project import XANYLABELING_PALETTE

        for index, class_def in enumerate(self.classes):
            class_def.color = XANYLABELING_PALETTE[index % len(XANYLABELING_PALETTE)]
        self._reload()
