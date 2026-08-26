"""类别管理对话框（仿 X-AnyLabeling）：表格化编辑 + 排序 + 自定义快捷键。

能力：
- 表格列：色块 / 类别名 / 数字快捷键
- 添加、删除、重命名（单元格编辑）
- 上移/下移调整顺序（行序 = YOLO 导出的类别 id，同步生效）
- 双击颜色格弹取色器；双击快捷键格直接键入新键
- 「重置自动配色」按行序套用 X-AnyLabeling 31 色表

保存语义：确定后由调用方把 self.classes 写回工程并重建渲染。
"""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QBrush
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..core.project import ClassDef, XANYLABELING_PALETTE

_COLS = ("颜色", "类别名", "快捷键")
_COL_COLOR, _COL_NAME, _COL_HOTKEY = 0, 1, 2


class ClassDialog(QDialog):
    """工程类别表编辑（确定后由调用方重建渲染与持久化）。"""

    def __init__(self, classes: list[ClassDef], parent=None):
        super().__init__(parent)
        self.setWindowTitle("类别管理")
        self.resize(520, 460)
        # 深拷贝：取消时不动原数据
        self.classes = [ClassDef(c.name, c.color, c.hotkey) for c in classes]

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.horizontalHeader().setSectionResizeMode(_COL_COLOR, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(_COL_NAME, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(_COL_HOTKEY, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.table, 1)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.table.cellChanged.connect(self._on_cell_changed)

        edit_row = QHBoxLayout()
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("新类别名称")
        self.btn_add = QPushButton("添加")
        self.btn_add.clicked.connect(self._add)
        btn_del = QPushButton("删除选中")
        btn_del.clicked.connect(self._delete)
        btn_up = QPushButton("上移 ↑")
        btn_up.clicked.connect(lambda: self._move(-1))
        btn_down = QPushButton("下移 ↓")
        btn_down.clicked.connect(lambda: self._move(1))
        btn_auto = QPushButton("重置自动配色")
        btn_auto.setToolTip("按行序用 X-AnyLabeling 调色板重排全部类别颜色")
        btn_auto.clicked.connect(self._reassign_colors)
        for w in (self.edit_name, self.btn_add, btn_del, btn_up, btn_down, btn_auto):
            edit_row.addWidget(w)
        layout.addLayout(edit_row)

        hint = QLabel(
            "行序 = YOLO/DOTA 导出的类别 id；上移/下移即调整 id。"
            "双击「颜色」换色，双击「快捷键」输入单字符（画布选中后按键快速改类）。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._loading = False
        self._reload()

    # ------------------------------------------------------------------ 表格

    def _reload(self):
        """从 self.classes 全量重建表格。"""
        self._loading = True
        self.table.setRowCount(len(self.classes))
        for row, class_def in enumerate(self.classes):
            color_item = QTableWidgetItem()
            color_item.setBackground(QBrush(QColor(class_def.color or "#ff77ff")))
            color_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            color_item.setData(Qt.ItemDataRole.UserRole, class_def.color)
            self.table.setItem(row, _COL_COLOR, color_item)

            name_item = QTableWidgetItem(class_def.name)
            name_item.setForeground(QBrush(QColor(class_def.color or "#ffffff")))
            self.table.setItem(row, _COL_NAME, name_item)

            hotkey_item = QTableWidgetItem(class_def.hotkey)
            self.table.setItem(row, _COL_HOTKEY, hotkey_item)
        self._loading = False

    def _on_cell_double_clicked(self, row: int, col: int):
        if col == _COL_COLOR:
            current = QColor(self.classes[row].color or "#ff77ff")
            color = QColorDialog.getColor(current, self, "类别颜色")
            if color.isValid():
                self.classes[row].color = color.name()
                self._reload()

    def _on_cell_changed(self, row: int, col: int):
        if self._loading or not (0 <= row < len(self.classes)):
            return
        text = self.table.item(row, col).text().strip()
        if col == _COL_NAME:
            if text and self._name_available(text, except_row=row):
                self.classes[row].name = text
                self.table.item(row, _COL_NAME).setForeground(
                    QBrush(QColor(self.classes[row].color or "#ffffff"))
                )
            else:
                self._flash_cell(row, col)
                self._reload()  # 还原名
                return
        elif col == _COL_HOTKEY:
            key = text[:1]  # 只取一个字符
            if key and self._hotkey_available(key, except_row=row):
                self.classes[row].hotkey = key
            else:
                self._flash_cell(row, col)
                self._reload()
                return
            item = self.table.item(row, _COL_HOTKEY)
            if item.text() != self.classes[row].hotkey:
                item.setText(self.classes[row].hotkey)

    def _flash_cell(self, row: int, col: int):
        """非法输入的行提示（背景短暂变红由 reload 复位）。"""
        item = self.table.item(row, col)
        if item is not None:
            item.setBackground(QBrush(QColor(255, 90, 90)))

    def _name_available(self, name: str, except_row: int) -> bool:
        return all(c.name != name for i, c in enumerate(self.classes) if i != except_row)

    def _hotkey_available(self, key: str, except_row: int) -> bool:
        return all(c.hotkey != key for i, c in enumerate(self.classes) if i != except_row)

    # ------------------------------------------------------------------ 操作

    def _add(self):
        name = self.edit_name.text().strip()
        if not name:
            return
        if not self._name_available(name, -1):
            QMessageBox.warning(self, "类别管理", f"类别已存在: {name}")
            return
        color = XANYLABELING_PALETTE[len(self.classes) % len(XANYLABELING_PALETTE)]
        hotkey = ""
        for digit in "123456789":
            if all(c.hotkey != digit for c in self.classes):
                hotkey = digit
                break
        self.classes.append(ClassDef(name, color, hotkey))
        self.edit_name.clear()
        self._reload()

    def _delete(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.classes):
            self.classes.pop(row)
            self._reload()

    def _move(self, delta: int):
        row = self.table.currentRow()
        target = row + delta
        if not (0 <= row < len(self.classes)) or not (0 <= target < len(self.classes)):
            return
        self.classes[row], self.classes[target] = (
            self.classes[target],
            self.classes[row],
        )
        self._reload()
        self.table.selectRow(target)

    def _reassign_colors(self):
        for index, class_def in enumerate(self.classes):
            class_def.color = XANYLABELING_PALETTE[index % len(XANYLABELING_PALETTE)]
        self._reload()
