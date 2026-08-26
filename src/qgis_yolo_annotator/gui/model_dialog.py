"""模型管理对话框：ONNX 登记 + .pt 权重自动转换（外部 AI 环境可配置）。"""

from __future__ import annotations

from pathlib import Path

from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from qgis.core import QgsApplication

from ..core.inference import VALID_TASKS
from ..core.model_registry import ModelConfig, ModelRegistry, read_onnx_metadata
from ..core.pt_converter import find_ai_env_python, python_has_ultralytics
from .controller import models_cache_dir_hosted

_AI_PY_SETTING = "qgis_yolo_annotator/ai_env_python"


def resolve_ai_env_python() -> str:
    """解析转换用 AI 环境解释器：用户配置（QSettings）优先，否则自动探测。"""
    settings = QSettings()
    configured = settings.value(_AI_PY_SETTING, "", type=str).strip()
    if configured and Path(configured).is_file():
        return configured
    detected = find_ai_env_python()
    return str(detected) if detected is not None else ""


class ModelDialog(QDialog):
    """模型注册表管理（列表 + 表单编辑 + pt 转换）。"""

    def __init__(self, registry: ModelRegistry, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.setWindowTitle("模型管理")
        self.resize(680, 560)
        self._current: ModelConfig | None = None
        self._convert_task = None

        layout = QVBoxLayout(self)

        # ---- AI 环境配置（pt 转换用）
        ai_box = QWidget()
        ai_form = QFormLayout(ai_box)
        ai_row = QHBoxLayout()
        self.edit_ai_python = QLineEdit()
        self.edit_ai_python.setPlaceholderText(
            "选择 .pt 转换用的 Python（需已安装 ultralytics，如 conda ai_env）"
        )
        btn_ai_pick = QPushButton("浏览…")
        btn_ai_detect = QPushButton("检测")
        btn_ai_auto = QPushButton("自动探测")
        btn_ai_pick.clicked.connect(self._pick_ai_python)
        btn_ai_detect.clicked.connect(self._test_ai_python)
        btn_ai_auto.clicked.connect(self._auto_detect_ai_python)
        ai_row.addWidget(self.edit_ai_python, 1)
        ai_row.addWidget(btn_ai_pick)
        ai_row.addWidget(btn_ai_detect)
        ai_row.addWidget(btn_ai_auto)
        ai_form.addRow("AI 环境", ai_row)
        layout.addWidget(ai_box)

        body = QHBoxLayout()
        layout.addLayout(body, stretch=1)

        self.list_widget = QListWidget()
        body.addWidget(self.list_widget, 1)

        form_host = QVBoxLayout()
        body.addLayout(form_host, 1)

        self.edit_name = QLineEdit()
        self.combo_task = QComboBox()
        self.combo_task.addItems(VALID_TASKS)
        self.edit_path = QLineEdit()
        btn_pick = QPushButton("选择模型…")
        btn_pick.clicked.connect(self._pick_model_file)
        path_row = QHBoxLayout()
        path_row.addWidget(self.edit_path, 1)
        path_row.addWidget(btn_pick)
        self.edit_labels = QPlainTextEdit()
        self.edit_labels.setPlaceholderText("每行一个类别（行序=类别id）")
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(64, 4096)
        self.spin_imgsz.setSingleStep(32)
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 0.99)
        self.spin_conf.setSingleStep(0.05)
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.05, 0.95)
        self.spin_iou.setSingleStep(0.05)

        form = QFormLayout()
        form.addRow("名称", self.edit_name)
        form.addRow("任务", self.combo_task)
        form.addRow("模型文件", path_row)
        form.addRow("类别表", self.edit_labels)
        form.addRow("输入尺寸", self.spin_imgsz)
        form.addRow("置信度阈值", self.spin_conf)
        form.addRow("NMS IoU", self.spin_iou)
        form_host.addLayout(form)
        form_host.addStretch(1)

        self.label_meta = QLabel("")
        self.label_meta.setWordWrap(True)
        form_host.addWidget(self.label_meta)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close, Qt.Orientation.Horizontal
        )
        self.btn_add = QPushButton("保存当前")
        self.btn_del = QPushButton("删除")
        buttons.addButton(self.btn_add, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self.btn_del, QDialogButtonBox.ButtonRole.ActionRole)
        self.btn_add.clicked.connect(self._save_current)
        self.btn_del.clicked.connect(self._delete_current)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.list_widget.currentItemChanged.connect(self._on_select)
        self._reload()

        configured = QSettings().value(_AI_PY_SETTING, "", type=str).strip()
        if configured:
            self.edit_ai_python.setText(configured)

    # ------------------------------------------------------------------ AI 环境

    def _persist_ai_python(self) -> str:
        """把输入框路径写入 QSettings 并返回。"""
        path = self.edit_ai_python.text().strip()
        QSettings().setValue(_AI_PY_SETTING, path)
        QSettings().sync()
        return path

    def _pick_ai_python(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "选择 Python 解释器（AI 环境）", "", "Python (python.exe python)"
        )
        if path:
            self.edit_ai_python.setText(str(Path(path).resolve()))
            self._persist_ai_python()
            self._test_ai_python()

    def _auto_detect_ai_python(self):
        detected = find_ai_env_python()
        if detected is None:
            QMessageBox.information(
                self,
                "AI 环境",
                "常见位置未找到带 ultralytics 的 Python，请手工浏览选择。",
            )
            return
        self.edit_ai_python.setText(str(detected))
        self._persist_ai_python()
        self.label_meta.setText(f"已探测 AI 环境: {detected}")

    def _test_ai_python(self):
        """校验输入框解释器可 import ultralytics 并持久化。"""
        path = self._persist_ai_python()
        if not path:
            QMessageBox.warning(self, "AI 环境", "请先填写 Python 路径")
            return
        if not Path(path).is_file():
            QMessageBox.warning(self, "AI 环境", f"文件不存在: {path}")
            return
        if python_has_ultralytics(Path(path)):
            self.label_meta.setText(f"✓ AI 环境可用: {path}")
        else:
            QMessageBox.warning(
                self, "AI 环境", f"该环境无法 import ultralytics:\n{path}"
            )

    # ------------------------------------------------------------------ 列表

    def _reload(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for config in self.registry.list_models():
            item = QListWidgetItem(f"{config.name}  [{config.task}]")
            item.setData(Qt.ItemDataRole.UserRole, config.name)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def _on_select(self, current, _previous):
        if current is None:
            return
        config = self.registry.get(current.data(Qt.ItemDataRole.UserRole))
        if config is None:
            return
        self._current = config
        self.edit_name.setText(config.name)
        self.combo_task.setCurrentText(config.task)
        self.edit_path.setText(config.file_path)
        self.edit_labels.setPlainText("\n".join(config.labels))
        self.spin_imgsz.setValue(config.imgsz)
        self.spin_conf.setValue(config.conf)
        self.spin_iou.setValue(config.iou)

    # ------------------------------------------------------------------ 模型文件

    def _pick_model_file(self):
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "选择模型",
            "",
            "模型文件 (*.onnx *.pt);;ONNX (*.onnx);;YOLO 权重 (*.pt)",
        )
        if not path:
            return
        if not self.edit_name.text().strip():
            self.edit_name.setText(Path(path).stem)
        if path.lower().endswith(".pt"):
            self._start_pt_conversion(path)
            return
        self._fill_from_onnx(path)

    def _fill_from_onnx(self, path: str):
        """读取 onnx metadata 预填表单。"""
        self.edit_path.setText(path)
        meta = read_onnx_metadata(path)
        if meta["task"]:
            self.combo_task.setCurrentText(meta["task"])
        if meta["imgsz"]:
            self.spin_imgsz.setValue(meta["imgsz"])
        if meta["labels"]:
            self.edit_labels.setPlainText("\n".join(meta["labels"]))
            self.label_meta.setText(f"已从模型 metadata 读取 {len(meta['labels'])} 个类别")
        else:
            self.label_meta.setText("模型未内嵌类别表，请手工填写（或从 classes.txt 粘贴）")

    # ------------------------------------------------------------------ pt 转换

    def _start_pt_conversion(self, pt_path: str):
        """启动后台 pt→onnx 转换（需可用 AI 环境）。"""
        python_exe = self.edit_ai_python.text().strip() or resolve_ai_env_python()
        if not python_exe or not Path(python_exe).is_file():
            QMessageBox.warning(
                self,
                "pt 转换",
                "未配置 AI 环境解释器。\n请在上方「AI 环境」选择带 ultralytics 的 Python（如 conda ai_env 的 python.exe）。",
            )
            return
        self.edit_ai_python.setText(python_exe)
        self._persist_ai_python()

        from ..tasks.pt_convert_task import PtConvertTask

        self.btn_add.setEnabled(False)
        self.label_meta.setText(f"⏳ 转换 {Path(pt_path).name} 中（后台执行，可稍候）…")
        task = PtConvertTask(
            pt_path=pt_path,
            python_exe=python_exe,
            fallback_imgsz=self.spin_imgsz.value(),
            cache_dir=str(models_cache_dir_hosted()),
        )
        task.succeeded.connect(self._on_pt_converted)
        task.failed.connect(self._on_pt_convert_failed)
        task.progress_text.connect(self.label_meta.setText)
        QgsApplication.taskManager().addTask(task)
        self._convert_task = task

    def _on_pt_converted(self, result: dict):
        """转换成功：用缓存 onnx 填表单。"""
        self.btn_add.setEnabled(True)
        self._fill_from_onnx(result["onnx_path"])
        self.label_meta.setText(
            f"✓ pt 转换完成 → {Path(result['onnx_path']).name}（imgsz={result['imgsz']}，"
            f"{len(result['labels'])} 类别）。点「保存当前」登记。"
        )

    def _on_pt_convert_failed(self, reason: str):
        self.btn_add.setEnabled(True)
        QMessageBox.warning(self, "pt 转换", f"转换失败:\n{reason}")

    # ------------------------------------------------------------------ 保存/删除

    def _save_current(self):
        name = self.edit_name.text().strip()
        path = self.edit_path.text().strip()
        if not name or not path:
            QMessageBox.warning(self, "模型管理", "名称与模型文件不能为空")
            return
        if not Path(path).is_file():
            QMessageBox.warning(self, "模型管理", f"文件不存在: {path}")
            return
        labels = [
            line for line in self.edit_labels.toPlainText().splitlines() if line.strip()
        ]
        config = ModelConfig(
            name=name,
            task=self.combo_task.currentText(),
            file_path=path,
            labels=labels,
            imgsz=self.spin_imgsz.value(),
            conf=self.spin_conf.value(),
            iou=self.spin_iou.value(),
        )
        try:
            self.registry.add(config)
        except ValueError as exc:
            QMessageBox.warning(self, "模型管理", str(exc))
            return
        self._reload()

    def _delete_current(self):
        if self._current is None:
            return
        if self.registry.remove(self._current.name):
            self._current = None
            self._reload()
