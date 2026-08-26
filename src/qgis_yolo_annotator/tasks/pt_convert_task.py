"""pt→onnx 转换后台任务（QgsTask）。"""

from __future__ import annotations

from pathlib import Path

from qgis.core import Qgis, QgsMessageLog, QgsTask
from qgis.PyQt.QtCore import pyqtSignal

from ..core.pt_converter import convert_pt_to_onnx, read_pt_metadata

_TAB = "YOLO Annotator"


class PtConvertTask(QgsTask):
    """读取 pt 元数据并导出 ONNX（后台线程，子进程复用外部 AI 环境）。

    Signals:
        progress_text(str): 进度消息。
        succeeded(object): 转换结果 dict（onnx_path / task / labels / imgsz）。
        failed(str): 失败原因（含子进程 stderr 摘要）。
    """

    progress_text = pyqtSignal(str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        pt_path: str,
        python_exe: str,
        fallback_imgsz: int,
        cache_dir: str,
    ):
        super().__init__(f"转换 {Path(pt_path).name}", QgsTask.Flag.CanCancel)
        self._pt_path = pt_path
        self._python = python_exe
        self._fallback_imgsz = int(fallback_imgsz)
        self._cache_dir = cache_dir
        self._result: dict = {}
        self._exception: Exception | None = None

    def run(self) -> bool:
        """后台执行：读 pt 元数据（task/类别/训练尺寸）→ 导出 ONNX 落缓存。"""
        try:
            self.progress_text.emit(f"读取 {Path(self._pt_path).name} 元数据…")
            self.setProgress(5)
            meta = read_pt_metadata(self._pt_path, self._python)
            imgsz = meta.get("imgsz") or self._fallback_imgsz
            self.progress_text.emit(f"导出 ONNX（imgsz={imgsz}），模型较大时需数十秒…")
            self.setProgress(15)
            onnx_path = convert_pt_to_onnx(
                self._pt_path, self._python, imgsz, self._cache_dir
            )
            self._result = {
                "onnx_path": str(onnx_path),
                "task": meta.get("task"),
                "labels": meta.get("labels") or [],
                "imgsz": imgsz,
            }
            self.setProgress(100)
            return True
        except Exception as exc:  # noqa: BLE001  后台线程需捕获全部异常回传
            self._exception = exc
            return False

    def finished(self, success: bool):
        """主线程回调：分发结果。"""
        if success:
            self.progress_text.emit("转换完成")
            self.succeeded.emit(self._result)
        else:
            reason = str(self._exception) if self._exception else "已取消"
            self.failed.emit(reason)
            QgsMessageLog.logMessage(
                f"pt 转换失败: {reason}", _TAB, Qgis.MessageLevel.Warning
            )
