"""数据集导出后台任务（QgsTask）。"""

from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog, QgsTask
from qgis.PyQt.QtCore import pyqtSignal

from ..core.raster_io import RasterRef
from ..export.chip_export import ExportOptions, ExportStats, export_dataset

_TAB = "YOLO Annotator"


class ExportTask(QgsTask):
    """对工程影像集合执行切片导出。

    Signals:
        progress_text(str): 进度消息。
        done(object): ExportStats。
        failed(str): 失败原因。
    """

    progress_text = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        image_jobs: list[tuple[RasterRef, list]],
        class_names: list[str],
        out_dir: str,
        options: ExportOptions,
    ):
        super().__init__(f"导出 {options.format} 数据集", QgsTask.Flag.CanCancel)
        self._jobs = image_jobs
        self._class_names = class_names
        self._out_dir = out_dir
        self._options = options
        self._exception: Exception | None = None

    def run(self) -> bool:
        """后台执行导出（GDAL 句柄线程内重开，见 infer_task 同款说明）。"""
        try:
            for raster_ref, _shapes in self._jobs:
                raster_ref.close()
            stats = export_dataset(
                self._jobs,
                self._class_names,
                self._out_dir,
                self._options,
                progress_cb=lambda done, total, msg: (
                    self.setProgress(int(100 * done / max(total, 1))),
                    self.progress_text.emit(msg),
                ),
            )
            self._stats = stats
            return True
        except Exception as exc:  # noqa: BLE001  后台线程需捕获全部异常回传
            self._exception = exc
            return False

    def finished(self, success: bool):
        """主线程回调。"""
        if success:
            stats: ExportStats = getattr(self, "_stats", ExportStats())
            self.done.emit(stats)
        else:
            reason = str(self._exception) if self._exception else "已取消"
            self.failed.emit(reason)
            QgsMessageLog.logMessage(
                f"导出失败: {reason}", _TAB, Qgis.MessageLevel.Warning
            )
