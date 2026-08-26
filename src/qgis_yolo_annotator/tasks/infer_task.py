"""选中场景的滑窗推理后台任务（QgsTask）。"""

from __future__ import annotations

from qgis.core import Qgis, QgsMessageLog, QgsTask
from qgis.PyQt.QtCore import pyqtSignal

from ..core.model_registry import ModelConfig
from ..core.project import SceneDef
from ..core.raster_io import RasterRef
from ..core.scene_infer import SceneInferOptions, infer_scene

_TAB = "YOLO Annotator"


class SceneInferTask(QgsTask):
    """对一组场景执行滑窗推理，完成后由上层把 shapes 写入标注图层。

    Signals:
        progress_text(str): 进度消息。
        scene_done(str, list, object): 单场景完成（场景名, shapes 像素坐标, 栅格对象）。
        failed(str): 任务失败原因。
    """

    progress_text = pyqtSignal(str)
    scene_done = pyqtSignal(str, list, object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        description: str,
        raster: RasterRef,
        scenes: list[SceneDef],
        model_session,
        model_cfg: ModelConfig,
        options: SceneInferOptions,
    ):
        super().__init__(description, QgsTask.Flag.CanCancel)
        self._raster = raster
        self._scenes = scenes
        self._session = model_session
        self._cfg = model_cfg
        self._options = options
        self._exception: Exception | None = None

    def run(self) -> bool:
        """后台执行（QgsTask 线程）：逐场景滑窗推理。

        GDAL Dataset 跨线程不安全：进入任务线程后关闭主线程句柄，
        由 read_window_bgr 在本线程惰性重开。

        Returns:
            是否成功完成（失败信息经 failed 信号传递）。
        """
        try:
            self._raster.close()
            for index, scene in enumerate(self._scenes):
                if self.isCanceled():
                    return False

                def _progress(done: int, total: int, message: str):
                    self.setProgress(
                        int(100 * (index + done / max(total, 1)) / len(self._scenes))
                    )
                    self.progress_text.emit(message)

                shapes = infer_scene(
                    self._raster,
                    scene,
                    self._session,
                    self._cfg,
                    self._options,
                    progress_cb=_progress,
                )
                self.scene_done.emit(scene.name, shapes, self._raster)
            return True
        except Exception as exc:  # noqa: BLE001  后台线程需捕获全部异常回传
            self._exception = exc
            return False

    def finished(self, success: bool):
        """主线程回调：报告结果。"""
        if success:
            self.progress_text.emit(
                f"推理完成：{len(self._scenes)} 个场景（{self._cfg.name}）"
            )
        else:
            reason = str(self._exception) if self._exception else "已取消"
            self.failed.emit(reason)
            QgsMessageLog.logMessage(
                f"场景推理失败: {reason}", _TAB, Qgis.MessageLevel.Warning
            )
