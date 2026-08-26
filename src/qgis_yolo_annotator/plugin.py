"""插件入口：注册菜单、工具栏与主 Dock 面板。"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QDockWidget

from .gui.main_dock import MainDock


class YoloAnnotatorPlugin:
    """YOLO 智能标注插件生命周期管理。"""

    def __init__(self, iface):
        self.iface = iface
        self._dock: QDockWidget | None = None
        self._panel: MainDock | None = None
        self._action: QAction | None = None

    def initGui(self):  # noqa: N802  QGIS 约定的初始化回调
        """QGIS 加载插件时创建 UI 入口。"""
        self._action = QAction(QIcon(), "YOLO 智能标注", self.iface.mainWindow())
        self._action.setToolTip("打开 YOLO 智能标注面板")
        self._action.setCheckable(True)
        self._action.triggered.connect(self._toggle_dock)
        self.iface.addPluginToMenu("&YOLO Annotator", self._action)
        self.iface.pluginToolBar().addAction(self._action)

    def unload(self):
        """QGIS 卸载插件时清理资源。"""
        if self._panel is not None:
            self._panel.controller.save_project()
        if self._dock is not None:
            self.iface.removeDockWidget(self._dock)
            self._dock.deleteLater()
            self._dock = None
            self._panel = None
        if self._action is not None:
            self.iface.removePluginMenu("&YOLO Annotator", self._action)
            self.iface.pluginToolBar().removeAction(self._action)
            self._action = None

    def _toggle_dock(self, checked: bool):
        if self._dock is None:
            self._panel = MainDock(self.iface)
            self._dock = QDockWidget("YOLO 智能标注", self.iface.mainWindow())
            self._dock.setObjectName("YoloAnnotatorDock")
            self._dock.setWidget(self._panel)
            self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock)
            self._dock.visibilityChanged.connect(self._action.setChecked)
        self._dock.setVisible(checked)
