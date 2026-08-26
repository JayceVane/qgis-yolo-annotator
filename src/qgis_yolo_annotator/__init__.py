"""qgis_yolo_annotator — QGIS 遥感影像 YOLO 智能标注插件。

入口模块：QGIS 通过 classFactory 加载插件。
"""

from .plugin import YoloAnnotatorPlugin


def classFactory(iface):  # noqa: N802  QGIS 约定的工厂函数名
    """QGIS 插件工厂。

    Args:
        iface: QgisInterface，QGIS 主界面接口。

    Returns:
        YoloAnnotatorPlugin: 插件实例。
    """
    return YoloAnnotatorPlugin(iface)
