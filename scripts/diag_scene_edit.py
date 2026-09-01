"""update_scene_extent 无头实测：临时工程验证 xyz 标注网格迁移与文件场景换算。

独立 QgsApplication 进程运行，不触碰正在使用的 QGIS 会话与真实工程。
"""

import json
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsProject,
    QgsRectangle,
)

app = QgsApplication([], False)
app.initQgis()

try:
    from qgis_yolo_annotator.core.xyz_source import (
        XyzSourceConfig,
        meters_per_pixel,
    )
    from qgis_yolo_annotator.gui.controller import Controller

    class _FakeMapSettings:
        def destinationCrs(self):
            return QgsCoordinateReferenceSystem("EPSG:3857")

    class _FakeCanvas:
        def __init__(self):
            self.extents = []
            self.refreshed = 0

        def mapSettings(self):
            return _FakeMapSettings()

        def setExtent(self, rect):
            self.extents.append(QgsRectangle(rect))

        def refresh(self):
            self.refreshed += 1

    class _FakeIface:
        def __init__(self):
            self._canvas = _FakeCanvas()

        def mapCanvas(self):
            return self._canvas

        def setActiveLayer(self, _layer):
            pass

    root = SRC.parent / ".tmp" / "scene_edit_headless"
    if root.exists():
        shutil.rmtree(root)

    ctrl = Controller(_FakeIface())
    root.mkdir(parents=True, exist_ok=True)
    from qgis_yolo_annotator.core.project import AnnotationProject

    ctrl.project = AnnotationProject.create(root, "headless 测试工程")
    ctrl.project_changed.emit()
    cfg = XyzSourceConfig(
        url_template="https://example.com/{z}/{x}/{y}.png", title="t"
    )
    ctrl.attach_xyz_workset(cfg, "t")
    scene = ctrl.project.add_scene(
        "xyz://t",
        [0, 0, 0, 0],
        kind="xyz",
        map_bbox=[1000.0, 2000.0, 1100.0, 2100.0],
        zoom=20,
        source=cfg.to_dict(),
    )
    ctrl._rebuild_scene_features()
    ctrl.load_scene(scene)

    shapes = [
        {
            "label": "car",
            "shape_type": "rotation",
            "points": [
                [10.0, 10.0],
                [20.0, 10.0],
                [20.0, 30.0],
                [10.0, 30.0],
            ],
        }
    ]
    ctrl.project.save_image_labels("xyz://t", shapes, 100, 100, scene_name=scene.name)
    ctrl._rebuild_annotation_features()

    res = meters_per_pixel(20)  # 3857 单位/像素
    # ---- 用例 1：向西扩 50 m、向北扩 25 m（原点变更 → 标注需迁移）
    assert ctrl.update_scene_extent(
        scene.name, QgsRectangle(950.0, 2000.0, 1100.0, 2125.0)
    ), "用例 1 应成功"
    assert scene.map_bbox == [950.0, 2000.0, 1100.0, 2125.0], scene.map_bbox
    exp_w, exp_h = 150.0 / res, 125.0 / res
    assert abs(scene.bbox[2] - exp_w) < 1e-6 and abs(scene.bbox[3] - exp_h) < 1e-6
    doc = json.loads(
        (root / "labels" / f"{scene.name}.json").read_text(encoding="utf-8")
    )
    p0 = doc["shapes"][0]["points"][0]
    assert abs(p0[0] - (10.0 + 50.0 / res)) < 1e-6, p0
    assert abs(p0[1] - (10.0 + 25.0 / res)) < 1e-6, p0
    # 虚拟网格与标注图层随新网格重建
    assert ctrl.raster is not None and ctrl.current_scene.name == scene.name
    assert ctrl.ann_layer.featureCount() == 1
    print(f"用例 1 OK：西/北扩 → 标注平移 (+{50.0/res:.1f}, +{25.0/res:.1f}) px")

    # ---- 用例 2：整体收缩平移 → 标注越出新范围，保留并计数
    assert ctrl.update_scene_extent(
        scene.name, QgsRectangle(970.0, 2010.0, 1090.0, 2110.0)
    )
    doc = json.loads(
        (root / "labels" / f"{scene.name}.json").read_text(encoding="utf-8")
    )
    assert len(doc["shapes"]) == 1, "越界标注应保留"
    print("用例 2 OK：收缩后标注保留")

    # ---- 用例 3：范围过小被拒
    assert not ctrl.update_scene_extent(
        scene.name, QgsRectangle(1000.0, 2000.0, 1000.01, 2000.01)
    )
    print("用例 3 OK：过小范围被拒绝")

    # ---- 用例 3.5：无网格状态（未推理/未加载场景）下 ensure_scene_context 自动恢复
    ctrl.current_scene = None
    if ctrl.raster is not None:
        ctrl.raster.close()
    ctrl.raster = None
    assert ctrl.ensure_scene_context(), "ensure_scene_context 应自动加载场景"
    assert ctrl.raster is not None and ctrl.current_scene is not None
    assert ctrl.ann_layer.featureCount() == 1, "恢复后标注图层应载入迁移后 JSON"
    print("用例 3.5 OK：无网格状态自动恢复场景上下文")

    # ---- 用例 4：文件影像场景（identity geotiff，像素=地图）
    import numpy as np
    from osgeo import gdal

    tif = root / "img.tif"
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(tif), 200, 100, 3, gdal.GDT_Byte)
    ds.SetGeoTransform((0.0, 1.0, 0.0, 100.0, 0.0, -1.0))
    ds.SetProjection(
        QgsCoordinateReferenceSystem("EPSG:3857").toWkt()
    )
    ds.WriteRaster(0, 0, 200, 100, np.zeros((100, 200, 3), np.uint8).tobytes())
    ds = None
    ctrl.load_image(str(tif))
    fscene = ctrl.project.add_scene(ctrl.current_image, [10.0, 10.0, 110.0, 60.0])
    assert ctrl.update_scene_extent(
        fscene.name, QgsRectangle(20.0, 50.0, 150.0, 90.0)
    )
    # clip 到影像 200×100：bbox 应为 [20, 10, 150, 50]（行 10~50 = y 90~50）
    assert fscene.bbox == [20.0, 10.0, 150.0, 50.0], fscene.bbox
    print("用例 4 OK：文件场景 clip 正确")

    # ---- 用例 5：手工画框落盘后场景状态联动（未标注→已标注，已审核不打回）
    from qgis_yolo_annotator.core.project import (
        SCENE_STATUS_ANNOTATED,
        SCENE_STATUS_UNANNOTATED,
        SCENE_STATUS_VERIFIED,
    )

    ctrl.load_scene(scene)  # 切回 xyz 场景
    scene.status = SCENE_STATUS_UNANNOTATED
    ctrl.save_current_labels()
    assert scene.status == SCENE_STATUS_ANNOTATED, scene.status
    scene.status = SCENE_STATUS_VERIFIED
    ctrl.save_current_labels()
    assert scene.status == SCENE_STATUS_VERIFIED, "已审核不应被自动降级"
    print("用例 5 OK：手工标注联动场景状态（只升级不降级）")

    # ---- 用例 6：新建工程后旧工程场景框/标注框从画布清空
    from qgis_yolo_annotator.core.project import AnnotationProject as _AP

    root2 = root.parent / "scene_edit_headless_2"
    if root2.exists():
        shutil.rmtree(root2)
    ctrl.create_project(str(root2), "第二个工程")
    assert ctrl.current_image is None and ctrl.current_scene is None
    assert ctrl.ann_layer is None and ctrl.scene_layer is None
    leftovers = [
        layer.name()
        for layer in QgsProject.instance().mapLayers().values()
        if layer.name().startswith("yolo_annotator")
    ]
    assert not leftovers, f"旧图层残留: {leftovers}"
    assert ctrl.project.name == "第二个工程"
    print("用例 6 OK：新建工程清空旧工程画布图层")

    print("ALL OK")
finally:
    app.exitQgis()
