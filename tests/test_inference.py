"""inference 几何单测：旋转框角点、多边形 IoU、NMS（不需要模型）。"""

import numpy as np
import pytest

from qgis_yolo_annotator.core.inference import (
    nms_polygons,
    polygon_area,
    rotate_box_points,
    rotated_iou,
)


def test_rotate_box_points_identity():
    pts = rotate_box_points(10, 20, 8, 4, 0.0)
    expected = np.array([[6, 18], [14, 18], [14, 22], [6, 22]], dtype=float)
    assert np.allclose(np.asarray(pts), expected)


def test_rotate_box_points_90deg():
    pts = rotate_box_points(0, 0, 8, 4, np.pi / 2)
    # 8×4 旋转 90° 后：x 方向跨度 4、y 方向跨度 8
    arr = np.asarray(pts)
    assert np.ptp(arr[:, 0]) == pytest.approx(4.0, abs=1e-9)
    assert np.ptp(arr[:, 1]) == pytest.approx(8.0, abs=1e-9)
    assert np.allclose(arr[0], [2, -4])


def test_polygon_area_unit_square():
    assert polygon_area(np.array([[0, 0], [1, 0], [1, 1], [0, 1]])) == 1.0


def test_rotated_iou_identical():
    pts = np.asarray(rotate_box_points(50, 50, 20, 10, 0.3))
    assert rotated_iou(pts, pts) == pytest.approx(1.0)


def test_rotated_iou_disjoint():
    a = np.asarray(rotate_box_points(10, 10, 8, 8, 0.0))
    b = np.asarray(rotate_box_points(50, 50, 8, 8, 0.0))
    assert rotated_iou(a, b) == 0.0


def test_rotated_iou_half_overlap():
    a = np.asarray(rotate_box_points(10, 10, 10, 10, 0.0))  # [5,15]^2
    b = np.asarray(rotate_box_points(15, 10, 10, 10, 0.0))  # [10,20]×[5,15]
    assert rotated_iou(a, b) == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_rotated_iou_cross_rotation():
    # 同中心 10×10 水平框 vs 45° 旋转框：交集明显大于一半、小于全重合
    a = np.asarray(rotate_box_points(0, 0, 10, 10, 0.0))
    b = np.asarray(rotate_box_points(0, 0, 10, 10, np.pi / 4))
    val = rotated_iou(a, b)
    assert 0.7 < val < 0.95


def test_nms_polygons_suppresses_overlap():
    high = np.asarray(rotate_box_points(20, 20, 20, 10, 0.2))
    low = np.asarray(rotate_box_points(21, 20, 20, 10, 0.2))  # 几乎重合
    far = np.asarray(rotate_box_points(100, 100, 20, 10, 0.2))
    keep = nms_polygons([high, low, far], [0.9, 0.8, 0.7], iou_threshold=0.5)
    assert sorted(keep) == [0, 2]


def test_nms_polygons_different_class_not_separated():
    # NMS 本身不区分类别（与 YOLO 逐类 NMS 略有差异，跨窗去重由上层处理）
    a = np.asarray(rotate_box_points(20, 20, 20, 10, 0.0))
    b = np.asarray(rotate_box_points(20, 20, 20, 10, 0.0))
    keep = nms_polygons([a, b], [0.9, 0.8], 0.5)
    assert keep == [0]
