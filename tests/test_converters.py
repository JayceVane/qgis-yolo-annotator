"""converters 单测：DOTA / YOLO-OBB / YOLO-det / VOC 转换与边界策略。"""

import xml.etree.ElementTree as ET

import pytest

from qgis_yolo_annotator.export import converters

CLASSES = ["Small Car", "Truck"]


def obb(cx, cy, w, h, angle_deg=0.0):
    """构造旋转框 shape（角度制输入便于阅读）。"""
    import math

    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = w / 2, h / 2
    local = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    pts = [[cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a] for x, y in local]
    return {"label": "Small Car", "points": pts, "difficult": False}


def test_dota_line_format():
    shape = obb(100, 100, 40, 20)
    lines = converters.dota_lines([shape], CLASSES, 200, 200)
    assert len(lines) == 1
    parts = lines[0].split()
    assert len(parts) == 10
    assert parts[8] == "Small_Car"  # DOTA 约定：label 空格转下划线
    assert parts[9] == "0"
    assert parts[0] == "80.00" and parts[1] == "90.00"  # 左上角点


def test_dota_difficult_flag():
    shape = obb(100, 100, 40, 20)
    shape["difficult"] = True
    lines = converters.dota_lines([shape], CLASSES, 200, 200)
    assert lines[0].endswith(" 1")


def test_boundary_skip_drops_out_of_bounds():
    shape = obb(5, 100, 40, 20)  # xmin=-15 越界
    assert converters.dota_lines([shape], CLASSES, 200, 200, "skip") == []
    assert converters.dota_lines([shape], CLASSES, 200, 200, "clip")


def test_boundary_clip_clamps_coords():
    shape = obb(5, 100, 40, 20)
    lines = converters.dota_lines([shape], CLASSES, 200, 200, "clip")
    coords = [float(v) for v in lines[0].split()[:8]]
    assert all(c >= 0.0 for c in coords)


def test_unknown_class_skipped():
    shape = obb(100, 100, 40, 20)
    shape["label"] = "Ship"
    assert converters.dota_lines([shape], CLASSES, 200, 200) == []


def test_yolo_obb_normalized():
    shape = obb(100, 100, 40, 20, angle_deg=30)
    lines = converters.yolo_obb_lines([shape], CLASSES, 200, 200)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"
    vals = [float(v) for v in parts[1:]]
    assert len(vals) == 8
    assert all(0.0 <= v <= 1.0 for v in vals)
    # 角点几何保持：相邻边长分别对应 40/200 与 20/200
    import math

    p = [(vals[0] * 200, vals[1] * 200), (vals[2] * 200, vals[3] * 200)]
    edge0 = math.dist(p[0], p[1])
    assert edge0 == pytest.approx(40.0, abs=0.01)


def test_yolo_det_hbb_from_obb():
    shape = obb(100, 100, 40, 20, angle_deg=45)
    lines = converters.yolo_det_lines([shape], CLASSES, 200, 200)
    parts = lines[0].split()
    cx, cy, bw, bh = (float(v) for v in parts[1:])
    # 45° 旋转 40×20 框的外接框约 42.43
    assert bw * 200 == pytest.approx(42.43, abs=0.1)
    assert bh * 200 == pytest.approx(42.43, abs=0.1)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)


def test_voc_hbb_mode():
    shape = obb(100, 100, 40, 20)
    root = converters.voc_xml([shape], CLASSES, "labels", "a.tif", 200, 200)
    assert root.tag == "annotation"
    obj = root.find("object")
    assert obj.find("name").text == "Small Car"
    bnd = obj.find("bndbox")
    assert bnd.find("xmin").text == "80.00"
    assert bnd.find("xmax").text == "120.00"
    assert obj.find("polygon") is None


def test_voc_polygon_mode():
    shape = obb(100, 100, 40, 20, angle_deg=10)
    root = converters.voc_xml(
        [shape], CLASSES, "labels", "a.tif", 200, 200, obb_mode="polygon"
    )
    obj = root.find("object")
    assert obj.find("bndbox") is None
    poly = obj.find("polygon")
    assert [c.tag for c in poly] == ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


def test_empty_shapes_produce_empty_output():
    assert converters.dota_lines([], CLASSES, 100, 100) == []
    root = converters.voc_xml([], CLASSES, "l", "f", 100, 100)
    assert root.find("object") is None
