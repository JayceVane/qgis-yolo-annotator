"""场景范围调整的标注迁移纯函数单测。"""

from qgis_yolo_annotator.core.label_store import (
    count_shapes_outside,
    shift_shapes,
)


def _shape(points):
    return {"label": "car", "shape_type": "rotation", "points": points}


def test_shift_shapes_translates_all_points():
    shapes = [_shape([[10.0, 20.0], [30.0, 40.0]])]
    moved = shift_shapes(shapes, 5.0, -3.0)
    assert moved[0]["points"] == [[15.0, 17.0], [35.0, 37.0]]


def test_shift_shapes_does_not_mutate_input():
    shapes = [_shape([[1.0, 2.0]])]
    shift_shapes(shapes, 100.0, 100.0)
    assert shapes[0]["points"] == [[1.0, 2.0]]


def test_shift_shapes_keeps_other_fields():
    shapes = [_shape([[0.0, 0.0]])]
    shapes[0]["score"] = 0.9
    moved = shift_shapes(shapes, 1.0, 1.0)
    assert moved[0]["score"] == 0.9
    assert moved[0] is not shapes[0]


def test_shift_shapes_empty_points():
    assert shift_shapes([_shape([])], 1.0, 1.0)[0]["points"] == []


def test_count_shapes_outside_variants():
    shapes = [
        _shape([[5.0, 5.0]]),                       # 完全在内
        _shape([[5.0, 5.0], [15.0, 5.0]]),          # 部分越界
        _shape([[20.0, 20.0]]),                     # 完全在外
    ]
    assert count_shapes_outside(shapes, 10.0, 10.0) == 2
    assert count_shapes_outside(shapes, 20.0, 20.0) == 0
    assert count_shapes_outside([], 10.0, 10.0) == 0
