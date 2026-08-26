"""label_store / project 单测。"""

import math

import pytest

from qgis_yolo_annotator.core.label_store import (
    import_dota,
    load_label,
    make_label_doc,
    make_shape,
    rotation_direction,
    save_label,
)
from qgis_yolo_annotator.core.project import (
    SCENE_STATUS_ANNOTATED,
    SCENE_STATUS_UNANNOTATED,
    SCENE_STATUS_VERIFIED,
    AnnotationProject,
)


def test_make_rotation_shape_direction():
    points = [[0, 0], [10, 0], [10, 5], [0, 5]]
    shape = make_shape("Car", points, "rotation", score=0.87)
    assert shape["shape_type"] == "rotation"
    assert shape["direction"] == pytest.approx(0.0)
    assert shape["score"] == 0.87
    assert shape["difficult"] is False


def test_rotation_direction_45deg():
    points = [[0, 0], [1, 1], [2, 0], [1, -1]]
    assert rotation_direction(points) == pytest.approx(math.pi / 4)


def test_make_shape_invalid():
    with pytest.raises(ValueError, match="4 个点"):
        make_shape("Car", [[0, 0]], "rotation")
    with pytest.raises(ValueError, match="shape_type"):
        make_shape("Car", [[0, 0]], "cube")


def test_save_and_load_roundtrip(tmp_path):
    doc = make_label_doc("img.tif", 100, 80, [make_shape("Car", [[1, 1], [9, 1], [9, 5], [1, 5]], "rotation")])
    path = tmp_path / "labels" / "img.json"
    save_label(path, doc)
    loaded = load_label(path)
    assert loaded["imageWidth"] == 100
    assert len(loaded["shapes"]) == 1
    assert loaded["shapes"][0]["label"] == "Car"


def test_load_missing_returns_none(tmp_path):
    assert load_label(tmp_path / "nope.json") is None


def test_load_corrupt_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="解析失败"):
        load_label(p)


def test_import_dota(tmp_path):
    p = tmp_path / "label.txt"
    p.write_text(
        "10.0 20.0 30.0 20.0 30.0 40.0 10.0 40.0 Small_Car 0\n"
        "# comment\n"
        "1 2 3 4 5 6 7 8 Truck 1\n",
        encoding="utf-8",
    )
    shapes = import_dota(p)
    assert len(shapes) == 2
    assert shapes[0]["label"] == "Small_Car"
    assert shapes[0]["difficult"] is False
    assert shapes[1]["difficult"] is True
    assert shapes[1]["points"] == [[1, 2], [3, 4], [5, 6], [7, 8]]


# ------------------------------------------------------------------ project


@pytest.fixture
def project(tmp_path):
    p = AnnotationProject.create(tmp_path / "proj", "测试工程")
    p.add_class("Small Car")
    p.add_class("Truck")
    return p


def test_project_create_open_roundtrip(project, tmp_path):
    project.save()
    reopened = AnnotationProject.open(project.root)
    assert reopened.name == "测试工程"
    assert [c.name for c in reopened.classes] == ["Small Car", "Truck"]


def test_project_create_rejects_existing(tmp_path):
    root = tmp_path / "p2"
    AnnotationProject.create(root, "a")
    with pytest.raises(FileExistsError):
        AnnotationProject.create(root, "b")


def test_image_add_remove(project, tmp_path):
    img = tmp_path / "a.tif"
    img.write_bytes(b"stub")
    entry = project.add_image(img)
    assert entry.scenes == []
    assert project.add_image(img) is entry  # 去重
    assert project.remove_image(img)
    assert not project.remove_image(img)


def test_scene_lifecycle(project, tmp_path):
    img = tmp_path / "a.tif"
    img.write_bytes(b"stub")
    project.add_image(img)
    s1 = project.add_scene(img, [10, 10, 100, 80])
    assert s1.name == "scene_001"
    assert s1.status == SCENE_STATUS_UNANNOTATED
    s2 = project.add_scene(img, [50, 50, 300, 200], name="重点区域")
    assert s2.name == "重点区域"
    project.set_scene_status(img, "重点区域", SCENE_STATUS_ANNOTATED)
    project.set_scene_status(img, "scene_001", SCENE_STATUS_VERIFIED)
    counts = project.scene_status_counts()
    assert counts[SCENE_STATUS_ANNOTATED] == 1
    assert counts[SCENE_STATUS_VERIFIED] == 1
    assert project.scene_count() == 2
    with pytest.raises(ValueError):
        project.set_scene_status(img, "scene_001", "bogus")
    with pytest.raises(ValueError):
        project.set_scene_status(img, "nope", SCENE_STATUS_ANNOTATED)


def test_scene_invalid_bbox(project, tmp_path):
    img = tmp_path / "a.tif"
    img.write_bytes(b"stub")
    project.add_image(img)
    with pytest.raises(ValueError, match="bbox"):
        project.add_scene(img, [10, 10, 5, 80])
    with pytest.raises(ValueError):
        project.add_scene(tmp_path / "not_in_project.tif", [0, 0, 1, 1])


def test_scenes_persist_roundtrip(project, tmp_path):
    img = tmp_path / "a.tif"
    img.write_bytes(b"stub")
    project.add_image(img)
    project.add_scene(img, [10, 10, 100, 80])
    project.save()
    reopened = AnnotationProject.open(project.root)
    entry = reopened.images[0]
    assert len(entry.scenes) == 1
    assert entry.scenes[0].bbox == [10.0, 10.0, 100.0, 80.0]


def test_add_image_rejects_extension(project, tmp_path):
    with pytest.raises(ValueError, match="格式"):
        project.add_image(tmp_path / "x.exe")


def test_class_helpers(project):
    assert project.class_index("Truck") == 1
    assert project.class_index("Nope") is None
    hotkey_cls = project.class_by_hotkey("1")
    assert hotkey_cls is not None and hotkey_cls.name == "Small Car"
    # 快捷键自动分配不冲突
    for name in ("Bus", "Van"):
        project.add_class(name)
    hotkeys = [c.hotkey for c in project.classes]
    assert len(hotkeys) == len(set(hotkeys))


def test_label_io_via_project(project, tmp_path):
    img = tmp_path / "b.png"
    img.write_bytes(b"stub")
    project.add_image(img)
    shapes = [make_shape("Truck", [[0, 0], [9, 0], [9, 4], [0, 4]], "rotation")]
    project.save_image_labels(img, shapes, 100, 50)
    loaded = project.load_image_labels(img)
    assert loaded[0]["label"] == "Truck"
    assert project.label_path(img).name == "b.json"
