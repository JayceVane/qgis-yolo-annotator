"""语法检查：obb_edit_tool.py。"""

import ast

for rel in (
    "src/qgis_yolo_annotator/gui/obb_edit_tool.py",
    "src/qgis_yolo_annotator/gui/controller.py",
    "src/qgis_yolo_annotator/gui/main_dock.py",
    "src/qgis_yolo_annotator/core/xyz_source.py",
    "src/qgis_yolo_annotator/core/project.py",
):
    src = open(rel, encoding="utf-8").read()
    ast.parse(src)
    print("ok:", rel)
