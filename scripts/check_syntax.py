"""语法 + 未定义名检查：插件全部模块。

pyflakes 已安装时追加未定义名扫描（捕获 QWidget 类 NameError 的漏网）；
未安装则跳过（仅语法检查）。
"""

import ast
from pathlib import Path

SRC_ROOT = Path("src/qgis_yolo_annotator")

for path in sorted(SRC_ROOT.rglob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print("ok:", path)

try:
    import pyflakes.checker as _pf_checker
except ImportError:
    print("note: pyflakes 未安装，跳过未定义名扫描（pip install pyflakes 启用）")
else:
    failed = False
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        checker = _pf_checker.Checker(tree, filename=str(path))
        for message in checker.messages:
            # 只拦未定义名（undefined name），未用导入等不阻塞
            if "undefined name" in str(message):
                print("FAIL:", message)
                failed = True
    if failed:
        raise SystemExit("发现未定义名称，请修复")
    print("undefined-name scan: clean")
