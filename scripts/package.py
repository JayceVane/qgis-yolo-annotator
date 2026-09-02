"""打包插件 zip：dist/qgis_yolo_annotator-<version>.zip。

用 zipfile 而非 PowerShell Compress-Archive：后者写入反斜杠路径，
违反 ZIP 规范，plugins.qgis.org 上传校验会拒绝。
"""

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "qgis_yolo_annotator"


def main() -> None:
    meta = (SRC / "metadata.txt").read_text(encoding="utf-8")
    version = re.search(r"^version=(.+)$", meta, re.MULTILINE).group(1).strip()
    out_dir = ROOT / "dist"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"qgis_yolo_annotator-{version}.zip"
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            arcname = f"qgis_yolo_annotator/{path.relative_to(SRC).as_posix()}"
            zf.write(path, arcname)
        license_path = ROOT / "LICENSE"
        if license_path.is_file():
            # plugins.qgis.org 要求包内含 LICENSE（仓库根为唯一权威副本）
            zf.write(license_path, "qgis_yolo_annotator/LICENSE")
    print(f"packaged: {out}")


if __name__ == "__main__":
    main()
