"""pt 权重 → ONNX 转换桥：调用外部 AI 环境的 ultralytics 导出，结果落插件缓存目录。

不在 QGIS Python 内安装 torch/ultralytics（重量级且易与 QGIS 冲突），
而是以子进程方式复用 conda AI 环境（如 ai_env）。本模块为纯逻辑
（subprocess + pathlib），不依赖 qgis。

注意：QGIS 的 python-qgis.bat 会设置 PYTHONHOME/PYTHONPATH 并被子进程继承，
导致外部 Python 被劫持为 QGIS 环境——所有子进程调用必须剥离这些变量。
"""

from __future__ import annotations

import ast
import os
import subprocess  # nosec B404: pt→onnx 桥本身就依赖外部解释器子进程
import sys
from pathlib import Path


def _clean_env() -> dict:
    """构造剥离 PYTHONHOME/PYTHONPATH 的子进程环境。"""
    return {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in ("PYTHONHOME", "PYTHONPATH")
    }


def _spawn_flags() -> int:
    """Windows 下隐藏子进程控制台窗口的 creationflags。"""
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


# 子进程导出脚本：路径/尺寸经 argv 传入（不以文本拼接进代码，杜绝引号注入）
_EXPORT_SCRIPT = """
import sys
from ultralytics import YOLO

model = YOLO(sys.argv[1])
path = model.export(format="onnx", imgsz=int(sys.argv[2]), dynamic=False, half=False, simplify=True)
print("EXPORT_RESULT:", path)
"""

# 权重元数据读取脚本：路径经 argv 传入
_META_SCRIPT = """
import sys
from ultralytics import YOLO

m = YOLO(sys.argv[1])
names = m.model.names or {}
items = sorted(names.items()) if isinstance(names, dict) else list(enumerate(names))
print("PT_META_TASK:", m.task)
print("PT_META_NAMES:", [v for _k, v in items])
print("PT_META_IMGSZ:", getattr(m, "overrides", {}).get("imgsz"))
"""

# 转换结果标记行（stdout 解析用）
_RESULT_MARKER = "EXPORT_RESULT:"

# 常见 AI 环境解释器候选路径（按序探测）
_DEFAULT_PYTHON_CANDIDATES = (
    r"D:\DevKit\anaconda3\envs\ai_env\python.exe",
    r"D:\DevKit\anaconda3\envs\yolo\python.exe",
    r"D:\DevKit\anaconda3\python.exe",
)

_CONVERT_TIMEOUT_SEC = 600


def find_ai_env_python() -> Path | None:
    """按候选路径探测带 ultralytics 的外部 Python。

    Returns:
        可用解释器路径；找不到返回 None。
    """
    for candidate in _DEFAULT_PYTHON_CANDIDATES:
        path = Path(candidate)
        if path.is_file() and _has_ultralytics(path):
            return path
    return None


def python_has_ultralytics(python_exe: Path) -> bool:
    """检查解释器是否可 import ultralytics（公开接口）。"""
    return _has_ultralytics(python_exe)


def _has_ultralytics(python_exe: Path) -> bool:
    """检查解释器是否可 import ultralytics。"""
    if not python_exe.is_file():
        return False
    try:
        result = subprocess.run(  # nosec B603: 本地已验证解释器，列表参数无 shell
            [str(python_exe), "-c", "import ultralytics"],
            capture_output=True,
            timeout=60,
            env=_clean_env(),
            creationflags=_spawn_flags(),
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def convert_pt_to_onnx(
    pt_path: str | Path,
    python_exe: str | Path,
    imgsz: int,
    cache_dir: str | Path,
    progress_cb=None,
) -> Path:
    """用外部 AI 环境将 .pt 权重导出为 ONNX 并落缓存目录。

    Args:
        pt_path: ultralytics 格式权重路径。
        python_exe: 带 ultralytics 的 Python 解释器（如 ai_env）。
        imgsz: 导出输入尺寸（与训练一致最佳）。
        cache_dir: 缓存目录（由调用方提供，如 profile 下 models_cache）。
        progress_cb: 可选回调 fn(line: str)（透传子进程 stderr 关键行）。

    Returns:
        缓存目录中的 onnx 路径。

    Raises:
        FileNotFoundError: 权重或解释器不存在。
        RuntimeError: 导出失败（含子进程 stderr 摘要）。
    """
    pt_path = Path(pt_path)
    python_exe = Path(python_exe)
    if not pt_path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {pt_path}")
    if not python_exe.is_file():
        raise FileNotFoundError(f"AI 环境解释器不存在: {python_exe}")

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target_onnx = cache / f"{pt_path.stem}_{imgsz}.onnx"
    if target_onnx.is_file():
        return target_onnx  # 缓存命中

    # 复制 pt 到缓存目录再导出：ultralytics 把 onnx 输出到 pt 旁边，
    # 这样产物天然落在缓存内（不污染源目录，也规避跨盘 rename 限制）
    local_pt = cache / pt_path.name
    if not local_pt.is_file() or local_pt.stat().st_mtime < pt_path.stat().st_mtime:
        import shutil

        shutil.copy2(pt_path, local_pt)

    script = _EXPORT_SCRIPT
    result = subprocess.run(  # nosec B603: 解释器已验证存在，路径经 argv 传入无拼接
        [str(python_exe), "-c", script, str(local_pt), str(int(imgsz))],
        capture_output=True,
        text=True,
        cwd=str(cache),
        timeout=_CONVERT_TIMEOUT_SEC,
        env=_clean_env(),
        creationflags=_spawn_flags(),
    )
    onnx_path = None
    for line in result.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            onnx_path = line[len(_RESULT_MARKER):].strip()
            break
    if result.returncode != 0 or not onnx_path:
        stderr_tail = "\n".join(result.stderr.splitlines()[-15:])
        raise RuntimeError(
            f"pt→onnx 导出失败（exit={result.returncode}）:\n{stderr_tail}"
        )
    exported = Path(onnx_path)
    if not exported.is_file():
        raise RuntimeError(f"导出声明成功但文件不存在: {exported}")
    if exported.resolve() != target_onnx.resolve():
        import shutil

        shutil.move(str(exported), target_onnx)  # 同盘 rename；跨盘 copy+delete
    # 导出成功后清理缓存中的临时 pt 副本（保留 onnx 即可）
    local_pt.unlink(missing_ok=True)
    return target_onnx


def read_pt_metadata(pt_path: str | Path, python_exe: str | Path) -> dict:
    """读取 .pt 权重的 task / 类别表 / 训练 imgsz（子进程，用于转换前预填表单）。

    Returns:
        {"task": str|None, "labels": list[str], "imgsz": int|None}；
        读取失败时字段为空，不抛异常。
    """
    empty = {"task": None, "labels": [], "imgsz": None}
    pt_path = Path(pt_path)
    python_exe = Path(python_exe)
    if not pt_path.is_file() or not python_exe.is_file():
        return empty
    try:
        result = subprocess.run(  # nosec B603: 两者均已验证存在，路径经 argv 传入
            [str(python_exe), "-c", _META_SCRIPT, str(pt_path)],
            capture_output=True,
            text=True,
            timeout=120,
            env=_clean_env(),
            creationflags=_spawn_flags(),
        )
    except (subprocess.SubprocessError, OSError):
        return empty
    meta = {"task": None, "labels": [], "imgsz": None}
    for line in result.stdout.splitlines():
        if line.startswith("PT_META_TASK:"):
            meta["task"] = line.split(":", 1)[1].strip() or None
        elif line.startswith("PT_META_NAMES:"):
            raw = line.split(":", 1)[1].strip()
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple)):
                    meta["labels"] = [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
        elif line.startswith("PT_META_IMGSZ:"):
            raw = line.split(":", 1)[1].strip()
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, int):
                    meta["imgsz"] = parsed
                elif isinstance(parsed, (list, tuple)) and parsed:
                    meta["imgsz"] = int(parsed[0])
            except (ValueError, SyntaxError, TypeError):
                pass
    return meta
