"""模型注册表：管理多个 ONNX 模型配置（task / labels / imgsz / conf / iou）。

注册表持久化在 QGIS profile 下的 qgis_yolo_annotator/models.json（跨工程共享）。
模型 labels 优先从 ultralytics 导出时写入 onnx 的 metadata 读取。
"""

from __future__ import annotations

import ast
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .inference import VALID_TASKS

_DEFAULT_IMGZ = 640
_DEFAULT_CONF = 0.25
_DEFAULT_IOU = 0.45


@dataclass
class ModelConfig:
    """单个模型的推理配置。"""

    name: str
    task: str  # det / obb / seg
    file_path: str
    labels: list[str] = field(default_factory=list)
    imgsz: int = _DEFAULT_IMGZ
    conf: float = _DEFAULT_CONF
    iou: float = _DEFAULT_IOU

    def __post_init__(self):
        if self.task not in VALID_TASKS:
            raise ValueError(f"未知任务类型: {self.task}（可选 {VALID_TASKS}）")
        self.imgsz = int(self.imgsz)
        self.conf = float(self.conf)
        self.iou = float(self.iou)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # noqa: SIM118
        return cls(**{k: v for k, v in data.items() if k in known})


def read_onnx_metadata(file_path: str | Path) -> dict:
    """读取 ultralytics 导出 onnx 的 metadata（names / imgsz / task 等）。

    Args:
        file_path: onnx 文件路径。

    Returns:
        {"labels": [...], "imgsz": int | None, "task": str | None}，
        无法解析的键缺省为 None / []。不抛异常（仅尽力解析）。
    """
    import onnxruntime as ort

    result: dict = {"labels": [], "imgsz": None, "task": None}
    try:
        session = ort.InferenceSession(str(file_path), providers=["CPUExecutionProvider"])
        meta = session.get_modelmeta().custom_metadata_map
    except Exception:
        return result
    raw_names = meta.get("names")
    if raw_names:
        try:
            parsed = ast.literal_eval(raw_names)
            if isinstance(parsed, dict):
                result["labels"] = [str(v) for _, v in sorted(parsed.items())]
            elif isinstance(parsed, (list, tuple)):
                result["labels"] = [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass
    raw_imgsz = meta.get("imgsz")
    if raw_imgsz:
        try:
            parsed = ast.literal_eval(raw_imgsz)
            if isinstance(parsed, (list, tuple)) and parsed:
                result["imgsz"] = int(parsed[0])
            elif isinstance(parsed, int):
                result["imgsz"] = parsed
        except (ValueError, SyntaxError):
            pass
    raw_task = meta.get("task")
    if raw_task in VALID_TASKS:
        result["task"] = raw_task
    return result


class ModelRegistry:
    """模型注册表（JSON 文件持久化，线程安全）。"""

    def __init__(self, store_path: str | Path):
        """初始化注册表。

        Args:
            store_path: JSON 存储路径（不存在时创建空表）。
        """
        self.store_path = Path(store_path)
        self._lock = threading.RLock()
        self._models: dict[str, ModelConfig] = {}
        self.load()

    # ------------------------------------------------------------------ CRUD

    def list_models(self) -> list[ModelConfig]:
        """全部模型配置（按名称排序）。"""
        with self._lock:
            return sorted(self._models.values(), key=lambda m: m.name)

    def get(self, name: str) -> ModelConfig | None:
        """按名称取模型配置。"""
        with self._lock:
            return self._models.get(name)

    def add(self, config: ModelConfig) -> None:
        """新增/覆盖模型配置并持久化。

        Raises:
            ValueError: 同名且指向不同文件的模型已存在。
        """
        with self._lock:
            existing = self._models.get(config.name)
            if existing is not None and Path(existing.file_path) != Path(config.file_path):
                raise ValueError(f"模型名已存在且指向不同文件: {config.name}")
            self._models[config.name] = config
            self.save()

    def remove(self, name: str) -> bool:
        """删除模型配置。返回是否存在并删除。"""
        from .inference import evict_session

        with self._lock:
            config = self._models.pop(name, None)
            if config is None:
                return False
            evict_session(config.file_path)
            self.save()
            return True

    # ------------------------------------------------------------------ IO

    def load(self) -> None:
        """从 JSON 加载注册表（文件损坏时重置为空表并保留备份）。"""
        self._models = {}
        if not self.store_path.is_file():
            return
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            for item in data.get("models", []):
                config = ModelConfig.from_dict(item)
                self._models[config.name] = config
        except (json.JSONDecodeError, ValueError, OSError):
            backup = self.store_path.with_suffix(".corrupt.json")
            try:
                self.store_path.replace(backup)
            except OSError:
                pass

    def save(self) -> None:
        """原子写 JSON（先写临时文件再替换）。"""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"models": [m.to_dict() for m in self._models.values()]}
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.store_path)
