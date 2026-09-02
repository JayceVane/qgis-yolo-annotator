"""标注工程管理：project.json + labels/*.json 目录结构。

工程目录布局：
    <project_dir>/
        project.json    # 类别表、影像/场景清单及状态、当前模型
        labels/         # 每影像一个 X-AnyLabeling JSON（像素坐标），按影像主名命名

影像以绝对路径引用（不拷贝）；类别行序 = 类别 id（导出 YOLO 系格式用）。

场景（Scene / AOI）：
- 用户在画布上画矩形框圈出的推理/标注范围，bbox 为**该影像的像素坐标**
- 状态挂在场景上：unannotated（未标注）→ annotated（已标注）→ verified（已审核）
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .label_store import load_label, make_label_doc, save_label

SCHEMA_VERSION = 2

SCENE_STATUS_UNANNOTATED = "unannotated"
SCENE_STATUS_ANNOTATED = "annotated"
SCENE_STATUS_VERIFIED = "verified"
ALL_SCENE_STATUSES = (
    SCENE_STATUS_UNANNOTATED,
    SCENE_STATUS_ANNOTATED,
    SCENE_STATUS_VERIFIED,
)
SCENE_STATUS_LABELS_ORDER = {
    SCENE_STATUS_UNANNOTATED: "未标注",
    SCENE_STATUS_ANNOTATED: "已标注",
    SCENE_STATUS_VERIFIED: "已审核",
}

# X-AnyLabeling 自动配色（label_colormap 基色表）：
# 新类别按行序循环分配，与 X-AnyLabeling 桌面版「shape_color: auto」一致
XANYLABELING_PALETTE = (
    "#aaaaff", "#ffaaaa", "#aaffaa", "#ffffaa", "#aaffff",
    "#ffaaff", "#55aaff", "#ffaa00", "#28ffff", "#00ff7f",
    "#ff69b4", "#7fffd4", "#ffd700", "#6495ed", "#ffb6c1",
    "#40e0d0", "#ffdfba", "#9370db", "#00bfff", "#f08080",
    "#98fb98", "#add8e6", "#ffc0cb", "#dda0dd", "#87cefa",
    "#fffacd", "#afeeee", "#fa8072", "#9acd32", "#20b2aa",
    "#ffa07a", "#b0e0e6",
)
DEFAULT_PALETTE = XANYLABELING_PALETTE

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class ClassDef:
    """类别定义（行序 = 类别 id）。"""

    name: str
    color: str = ""
    hotkey: str = ""  # 单字符快捷键（数字/字母），空为未绑定


@dataclass
class SceneDef:
    """场景（AOI）：推理/标注/导出的工作范围。

    kind="pixel"：文件影像内矩形，bbox 为该影像像素坐标；
    kind="xyz"：在线瓦片场景，map_bbox 为 EPSG:3857 地图矩形 + zoom +
    source（瓦片源配置 dict，见 core.xyz_source.XyzSourceConfig），
    像素网格由 (map_bbox, zoom) 惰性确定（bbox 留占位）。
    """

    name: str
    bbox: list[float]  # 像素坐标 [x0, y0, x1, y1]（xyz 场景为占位）
    status: str = SCENE_STATUS_UNANNOTATED
    kind: str = "pixel"  # pixel / xyz
    map_bbox: list[float] | None = None  # xyz: [xmin, ymin, xmax, ymax] EPSG:3857
    zoom: int | None = None  # xyz: 瓦片级别
    source: dict | None = None  # xyz: XyzSourceConfig.to_dict()

    def __post_init__(self):
        if self.status not in ALL_SCENE_STATUSES:
            raise ValueError(f"非法场景状态: {self.status}")
        if self.kind == "xyz":
            bbox = self.map_bbox
            if not bbox or len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
                raise ValueError(f"xyz 场景 map_bbox 非法: {self.map_bbox}")
            if self.zoom is None or self.source is None:
                raise ValueError("xyz 场景需要 zoom 与 source 配置")
        else:
            if len(self.bbox) != 4 or not (self.bbox[0] < self.bbox[2] and self.bbox[1] < self.bbox[3]):
                raise ValueError(f"场景 bbox 非法（需 [x0,y0,x1,y1] 且 x1>x0,y1>y0）: {self.bbox}")

    @property
    def width(self) -> float:
        return (self.map_bbox[2] - self.map_bbox[0]) if self.kind == "xyz" else (self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return (self.map_bbox[3] - self.map_bbox[1]) if self.kind == "xyz" else (self.bbox[3] - self.bbox[1])


@dataclass
class ImageEntry:
    """工程中的影像条目（本地文件或在线瓦片工作集；状态由其场景聚合表达）。"""

    path: str  # 文件绝对路径；xyz 工作集为 "xyz://<图层标题>"
    scenes: list[SceneDef] = field(default_factory=list)
    kind: str = "file"  # file / xyz


@dataclass
class AnnotationProject:
    """标注工程（内存态 + JSON 持久化）。"""

    root: Path
    name: str = ""
    classes: list[ClassDef] = field(default_factory=list)
    images: list[ImageEntry] = field(default_factory=list)
    active_model: str = ""
    schema_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------------ 工程级 IO

    @classmethod
    def create(cls, root: str | Path, name: str) -> "AnnotationProject":
        """创建新工程（目录 + 空 project.json）。

        Args:
            root: 工程根目录（已存在则要求为空或只含本工程文件）。
            name: 工程名。

        Returns:
            AnnotationProject 实例（已持久化）。

        Raises:
            FileExistsError: 目录已存在且含 project.json。
        """
        root = Path(root)
        marker = root / "project.json"
        if marker.is_file():
            raise FileExistsError(f"目录已是标注工程: {root}")
        root.mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(exist_ok=True)
        project = cls(root=root, name=name)
        project.save()
        return project

    @classmethod
    def open(cls, root: str | Path) -> "AnnotationProject":
        """打开已有工程（兼容 schema v1 的影像级状态：迁移为无场景影像）。

        Args:
            root: 工程根目录（含 project.json）。

        Returns:
            AnnotationProject 实例。

        Raises:
            FileNotFoundError: project.json 不存在。
            ValueError: JSON/结构损坏。
        """
        root = Path(root)
        marker = root / "project.json"
        if not marker.is_file():
            raise FileNotFoundError(f"不是标注工程目录（缺 project.json）: {root}")
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"project.json 解析失败: {exc}") from exc
        images: list[ImageEntry] = []
        for i in data.get("images", []):
            scenes = [
                SceneDef(
                    name=str(s.get("name", "")),
                    bbox=[float(v) for v in s.get("bbox", [0, 0, 0, 0])],
                    status=str(s.get("status", SCENE_STATUS_UNANNOTATED)),
                    kind=str(s.get("kind", "pixel")),
                    map_bbox=[float(v) for v in s["map_bbox"]] if s.get("map_bbox") else None,
                    zoom=int(s["zoom"]) if s.get("zoom") is not None else None,
                    source=s.get("source") or None,
                )
                for s in i.get("scenes", [])
            ]
            images.append(
                ImageEntry(
                    path=str(i["path"]),
                    scenes=scenes,
                    kind=str(i.get("kind", "file")),
                )
            )
        return cls(
            root=root,
            name=str(data.get("name", root.name)),
            classes=[
                ClassDef(
                    name=str(c.get("name", "")),
                    color=str(c.get("color", "")),
                    hotkey=str(c.get("hotkey", "")),
                )
                for c in data.get("classes", [])
            ],
            images=images,
            active_model=str(data.get("active_model", "")),
        )

    def save(self) -> None:
        """原子写 project.json。"""
        payload = {
            "schema_version": self.schema_version,
            "name": self.name,
            "classes": [asdict(c) for c in self.classes],
            "images": [asdict(i) for i in self.images],
            "active_model": self.active_model,
        }
        tmp = self.root / "project.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.root / "project.json")

    # ------------------------------------------------------------------ 影像管理

    def add_image(self, path: str | Path, kind: str = "file") -> ImageEntry:
        """添加影像到工程（去重；kind=xyz 时 path 为 "xyz://标题" 工作集标识）。

        Raises:
            ValueError: 扩展名不受支持（仅 file 条目校验）。
        """
        path_str = str(path)
        if kind == "file":
            resolved = str(Path(path).resolve())
            if Path(resolved).suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"不支持的影像格式: {Path(resolved).suffix}")
        else:
            resolved = path_str
        for entry in self.images:
            if entry.path == resolved and entry.kind == kind:
                return entry
        entry = ImageEntry(path=resolved, kind=kind)
        self.images.append(entry)
        return entry

    def add_image_folder(self, folder: str | Path) -> list[ImageEntry]:
        """批量导入目录下所有受支持影像（递归）。"""
        folder = Path(folder)
        added: list[ImageEntry] = []
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                added.append(self.add_image(p))
        return added

    def remove_image(self, path: str | Path) -> bool:
        """从工程移除影像（标注 JSON 保留不删）。"""
        resolved = str(Path(path).resolve())
        before = len(self.images)
        self.images = [i for i in self.images if i.path != resolved]
        return len(self.images) < before

    def find_image(self, path: str | Path) -> ImageEntry | None:
        """按路径查找影像条目（xyz 工作集按字符串精确匹配）。"""
        path_str = str(path)
        if path_str.startswith("xyz://"):
            for entry in self.images:
                if entry.path == path_str and entry.kind == "xyz":
                    return entry
            return None
        resolved = str(Path(path).resolve())
        for entry in self.images:
            if entry.path == resolved and entry.kind == "file":
                return entry
        return None

    # ------------------------------------------------------------------ 场景管理

    def add_scene(
        self,
        image_path: str | Path,
        bbox: list[float],
        name: str | None = None,
        *,
        kind: str = "pixel",
        map_bbox: list[float] | None = None,
        zoom: int | None = None,
        source: dict | None = None,
    ) -> SceneDef:
        """为影像添加场景（AOI）。

        Args:
            image_path: 影像路径或 xyz 工作集标识（须已在工程中）。
            bbox: 像素坐标（pixel 场景）；xyz 场景传占位 [0,0,0,0]。
            name: 场景名；缺省自动编号 scene_001。
            kind: pixel / xyz。
            map_bbox: xyz 场景的 EPSG:3857 矩形。
            zoom: xyz 场景瓦片级别。
            source: xyz 瓦片源配置 dict。

        Returns:
            新建的 SceneDef。

        Raises:
            ValueError: 影像不在工程中。
        """
        entry = self.find_image(image_path)
        if entry is None:
            raise ValueError(f"影像不在工程中: {image_path}")
        if name is None:
            used = {s.name for s in entry.scenes}
            counter = len(entry.scenes) + 1
            while f"scene_{counter:03d}" in used:
                counter += 1
            name = f"scene_{counter:03d}"
        scene = SceneDef(
            name=name,
            bbox=[float(v) for v in bbox],
            kind=kind,
            map_bbox=[float(v) for v in map_bbox] if map_bbox else None,
            zoom=zoom,
            source=source,
        )
        entry.scenes.append(scene)
        return scene

    def set_scene_status(self, image_path: str | Path, scene_name: str, status: str) -> None:
        """更新场景状态。

        Raises:
            ValueError: 状态非法、影像或场景不存在。
        """
        if status not in ALL_SCENE_STATUSES:
            raise ValueError(f"非法状态: {status}")
        entry = self.find_image(image_path)
        if entry is None:
            raise ValueError(f"影像不在工程中: {image_path}")
        for scene in entry.scenes:
            if scene.name == scene_name:
                scene.status = status
                return
        raise ValueError(f"场景不存在: {scene_name}")

    def scene_status_counts(self) -> dict[str, int]:
        """全工程各状态场景计数。"""
        counts = {s: 0 for s in ALL_SCENE_STATUSES}
        for entry in self.images:
            for scene in entry.scenes:
                counts[scene.status] += 1
        return counts

    def scene_count(self) -> int:
        """全工程场景总数。"""
        return sum(len(i.scenes) for i in self.images)

    # ------------------------------------------------------------------ 标注 IO

    def label_path(self, image_path: str | Path, scene_name: str | None = None) -> Path:
        """标注 JSON 路径：文件影像用 labels/<影像主名>.json；
        xyz 场景按场景独立存 labels/<场景名>.json（scene_name 指定时优先）。"""
        if scene_name:
            return self.root / "labels" / f"{scene_name}.json"
        stem = Path(image_path).stem
        return self.root / "labels" / f"{stem}.json"

    def load_image_labels(
        self, image_path: str | Path, scene_name: str | None = None
    ) -> list[dict]:
        """读取标注 shapes；无标注文件返回空列表（xyz 场景传 scene_name）。"""
        doc = load_label(self.label_path(image_path, scene_name))
        return doc["shapes"] if doc else []

    def save_image_labels(
        self,
        image_path: str | Path,
        shapes: list[dict],
        image_width: int,
        image_height: int,
        scene_name: str | None = None,
    ) -> None:
        """写影像/场景标注（原子写）。"""
        doc = make_label_doc(image_path, image_width, image_height, shapes)
        if scene_name:
            doc["imagePath"] = scene_name
        save_label(self.label_path(image_path, scene_name), doc)

    # ------------------------------------------------------------------ 类别管理

    def class_index(self, name: str) -> int | None:
        """类别名 → id（行序）；不存在返回 None。"""
        for idx, c in enumerate(self.classes):
            if c.name == name:
                return idx
        return None

    def replace_label(
        self, old_name: str, new_name: str, exclude_path: Path | None = None
    ) -> int:
        """工程级类别替换：把所有标注 JSON 中 label==old_name 的 shape 改为 new_name。

        Args:
            old_name: 原类别名（与 new_name 相同直接返回 0）。
            new_name: 目标类别名。
            exclude_path: 排除的 JSON 路径（如当前场景由图层编辑命令处理时）。

        Returns:
            替换的 shape 总数（遍历 labels/*.json，原子写回；损坏文件跳过）。
        """
        if old_name == new_name:
            return 0
        labels_dir = self.root / "labels"
        if not labels_dir.is_dir():
            return 0
        exclude = exclude_path.resolve() if exclude_path is not None else None
        total = 0
        for path in sorted(labels_dir.glob("*.json")):
            if exclude is not None and path.resolve() == exclude:
                continue
            try:
                doc = load_label(path)
            except ValueError:
                continue
            if doc is None:
                continue
            changed = 0
            for shape in doc.get("shapes", []):
                if shape.get("label") == old_name:
                    shape["label"] = new_name
                    changed += 1
            if changed:
                save_label(path, doc)
                total += changed
        return total

    def label_counts(self) -> dict[str, int]:
        """全工程各类别标注数量统计（labels/*.json 遍历）。"""
        labels_dir = self.root / "labels"
        counts: dict[str, int] = {}
        if not labels_dir.is_dir():
            return counts
        for path in sorted(labels_dir.glob("*.json")):
            try:
                doc = load_label(path)
            except ValueError:
                continue
            if doc is None:
                continue
            for shape in doc.get("shapes", []):
                name = str(shape.get("label", ""))
                counts[name] = counts.get(name, 0) + 1
        return counts

    def class_by_hotkey(self, key: str) -> ClassDef | None:
        """快捷键 → 类别定义。"""
        for c in self.classes:
            if c.hotkey == key:
                return c
        return None

    def add_class(self, name: str, color: str = "", hotkey: str = "") -> ClassDef:
        """新增类别（重名直接返回已有定义）。

        颜色按 X-AnyLabeling 规则自动分配：第 N 个类别取调色板第 N 色（循环）；
        新类别自动分配下一个未占用的数字快捷键。
        """
        for c in self.classes:
            if c.name == name:
                return c
        if not color:
            color = XANYLABELING_PALETTE[len(self.classes) % len(XANYLABELING_PALETTE)]
        if not hotkey:
            for digit in "123456789":
                if all(c.hotkey != digit for c in self.classes):
                    hotkey = digit
                    break
        definition = ClassDef(name=name, color=color, hotkey=hotkey)
        self.classes.append(definition)
        return definition

    def reassign_auto_colors(self) -> None:
        """按行序用 X-AnyLabeling 调色板重排全部类别颜色。"""
        for index, class_def in enumerate(self.classes):
            class_def.color = XANYLABELING_PALETTE[index % len(XANYLABELING_PALETTE)]
