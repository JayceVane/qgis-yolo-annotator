"""YOLO 检测（HBB）/ 旋转框（OBB）/ 实例分割（Seg）ONNX 推理。

移植自 t1_xanylabeling_web（已验证的推理实现），适配插件环境：
- 支持 YOLOv8 / YOLO11 系列 export 的 onnx
- det: 输出 [1, 4+nc, N]（自动兼容 [1, N, 4+nc] 转置布局）
- obb: 输出 [1, 5+nc, N]，多一维角度（弧度）
- seg: 输出 [1, 4+nc+32, N] + proto [1, 32, mh, mw]
- OBB NMS 自实现凸多边形 IoU（Sutherland-Hodgman 裁剪），
  不依赖 cv2.dnn.NMSBoxesRotated 的版本行为差异（OpenCV 4/5 表现不一致）
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # 允许未装 onnxruntime 时加载插件（推理时报错提示）
    ort = None

VALID_TASKS = ("det", "obb", "seg")

_SEG_COEFFS = 32  # YOLO-seg mask 系数维度（官方固定 32）
_MASK_BIN_THRESHOLD = 0.5
_MIN_CONTOUR_AREA = 25.0
_CONTOUR_EPSILON = 1.5


@dataclass
class Detection:
    """单个检测结果（图像像素坐标系）。"""

    points: list[list[float]]  # det: 四点顺时针；obb: 角点；seg: 多边形顶点
    score: float
    class_index: int


class InferenceUnavailable(RuntimeError):
    """onnxruntime 未安装或模型不可用。"""


def rotate_box_points(cx: float, cy: float, w: float, h: float, angle_rad: float) -> list[list[float]]:
    """旋转矩形（中心+宽高+弧度角）→ 四个角点。

    与 cv2.boxPoints 语义一致：起点为 (cx - w/2, cy - h/2) 绕中心旋转。

    Args:
        cx, cy: 中心坐标。
        w, h: 宽（沿角度方向）、高。
        angle_rad: 旋转角（弧度）。

    Returns:
        4 个 [x, y] 角点。
    """
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    dx, dy = w / 2.0, h / 2.0
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = local @ rot.T + np.array([cx, cy])
    return [[float(p[0]), float(p[1])] for p in pts]


# ---------------------------------------------------------------------------
# 凸多边形 IoU（OBB NMS 与跨窗去重共用）
# ---------------------------------------------------------------------------

def polygon_area(pts: np.ndarray) -> float:
    """shoelace 多边形有向面积的绝对值。

    Args:
        pts: (N, 2) 顶点序列。

    Returns:
        面积（像素²）。
    """
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman 凸多边形裁剪（subject 按凸多边形 clip 裁剪）。

    Args:
        subject: (N, 2) 被裁剪多边形顶点。
        clip: (M, 2) 裁剪窗口（凸）顶点。

    Returns:
        (K, 2) 交集多边形顶点；无交集时形状为 (0, 2)。
    """
    output = subject.astype(np.float64)
    m = len(clip)
    for i in range(m):
        if len(output) == 0:
            break
        a, b = clip[i], clip[(i + 1) % m]
        edge = b - a
        # 内侧判定：叉积 >= 0（clip 顶点为逆/顺时针一致时统一成立）
        side = lambda p: edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
        inside = np.array([side(p) >= 0 for p in output])
        input_pts = output
        output = []
        for j in range(len(input_pts)):
            cur, nxt = input_pts[j], input_pts[(j + 1) % len(input_pts)]
            cur_in, nxt_in = inside[j], inside[(j + 1) % len(input_pts)]
            if cur_in:
                output.append(cur)
            if cur_in != nxt_in:
                t = side(cur) / (side(cur) - side(nxt))
                output.append(cur + t * (nxt - cur))
        output = np.array(output) if output else np.zeros((0, 2))
    return output


def rotated_iou(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """两个凸多边形（矩形）的 IoU。

    Args:
        pts_a, pts_b: (4, 2) 角点。

    Returns:
        IoU ∈ [0, 1]。
    """
    inter = _clip_convex(np.asarray(pts_a, dtype=np.float64), np.asarray(pts_b, dtype=np.float64))
    if len(inter) < 3:
        return 0.0
    inter_area = polygon_area(inter)
    union_area = polygon_area(np.asarray(pts_a)) + polygon_area(np.asarray(pts_b)) - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class YoloOrtModel:
    """单模型 ONNX 会话（进程内缓存，线程锁串行推理）。"""

    def __init__(self, path: str | Path, task: str):
        """初始化推理会话。

        Args:
            path: onnx 模型文件路径。
            task: 任务类型，det / obb / seg。

        Raises:
            InferenceUnavailable: onnxruntime 缺失、文件不存在或会话创建失败。
        """
        if ort is None:
            raise InferenceUnavailable("QGIS Python 未安装 onnxruntime，无法推理")
        path = Path(path)
        if not path.is_file():
            raise InferenceUnavailable(f"模型文件不存在: {path}")
        if task not in VALID_TASKS:
            raise InferenceUnavailable(f"未知任务类型: {task}（可选 {VALID_TASKS}）")
        self.task = task
        self.path = path
        self._lock = threading.Lock()
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # e.g. [1,3,640,640]
        self.output_names = [o.name for o in self.session.get_outputs()]

    @property
    def static_imgsz(self) -> int | None:
        """模型静态输入尺寸（导出时固定 imgsz）；动态尺寸返回 None。"""
        shape = self.input_shape
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] == shape[3]:
            return int(shape[2])
        return None

    def infer(
        self, image_bgr: np.ndarray, imgsz: int, conf: float, iou: float
    ) -> list[Detection]:
        """对单张 BGR 图像推理，返回图像坐标系的检测列表。

        Args:
            image_bgr: (h, w, 3) BGR uint8。
            imgsz: 模型输入边长（letterbox 目标）。
            conf: 置信度阈值。
            iou: NMS IoU 阈值。

        Returns:
            Detection 列表（坐标已逆 letterbox 回原图）。
        """
        effective_imgsz = self.static_imgsz or imgsz  # 静态输入模型强制用导出尺寸
        tensor, params = self._letterbox(image_bgr, effective_imgsz)
        with self._lock:
            outputs = self.session.run(self.output_names, {self.input_name: tensor})
        pred = outputs[0]
        # 兼容 [1, C, N] / [1, N, C] 布局，统一为 [N, C]
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        num_classes = pred.shape[1] - 4 - (
            _SEG_COEFFS if self.task == "seg" else (1 if self.task == "obb" else 0)
        )
        if num_classes <= 0:
            raise InferenceUnavailable(
                f"输出通道数 {pred.shape[1]} 与任务 {self.task} 不匹配，请核对模型任务类型"
            )
        boxes_xywh = pred[:, :4]
        coeffs = pred[:, 4 + num_classes :] if self.task == "seg" else None

        # OBB 输出布局自适应（类别分数由图内 Sigmoid 产生恒非负，角度可负）：
        # - 新（ultralytics 8.4+）：[xywh, cls(nc), angle]，角度在最后一行
        # - 旧（8.3 及以前）：[xywh, angle, cls(nc)]，角度在 row4
        if self.task == "obb":
            cls_probe = pred[:, 4 : 4 + num_classes]
            if cls_probe.size and float(cls_probe.min()) < 0.0:
                score_offset, angle = 5, pred[:, 4]
            else:
                score_offset, angle = 4, pred[:, -1]
        else:
            score_offset, angle = 4, None
        scores_all = pred[:, score_offset : score_offset + num_classes]
        # 兼容未在图内做 sigmoid 的第三方导出（真 raw logits）
        if scores_all.size and float(scores_all.max()) > 1.0:
            scores_all = 1.0 / (1.0 + np.exp(-scores_all))

        class_ids = scores_all.argmax(axis=1)
        confidences = scores_all.max(axis=1)

        keep = confidences >= conf
        boxes_xywh = boxes_xywh[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]
        angle = angle[keep] if angle is not None else None
        coeffs = coeffs[keep] if coeffs is not None else None
        if len(confidences) == 0:
            return []

        if self.task == "obb":
            boxes_pts = [
                np.asarray(
                    rotate_box_points(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(a))
                )
                for b, a in zip(boxes_xywh, angle)
            ]
            keep_idx = nms_polygons(boxes_pts, confidences.tolist(), iou)
        else:
            keep_idx = self._nms(boxes_xywh, confidences, iou)

        proto = None
        if self.task == "seg":
            for out in outputs[1:]:
                if out.ndim == 4 and out.shape[1] == _SEG_COEFFS:
                    proto = out[0]
                    break
            if proto is None:
                raise InferenceUnavailable(
                    "seg 模型输出中未找到 proto 掩码 [1,32,mh,mw]，请确认 onnx 为 YOLO-seg 导出"
                )

        scale, pad_x, pad_y = params
        detections: list[Detection] = []
        for idx in keep_idx:
            x, y, w, h = boxes_xywh[idx]
            score = float(confidences[idx])
            cid = int(class_ids[idx])
            if self.task == "obb":
                points = rotate_box_points(
                    float(x), float(y), float(w), float(h), float(angle[idx])
                )
            elif self.task == "seg":
                box_img = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
                points = self._seg_polygon(
                    proto, coeffs[idx], box_img, params, image_bgr.shape[1], image_bgr.shape[0]
                )
                if points is None:
                    continue  # 掩码面积过小，跳过该实例
            else:
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            # letterbox 逆变换回原图坐标（seg 掩码已 resize 到有效区，仅除 scale）
            if self.task != "seg":
                points = [[(px - pad_x) / scale, (py - pad_y) / scale] for px, py in points]
            else:
                points = [[px / scale, py / scale] for px, py in points]
            detections.append(Detection(points=points, score=score, class_index=cid))
        return detections

    # ------------------------------------------------------------------ 内部

    @staticmethod
    def _seg_polygon(
        proto: np.ndarray,
        coeff: np.ndarray,
        box: list[float],
        params: tuple[float, float, float],
        orig_w: int,
        orig_h: int,
    ) -> list[list[float]] | None:
        """掩码系数与 proto 合成实例掩码，提取最大外轮廓为多边形。

        Args:
            proto: [32, mh, mw] proto 输出。
            coeff: [32] 实例掩码系数。
            box: letterbox 坐标系 xyxy。
            params: (scale, pad_x, pad_y) letterbox 参数。
            orig_w, orig_h: 原图尺寸。

        Returns:
            原图坐标多边形顶点；掩码面积过小时 None。
        """
        scale, pad_x, pad_y = params
        mh, mw = proto.shape[1], proto.shape[2]
        mask = coeff @ proto.reshape(_SEG_COEFFS, -1)
        mask = 1.0 / (1.0 + np.exp(-mask))  # sigmoid
        mask = mask.reshape(mh, mw)
        new_w = max(1, round(orig_w * scale))
        new_h = max(1, round(orig_h * scale))
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        x1 = max(0, int(box[0] - pad_x))
        y1 = max(0, int(box[1] - pad_y))
        x2 = min(new_w, int(round(box[2] - pad_x)))
        y2 = min(new_h, int(round(box[3] - pad_y)))
        if x2 <= x1 or y2 <= y1:
            return None
        canvas = np.zeros((new_h, new_w), dtype=np.float32)
        canvas[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        binary = (canvas >= _MASK_BIN_THRESHOLD).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < _MIN_CONTOUR_AREA * scale * scale:
            return None
        approx = cv2.approxPolyDP(largest, _CONTOUR_EPSILON, True)
        points = [[float(p[0][0] / scale), float(p[0][1] / scale)] for p in approx]
        return points if len(points) >= 3 else None

    @staticmethod
    def _letterbox(
        image: np.ndarray, imgsz: int
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        """等比缩放 + 居中填充到 imgsz×imgsz，返回 (NCHW tensor, (scale, pad_x, pad_y))。"""
        height, width = image.shape[:2]
        scale = min(imgsz / width, imgsz / height)
        new_w, new_h = round(width * scale), round(height * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (imgsz - new_w) / 2, (imgsz - new_h) / 2
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        canvas[
            round(pad_y) : round(pad_y) + new_h, round(pad_x) : round(pad_x) + new_w
        ] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        return tensor, (scale, pad_x, pad_y)

    @staticmethod
    def _nms(boxes_xywh: np.ndarray, confidences: np.ndarray, iou: float) -> list[int]:
        """HBB NMS（轴对齐 IoU，cv2.dnn.NMSBoxes）。"""
        boxes_xyxy = np.concatenate(
            [boxes_xywh[:, :2] - boxes_xywh[:, 2:] / 2, boxes_xywh[:, :2] + boxes_xywh[:, 2:] / 2],
            axis=1,
        )
        idx = cv2.dnn.NMSBoxes(
            bboxes=boxes_xyxy.tolist(),
            scores=confidences.tolist(),
            score_threshold=0.0,
            nms_threshold=iou,
        )
        return [int(i) for i in (idx.flatten() if hasattr(idx, "flatten") else idx)]


def nms_polygons(
    polygons: list[np.ndarray],
    scores: list[float],
    iou_threshold: float,
    max_candidates: int = 512,
) -> list[int]:
    """凸多边形 IoU NMS（按分数降序贪心抑制，带 bbox 预筛加速）。

    性能策略：
    1. 候选超过 max_candidates 时仅保留分数 topk（模型输出常含大量低分噪声）
    2. 轴对齐外接框不相交的对直接跳过精确多边形 IoU（O(N²) 的 bbox 广播预筛）

    Args:
        polygons: 每个 (4+, 2) 凸多边形顶点。
        scores: 对应分数。
        iou_threshold: 抑制阈值。
        max_candidates: 参与精确 NMS 的最大候选数。

    Returns:
        保留的原始索引列表。
    """
    if not polygons:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if len(order) > max_candidates:
        order = order[:max_candidates]
    # 外接框预筛矩阵
    boxes = np.full((len(order), 4), np.nan)
    for row, idx in enumerate(order):
        pts = polygons[idx]
        boxes[row] = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)

    keep: list[int] = []
    suppressed = np.zeros(len(order), dtype=bool)
    for row in range(len(order)):
        if suppressed[row]:
            continue
        best = order[row]
        keep.append(best)
        rest = np.where(~suppressed)[0]
        rest = rest[rest > row]
        if rest.size == 0:
            continue
        # bbox 相交判定（向量化）
        ix1 = np.maximum(boxes[row, 0], boxes[rest, 0])
        iy1 = np.maximum(boxes[row, 1], boxes[rest, 1])
        ix2 = np.minimum(boxes[row, 2], boxes[rest, 2])
        iy2 = np.minimum(boxes[row, 3], boxes[rest, 3])
        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        union = areas[row] + areas[rest] - inter
        bbox_iou = np.where(union > 0, inter / union, 0.0)
        candidates = rest[bbox_iou > 0]
        for ci in candidates:
            if rotated_iou(polygons[best], polygons[order[ci]]) >= iou_threshold:
                suppressed[ci] = True
    return keep


# ---------------------------------------------------------------------------
# 会话缓存（模型文件路径 -> YoloOrtModel）
# ---------------------------------------------------------------------------

_sessions: dict[str, YoloOrtModel] = {}
_sessions_lock = threading.Lock()


def get_session(file_path: str, task: str) -> YoloOrtModel:
    """获取（或创建）模型的推理会话。

    Args:
        file_path: onnx 模型路径。
        task: det / obb / seg。

    Returns:
        YoloOrtModel 会话实例。
    """
    key = f"{task}:{file_path}"
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = YoloOrtModel(file_path, task)
            _sessions[key] = session
        return session


def evict_session(file_path: str) -> None:
    """模型删除后清理缓存会话。"""
    with _sessions_lock:
        for key in [k for k in _sessions if k.endswith(file_path)]:
            _sessions.pop(key, None)
