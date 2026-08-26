"""几何小工具：滑窗起点序列（chip_export 与 scene_infer 共用）。"""

from __future__ import annotations


def slide_starts(total: int, chip: int, step: int) -> list[int]:
    """生成滑窗起点序列（覆盖 [0, total)，末窗对齐右边界）。

    策略：从 0 按 step 步进生成完整 chip 窗口起点；若末尾仍有未覆盖区域，
    追加一个贴 total 右边界的完整窗口（保证全覆盖且窗口尺寸统一）。

    Args:
        total: 总长度（像素）。
        chip: 窗口尺寸（像素）。
        step: 步长（像素）。

    Returns:
        起点列表；total <= chip 时为 [0]。
    """
    if total <= chip:
        return [0]
    starts = list(range(0, total - chip + 1, step))
    if not starts or starts[-1] + chip < total:
        starts.append(total - chip)
    return starts
