import sys
import time

sys.path.insert(0, r"D:\Workspace\SarDetection\t2_qgis_intelligent\src")
import numpy as np

from qgis_yolo_annotator.core.inference import nms_polygons, rotate_box_points

rng = np.random.default_rng(0)
n = 2000
polys = [
    np.asarray(
        rotate_box_points(
            rng.uniform(0, 2500), rng.uniform(0, 2500),
            rng.uniform(20, 80), rng.uniform(10, 40), rng.uniform(-3.14, 3.14),
        )
    )
    for _ in range(n)
]
scores = list(rng.uniform(0.15, 0.99, n))
t0 = time.time()
keep = nms_polygons(polys, scores, 0.5)
dt = time.time() - t0
print(f"nms 2000 candidates: {dt:.2f}s, keep={len(keep)}")
