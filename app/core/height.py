from __future__ import annotations

import numpy as np

from .types import Detection


class HeightEstimator:
    def __init__(self, config: dict):
        self.config = config
        self.table_depth_mm = float(config["height"]["table_depth_mm"])

    def calibrate_table(self, depth_mm: np.ndarray) -> float:
        valid = depth_mm[np.isfinite(depth_mm) & (depth_mm > 0)]
        if valid.size:
            self.table_depth_mm = float(np.median(valid))
        return self.table_depth_mm

    def estimate(self, depth_mm: np.ndarray, detection: Detection) -> float:
        x1, y1, x2, y2 = detection.bbox
        roi_ratio = float(self.config["height"].get("roi_ratio", 0.35))
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        cx, cy = detection.center
        half_w = max(2, int(width * roi_ratio / 2))
        half_h = max(2, int(height * roi_ratio / 2))
        rx1 = max(0, cx - half_w)
        rx2 = min(depth_mm.shape[1], cx + half_w)
        ry1 = max(0, cy - half_h)
        ry2 = min(depth_mm.shape[0], cy + half_h)
        roi = depth_mm[ry1:ry2, rx1:rx2]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        if valid.size == 0:
            return 0.0
        top_depth = float(np.percentile(valid, 35))
        object_height = self.table_depth_mm - top_depth
        min_h = float(self.config["height"].get("min_height_mm", 15.0))
        max_h = float(self.config["height"].get("max_height_mm", 60.0))
        return float(np.clip(object_height, min_h, max_h))
