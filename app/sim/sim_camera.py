from __future__ import annotations

from time import time

import numpy as np

from app.core.types import CameraFrame, CameraIntrinsics


class SimulatedCamera:
    def __init__(self, config: dict):
        self.config = config
        cam_cfg = config["camera"]
        cal_cfg = config["calibration"]["camera_matrix"]
        self.width = int(cam_cfg["width"])
        self.height = int(cam_cfg["height"])
        self.frame_index = 0
        self.intrinsics = CameraIntrinsics(float(cal_cfg["fx"]), float(cal_cfg["fy"]), float(cal_cfg["cx"]), float(cal_cfg["cy"]))
        self.labels = list(config["model"]["class_names"])

    def start(self) -> None:
        self.frame_index = 0

    def read(self) -> CameraFrame:
        import cv2
        self.frame_index += 1
        color = np.full((self.height, self.width, 3), (35, 45, 50), dtype=np.uint8)
        depth = np.full((self.height, self.width), float(self.config["height"]["table_depth_mm"]), dtype=np.float32)
        centers = [(180, 180), (315, 165), (440, 260), (240, 315)]
        radii = [32, 30, 28, 31]
        heights = [35.0, 48.0, 26.0, 55.0]
        colors = [(0, 0, 220), (0, 220, 220), (40, 40, 230), (160, 60, 160)]
        for idx, (center, radius, height, bgr) in enumerate(zip(centers, radii, heights, colors)):
            cx = center[0] + int(6 * np.sin((self.frame_index + idx * 9) / 20.0))
            cy = center[1]
            cv2.circle(color, (cx, cy), radius + 7, (235, 235, 235), -1)
            cv2.circle(color, (cx, cy), radius, bgr, -1)
            cv2.putText(color, self.labels[idx][0].upper(), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius + 5, 255, -1)
            depth[mask > 0] = float(self.config["height"]["table_depth_mm"]) - height
        cv2.putText(color, "SIMULATION MODE", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2)
        return CameraFrame(color=color, depth_mm=depth, intrinsics=self.intrinsics, timestamp=time())

    def stop(self) -> None:
        pass
