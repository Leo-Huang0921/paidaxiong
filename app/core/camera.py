from __future__ import annotations

from pathlib import Path
from time import time

import numpy as np

from .types import CameraFrame, CameraIntrinsics


class CameraError(RuntimeError):
    pass


class RealSenseCamera:
    def __init__(self, config: dict):
        self.config = config
        self.pipeline = None
        self.align = None
        self.profile = None
        self.intrinsics = CameraIntrinsics(
            fx=float(config["calibration"]["camera_matrix"]["fx"]),
            fy=float(config["calibration"]["camera_matrix"]["fy"]),
            cx=float(config["calibration"]["camera_matrix"]["cx"]),
            cy=float(config["calibration"]["camera_matrix"]["cy"]),
        )

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            raise CameraError("未安装或未检测到 pyrealsense2，无法启动 RealSense") from exc
        camera_cfg = self.config["camera"]
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, int(camera_cfg["width"]), int(camera_cfg["height"]), rs.format.z16, int(camera_cfg["fps"]))
        rs_config.enable_stream(rs.stream.color, int(camera_cfg["width"]), int(camera_cfg["height"]), rs.format.bgr8, int(camera_cfg["fps"]))
        self.profile = self.pipeline.start(rs_config)
        self.align = rs.align(rs.stream.color)
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.intrinsics = CameraIntrinsics(float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy))

    def read(self) -> CameraFrame:
        if self.pipeline is None or self.align is None:
            raise CameraError("RealSense 未启动")
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise CameraError("未获取到有效 RGB/Depth 帧")
        color = np.asanyarray(color_frame.get_data())
        depth_mm = np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(depth_frame.get_units()) * 1000.0
        return CameraFrame(color=color, depth_mm=depth_mm, intrinsics=self.intrinsics, timestamp=time())

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        self.pipeline = None
        self.align = None


def save_capture(frame: CameraFrame, output_dir: str | Path, prefix: str | None = None) -> dict[str, Path]:
    import cv2
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or time().__format__(".3f").replace(".", "_")
    rgb_path = output_dir / f"{prefix}_rgb.jpg"
    depth_path = output_dir / f"{prefix}_depth.npy"
    depth_vis_path = output_dir / f"{prefix}_depth_vis.png"
    cv2.imwrite(str(rgb_path), frame.color)
    np.save(depth_path, frame.depth_mm)
    valid = frame.depth_mm[np.isfinite(frame.depth_mm) & (frame.depth_mm > 0)]
    max_depth = float(np.percentile(valid, 95)) if valid.size else 1000.0
    depth_vis = np.clip(frame.depth_mm / max_depth * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(depth_vis_path), depth_vis)
    return {"rgb": rgb_path, "depth": depth_path, "depth_vis": depth_vis_path}
