from __future__ import annotations

import json
from pathlib import Path
from time import strftime

import numpy as np

from .camera import RealSenseCamera, save_capture
from .detector import FruitDetector
from .height import HeightEstimator
from .robot_client import DobotClient
from .transform import CoordinateTransformer
from .types import Detection, TargetResult
from app.sim.sim_camera import SimulatedCamera


class CompetitionWorkflow:
    def __init__(self, config: dict, logger=print):
        self.config = config
        self.log = logger
        self.simulation = config.get("mode") == "simulation"
        self.camera = SimulatedCamera(config) if self.simulation else RealSenseCamera(config)
        self.detector = FruitDetector(config, simulation=self.simulation)
        self.height_estimator = HeightEstimator(config)
        self.transformer = CoordinateTransformer(config)
        self.robot = DobotClient(config, simulation=self.simulation)
        self.frame = None
        self.detections: list[Detection] = []
        self.current_target: TargetResult | None = None

    def start_camera(self) -> None:
        self.camera.start()
        self.frame = self.camera.read()
        self.log("相机已启动：RGB/Depth 双流就绪" + ("（模拟模式）" if self.simulation else ""))

    def read_frame(self):
        self.frame = self.camera.read()
        return self.frame

    def save_current_frame(self) -> dict[str, Path]:
        if self.frame is None:
            self.read_frame()
        output_dir = Path(__file__).resolve().parents[2] / "runs" / "captures" / strftime("%Y%m%d")
        paths = save_capture(self.frame, output_dir)
        self.log(f"图像已保存：{paths['rgb'].name} / {paths['depth'].name}")
        return paths

    def detect_once(self) -> list[Detection]:
        if self.frame is None:
            self.read_frame()
        self.detections = self.detector.detect(self.frame.color)
        if not self.detections:
            self.current_target = None
            self.log("未识别到有效目标")
            return []
        target = self.detections[0]
        object_height = self.height_estimator.estimate(self.frame.depth_mm, target)
        cx, cy = target.center
        depth = self._depth_at(cx, cy)
        camera_xyz = self.transformer.pixel_to_camera(cx, cy, depth, self.frame.intrinsics)
        base_pose = self.transformer.camera_to_base_pose(camera_xyz, self.robot.current_pose, [180.0, 0.0, 0.0])
        base_pose[2] = max(20.0, base_pose[2] - object_height)
        self.current_target = TargetResult(target, object_height, camera_xyz, base_pose)
        self.log(f"识别成功：{target.label} conf={target.confidence:.2f} pixel={target.center} height={object_height:.1f}mm")
        return self.detections

    def _depth_at(self, u: int, v: int) -> float:
        if self.frame is None:
            raise RuntimeError("没有相机帧")
        height, width = self.frame.depth_mm.shape[:2]
        x1, x2 = max(0, u - 3), min(width, u + 4)
        y1, y2 = max(0, v - 3), min(height, v + 4)
        roi = self.frame.depth_mm[y1:y2, x1:x2]
        valid = roi[np.isfinite(roi) & (roi > 0)]
        if valid.size == 0:
            return float(self.config["height"]["table_depth_mm"])
        return float(np.median(valid))

    def connect_robot(self) -> None:
        try:
            self.robot.connect()
            self.robot.enable()
            self.robot.get_pose()
        except Exception:
            self.robot.close()
            raise
        self.log("机器人已连接并使能" + ("（模拟模式）" if self.simulation else ""))

    def fixed_point_test(self) -> None:
        self._ensure_robot_ready()
        grasp_pose = list(self.config["robot"]["fixed_test_pose"])
        bins = self.config["robot"].get("bins", {})
        if not bins:
            raise RuntimeError("未配置任何置物盒坐标，无法完成定点抓取测试")
        target_label = next(iter(bins.keys()))
        sequence = self.robot.build_grasp_sequence(grasp_pose, target_label)
        self.log(f"定点抓取测试开始：固定抓取点={grasp_pose}，目标置物盒={target_label}")
        self._execute_sequence(sequence)
        self.log(f"定点抓取测试完成：已从固定点抓取并放置到 {target_label} 置物盒")

    def calculate_grasp(self) -> TargetResult:
        if self.current_target is None:
            self.detect_once()
        if self.current_target is None:
            raise RuntimeError("无有效目标，无法计算抓取坐标")
        payload = {
            "cmd": "grasp_target",
            "label": self.current_target.detection.label,
            "pixel": list(self.current_target.detection.center),
            "height_mm": round(self.current_target.height_mm, 2),
            "base_pose": [round(value, 3) for value in self.current_target.base_pose],
        }
        self.log("抓取坐标JSON: " + json.dumps(payload, ensure_ascii=False))
        return self.current_target

    def execute_grasp(self) -> None:
        self._ensure_robot_ready()
        target = self.calculate_grasp()
        if target.detection.label not in self.config["robot"]["bins"]:
            raise RuntimeError(f"未配置 {target.detection.label} 的料盒坐标")
        sequence = self.robot.build_grasp_sequence(target.base_pose, target.detection.label)
        self._execute_sequence(sequence)
        self.log(f"单物料分拣完成：{target.detection.label}")

    def _execute_sequence(self, sequence: list[tuple[str, list[float] | bool]]) -> None:
        for action, value in sequence:
            if action == "movj":
                self.robot.movj(value)
            elif action == "movl":
                self.robot.movl(value)
            elif action == "suction":
                self.robot.suction(bool(value))
                self.log("吸盘得电吸取" if value else "吸盘断电释放")
            self.log(f"执行：{action} {value}")

    def auto_run(self) -> int:
        self._ensure_robot_ready()
        completed = 0
        max_objects = int(self.config["workflow"]["auto_max_objects"])
        empty_frames = 0
        while completed < max_objects:
            self.read_frame()
            self.detect_once()
            if self.current_target is None:
                empty_frames += 1
                if empty_frames >= int(self.config["workflow"]["auto_empty_frames_to_finish"]):
                    break
                continue
            empty_frames = 0
            self.execute_grasp()
            completed += 1
        self.log(f"自动分拣结束：完成 {completed} 个目标")
        return completed

    def _ensure_robot_ready(self) -> None:
        if not self.robot.connected:
            self.connect_robot()
        if not self.robot.enabled:
            self.robot.enable()

    def emergency_stop(self) -> None:
        errors: list[str] = []
        try:
            if self.robot.connected:
                try:
                    self.robot.suction(False)
                except Exception as exc:
                    errors.append(f"关闭吸盘失败: {exc}")
                try:
                    mode = str(self.config["robot"].get("emergency_stop_mode", "stop")).lower()
                    if mode == "emergency_stop":
                        self.robot.emergency_stop()
                        self.log("急停触发：已发送 EmergencyStop(1)，机器人会下使能并报警")
                    else:
                        self.robot.stop_motion()
                        self.log("急停触发：已发送 Stop()，运动队列已请求停止")
                except Exception as exc:
                    errors.append(f"停止运动失败: {exc}")
        finally:
            self.robot.enabled = False
            if errors:
                self.log("；".join(errors))
