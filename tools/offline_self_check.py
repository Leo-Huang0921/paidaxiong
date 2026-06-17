from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.core.workflow import CompetitionWorkflow


def main() -> int:
    config = load_config(ROOT / "configs" / "competition.yaml")
    config["mode"] = "simulation"
    logs: list[str] = []
    workflow = CompetitionWorkflow(config, logger=logs.append)
    workflow.start_camera()
    workflow.read_frame()
    paths = workflow.save_current_frame()
    detections = workflow.detect_once()
    if not detections:
        raise RuntimeError("模拟检测失败")
    workflow.connect_robot()
    workflow.fixed_point_test()
    target = workflow.calculate_grasp()
    if target.height_mm <= 0:
        raise RuntimeError("高度估计失败")
    workflow.execute_grasp()
    print("OFFLINE_SELF_CHECK_OK")
    print(f"detections={len(detections)} target={target.detection.label} height={target.height_mm:.1f}mm")
    print("capture_rgb=" + str(paths["rgb"]))
    print("last_commands=" + " | ".join(workflow.robot.sent_commands[-5:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
