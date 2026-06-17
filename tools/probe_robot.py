from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.core.robot_client import DobotClient


def try_step(name, func):
    print(f"\n== {name} ==")
    try:
        result = func()
        print("OK:", result)
        return True
    except Exception as exc:
        print("FAIL:", repr(exc))
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="DOBOT TCP 快速探测：连接/RequestControl/Enable/GetPose")
    parser.add_argument("--ip", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-request-control", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "competition.yaml")
    config["mode"] = "hardware"
    if args.ip:
        config["robot"]["ip"] = args.ip
    if args.port:
        config["robot"]["dashboard_port"] = args.port
    if args.no_request_control:
        config["robot"]["request_control_before_enable"] = False

    client = DobotClient(config, simulation=False)
    ip = config["robot"]["ip"]
    port = int(config["robot"]["dashboard_port"])
    print(f"Target: {ip}:{port}")

    if not try_step("socket connect", lambda: client.connect().raw):
        return 2
    if not args.no_request_control:
        try_step("RequestControl", lambda: client.request_control().raw)
    try_step("EnableRobot", lambda: client.enable().raw)
    try_step("GetPose", lambda: client.get_pose())
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
