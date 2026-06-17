from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.core.robot_client import DobotClient


def main() -> int:
    parser = argparse.ArgumentParser(description="DOBOT 29999 TCP/IP 现场通讯诊断")
    parser.add_argument("--ip", default=None, help="机器人IP，默认读取 configs/competition.yaml")
    parser.add_argument("--enable", action="store_true", help="是否执行 EnableRobot()")
    parser.add_argument("--suction-test", action="store_true", help="是否测试 ToolDOInstant 吸盘开关")
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "competition.yaml")
    config["mode"] = "hardware"
    if args.ip:
        config["robot"]["ip"] = args.ip

    client = DobotClient(config, simulation=False)
    print(f"连接机器人 {config['robot']['ip']}:{config['robot']['dashboard_port']} ...")
    client.connect()
    print("连接成功")

    if args.enable:
        reply = client.enable()
        print("EnableRobot:", reply.raw)

    pose = client.get_pose()
    print("GetPose:", pose)

    if args.suction_test:
        print("打开吸盘:", client.suction(True).raw)
        print("关闭吸盘:", client.suction(False).raw)

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
