# 2026 睿抗机器人大赛黑龙江赛区视觉分拣系统

本目录下新增 `app/` 作为比赛主程序，目标是覆盖任务书 7 个流程：系统唤醒、图像采集、单目标识别与高度测算、固定点位运动、坐标转换与指令下发、单物料抓取分类、多目标连续分拣。

## 运行方式

离线自检：

```bash
cd robotgame
python tools/offline_self_check.py
```

启动 GUI：

```bash
cd robotgame
python -m app.main
```

默认配置位于 `configs/competition.yaml`，当前 `mode: simulation`。现场接入真实硬件时改为 `mode: hardware`，并确认机器人 IP、料盒坐标、吸盘 IO、手眼矩阵和 YOLO 权重路径。

## 模块说明

`app/main.py` 是 PyQt5 全屏 GUI。A 区显示 RGB/Depth 双画面，B 区提供纯文本控制按钮，C 区显示状态日志和识别/高度/坐标结果。

`app/core/camera.py` 封装 RealSense D435，深度帧对齐到彩色帧，并支持保存 RGB、Depth 原始矩阵和深度可视化图。

`app/core/detector.py` 封装 YOLOv8 推理；模拟模式下返回 4 类水果的固定检测框，用于无硬件自检。

`app/core/height.py` 使用“桌面基准深度 - 物料顶面深度”的 ROI 中位/分位估计方式计算 15-60mm 高度。

`app/core/transform.py` 实现像素到相机坐标、相机到末端、末端到基坐标的转换，优先读取现有 `configs/20260610.yaml` 手眼矩阵。

`app/core/robot_client.py` 封装 DOBOT 29999 TCP/IP 指令，包括 `EnableRobot()`、`GetPose()`、`MovJ()`、`MovL()`、`ToolDOInstant()`。

`app/core/workflow.py` 是竞赛流程状态机，负责按钮动作、JSON 日志、抓取序列和自动分拣闭环。

## 现场联调顺序

先保持机械臂远离物料，单独验证相机双流和保存图像；然后验证 YOLO 检测与高度误差；再连接机器人测试 `GetPose()`、`EnableRobot()` 和吸盘 `ToolDOInstant()`；随后低速测试固定点位；最后只用单物料测试抓取轨迹，确认安全后再开启自动分拣。

## 待现场确认参数

料盒坐标、抓取姿态、吸盘 IO 编号、桌面基准深度、YOLO 最终权重、手眼标定矩阵都在 `configs/competition.yaml` 中集中配置。当前数值是安全占位或已有标定结果，必须在真实赛台上复核。
