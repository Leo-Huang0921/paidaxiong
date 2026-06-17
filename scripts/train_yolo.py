from ultralytics import YOLO
from pathlib import Path
import os

if __name__ == '__main__':
    # 获取项目根目录 (robotgame/)
    root_dir = Path(__file__).resolve().parent.parent

    # 动态构建指向 models 和 configs 的绝对路径
    model_path = root_dir / 'models' / 'yolov8s.pt'
    data_config = root_dir / 'configs' / 'fruit.yaml'

    model = YOLO(str(model_path))

    print("🚀 开始使用高阶数据增强训练【工业级小目标】模型...")

    results = model.train(
        data=str(data_config),  # 使用绝对路径
        epochs=100,
        imgsz=1280,
        batch=8,
        workers=4,
        device=0,
        project='runs/train',
        name='fruit_yolov8s_advanced',

        # ================= 高阶数据增强参数 =================
        mosaic=1.0,
        scale=0.5,
        degrees=15.0,
        hsv_s=0.5,
        fliplr=0.5
    )

    print("🎉 进阶版模型训练彻底完成！")