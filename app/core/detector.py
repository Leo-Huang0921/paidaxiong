from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import Detection


class FruitDetector:
    def __init__(self, config: dict, simulation: bool = False):
        self.config = config
        self.simulation = simulation
        self.model = None
        self.class_names = list(config["model"]["class_names"])
        if not simulation:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("未安装 ultralytics，无法加载 YOLO 模型") from exc
        weights = Path(self.config["model"]["weights"])
        if not weights.exists():
            weights = Path(self.config["model"]["fallback_weights"])
        self.model = YOLO(str(weights))

    def detect(self, color_image: np.ndarray) -> list[Detection]:
        if self.simulation or self.model is None:
            return self._detect_simulated(color_image)
        conf_threshold = float(self.config["model"]["conf_threshold"])
        results = self.model.predict(color_image, conf=conf_threshold, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].cpu().item())
                label = str(names.get(cls_id, self.class_names[cls_id] if cls_id < len(self.class_names) else cls_id))
                confidence = float(box.conf[0].cpu().item())
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append(Detection(label=label, confidence=confidence, bbox=(x1, y1, x2, y2), center=((x1 + x2) // 2, (y1 + y2) // 2)))
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    def _detect_simulated(self, color_image: np.ndarray) -> list[Detection]:
        height, width = color_image.shape[:2]
        boxes = [
            ("apple", 0.94, (140, 140, 220, 220)),
            ("banana", 0.91, (275, 125, 355, 205)),
            ("strawberry", 0.88, (400, 220, 480, 300)),
            ("grape", 0.86, (200, 275, 280, 355)),
        ]
        detections = []
        for label, conf, (x1, y1, x2, y2) in boxes:
            x1, x2 = max(0, x1), min(width - 1, x2)
            y1, y2 = max(0, y1), min(height - 1, y2)
            detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2), center=((x1 + x2) // 2, (y1 + y2) // 2)))
        return detections
