from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class CameraFrame:
    color: Any
    depth_mm: Any
    intrinsics: CameraIntrinsics
    timestamp: float = field(default_factory=time)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]


@dataclass
class TargetResult:
    detection: Detection
    height_mm: float
    camera_xyz_mm: tuple[float, float, float]
    base_pose: list[float]
