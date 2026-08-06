from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict

from .base_config import BaseConfig


class DetectorConfig(BaseConfig):
    """Configuration for YOLO detector training."""

    model_config = ConfigDict(frozen=True)

    weights: Path = Path("src/allium_cepa_classifier/weights/yolo11n.pt")
    data: Path = Path("datasets/yolo_dataset/data.yaml")
    epochs: int = 200
    imgsz: int = 640
    device: str = "0"
    out: Path = Path("src/allium_cepa_classifier/weights/object_detection.pt")
    tensorboard: bool = True
