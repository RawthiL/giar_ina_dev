from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()


class HeadConfig(BaseModel):
    hidden_dims: list[int] = [512, 256, 128]
    dropouts: list[float] = [0.3, 0.2, 0.0]
    activation: Literal["leaky_relu", "relu", "gelu"] = "leaky_relu"


class ModelConfig(BaseModel):
    arch: Literal["efficientnet_b1", "efficientnet_b2", "resnet50", "vgg19"] = "efficientnet_b1"
    pretrained: bool = True
    freeze_stages: int = 2
    head: HeadConfig = HeadConfig()


class LRSchedulerConfig(BaseModel):
    factor: float = 0.2
    patience: int = 5
    min_lr: float = 1e-6


class TrainingHPConfig(BaseModel):
    epochs: int = 30
    lr: float = 1e-5
    early_stopping_patience: int = 10
    class_weight_multipliers: dict[str, float] = {"mitosis": 2.0, "no_mitosis": 0.5}
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    augmentation: list[str] = ["hflip", "vflip", "color_jitter"]
    tensorboard: bool = True


class DataConfig(BaseModel):
    image_size: tuple[int, int] = (260, 260)
    batch_size: int = 32
    seed: int = 42
    normalize_mean: list[float] = [0.485, 0.456, 0.406]
    normalize_std: list[float] = [0.229, 0.224, 0.225]
    binary_classifier_crops_dir: Path = _ROOT / "datasets/crops/binary_classifier"
    experiments_dir: Path = _ROOT / "experiments"


class ExperimentConfig(BaseConfig):
    experiment_name: str
    model: ModelConfig = ModelConfig()
    training: TrainingHPConfig = TrainingHPConfig()
    data: DataConfig = DataConfig()
