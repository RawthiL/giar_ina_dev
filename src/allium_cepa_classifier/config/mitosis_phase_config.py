from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .base_config import BaseConfig, find_project_root
from .experiment_config import HeadConfig, LRSchedulerConfig

_ROOT = find_project_root()

MITOSIS_CLASSES = ["prophase", "metaphase", "anaphase", "telophase"]


class MitosisModelConfig(BaseModel):
    arch: Literal["efficientnet_b1", "efficientnet_b2", "resnet50", "vgg19"] = "efficientnet_b2"
    pretrained: bool = True
    freeze_stages: int = 2
    head: HeadConfig = HeadConfig()


class MitosisTrainingConfig(BaseModel):
    epochs: int = 30
    lr: float = 1e-4
    early_stopping_patience: int = 10
    use_balanced_class_weights: bool = True
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    augmentation: list[str] = ["hflip", "vflip", "color_jitter"]
    tensorboard: bool = True


class MitosisDataConfig(BaseModel):
    image_size: tuple[int, int] = (260, 260)
    batch_size: int = 32
    seed: int = 42
    normalize_mean: list[float] = [0.485, 0.456, 0.406]
    normalize_std: list[float] = [0.229, 0.224, 0.225]
    classes: list[str] = MITOSIS_CLASSES
    # ImageFolder root with {train,validation,test}/{phase}/ subdirs (real-only).
    crops_dir: Path = _ROOT / "datasets/crops/mitosis_phase"
    # When True, train adds images from synthetic_dir; validation/test stay real-only
    # so the synthetic benefit is never confounded by synthetic eval data.
    include_synthetic: bool = False
    synthetic_dir: Path = _ROOT / "datasets/crops/mitosis_phase_synth"
    experiments_dir: Path = _ROOT / "experiments"


class MitosisPhaseConfig(BaseConfig):
    experiment_name: str
    model: MitosisModelConfig = MitosisModelConfig()
    training: MitosisTrainingConfig = MitosisTrainingConfig()
    data: MitosisDataConfig = MitosisDataConfig()
