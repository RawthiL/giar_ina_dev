from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()


class SDVAEModelConfig(BaseModel):
    pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4"
    subfolder: str = "vae"
    resolution: int = 200
    decoder_only: bool = True
    weight_l2: float = 0.5
    weight_kl: float = 0.001
    weight_lpips: float = 0.002


class LRSchedulerConfig(BaseModel):
    factor: float = 0.2
    patience: int = 7
    min_lr: float = 1e-6


class SDVAETrainingConfig(BaseModel):
    epochs: int = 20
    lr: float = 1e-4
    batch_size: int = 16
    mixed_precision: bool = False
    early_stopping_patience: int = 7
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    tensorboard: bool = True


class SDVAEDataConfig(BaseModel):
    vae_crops_dir: Path = _ROOT / "datasets/crops/vae"
    sources: list[Literal["tagged", "untagged"]] = ["tagged", "untagged"]
    seed: int = 42


class SDVAEExperimentConfig(BaseConfig):
    experiment_name: str
    model: SDVAEModelConfig = SDVAEModelConfig()
    training: SDVAETrainingConfig = SDVAETrainingConfig()
    data: SDVAEDataConfig = SDVAEDataConfig()
