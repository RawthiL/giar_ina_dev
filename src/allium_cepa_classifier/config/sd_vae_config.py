from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()


class SDVAELoRAConfig(BaseModel):
    enabled: bool = True
    r: int = 8
    alpha: float = 8.0
    target_modules: list[str] = ["conv", "Conv2d", "Linear"]
    dropout: float = 0.0
    merge_and_save_full: bool = True


class SDVAEModelConfig(BaseModel):
    pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4"
    subfolder: str = "vae"
    resolution: int = 200
    decoder_only: bool = True
    freeze_quant_conv: bool = True
    weight_l2: float = 0.5
    weight_kl: float = 0.001
    weight_lpips: float = 0.002
    lora: SDVAELoRAConfig = SDVAELoRAConfig()


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
    log_every_n_steps: int = 50
    log_images_every_n_steps: int = 0
    log_images_n_latents: int = 4
    eval_batch_size: int = 8
    random_sample_batch_size: int = 8


class SDVAEValidationConfig(BaseModel):
    images: list[Path] | None = None


class SDVAEDataConfig(BaseModel):
    vae_crops_dir: Path = _ROOT / "datasets/crops/vae"
    sources: list[Literal["tagged", "untagged"]] = ["tagged", "untagged"]
    seed: int = 42
    balanced_sampling: bool = False
    untagged_prob: float = 0.5
    balanced_epoch_multiplier: int = 5


class SDVAEExperimentConfig(BaseConfig):
    experiment_name: str
    model: SDVAEModelConfig = SDVAEModelConfig()
    training: SDVAETrainingConfig = SDVAETrainingConfig()
    data: SDVAEDataConfig = SDVAEDataConfig()
    validation: SDVAEValidationConfig = SDVAEValidationConfig()
