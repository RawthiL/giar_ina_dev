from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()


class VAEModelConfig(BaseModel):
    latent_dim: int = 32
    image_size: tuple[int, int] = (200, 200)
    in_channels: int = 1
    encoder_filters: list[int] = [32, 64, 128, 256]
    decoder_filters: list[int] = [256, 128, 64, 32]


class KLAnnealingConfig(BaseModel):
    enabled: bool = False
    start: float = 0.0
    end: float = 1.0
    duration_steps: int = 1000


class LRSchedulerConfig(BaseModel):
    factor: float = 0.2
    patience: int = 7
    min_lr: float = 1e-6


class VAETrainingConfig(BaseModel):
    epochs: int = 20
    lr: float = 1e-4
    batch_size: int = 64
    recon_loss: Literal["mse", "bce", "edge"] = "edge"
    recon_gamma: float = 0.8
    beta: float = 1.0
    learnable_prior: bool = True
    early_stopping_patience: int = 7
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    kl_annealing: KLAnnealingConfig = KLAnnealingConfig()
    tensorboard: bool = True


class VAEDataConfig(BaseModel):
    sources: list[Literal["tagged", "untagged"]] = ["tagged", "untagged"]
    vae_crops_dir: Path = _ROOT / "datasets/crops/vae"
    online_augment: bool = False
    seed: int = 42


class VAEExperimentConfig(BaseConfig):
    experiment_name: str
    model: VAEModelConfig = VAEModelConfig()
    training: VAETrainingConfig = VAETrainingConfig()
    data: VAEDataConfig = VAEDataConfig()
