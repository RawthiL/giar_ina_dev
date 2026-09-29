from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .base_config import BaseConfig, find_project_root
from .sd_vae_config import SDVAEExperimentConfig

_ROOT = find_project_root()


class AdversarialConfig(BaseModel):
    lambda_adv: float = 1.0
    recon_steps_per_disc_update: int = 1
    use_recon_mse: bool = True
    use_recon_lpips: bool = True
    weight_l2: float = 0.5
    weight_lpips: float = 0.002
    latent_dataset: Path = _ROOT / "datasets/latents/diffuser_latents.parquet"
    latent_config: Path = _ROOT / "datasets/latents/diffuser_latents.config.json"


class DiscDownsampleConfig(BaseModel):
    mode: Literal["bilinear", "bicubic", "area"] = "bilinear"
    antialias: bool = True


class DiscriminatorConfig(BaseModel):
    checkpoint: Path = _ROOT / "src/allium_cepa_classifier/weights/classifier_calibrated.pt"
    arch: Literal["efficientnet_b1", "efficientnet_b2", "resnet50", "vgg19"] = "efficientnet_b2"
    trainable: Literal["all", "head"] = "all"
    image_size: int = 260
    downsample: DiscDownsampleConfig = DiscDownsampleConfig()
    lr: float = 1e-4


class SDVAEAdversarialExperimentConfig(SDVAEExperimentConfig):
    adversarial: AdversarialConfig = AdversarialConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()
