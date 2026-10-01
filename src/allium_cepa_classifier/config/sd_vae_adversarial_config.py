from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .base_config import find_project_root
from .sd_vae_config import SDVAEExperimentConfig

_ROOT = find_project_root()


class AdversarialConfig(BaseModel):
    lambda_adv: float = 1.0
    recon_steps_per_disc_update: int = 1
    grad_accum_steps: int = 1
    lambda_ramp_steps: int = 0
    use_recon_mse: bool = True
    use_recon_lpips: bool = True
    weight_l2: float = 1.0
    weight_lpips: float = 1.0
    latent_dataset: Path = _ROOT / "datasets/latents/diffuser_latents.parquet"
    # Optional sidecar metadata written by generate_diffuser_latent_dataset.py.
    # When None, the sibling "<parquet_stem>.config.json" is used.
    latent_config: Path | None = None


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
    warmup_steps: int = 0


class SDVAEAdversarialExperimentConfig(SDVAEExperimentConfig):
    adversarial: AdversarialConfig = AdversarialConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()
