from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .base_config import find_project_root
from .sd_vae_config import SDVAEExperimentConfig

_ROOT = find_project_root()


class AdversarialConfig(BaseModel):
    lambda_adv: float = 0.02
    recon_steps_per_disc_update: int = 1
    grad_accum_steps: int = 4
    lambda_ramp_steps: int = 2000
    use_recon_l1: bool = True
    use_recon_lpips: bool = True
    weight_l1: float = 2.0
    weight_lpips: float = 0.02
    # Global-norm clip on generator grads at each optimizer step (0 disables).
    grad_clip_norm: float = 5.0
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
    backbone_lr: float = 1e-5
    label_smoothing: float = 0.1
    freeze_norm_stats: bool = True
    warmup_steps: int = 0


class SDVAEAdversarialExperimentConfig(SDVAEExperimentConfig):
    adversarial: AdversarialConfig = AdversarialConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()
