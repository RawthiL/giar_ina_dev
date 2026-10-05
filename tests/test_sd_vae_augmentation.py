"""Regression tests for the SD-VAE online augmentation (reflect-padded rotation)."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from allium_cepa_classifier.config.sd_vae_config import (
    SDVAEAugmentationConfig,
    SDVAEExperimentConfig,
)
from allium_cepa_classifier.training.sd_vae_trainer import _build_train_transform

_ROOT = Path(__file__).resolve().parent.parent


def _solid(res: int, val: int = 200) -> Image.Image:
    return Image.fromarray(np.full((res, res, 3), val, dtype=np.uint8))


def test_reflect_rotation_keeps_solid_uniform():
    """Mirror fill must not inject a constant corner block (which would break uniformity)."""
    aug = SDVAEAugmentationConfig(enabled=True, rotation_degrees=15.0)
    tfm = _build_train_transform(128, aug)
    t = tfm(_solid(128))
    assert t.shape == (3, 128, 128)
    assert (t - t.mean()).abs().max() < 1e-5, "non-uniform pixels => constant fill leaked"


def test_vae256_config_augment_loads_and_shapes():
    cfg = SDVAEExperimentConfig.from_yaml(
        _ROOT / "experiments" / "sd_vae_finetune" / "vae_256" / "config.yaml"
    )
    assert not hasattr(cfg.data.augmentation, "rotation_fill")
    assert cfg.data.augmentation.rotation_degrees == 5.0
    tfm = _build_train_transform(cfg.model.resolution, cfg.data.augmentation)
    t = tfm(_solid(cfg.model.resolution))
    assert tuple(t.shape) == (3, cfg.model.resolution, cfg.model.resolution)
    assert torch.is_tensor(t) and t.min() >= -1.0 and t.max() <= 1.0
