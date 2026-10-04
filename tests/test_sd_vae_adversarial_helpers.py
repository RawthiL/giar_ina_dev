"""Unit tests for the SD-VAE adversarial trainer helpers (metrics, norm pinning, D inputs)."""

import pytest
import torch
import torch.nn as nn

from allium_cepa_classifier.config.sd_vae_adversarial_config import SDVAEAdversarialExperimentConfig
from allium_cepa_classifier.training.sd_vae_adversarial_trainer import (
    DiscPreprocess,
    _degrade_roundtrip,
    _disc_input_transform,
    _freeze_norm_stats,
    _hf_energy,
    _MetricAgg,
)


def test_generator_window_ignores_discriminator_warmup_steps():
    wd, wg = _MetricAgg(), _MetricAgg()
    for _ in range(5):  # discriminator warmup: D updates, G does not
        wd.push(2, loss=1.38, acc_r=0.5, acc_f=0.5)
    for _ in range(10):
        wd.push(2, loss=1.20, acc_r=0.8, acc_f=0.7)
        wg.push(2, l1=0.01, adv=0.02)

    assert wd.weight == 30
    assert wg.weight == 20
    assert wg.mean("l1") == pytest.approx(0.01)
    assert wd.mean("loss") == pytest.approx((1.38 * 10 + 1.20 * 20) / 30)

    # Diluting G metrics by every step (warmup included) is what made the curves climb.
    assert wg.mean("l1") > (0.01 * 20) / 30


def test_metric_agg_window_reset_and_missing_keys():
    agg = _MetricAgg()
    assert agg.mean("adv") == 0.0
    agg.push(4, adv=0.25)
    assert agg.mean("adv") == pytest.approx(0.25)
    # Keys never pushed (loss term disabled) report 0.0 instead of raising
    assert agg.mean("l1") == 0.0
    agg.reset()
    assert agg.weight == 0.0
    assert agg.mean("adv") == 0.0


def _conv_bn() -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.BatchNorm2d(2))


def test_freeze_norm_stats_pins_running_stats_across_train_calls():
    model = _conv_bn()
    conv, bn = model[0], model[1]
    with torch.no_grad():
        bn.running_mean.fill_(0.5)
        bn.running_var.fill_(1.0)
    mean_before = bn.running_mean.clone()

    _freeze_norm_stats(model)
    # The trainer calls module.train() at the start of every epoch, then re-pins.
    model.train()
    _freeze_norm_stats(model)
    assert bn.momentum == 0.0
    assert not bn.training

    x = torch.randn(1, 1, 4, 4)
    with torch.no_grad():
        out = model(x)
        h = conv(x)

    assert torch.equal(bn.running_mean, mean_before)
    # Normalization must use the pinned running stats, not this batch of 2 images
    expected = (h - bn.running_mean.view(1, -1, 1, 1)) / torch.sqrt(
        bn.running_var.view(1, -1, 1, 1) + bn.eps
    )
    assert torch.allclose(out, expected, atol=1e-5)


def test_adversarial_defaults_use_l1_and_freeze_norm_stats():
    cfg = SDVAEAdversarialExperimentConfig(experiment_name="unit_test")
    assert not hasattr(cfg.adversarial, "weight_l2")
    assert not hasattr(cfg.adversarial, "use_recon_mse")
    assert cfg.adversarial.use_recon_l1
    assert cfg.adversarial.weight_l1 > 0
    assert cfg.adversarial.lambda_adv < cfg.adversarial.weight_l1
    assert cfg.adversarial.degrade_roundtrip_mid is None
    assert cfg.discriminator.freeze_norm_stats
    assert 0.0 <= cfg.discriminator.label_smoothing < 0.5
    assert cfg.discriminator.backbone_lr <= cfg.discriminator.lr


def _checker(size: int = 512, period: int = 4) -> torch.Tensor:
    idx = torch.arange(size)
    parity = ((idx[None, :] // period + idx[:, None] // period) % 2).float()
    return parity.view(1, 1, size, size).repeat(1, 3, 1, 1) * 2 - 1


def _pp(degrade_mid: int | None = None) -> DiscPreprocess:
    return DiscPreprocess(
        image_size=512,
        mode="bilinear",
        antialias=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        degrade_mid=degrade_mid,
    )


def test_roundtrip_degradation_attenuates_high_frequency():
    x = _checker(512, period=4)
    degraded = _disc_input_transform(x, _pp(200))
    raw = _disc_input_transform(x, _pp(None))
    assert degraded.shape == raw.shape
    assert _hf_energy(degraded) < 0.5 * _hf_energy(raw)


def test_roundtrip_is_identity_when_mid_not_smaller_and_off_by_default():
    x = _checker(512, period=8)
    assert torch.equal(_degrade_roundtrip(x, 512), x)
    assert _pp().degrade_mid is None
    assert torch.equal(_disc_input_transform(x, _pp(None)), _disc_input_transform(x, _pp()))


def test_hf_energy_tracks_spatial_frequency():
    fine = _disc_input_transform(_checker(512, 4), _pp())
    coarse = _disc_input_transform(_checker(512, 32), _pp())
    flat = _disc_input_transform(torch.zeros(1, 3, 512, 512), _pp())
    assert _hf_energy(fine) > _hf_energy(coarse) > _hf_energy(flat)
    assert _hf_energy(flat) == pytest.approx(0.0)
