"""Unit tests for VAE model architecture and KLAnnealer."""

import torch

from allium_cepa_classifier.config.vae_config import (
    KLAnnealingConfig,
    VAEModelConfig,
    VAETrainingConfig,
)
from allium_cepa_classifier.training.vae_model import VAE, Decoder, Encoder
from allium_cepa_classifier.training.vae_trainer import (
    KLAnnealer,
    _sobel_edges,
    compute_vae_loss,
)


def _cfg() -> VAEModelConfig:
    return VAEModelConfig()


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def test_encoder_output_shapes():
    enc = Encoder(_cfg())
    x = torch.zeros(3, 1, 200, 200)
    z_mean, z_log_var = enc(x)
    assert z_mean.shape == (3, 32)
    assert z_log_var.shape == (3, 32)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def test_decoder_output_shape_and_range():
    dec = Decoder(_cfg())
    z = torch.randn(3, 32)
    out = dec(z)
    assert out.shape == (3, 1, 200, 200)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


# ---------------------------------------------------------------------------
# Full VAE forward
# ---------------------------------------------------------------------------


def test_vae_forward_recon_shape():
    vae = VAE(_cfg())
    x = torch.zeros(2, 1, 200, 200)
    z_mean, z_log_var, recon = vae(x)
    assert recon.shape == x.shape
    assert z_mean.shape == (2, 32)
    assert z_log_var.shape == (2, 32)


def test_vae_decode_shortcut():
    vae = VAE(_cfg())
    z = torch.randn(4, 32)
    out = vae.decode(z)
    assert out.shape == (4, 1, 200, 200)


def test_vae_learnable_prior_is_parameter():
    vae = VAE(_cfg(), learnable_prior=True)
    param_names = {n for n, _ in vae.named_parameters()}
    assert "prior_mean" in param_names
    assert "prior_log_var" in param_names


def test_vae_fixed_prior_is_buffer():
    vae = VAE(_cfg(), learnable_prior=False)
    buffer_names = {n for n, _ in vae.named_buffers()}
    assert "prior_mean" in buffer_names
    assert "prior_log_var" in buffer_names
    param_names = {n for n, _ in vae.named_parameters()}
    assert "prior_mean" not in param_names


# ---------------------------------------------------------------------------
# Edge reconstruction loss
# ---------------------------------------------------------------------------


def test_sobel_edges_shape():
    x = torch.rand(2, 1, 32, 32)
    edges = _sobel_edges(x)
    # one [dy, dx] pair per input channel
    assert edges.shape == (2, 2, 32, 32)


def test_sobel_edges_zero_on_flat_image_interior():
    # A constant image has no interior gradients → edge response is ~0 away from
    # the borders (zero-padding produces border artifacts, mirroring tf.image.sobel_edges).
    x = torch.full((2, 1, 32, 32), 0.5)
    interior = _sobel_edges(x)[:, :, 1:-1, 1:-1]
    assert torch.allclose(interior, torch.zeros_like(interior), atol=1e-5)


def test_edge_loss_zero_for_identical_images():
    cfg = VAETrainingConfig(recon_loss="edge")
    img = torch.rand(2, 1, 200, 200)

    class _Identity(torch.nn.Module):
        prior_mean = torch.zeros(32)
        prior_log_var = torch.zeros(32)

        def forward(self, x):
            return torch.zeros(x.size(0), 32), torch.zeros(x.size(0), 32), x

    _, recon_loss, _ = compute_vae_loss(_Identity(), img, cfg, beta=1.0)
    assert recon_loss.item() == 0.0


def test_compute_vae_loss_edge_backward_populates_grads():
    vae = VAE(_cfg(), learnable_prior=True)
    cfg = VAETrainingConfig(recon_loss="edge")
    x = torch.rand(2, 1, 200, 200)
    total, recon_loss, kl_loss = compute_vae_loss(vae, x, cfg, beta=2.0)

    assert torch.isfinite(total)
    assert torch.isfinite(recon_loss)
    assert torch.isfinite(kl_loss)

    total.backward()
    assert vae.prior_mean.grad is not None
    assert any(p.grad is not None for p in vae.encoder.parameters())
    assert any(p.grad is not None for p in vae.decoder.parameters())


# ---------------------------------------------------------------------------
# KLAnnealer
# ---------------------------------------------------------------------------


def test_kl_annealer_fixed_beta():
    cfg = VAETrainingConfig(beta=2.0, kl_annealing=KLAnnealingConfig(enabled=False))
    annealer = KLAnnealer(cfg)
    assert annealer.beta == 2.0
    for _ in range(100):
        annealer.step()
    assert annealer.beta == 2.0


def test_kl_annealer_reaches_target_at_duration():
    cfg = VAETrainingConfig(
        beta=2.0,
        kl_annealing=KLAnnealingConfig(enabled=True, start=0.0, duration_steps=10),
    )
    annealer = KLAnnealer(cfg)
    assert annealer.beta == 0.0
    for _ in range(10):
        annealer.step()
    assert abs(annealer.beta - 2.0) < 1e-6


def test_kl_annealer_does_not_exceed_target():
    cfg = VAETrainingConfig(
        beta=1.0,
        kl_annealing=KLAnnealingConfig(enabled=True, start=0.0, duration_steps=5),
    )
    annealer = KLAnnealer(cfg)
    for _ in range(100):
        annealer.step()
    assert annealer.beta <= 1.0 + 1e-9
