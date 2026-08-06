from __future__ import annotations

import torch
import torch.nn as nn

from allium_cepa_classifier.config.vae_config import VAEModelConfig


class Encoder(nn.Module):
    """Encodes an image to (z_mean, z_log_var) in latent space.

    Input:  (N, in_channels, 200, 200)
    Output: two tensors of shape (N, latent_dim)

    Four strided conv blocks halve spatial dims: 200→100→50→25→13.
    A fully-connected bottleneck projects to 256, then two parallel heads
    produce z_mean and z_log_var independently.
    """

    def __init__(self, cfg: VAEModelConfig):
        super().__init__()
        filters = cfg.encoder_filters
        in_ch = cfg.in_channels
        conv_blocks: list[nn.Module] = []
        ch = in_ch
        for out_ch in filters:
            conv_blocks += [
                nn.Conv2d(ch, out_ch, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(out_ch),
            ]
            ch = out_ch
        self.conv = nn.Sequential(*conv_blocks)
        flat = filters[-1] * 13 * 13
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.fc_mean = nn.Linear(256, cfg.latent_dim)
        self.fc_log_var = nn.Linear(256, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(self.conv(x))
        return self.fc_mean(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """Decodes a latent vector back to an image.

    Input:  (N, latent_dim)
    Output: (N, in_channels, 200, 200)  values in [0, 1]

    Two FC layers project and reshape to (N, 256, 13, 13).
    Four ConvTranspose2d blocks double spatial dims: 13→26→52→104→208.
    output_padding=1 is required so stride=2 deconv doubles exactly.
    Crop from 208→200 by slicing 4px off each side, mirroring Keras Cropping2D.
    Final conv + Sigmoid produces pixel probabilities.
    """

    def __init__(self, cfg: VAEModelConfig):
        super().__init__()
        filters = cfg.decoder_filters
        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, filters[0] * 13 * 13),
            nn.LeakyReLU(0.2),
        )
        deconv_blocks: list[nn.Module] = []
        ch = filters[0]
        for out_ch in filters[1:]:
            deconv_blocks += [
                nn.ConvTranspose2d(ch, out_ch, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(out_ch),
            ]
            ch = out_ch
        # extra upsample block: keeps same channel count (ch→ch), brings 104→208
        deconv_blocks += [
            nn.ConvTranspose2d(ch, ch, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(ch),
        ]
        self.deconv = nn.Sequential(*deconv_blocks)
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(ch, cfg.in_channels, 5, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), -1, 13, 13)
        x = self.deconv(h)
        x = x[:, :, 4:-4, 4:-4]  # crop 208 → 200, mirrors Keras Cropping2D((4,4),(4,4))
        return self.out_conv(x)


class VAE(nn.Module):
    """Variational Autoencoder combining Encoder + Decoder with a (optionally learnable) prior.

    forward() returns (z_mean, z_log_var, reconstruction) so the trainer can
    compute both reconstruction loss and KL divergence against the prior.

    The reparameterization trick: instead of sampling z ~ N(z_mean, exp(z_log_var))
    directly (which would block gradients), we compute z = z_mean + std * eps
    where std = exp(0.5 * z_log_var) and eps ~ N(0,I). Gradients flow through
    z_mean and z_log_var; eps is just noise.
    """

    def __init__(self, cfg: VAEModelConfig, learnable_prior: bool = True):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        if learnable_prior:
            self.prior_mean = nn.Parameter(torch.zeros(cfg.latent_dim))
            self.prior_log_var = nn.Parameter(torch.zeros(cfg.latent_dim))
        else:
            self.register_buffer("prior_mean", torch.zeros(cfg.latent_dim))
            self.register_buffer("prior_log_var", torch.zeros(cfg.latent_dim))

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = (0.5 * log_var).exp()
        return mean + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, self.decoder(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
