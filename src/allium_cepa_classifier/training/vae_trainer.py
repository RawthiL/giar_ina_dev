from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms

from allium_cepa_classifier.config.vae_config import VAEExperimentConfig, VAETrainingConfig
from allium_cepa_classifier.training.vae_model import VAE

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except Exception:
    _SummaryWriter = None

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class FlatImageDataset(Dataset):
    """Loads all images from a flat directory (no class subdirs)."""

    def __init__(self, root: Path, transform):
        self.paths = sorted(p for p in root.iterdir() if p.suffix.lower() in _IMG_EXTS)
        self.transform = transform
        if not self.paths:
            raise ValueError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img)


def _build_train_transform(cfg: VAEExperimentConfig) -> transforms.Compose:
    h, w = cfg.model.image_size
    aug: list = []
    if cfg.data.online_augment:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    return transforms.Compose(
        [transforms.Resize((h, w)), transforms.Grayscale(), *aug, transforms.ToTensor()]
    )


def _build_eval_transform(cfg: VAEExperimentConfig) -> transforms.Compose:
    h, w = cfg.model.image_size
    return transforms.Compose(
        [transforms.Resize((h, w)), transforms.Grayscale(), transforms.ToTensor()]
    )


def _load_split(split_dir: Path, sources: list[str], transform) -> Dataset:
    parts: list[Dataset] = []
    if "tagged" in sources:
        tagged = split_dir / "tagged"
        if tagged.exists():
            parts.append(_LabelDropWrapper(datasets.ImageFolder(str(tagged), transform=transform)))
    if "untagged" in sources:
        untagged = split_dir / "untagged"
        if untagged.exists():
            parts.append(FlatImageDataset(untagged, transform=transform))
    if not parts:
        raise ValueError(f"No data found in {split_dir} for sources {sources}")
    return ConcatDataset(parts) if len(parts) > 1 else parts[0]


def _build_loaders(cfg: VAEExperimentConfig) -> tuple[DataLoader, DataLoader]:
    vae_dir = cfg.data.vae_crops_dir
    train_ds = _load_split(vae_dir / "train", cfg.data.sources, _build_train_transform(cfg))
    val_ds = _load_split(vae_dir / "val", cfg.data.sources, _build_eval_transform(cfg))

    log.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}")

    kw = {"batch_size": cfg.training.batch_size, "num_workers": 4, "pin_memory": True}
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    return train_loader, val_loader


class _LabelDropWrapper(Dataset):
    """Strips the integer label returned by ImageFolder, returning only the tensor."""

    def __init__(self, ds):
        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self._ds[idx]
        return img


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

# Sobel kernels matching tf.image.sobel_edges (un-normalized). Row order is
# [dy, dx]; dy detects horizontal edges, dx detects vertical edges.
_SOBEL_KERNELS = torch.tensor(
    [
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],  # dy
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],  # dx
    ]
)


def _gamma(x: torch.Tensor, gamma: float, eps: float = 1e-6) -> torch.Tensor:
    """Gamma-correct an image in [0, 1]. clamp_min avoids an infinite gradient at 0."""
    return x.clamp_min(eps).pow(gamma)


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Depthwise Sobel edges, mirroring tf.image.sobel_edges.

    Input  (N, C, H, W) → output (N, 2*C, H, W) with [dy, dx] per channel.
    Kernels are un-normalized and use SAME (padding=1) convolution.
    """
    c = x.size(1)
    # (2, 1, 3, 3) → repeat per input channel for a depthwise (grouped) conv
    weight = _SOBEL_KERNELS.to(x.device, x.dtype).unsqueeze(1).repeat(c, 1, 1, 1)
    return F.conv2d(x, weight, padding=1, groups=c)


def _reconstruction_loss(
    recon: torch.Tensor, x: torch.Tensor, cfg: VAETrainingConfig
) -> torch.Tensor:
    """Reconstruction term, dispatched on cfg.recon_loss.

    - "edge": gamma-correct then MSE on Sobel edge maps, scaled by H*W (mirrors the
      TF notebook: tf.reduce_mean(MSE(sobel(true), sobel(recon))) * IMG_H*IMG_W).
    - "mse"/"bce": per-pixel loss summed over spatial dims, averaged over the batch.
    """
    if cfg.recon_loss == "edge":
        h, w = x.shape[-2:]
        et = _sobel_edges(_gamma(x, cfg.recon_gamma))
        er = _sobel_edges(_gamma(recon, cfg.recon_gamma))
        return F.mse_loss(er, et) * (h * w)

    loss_fn = F.mse_loss if cfg.recon_loss == "mse" else F.binary_cross_entropy
    return loss_fn(recon, x, reduction="none").sum(dim=[1, 2, 3]).mean()


class KLAnnealer:
    """Linearly ramps beta from start → cfg.training.beta over duration_steps.

    If kl_annealing.enabled is False, beta is fixed at cfg.training.beta.
    Call .step() after every optimizer step to advance the counter.
    """

    def __init__(self, cfg: VAETrainingConfig):
        self.fixed_beta = cfg.beta
        self.enabled = cfg.kl_annealing.enabled
        self.start = cfg.kl_annealing.start
        self.end = cfg.beta
        self.duration = cfg.kl_annealing.duration_steps
        self._step = 0

    @property
    def beta(self) -> float:
        if not self.enabled:
            return self.fixed_beta
        ratio = min(1.0, self._step / max(1, self.duration))
        return self.start + (self.end - self.start) * ratio

    def step(self) -> None:
        self._step += 1


def compute_vae_loss(
    model: VAE,
    x: torch.Tensor,
    train_cfg: VAETrainingConfig,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass + loss.

    KL divergence against the (possibly learnable) prior:
      KL = 0.5 * sum_per_dim( prior_log_var - z_log_var
                               + (exp(z_log_var) + (z_mean - prior_mean)^2)
                                 / exp(prior_log_var)
                               - 1 ).mean_over_batch()

    Returns (total_loss, recon_loss, kl_loss) — all scalars.
    """
    z_mean, z_log_var, recon = model(x)
    recon_loss = _reconstruction_loss(recon, x, train_cfg)

    prior_mean = model.prior_mean
    prior_log_var = model.prior_log_var

    kl_loss = (
        0.5
        * (
            prior_log_var
            - z_log_var
            + (z_log_var.exp() + (z_mean - prior_mean).pow(2)) / prior_log_var.exp()
            - 1.0
        )
        .sum(dim=1)
        .mean()
    )

    total = recon_loss + beta * kl_loss
    return total, recon_loss.detach(), kl_loss.detach()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_training(cfg: VAEExperimentConfig, run_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_loader, val_loader = _build_loaders(cfg)

    model = VAE(cfg.model, learnable_prior=cfg.training.learnable_prior).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable params: {trainable:,} / {total:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    sched_cfg = cfg.training.lr_scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg.factor,
        patience=sched_cfg.patience,
        min_lr=sched_cfg.min_lr,
    )

    annealer = KLAnnealer(cfg.training)

    best_val_loss = float("inf")
    best_state: dict | None = None
    patience_counter = 0
    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_recon": [],
        "val_recon": [],
        "train_kl": [],
        "val_kl": [],
        "beta": [],
    }

    writer = None
    if cfg.training.tensorboard and _SummaryWriter is not None:
        try:
            writer = _SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        except Exception as e:
            log.warning(f"TensorBoard writer failed to open, logging disabled: {e}")

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        model.train()

        train_loss = train_recon = train_kl = 0.0
        n_train = 0
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss, r, kl = compute_vae_loss(model, x, cfg.training, annealer.beta)
            loss.backward()
            optimizer.step()
            annealer.step()

            bs = x.size(0)
            train_loss += loss.item() * bs
            train_recon += r.item() * bs
            train_kl += kl.item() * bs
            n_train += bs

        model.eval()
        val_loss = val_recon = val_kl = 0.0
        n_val = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                loss, r, kl = compute_vae_loss(model, x, cfg.training, annealer.beta)
                bs = x.size(0)
                val_loss += loss.item() * bs
                val_recon += r.item() * bs
                val_kl += kl.item() * bs
                n_val += bs

        avg_train = train_loss / n_train
        avg_val = val_loss / n_val
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["train_recon"].append(train_recon / n_train)
        history["val_recon"].append(val_recon / n_val)
        history["train_kl"].append(train_kl / n_train)
        history["val_kl"].append(val_kl / n_val)
        history["beta"].append(annealer.beta)

        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        lr = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch {epoch:02d}/{cfg.training.epochs} "
            f"| {time.time() - t0:.1f}s "
            f"| train={avg_train:.4f} (r={train_recon / n_train:.4f} kl={train_kl / n_train:.4f}) "
            f"| val={avg_val:.4f} (r={val_recon / n_val:.4f} kl={val_kl / n_val:.4f}) "
            f"| beta={annealer.beta:.3f} lr={lr:.2e} patience={patience_counter}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", avg_train, epoch)
            writer.add_scalar("Loss/val", avg_val, epoch)
            writer.add_scalar("Recon/train", train_recon / n_train, epoch)
            writer.add_scalar("Recon/val", val_recon / n_val, epoch)
            writer.add_scalar("KL/train", train_kl / n_train, epoch)
            writer.add_scalar("KL/val", val_kl / n_val, epoch)
            writer.add_scalar("Beta", annealer.beta, epoch)
            writer.add_scalar("LR", lr, epoch)

        if patience_counter >= cfg.training.early_stopping_patience:
            log.info(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    log.info(f"Restored best weights (val_loss={best_val_loss:.4f})")

    if writer is not None:
        writer.add_hparams(
            {"latent_dim": cfg.model.latent_dim, "beta": cfg.training.beta, "lr": cfg.training.lr},
            {"hparam/val_loss": best_val_loss},
        )
        writer.close()

    weights_path = run_dir / "weights" / "vae.pt"
    torch.save(
        {
            "encoder_state_dict": model.encoder.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
            "prior_mean": model.prior_mean.detach().cpu(),
            "prior_log_var": model.prior_log_var.detach().cpu(),
            "latent_dim": cfg.model.latent_dim,
            "image_size": cfg.model.image_size,
            "in_channels": cfg.model.in_channels,
        },
        weights_path,
    )
    log.info(f"Weights saved → {weights_path}")

    metrics = {
        "train_loss": history["train_loss"][-1],
        "val_loss": best_val_loss,
        "train_recon_loss": history["train_recon"][-1],
        "val_recon_loss": history["val_recon"][-1],
        "train_kl_loss": history["train_kl"][-1],
        "val_kl_loss": history["val_kl"][-1],
        "epochs_run": len(history["train_loss"]),
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    from allium_cepa_classifier.training.vae_evaluator import run_evaluation

    run_evaluation(model, history, cfg, run_dir, val_loader, device)

    return metrics
