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

from allium_cepa_classifier.config.sd_vae_config import SDVAEExperimentConfig, SDVAETrainingConfig
import lpips

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except Exception:
    _SummaryWriter = None

log = logging.getLogger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class FlatImageDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.paths = sorted(p for p in root.iterdir() if p.suffix.lower() in _IMG_EXTS)
        self.transform = transform
        if not self.paths:
            raise ValueError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def _build_train_transform(resolution: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _build_eval_transform(resolution: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


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


def _build_loaders(cfg: SDVAEExperimentConfig) -> tuple[DataLoader, DataLoader]:
    vae_dir = cfg.data.vae_crops_dir
    train_transform = _build_train_transform(cfg.model.resolution)
    eval_transform = _build_eval_transform(cfg.model.resolution)
    
    train_ds = _load_split(vae_dir / "train", cfg.data.sources, train_transform)
    val_ds = _load_split(vae_dir / "val", cfg.data.sources, eval_transform)

    log.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}")

    kw = {"batch_size": cfg.training.batch_size, "num_workers": 4, "pin_memory": True}
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    return train_loader, val_loader


class _LabelDropWrapper(Dataset):
    def __init__(self, ds):
        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self._ds[idx]
        return img


def _kl_loss(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()


def compute_sdvae_loss(
    model,
    x: torch.Tensor,
    cfg: SDVAETrainingConfig,
    lpips_fn,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Encode
    posterior = model.encode(x)
    z = posterior.latent_dist.sample()
    
    # Decode
    recon = model.decode(z).sample
    
    # MSE reconstruction loss
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    
    # KL loss
    mean, logvar = posterior.latent_dist.mean, posterior.latent_dist.logvar
    kl = _kl_loss(mean, logvar)
    
    # LPIPS loss
    lpips_loss = torch.tensor(0.0, device=device)
    if cfg.model.weight_lpips > 0 if hasattr(cfg, "model") else True:
        lpips_loss = lpips_fn(x, recon).mean()
    
    # Weighted sum
    total = (
        cfg.model.weight_l2 * recon_loss
        + cfg.model.weight_kl * kl
        + cfg.model.weight_lpips * lpips_loss
    ) if hasattr(cfg, "model") else recon_loss + kl
    
    return total, recon_loss.detach(), kl.detach(), lpips_loss.detach()


def run_training(cfg: SDVAEExperimentConfig, run_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_loader, val_loader = _build_loaders(cfg)

    from diffusers import AutoencoderKL
    model = AutoencoderKL.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder=cfg.model.subfolder,
    ).to(device)

    if cfg.model.decoder_only:
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable params: {trainable:,} / {total:,}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.lr,
    )
    sched_cfg = cfg.training.lr_scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg.factor,
        patience=sched_cfg.patience,
        min_lr=sched_cfg.min_lr,
    )

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_recon": [],
        "val_recon": [],
        "train_kl": [],
        "val_kl": [],
        "train_lpips": [],
        "val_lpips": [],
    }

    writer = None
    if cfg.training.tensorboard and _SummaryWriter is not None:
        try:
            writer = _SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        except Exception as e:
            log.warning(f"TensorBoard writer failed: {e}")

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        model.train()
        
        train_loss = train_recon = train_kl = train_lpips = 0.0
        n_train = 0
        
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
                posterior = model.encode(x)
                z = posterior.latent_dist.sample()
                recon = model.decode(z).sample
                
                recon_loss = F.mse_loss(recon, x, reduction="mean")
                mean, logvar = posterior.latent_dist.mean, posterior.latent_dist.logvar
                kl = _kl_loss(mean, logvar)
                lpips_loss = lpips_fn(x, recon).mean() if cfg.model.weight_lpips > 0 else torch.tensor(0.0, device=device)
                
                total = (
                    cfg.model.weight_l2 * recon_loss
                    + cfg.model.weight_kl * kl
                    + cfg.model.weight_lpips * lpips_loss
                )
            
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            
            bs = x.size(0)
            train_loss += total.item() * bs
            train_recon += recon_loss.item() * bs
            train_kl += kl.item() * bs
            train_lpips += lpips_loss.item() * bs
            n_train += bs

        model.eval()
        val_loss = val_recon = val_kl = val_lpips = 0.0
        n_val = 0
        
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                posterior = model.encode(x)
                z = posterior.latent_dist.sample()
                recon = model.decode(z).sample
                
                recon_loss = F.mse_loss(recon, x, reduction="mean")
                mean, logvar = posterior.latent_dist.mean, posterior.latent_dist.logvar
                kl = _kl_loss(mean, logvar)
                lpips_loss = lpips_fn(x, recon).mean() if cfg.model.weight_lpips > 0 else torch.tensor(0.0, device=device)
                
                total = (
                    cfg.model.weight_l2 * recon_loss
                    + cfg.model.weight_kl * kl
                    + cfg.model.weight_lpips * lpips_loss
                )
                
                bs = x.size(0)
                val_loss += total.item() * bs
                val_recon += recon_loss.item() * bs
                val_kl += kl.item() * bs
                val_lpips += lpips_loss.item() * bs
                n_val += bs

        avg_train = train_loss / n_train
        avg_val = val_loss / n_val
        
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["train_recon"].append(train_recon / n_train)
        history["val_recon"].append(val_recon / n_val)
        history["train_kl"].append(train_kl / n_train)
        history["val_kl"].append(val_kl / n_val)
        history["train_lpips"].append(train_lpips / n_train)
        history["val_lpips"].append(val_lpips / n_val)

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
            f"| train={avg_train:.4f} (r={train_recon/n_train:.4f} kl={train_kl/n_train:.4f} lpips={train_lpips/n_train:.4f}) "
            f"| val={avg_val:.4f} (r={val_recon/n_val:.4f} kl={val_kl/n_val:.4f} lpips={val_lpips/n_val:.4f}) "
            f"| lr={lr:.2e} patience={patience_counter}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", avg_train, epoch)
            writer.add_scalar("Loss/val", avg_val, epoch)
            writer.add_scalar("Recon/train", train_recon / n_train, epoch)
            writer.add_scalar("Recon/val", val_recon / n_val, epoch)
            writer.add_scalar("KL/train", train_kl / n_train, epoch)
            writer.add_scalar("KL/val", val_kl / n_val, epoch)
            writer.add_scalar("LPIPS/train", train_lpips / n_train, epoch)
            writer.add_scalar("LPIPS/val", val_lpips / n_val, epoch)

        if patience_counter >= cfg.training.early_stopping_patience:
            log.info(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    log.info(f"Restored best weights (val_loss={best_val_loss:.4f})")

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(weights_dir)
    log.info(f"Weights saved → {weights_dir}")

    metrics = {
        "train_loss": history["train_loss"][-1],
        "val_loss": best_val_loss,
        "epochs_run": len(history["train_loss"]),
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    from allium_cepa_classifier.training.sd_vae_evaluator import run_evaluation
    run_evaluation(model, history, cfg, run_dir, val_loader, device)

    return metrics
