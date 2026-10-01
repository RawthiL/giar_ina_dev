from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from allium_cepa_classifier.config.sd_vae_config import SDVAEExperimentConfig, SDVAETrainingConfig
from allium_cepa_classifier.training.sd_vae_evaluator import get_validation_samples

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
    return transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _build_eval_transform(resolution: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
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


def _build_loaders(cfg: SDVAEExperimentConfig) -> tuple[DataLoader, DataLoader]:
    vae_dir = cfg.data.vae_crops_dir
    train_transform = _build_train_transform(cfg.model.resolution)
    eval_transform = _build_eval_transform(cfg.model.resolution)

    train_ds = _load_split(vae_dir / "train", cfg.data.sources, train_transform)
    val_ds = _load_split(vae_dir / "val", cfg.data.sources, eval_transform)

    log.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}")
    kw = {
        "batch_size": cfg.training.batch_size,
        "num_workers": cfg.training.dataloader_num_workers,
        "pin_memory": True,
    }
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
        (
            cfg.model.weight_l2 * recon_loss
            + cfg.model.weight_kl * kl
            + cfg.model.weight_lpips * lpips_loss
        )
        if hasattr(cfg, "model")
        else recon_loss + kl
    )

    return total, recon_loss.detach(), kl.detach(), lpips_loss.detach()


def run_training(cfg: SDVAEExperimentConfig, run_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_loader, val_loader = _build_loaders(cfg)

    from diffusers import AutoencoderKL
    from peft import LoraConfig, PeftModel, get_peft_model

    model = AutoencoderKL.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder=cfg.model.subfolder,
    ).to(device)

    # LoRA injection
    if cfg.model.lora.enabled:
        # Only target decoder layers
        target_modules = cfg.model.lora.target_modules

        lora_config = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            target_modules=target_modules,
            lora_dropout=cfg.model.lora.dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        log.info(
            f"LoRA injected with r={cfg.model.lora.r}, alpha={cfg.model.lora.alpha}, targets={target_modules}"
        )

        # Freeze base model, only LoRA params trainable
        for name, param in model.named_parameters():
            is_encoder_path = "encoder" in name or (
                cfg.model.freeze_quant_conv and "quant_conv" in name
            )
            is_decoder_base = "encoder" not in name and "lora" not in name
            if is_encoder_path:
                param.requires_grad = False
            elif is_decoder_base:
                if cfg.model.decoder_only:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    else:
        if cfg.model.decoder_only:
            for name, param in model.named_parameters():
                is_encoder_path = "encoder" in name or (
                    cfg.model.freeze_quant_conv and "quant_conv" in name
                )
                param.requires_grad = not is_encoder_path
        else:
            for param in model.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable params: {trainable:,} / {total:,}")
    log.info(
        f"decoder_only: {cfg.model.decoder_only}, freeze_quant_conv: {cfg.model.freeze_quant_conv}"
    )

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
    step_counter = 0
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
            # Auto-named TensorBoard run
            tb_run_name = time.strftime("%Y%m%d-%H%M%S")
            tb_log_dir = run_dir / "logs" / tb_run_name
            writer = _SummaryWriter(log_dir=str(tb_log_dir))
            log.info(f"TensorBoard logging to {tb_log_dir}")
        except Exception as e:
            log.warning(f"TensorBoard writer failed: {e}")

        # Write initial - step zero - images
        if cfg.training.log_images_every_n_steps > 0:
            with torch.no_grad():
                val_batch, _ = get_validation_samples(cfg, device)
                posterior = model.encode(val_batch)
                z = posterior.latent_dist.sample()
                recon = model.decode(z).sample

                # Denormalize from [-1,1] to [0,1] for TensorBoard
                val_vis = (val_batch + 1) / 2
                recon_vis = (recon + 1) / 2
                # 2x4 grid: top row originals, bottom row reconstructions, nrow=4
                grid = make_grid(torch.cat([val_vis, recon_vis], dim=0), nrow=4, pad_value=0.5)
                writer.add_image("Images/VAE/ReconstructionsPerPhase", grid, step_counter)

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        model.train()

        train_loss = train_recon = train_kl = train_lpips = 0.0
        n_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}", unit="batch")
        for x in pbar:
            x = x.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
                posterior = model.encode(x)
                z = posterior.latent_dist.sample()
                recon = model.decode(z).sample

                recon_loss = F.mse_loss(recon, x, reduction="mean")
                mean, logvar = posterior.latent_dist.mean, posterior.latent_dist.logvar
                kl = _kl_loss(mean, logvar)
                lpips_loss = (
                    lpips_fn(x, recon).mean()
                    if cfg.model.weight_lpips > 0
                    else torch.tensor(0.0, device=device)
                )

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

            step_counter += 1

            # Running averages for current epoch
            avg_loss = train_loss / n_train
            avg_recon = train_recon / n_train
            avg_kl = train_kl / n_train
            avg_lpips = train_lpips / n_train

            # Per-step logging
            if step_counter % cfg.training.log_every_n_steps == 0 and writer is not None:
                writer.add_scalar("Train/VAE/Loss_step", avg_loss, step_counter)
                writer.add_scalar("Train/VAE/Recon_MSE_step", avg_recon, step_counter)
                writer.add_scalar("Train/VAE/Recon_KL_step", avg_kl, step_counter)
                writer.add_scalar("Train/VAE/Recon_LPIPS_step", avg_lpips, step_counter)
                pbar.set_postfix(
                    {"loss": f"{avg_loss:.4f}", "recon": f"{avg_recon:.4f}", "kl": f"{avg_kl:.4f}"}
                )
                log.info(
                    f"\nStep {step_counter} | loss={avg_loss:.4f} (r={avg_recon:.4f} kl={avg_kl:.4f} lpips={avg_lpips:.4f})"
                )

            # Log images every N steps
            if (
                cfg.training.log_images_every_n_steps > 0
                and step_counter % cfg.training.log_images_every_n_steps == 0
                and writer is not None
            ):
                with torch.no_grad():
                    val_batch, _ = get_validation_samples(cfg, device)
                    posterior = model.encode(val_batch)
                    z = posterior.latent_dist.sample()
                    recon = model.decode(z).sample

                    # Denormalize from [-1,1] to [0,1] for TensorBoard
                    val_vis = (val_batch + 1) / 2
                    recon_vis = (recon + 1) / 2
                    # 2x4 grid: top row originals, bottom row reconstructions, nrow=4
                    grid = make_grid(torch.cat([val_vis, recon_vis], dim=0), nrow=4, pad_value=0.5)
                    writer.add_image("Images/VAE/ReconstructionsPerPhase", grid, step_counter)

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
                lpips_loss = (
                    lpips_fn(x, recon).mean()
                    if cfg.model.weight_lpips > 0
                    else torch.tensor(0.0, device=device)
                )

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
            f"| train={avg_train:.4f} (r={train_recon / n_train:.4f} kl={train_kl / n_train:.4f} lpips={train_lpips / n_train:.4f}) "
            f"| val={avg_val:.4f} (r={val_recon / n_val:.4f} kl={val_kl / n_val:.4f} lpips={val_lpips / n_val:.4f}) "
            f"| lr={lr:.2e} patience={patience_counter}"
        )

        if writer is not None:
            writer.add_scalar("Train/VAE/Loss_epoch", avg_train, epoch)
            writer.add_scalar("Val/VAE/Loss_epoch", avg_val, epoch)
            writer.add_scalar("Train/VAE/Recon_MSE_epoch", train_recon / n_train, epoch)
            writer.add_scalar("Val/VAE/Recon_MSE_epoch", val_recon / n_val, epoch)
            writer.add_scalar("Train/VAE/Recon_KL_epoch", train_kl / n_train, epoch)
            writer.add_scalar("Val/VAE/Recon_KL_epoch", val_kl / n_val, epoch)
            writer.add_scalar("Train/VAE/Recon_LPIPS_epoch", train_lpips / n_train, epoch)
            writer.add_scalar("Val/VAE/Recon_LPIPS_epoch", val_lpips / n_val, epoch)

        if patience_counter >= cfg.training.early_stopping_patience:
            log.info(f"Early stopping at epoch {epoch}.")
            break

    # Save best model
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if cfg.model.lora.enabled:
        # Save LoRA adapter only
        lora_dir = weights_dir / "lora"
        lora_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(lora_dir)
        log.info(f"LoRA adapter saved → {lora_dir}")

        # Optionally save merged full model
        if cfg.model.lora.merge_and_save_full:
            # Load base model fresh
            from diffusers import AutoencoderKL
            from peft import PeftModel

            base_model = AutoencoderKL.from_pretrained(
                cfg.model.pretrained_model_name_or_path,
                subfolder=cfg.model.subfolder,
            ).to(device)
            # Load LoRA weights
            merged_model = PeftModel.from_pretrained(base_model, str(lora_dir))
            merged_model = merged_model.merge_and_unload()
            merged_dir = weights_dir / "merged"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(merged_dir)
            log.info(f"Merged VAE saved → {merged_dir}")
    else:
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
