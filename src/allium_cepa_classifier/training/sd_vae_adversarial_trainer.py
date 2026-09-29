from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import lpips
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm

from allium_cepa_classifier.config.sd_vae_adversarial_config import SDVAEAdversarialExperimentConfig

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except Exception:
    _SummaryWriter = None

log = logging.getLogger(__name__)


class LatentParquetDataset(Dataset):
    def __init__(self, parquet_path: Path):
        table = pq.read_table(parquet_path)
        latent_col = table.column("latent")
        # FixedSizeList → flatten → reshape
        flat = latent_col.combine_chunks().flatten().to_numpy(zero_copy_only=False)
        # Infer shape from config or assume 4x64x64
        # We'll assume 4,64,64 as per generator
        self.latents = torch.from_numpy(flat).float().view(-1, 4, 64, 64)
        self.seeds = table.column("seed").to_numpy()
        self.phase_ids = table.column("phase_id").to_numpy()

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        return self.latents[idx]


def _build_real_loaders(cfg: SDVAEAdversarialExperimentConfig):
    from allium_cepa_classifier.training.sd_vae_trainer import _build_loaders

    train_loader, val_loader = _build_loaders(
        # _build_loaders expects SDVAEExperimentConfig; our config is subclass
        cfg  # type: ignore[arg-type]
    )
    return train_loader, val_loader


def _disc_input_transform(x: torch.Tensor, image_size: int, mode: str, antialias: bool):
    # x in [-1,1]
    x = (x + 1.0) / 2.0
    if x.shape[-2] != image_size or x.shape[-1] != image_size:
        x = F.interpolate(x, size=(image_size, image_size), mode=mode, antialias=antialias)
    # ImageNet normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def _load_discriminator(cfg, device):
    import timm

    ckpt = torch.load(cfg.discriminator.checkpoint, map_location=device, weights_only=False)
    timm_model_name = ckpt.get("timm_model_name", "efficientnet_b2")
    model = timm.create_model(timm_model_name, pretrained=False)
    # Load base weights from checkpoint
    state_dict = ckpt["model_state_dict"]
    base_state = {}
    for k, v in state_dict.items():
        if k.startswith("base_model."):
            base_state[k[len("base_model.") :]] = v
    model.load_state_dict(base_state, strict=False)

    # Replace final classifier layer to output 1 logit
    # Assume model.classifier is Sequential
    if isinstance(model.classifier, nn.Sequential):
        # Find last Linear
        for i in reversed(range(len(model.classifier))):
            m = model.classifier[i]
            if isinstance(m, nn.Linear):
                in_features = m.in_features
                new_linear = nn.Linear(in_features, 1)
                model.classifier[i] = new_linear
                break
    # Freeze / unfreeze
    if cfg.discriminator.trainable == "head":
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name
    # else all trainable

    model.to(device)
    return model


def _fixed_latent_batch(latent_ds: LatentParquetDataset, n: int, seed: int) -> torch.Tensor:
    n = max(1, min(n, len(latent_ds)))
    gen = torch.Generator().manual_seed(seed + 1337)
    idx = torch.randperm(len(latent_ds), generator=gen)[:n]
    return latent_ds.latents[idx].clone()


def _log_latent_samples(vae, fixed_latents, writer, device, step):
    with torch.no_grad():
        z = fixed_latents.to(device) / vae.config.scaling_factor
        x = vae.decode(z).sample
        vis = (x + 1) / 2
        grid = make_grid(vis, nrow=x.size(0), pad_value=0.5)
        writer.add_image("train/decoded_latents", grid, step)


def _next_latent_batch(latent_iter, latent_loader):
    try:
        z = next(latent_iter)
    except StopIteration:
        latent_iter = iter(latent_loader)
        z = next(latent_iter)
    return z, latent_iter


def run_training(cfg: SDVAEAdversarialExperimentConfig, run_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Real loaders
    train_loader, val_loader = _build_real_loaders(cfg)

    # Latent dataset
    latent_ds = LatentParquetDataset(cfg.adversarial.latent_dataset)
    latent_loader = DataLoader(
        latent_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # VAE
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder=cfg.model.subfolder,
    ).to(device)

    # Freeze encoder
    for name, param in vae.named_parameters():
        if "encoder" in name or (cfg.model.freeze_quant_conv and "quant_conv" in name):
            param.requires_grad = False
        else:
            # decoder only
            if "encoder" not in name:
                param.requires_grad = True

    # Discriminator
    disc = _load_discriminator(cfg, device)
    disc_optimizer = torch.optim.Adam(
        [p for p in disc.parameters() if p.requires_grad],
        lr=cfg.discriminator.lr,
    )

    vae_optimizer = torch.optim.Adam(
        [p for p in vae.parameters() if p.requires_grad],
        lr=cfg.training.lr,
    )

    lpips_fn = lpips.LPIPS(net="alex").to(device)

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

    step_counter = 0
    fixed_latents = None
    if writer is not None and cfg.training.log_images_every_n_steps > 0:
        fixed_latents = _fixed_latent_batch(
            latent_ds, cfg.training.log_images_n_latents, cfg.data.seed
        )
        _log_latent_samples(vae, fixed_latents, writer, device, step_counter)

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision)

    # Align real and latent loaders by iterator
    latent_iter = iter(latent_loader)
    for epoch in range(1, cfg.training.epochs + 1):
        vae.train()
        disc.train()
        train_loss = 0.0
        n = 0
        train_d = train_recon = train_adv = train_lpips = 0.0
        acc_real = acc_fake = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}", unit="batch")
        for x_real in pbar:
            x_real = x_real.to(device)
            z_fake_raw, latent_iter = _next_latent_batch(latent_iter, latent_loader)
            z_fake_raw = z_fake_raw.to(device)

            # Fake decode (for D step)
            scaling = vae.config.scaling_factor
            x_fake = vae.decode(z_fake_raw / scaling).sample

            # Disc inputs
            ds = cfg.discriminator.downsample
            real_disc = _disc_input_transform(
                x_real, cfg.discriminator.image_size, ds.mode, ds.antialias
            )
            fake_disc = _disc_input_transform(
                x_fake, cfg.discriminator.image_size, ds.mode, ds.antialias
            )

            # Discriminator step
            disc_real = disc(real_disc)
            disc_fake = disc(fake_disc.detach())
            loss_D = F.binary_cross_entropy_with_logits(
                disc_real, torch.ones_like(disc_real)
            ) + F.binary_cross_entropy_with_logits(disc_fake, torch.zeros_like(disc_fake))
            disc_optimizer.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(disc_optimizer)
            scaler.update()

            bs = x_real.size(0)

            acc_real += (disc_real.detach().sigmoid() > 0.5).float().mean().item()
            acc_fake += (disc_fake.sigmoid() < 0.5).float().mean().item()
            train_d += loss_D.item() * bs

            # Generator update: k fresh decode passes, single optimizer step
            k = max(1, cfg.adversarial.recon_steps_per_disc_update)
            loss_dec = None
            step_recon = step_lpips = 0.0
            n_dec_iters = 0
            for _ in range(k):
                x_recon = vae.decode(vae.encode(x_real).latent_dist.mean).sample
                recon_loss = 0.0
                if cfg.adversarial.use_recon_mse:
                    mse = F.mse_loss(x_recon, x_real)
                    recon_loss = recon_loss + mse
                    step_recon += mse.item()
                if cfg.adversarial.use_recon_lpips:
                    lp = lpips_fn(x_recon, x_real).mean()
                    recon_loss = recon_loss + lp
                    step_lpips += lp.item()

                z_fake_raw, latent_iter = _next_latent_batch(latent_iter, latent_loader)
                x_fake_g = vae.decode(z_fake_raw.to(device) / scaling).sample
                fake_disc_for_gen = disc(
                    _disc_input_transform(
                        x_fake_g, cfg.discriminator.image_size, ds.mode, ds.antialias
                    )
                )
                adv_loss = F.binary_cross_entropy_with_logits(
                    fake_disc_for_gen, torch.ones_like(fake_disc_for_gen)
                )
                step_loss = recon_loss + cfg.adversarial.lambda_adv * adv_loss
                loss_dec = step_loss if loss_dec is None else loss_dec + step_loss

                n_dec_iters += 1
                train_adv += adv_loss.item() * bs

            vae_optimizer.zero_grad()
            scaler.scale(loss_dec / k).backward()
            scaler.step(vae_optimizer)
            scaler.update()

            train_recon += (step_recon / max(n_dec_iters, 1)) * bs
            train_lpips += (step_lpips / max(n_dec_iters, 1)) * bs

            train_loss += loss_dec.mean().item() * bs
            n += bs

            step_counter += 1
            n_batches += 1
            avg_step = train_loss / n
            avg_d = train_d / n
            avg_recon = train_recon / n
            avg_adv = train_adv / n
            avg_lpips = train_lpips / n
            acc_r = acc_real / n_batches
            acc_f = acc_fake / n_batches

            if step_counter % cfg.training.log_every_n_steps == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{avg_step:.4f}",
                        "D": f"{avg_d:.4f}",
                        "acc": f"{acc_r:.2f}/{acc_f:.2f}",
                    }
                )
                if writer is not None:
                    writer.add_scalar("Loss/train_step", avg_step, step_counter)
                    writer.add_scalar("Loss/disc_step", avg_d, step_counter)
                    writer.add_scalar("Recon/train_step", avg_recon, step_counter)
                    writer.add_scalar("Adv/train_step", avg_adv, step_counter)
                    writer.add_scalar("LPIPS/train_step", avg_lpips, step_counter)
                    writer.add_scalar("Disc/acc_real_step", acc_r, step_counter)
                    writer.add_scalar("Disc/acc_fake_step", acc_f, step_counter)
                    writer.add_scalar(
                        "LR/dec_step", vae_optimizer.param_groups[0]["lr"], step_counter
                    )
                    writer.add_scalar(
                        "LR/disc_step", disc_optimizer.param_groups[0]["lr"], step_counter
                    )
                log.info(
                    f"\nStep {step_counter} | G={avg_step:.4f} (recon={avg_recon:.4f} lpips={avg_lpips:.4f} "
                    f"adv={avg_adv:.4f}) | D={avg_d:.4f} acc_r={acc_r:.2f} acc_f={acc_f:.2f}"
                )

            if (
                writer is not None
                and cfg.training.log_images_every_n_steps > 0
                and step_counter % cfg.training.log_images_every_n_steps == 0
            ):
                _log_latent_samples(vae, fixed_latents, writer, device, step_counter)

        # Validation (recon MSE, optional LPIPS)
        vae.eval()
        val_loss = 0.0
        val_lpips = 0.0
        n_val = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                posterior = vae.encode(x)
                z = posterior.latent_dist.mean
                recon = vae.decode(z).sample
                loss = F.mse_loss(recon, x, reduction="sum")
                val_loss += loss.item()
                if cfg.adversarial.use_recon_lpips:
                    val_lpips += lpips_fn(x, recon).mean().item() * x.size(0)
                n_val += x.numel() // 3
        avg_val = val_loss / n_val if n_val else float("inf")
        avg_val_lpips = val_lpips / n_val if n_val else 0.0
        avg_train = train_loss / n if n else float("inf")
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        log.info(f"Epoch {epoch}/{cfg.training.epochs} train={avg_train:.4f} val={avg_val:.4f}")

        if writer is not None:
            writer.add_scalar("Loss/train_epoch", avg_train, epoch)
            writer.add_scalar("Loss/val_epoch", avg_val, epoch)
            writer.add_scalar("Recon/val_epoch", avg_val, epoch)
            writer.add_scalar("Loss/disc_epoch", train_d / n if n else 0.0, epoch)
            writer.add_scalar("Recon/train_epoch", train_recon / n if n else 0.0, epoch)
            writer.add_scalar("Adv/train_epoch", train_adv / n if n else 0.0, epoch)
            writer.add_scalar("LPIPS/train_epoch", train_lpips / n if n else 0.0, epoch)
            writer.add_scalar("Disc/acc_real_epoch", acc_real / max(n_batches, 1), epoch)
            writer.add_scalar("Disc/acc_fake_epoch", acc_fake / max(n_batches, 1), epoch)
            if cfg.adversarial.use_recon_lpips:
                writer.add_scalar("LPIPS/val_epoch", avg_val_lpips, epoch)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            vae.save_pretrained(weights_dir)
            torch.save(disc.state_dict(), weights_dir / "discriminator.pt")

    if writer is not None:
        writer.close()

    metrics = {"train_loss": history["train_loss"][-1], "val_loss": best_val_loss}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
