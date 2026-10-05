from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import lpips
import numpy as np
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


def _resolve_latent_shape(
    parquet_path: Path, list_size: int, config_path: Path | None
) -> tuple[int, ...]:
    cfg_path = config_path if config_path and config_path.exists() else None
    if cfg_path is None and config_path is not None:
        log.warning(
            f"adversarial.latent_config not found at {config_path}; falling back to parquet sibling"
        )
    if cfg_path is None:
        sibling = parquet_path.parent / f"{parquet_path.stem}.config.json"
        cfg_path = sibling if sibling.exists() else None

    if cfg_path is not None:
        meta = json.loads(cfg_path.read_text())
        shape = tuple(int(s) for s in meta["latent_shape"])
        if int(np.prod(shape)) != list_size:
            raise ValueError(
                f"latent_shape {shape} from {cfg_path} does not match parquet list size "
                f"{list_size} ({int(np.prod(shape))} != {list_size})"
            )
        log.info(f"Latent shape {shape} from {cfg_path}")
        return shape

    side = int(round((list_size / 4) ** 0.5))
    if side * side * 4 != list_size:
        raise ValueError(
            f"Cannot infer latent shape from list size {list_size}: no sidecar config.json found "
            f"for {parquet_path} and the size is not a 4-channel square. "
            "Set adversarial.latent_config explicitly."
        )
    log.warning(
        f"No latent config.json found for {parquet_path}; assuming shape (4, {side}, {side}). "
        "Set adversarial.latent_config to be explicit."
    )
    return (4, side, side)


class LatentParquetDataset(Dataset):
    def __init__(self, parquet_path: Path, latent_config: Path | None = None):
        table = pq.read_table(parquet_path)
        latent_col = table.column("latent")
        # FixedSizeList → flatten → reshape
        flat = latent_col.combine_chunks().flatten().to_numpy(zero_copy_only=False)
        # FixedSizeList element count (pyarrow exposes it as `type.list_size`; derive it from the
        # flattened buffer so this works across pyarrow versions)
        list_size = flat.size // table.num_rows
        shape = _resolve_latent_shape(parquet_path, list_size, latent_config)
        self.latents = torch.from_numpy(flat).float().view(-1, *shape)
        self.seeds = table.column("seed").to_numpy()
        self.phase_ids = table.column("phase_id").to_numpy()

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        return self.latents[idx]


@dataclass
class DiscPreprocess:
    """Input spec for the discriminator: normalization from the checkpoint, degradation from config."""

    image_size: int
    mode: str
    antialias: bool
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    degrade_mid: int | None = None
    _stats_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def stats_for(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._stats_cache.get(device)
        if cached is None:
            mean = torch.tensor(self.mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(self.std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
            cached = (mean, std)
            self._stats_cache[device] = cached
        return cached


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _degrade_roundtrip(x: torch.Tensor, mid: int) -> torch.Tensor:
    hw = x.shape[-2:]
    if mid >= min(hw):
        return x
    x = F.interpolate(x, size=(mid, mid), mode="bilinear", antialias=True)
    return F.interpolate(x, size=hw, mode="bilinear", antialias=False)


@lru_cache(maxsize=8)
def _laplacian_kernel(device: str) -> torch.Tensor:
    k = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=torch.device(device)
    )
    return k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)


def _hf_energy(x: torch.Tensor) -> float:
    """Mean |Laplacian| of an RGB batch: cheap sharpness proxy, arbitrary fixed scale."""
    with torch.no_grad():
        edges = F.conv2d(x.float(), _laplacian_kernel(str(x.device)), groups=3) / 4.0
        return edges.abs().mean().item()


def _disc_input_transform(x: torch.Tensor, pp: DiscPreprocess):
    # x in [-1,1] (VAE range) -> [0,1] -> discriminator input space
    x = (x + 1.0) / 2.0
    # The single choke point for every tensor entering the D (real, fake-for-D, fake-for-G),
    # so both branches get the identical operator: real crops are natively ~200px upscaled to
    # the training resolution, while latents decode with genuine high-frequency content. Without
    # this round trip the D separates on sharpness, a cue the decoder cannot answer.
    if pp.degrade_mid:
        x = _degrade_roundtrip(x, pp.degrade_mid)
    if x.shape[-2] != pp.image_size or x.shape[-1] != pp.image_size:
        x = F.interpolate(
            x, size=(pp.image_size, pp.image_size), mode=pp.mode, antialias=pp.antialias
        )
    # The discriminator backbone is a classifier fine-tuned on ImageNet-normalized crops,
    # so these stats are required (they come from the checkpoint metadata).
    mean, std = pp.stats_for(x.device)
    return (x - mean) / std


def _freeze_norm_stats(model: nn.Module) -> int:
    """Pin norm running stats. momentum=0 keeps them pinned across model.train() calls."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
            if hasattr(m, "momentum"):
                m.momentum = 0.0
            m.eval()
            n += 1
    return n


@dataclass
class _MetricAgg:
    """Image-weighted running sums that can be reset per logging window.

    Generator metrics must be accumulated only over steps where a G update actually ran,
    otherwise discriminator-warmup steps dilute the means and the curves climb by construction.
    """

    sums: dict[str, float] = field(default_factory=dict)
    weight: float = 0.0

    def push(self, weight: float, **values: float) -> None:
        self.weight += weight
        for key, val in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + val * weight

    def mean(self, key: str) -> float:
        return self.sums.get(key, 0.0) / self.weight if self.weight else 0.0

    def reset(self) -> None:
        self.sums = {}
        self.weight = 0.0


def _load_discriminator(cfg, device) -> tuple[nn.Module, DiscPreprocess]:
    import timm

    ckpt = torch.load(cfg.discriminator.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    ckpt_arch = ckpt.get("timm_model_name")
    arch = cfg.discriminator.arch
    if ckpt_arch is None:
        log.info(f"Checkpoint has no timm_model_name; using discriminator.arch={arch}")
    elif ckpt_arch != arch:
        log.warning(
            f"Checkpoint arch={ckpt_arch} differs from discriminator.arch={arch}; using {arch}"
        )

    model = timm.create_model(arch, pretrained=False, num_classes=0)
    # Feature extractor only: the classifier head is rebuilt below as a 1-logit disc head.
    # Supports both key layouts: timm-native ("base_model.conv_stem") and
    # BackboneWithHead ("base_model.backbone.conv_stem").
    backbone_state = {}
    for k, v in state_dict.items():
        name = k[len("base_model.") :] if k.startswith("base_model.") else k
        if name.startswith("backbone."):
            name = name[len("backbone.") :]
        # Skip the 2-class mitosis head and the calibration temperature: only the
        # feature extractor is reused, the disc head is rebuilt below.
        if name == "temperature" or "classifier" in name or name.startswith("head."):
            continue
        backbone_state[name] = v

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Discriminator weights do not match model '{arch}': "
            f"missing={list(missing)[:6]} unexpected={list(unexpected)[:6]}"
        )
    log.info(f"Discriminator backbone loaded strictly: {len(backbone_state)} tensors")

    # Single real/fake logit head, randomly initialized (standard GAN practice)
    model.classifier = nn.Linear(model.num_features, 1)

    if cfg.discriminator.trainable == "head":
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name
    disc_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    disc_total = sum(p.numel() for p in model.parameters())
    log.info(f"Discriminator trainable params: {disc_trainable:,} / {disc_total:,}")

    mean = tuple(ckpt.get("imagenet_mean", _IMAGENET_MEAN))
    std = tuple(ckpt.get("imagenet_std", _IMAGENET_STD))
    ckpt_image_size = ckpt.get("image_size")
    if ckpt_image_size is not None:
        ckpt_hw = (
            int(ckpt_image_size[-1])
            if isinstance(ckpt_image_size, (list, tuple))
            else int(ckpt_image_size)
        )
        if ckpt_hw != cfg.discriminator.image_size:
            log.warning(
                f"Checkpoint image_size={ckpt_hw} differs from "
                f"discriminator.image_size={cfg.discriminator.image_size}; using the config value"
            )

    model.to(device)
    if cfg.discriminator.freeze_norm_stats:
        n_pinned = _freeze_norm_stats(model)
        log.info(
            f"Discriminator: pinned running stats on {n_pinned} norm layers "
            "(momentum=0, survives .train()); the checkpoint stats stay as calibrated"
        )
    degrade_mid = cfg.adversarial.degrade_roundtrip_mid
    if degrade_mid:
        log.info(
            f"Discriminator inputs: matched {degrade_mid}px round-trip degradation on "
            "both real and fake branches"
        )
    ds = cfg.discriminator.downsample
    pp = DiscPreprocess(
        image_size=cfg.discriminator.image_size,
        mode=ds.mode,
        antialias=ds.antialias,
        mean=mean,
        std=std,
        degrade_mid=degrade_mid,
    )
    return model, pp


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
        writer.add_image("Images/Generator/FixedLatentDecodes", grid, step)


def _log_real_vs_recon(vae, x_batch, writer, device, step, n: int = 4):
    """Paired real|recon grid on a fixed real batch: separates GAN sharpness from hallucinated
    texture. Genuinely-recovered structure looks like real-but-sharper; invented texture does not
    exist in the real panel at all."""
    with torch.no_grad():
        x = x_batch[:n].to(device)
        recon = vae.decode(vae.encode(x).latent_dist.mean).sample
    pairs = torch.stack([x, recon], dim=1).flatten(0, 1)
    grid = make_grid((pairs + 1) / 2, nrow=2, pad_value=0.5)
    writer.add_image("Images/Validation/RealVsRecon", grid, step)


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

    # Real loaders (balanced sampling + online augmentation handled in the shared loader)
    from allium_cepa_classifier.training.sd_vae_trainer import _build_loaders

    train_loader, val_loader = _build_loaders(cfg)

    # Latent dataset
    latent_ds = LatentParquetDataset(cfg.adversarial.latent_dataset, cfg.adversarial.latent_config)
    latent_loader = DataLoader(
        latent_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # VAE
    from diffusers import AutoencoderKL
    from peft import LoraConfig, PeftModel, get_peft_model

    vae = AutoencoderKL.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder=cfg.model.subfolder,
    ).to(device)

    # LoRA injection
    if cfg.model.lora.enabled:
        target_modules = cfg.model.lora.target_modules
        lora_config = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            target_modules=target_modules,
            lora_dropout=cfg.model.lora.dropout,
            bias="none",
        )
        vae = get_peft_model(vae, lora_config)
        log.info(
            f"LoRA injected with r={cfg.model.lora.r}, alpha={cfg.model.lora.alpha}, targets={target_modules}"
        )

        # Freeze base model, only LoRA params trainable
        for name, param in vae.named_parameters():
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
            # Freeze encode model, only decoder trainable
            for name, param in vae.named_parameters():
                is_encoder_path = "encoder" in name or (
                    cfg.model.freeze_quant_conv and "quant_conv" in name
                )
                param.requires_grad = not is_encoder_path
        else:
            for param in vae.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vae.parameters())
    log.info(f"Trainable params: {trainable:,} / {total:,}")
    log.info(
        f"decoder_only: {cfg.model.decoder_only}, freeze_quant_conv: {cfg.model.freeze_quant_conv}"
    )

    # Discriminator
    disc, disc_pp = _load_discriminator(cfg, device)
    head_params = [p for n, p in disc.named_parameters() if p.requires_grad and "classifier" in n]
    backbone_params = [
        p for n, p in disc.named_parameters() if p.requires_grad and "classifier" not in n
    ]
    disc_groups = [{"params": head_params, "lr": cfg.discriminator.lr}]
    if backbone_params:
        disc_groups.append({"params": backbone_params, "lr": cfg.discriminator.backbone_lr})
    disc_optimizer = torch.optim.Adam(disc_groups)
    log.info(
        f"Discriminator optimizer: head {sum(p.numel() for p in head_params):,} @ "
        f"{cfg.discriminator.lr:.1e}"
        + (
            f" | backbone {sum(p.numel() for p in backbone_params):,} @ "
            f"{cfg.discriminator.backbone_lr:.1e}"
            if backbone_params
            else ""
        )
    )

    vae_params = [p for p in vae.parameters() if p.requires_grad]
    vae_optimizer = torch.optim.Adam(vae_params, lr=cfg.training.lr)
    sched_cfg = cfg.training.lr_scheduler
    vae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        vae_optimizer,
        mode="min",
        factor=sched_cfg.factor,
        patience=sched_cfg.patience,
        min_lr=sched_cfg.min_lr,
    )

    lpips_fn = lpips.LPIPS(net="alex").to(device) if cfg.adversarial.use_recon_lpips else None

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
    fixed_val_images = None
    if writer is not None and cfg.training.log_images_every_n_steps > 0:
        fixed_latents = _fixed_latent_batch(
            latent_ds, cfg.training.log_images_n_latents, cfg.data.seed
        )
        # The val loader is seeded, so this batch is identical at every epoch and the grids
        # stay comparable across the run.
        fixed_val_images = next(iter(val_loader))[: cfg.training.log_images_n_latents]
        vae.eval()
        _log_latent_samples(vae, fixed_latents, writer, device, step_counter)
        _log_real_vs_recon(vae, fixed_val_images, writer, device, step_counter)
        vae.train()

    # Training loop
    mp = cfg.training.mixed_precision
    # Separate scalers: a D-side inf must not change the loss scale while G grads are still
    # accumulating inside the same gradient-accumulation window.
    scaler_d = torch.amp.GradScaler("cuda", enabled=mp)
    scaler_g = torch.amp.GradScaler("cuda", enabled=mp)
    accum = max(1, cfg.adversarial.grad_accum_steps)
    scaling = vae.config.scaling_factor
    warmup_steps = max(0, cfg.discriminator.warmup_steps)
    ramp_steps = max(0, cfg.adversarial.lambda_ramp_steps)
    label_smoothing = float(np.clip(cfg.discriminator.label_smoothing, 0.0, 0.5))
    real_target = 1.0 - label_smoothing
    fake_target = label_smoothing
    grad_clip = float(cfg.adversarial.grad_clip_norm)
    warmup_logged = warmup_steps <= 0

    use_lpips = lpips_fn is not None
    best_val = float("inf")
    best_mse_at_best: float | None = None
    patience_counter = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_mse": [],
        "val_lpips": [],
        "train_l1": [],
        "train_lpips": [],
        "train_adv": [],
        "disc_loss": [],
    }

    def _step_vae_optimizer() -> None:
        if grad_clip > 0:
            if scaler_g.is_enabled():
                scaler_g.unscale_(vae_optimizer)
            nn.utils.clip_grad_norm_(vae_params, grad_clip)
        scaler_g.step(vae_optimizer)
        scaler_g.update()
        vae_optimizer.zero_grad()

    def _save_state(dest_dir: Path, save_disc: bool, merge: bool = True) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if cfg.model.lora.enabled:
            lora_dir = dest_dir / "lora"
            lora_dir.mkdir(parents=True, exist_ok=True)
            vae.save_pretrained(lora_dir)
            log.info(f"LoRA adapter saved → {lora_dir}")
            if merge and cfg.model.lora.merge_and_save_full:
                base_model = AutoencoderKL.from_pretrained(
                    cfg.model.pretrained_model_name_or_path,
                    subfolder=cfg.model.subfolder,
                ).to(device)
                merged_model = PeftModel.from_pretrained(base_model, str(lora_dir))
                merged_model = merged_model.merge_and_unload()
                merged_dir = dest_dir / "merged"
                merged_dir.mkdir(parents=True, exist_ok=True)
                merged_model.save_pretrained(merged_dir)
                log.info(f"Merged VAE saved → {merged_dir}")
        else:
            vae.save_pretrained(dest_dir)
            log.info(f"VAE weights saved → {dest_dir}")
        if save_disc:
            torch.save(disc.state_dict(), dest_dir / "discriminator.pt")

    # Align real and latent loaders by iterator
    latent_iter = iter(latent_loader)
    for epoch in range(1, cfg.training.epochs + 1):
        vae.train()
        disc.train()
        if cfg.discriminator.freeze_norm_stats:
            _freeze_norm_stats(disc)
        vae_optimizer.zero_grad()
        micro_idx = 0
        n_batches = 0
        # D metrics are pushed every step; G metrics only on steps where G actually updated
        wd, wg = _MetricAgg(), _MetricAgg()
        ed, eg = _MetricAgg(), _MetricAgg()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}", unit="batch")
        for x_real in pbar:
            x_real = x_real.to(device)
            z_fake_raw, latent_iter = _next_latent_batch(latent_iter, latent_loader)
            z_fake_raw = z_fake_raw.to(device)

            # Fake decode for D step: no decoder grads needed, keep it out of the graph
            with torch.no_grad():
                x_fake = vae.decode(z_fake_raw / scaling).sample

            # Discriminator step
            with torch.autocast(device_type="cuda", enabled=mp):
                real_disc = _disc_input_transform(x_real, disc_pp)
                fake_disc = _disc_input_transform(x_fake, disc_pp)
                hf_real = _hf_energy(real_disc)
                hf_fake = _hf_energy(fake_disc)
                disc_real = disc(real_disc)
                disc_fake = disc(fake_disc)
                loss_D = F.binary_cross_entropy_with_logits(
                    disc_real, torch.full_like(disc_real, real_target)
                ) + F.binary_cross_entropy_with_logits(
                    disc_fake, torch.full_like(disc_fake, fake_target)
                )
            del x_fake, real_disc, fake_disc

            bs = x_real.size(0)
            d_val = loss_D.item()
            acc_r_batch = (disc_real.detach().sigmoid() > 0.5).float().mean().item()
            acc_f_batch = (disc_fake.detach().sigmoid() < 0.5).float().mean().item()

            disc_optimizer.zero_grad()
            scaler_d.scale(loss_D).backward()
            scaler_d.step(disc_optimizer)
            scaler_d.update()
            del disc_real, disc_fake, loss_D

            wd.push(
                bs, loss=d_val, acc_r=acc_r_batch, acc_f=acc_f_batch, hf_r=hf_real, hf_f=hf_fake
            )
            ed.push(
                bs, loss=d_val, acc_r=acc_r_batch, acc_f=acc_f_batch, hf_r=hf_real, hf_f=hf_fake
            )

            # Effective adv weight: 0 during D warmup, linear ramp afterwards
            if step_counter < warmup_steps:
                lambda_eff = 0.0
            elif ramp_steps > 0:
                progress = min(1.0, (step_counter - warmup_steps) / ramp_steps)
                lambda_eff = cfg.adversarial.lambda_adv * progress
            else:
                lambda_eff = cfg.adversarial.lambda_adv

            if not warmup_logged and step_counter >= warmup_steps:
                warmup_logged = True
                log.info(f"\nD warmup complete at step {step_counter}; G updates enabled.")
                if writer is not None:
                    writer.add_scalar("Events/Discriminator/WarmupComplete", 1, step_counter)

            # Generator update: k decode passes; recon and adv graphs are backwarded
            # separately so at most one full decoder graph is live at a time
            if step_counter >= warmup_steps:
                k = max(1, cfg.adversarial.recon_steps_per_disc_update)
                step_l1 = step_lpips = step_adv = step_adv_raw = step_g = 0.0
                for _ in range(k):
                    # Train over the real dataset too in order to keep the
                    # decoder bound to real data too
                    recon_loss = None
                    with torch.autocast(device_type="cuda", enabled=mp):
                        x_recon = vae.decode(vae.encode(x_real).latent_dist.mean).sample
                        if cfg.adversarial.use_recon_l1:
                            l1 = F.l1_loss(x_recon, x_real)
                            l1_term = cfg.adversarial.weight_l1 * l1
                            recon_loss = l1_term if recon_loss is None else recon_loss + l1_term
                            step_l1 += l1.item()
                        if use_lpips:
                            lp = lpips_fn(x_recon, x_real).mean()
                            lp_term = cfg.adversarial.weight_lpips * lp
                            recon_loss = lp_term if recon_loss is None else recon_loss + lp_term
                            step_lpips += lp.item()
                    if recon_loss is not None:
                        scaler_g.scale(recon_loss / (k * accum)).backward()
                        step_g += recon_loss.item()
                    del x_recon, recon_loss

                    # Now run on the sampled lattents to get the "fake"
                    # images into the classifier network
                    if lambda_eff > 0:
                        with torch.autocast(device_type="cuda", enabled=mp):
                            z_fake_raw, latent_iter = _next_latent_batch(latent_iter, latent_loader)
                            x_fake_g = vae.decode(z_fake_raw.to(device) / scaling).sample
                            fake_disc_for_gen = disc(_disc_input_transform(x_fake_g, disc_pp))
                            adv_loss = F.binary_cross_entropy_with_logits(
                                fake_disc_for_gen, torch.ones_like(fake_disc_for_gen)
                            )
                        adv_val = adv_loss.item()
                        scaler_g.scale(lambda_eff * adv_loss / (k * accum)).backward()
                        step_adv += lambda_eff * adv_val
                        step_adv_raw += adv_val
                        step_g += lambda_eff * adv_val
                        del x_fake_g, fake_disc_for_gen, adv_loss

                micro_idx += 1
                if micro_idx % accum == 0:
                    _step_vae_optimizer()

                g_metrics = {
                    "loss": step_g / k,
                    "l1": step_l1 / k,
                    "lpips": step_lpips / k,
                    "adv": step_adv / k,
                    "adv_raw": step_adv_raw / k,
                }
                wg.push(bs, **g_metrics)
                eg.push(bs, **g_metrics)

            step_counter += 1
            n_batches += 1
            avg_d = wd.mean("loss")
            acc_r = wd.mean("acc_r")
            acc_f = wd.mean("acc_f")
            hf_r = wd.mean("hf_r")
            hf_f = wd.mean("hf_f")
            avg_step = wg.mean("loss")
            avg_l1 = wg.mean("l1")
            avg_adv = wg.mean("adv")
            avg_adv_raw = wg.mean("adv_raw")
            avg_lpips = wg.mean("lpips")

            if step_counter % cfg.training.log_every_n_steps == 0:
                pbar.set_postfix(
                    {
                        "l1": f"{avg_l1:.4f}",
                        "adv": f"{avg_adv_raw:.3f}",
                        "D": f"{avg_d:.4f}",
                        "acc": f"{acc_r:.2f}/{acc_f:.2f}",
                        "hf": f"{hf_r:.2f}/{hf_f:.2f}",
                    }
                )
                if writer is not None:
                    writer.add_scalar("TrainSteps/Discriminator/Loss", avg_d, step_counter)
                    writer.add_scalar("TrainSteps/Discriminator/AccReal", acc_r, step_counter)
                    writer.add_scalar("TrainSteps/Discriminator/AccFake", acc_f, step_counter)
                    writer.add_scalar("TrainSteps/Discriminator/HFReal", hf_r, step_counter)
                    writer.add_scalar("TrainSteps/Discriminator/HFFake", hf_f, step_counter)

                    if step_counter >= warmup_steps:
                        writer.add_scalar("TrainSteps/Generator/Recon_L1", avg_l1, step_counter)
                        writer.add_scalar(
                            "TrainSteps/Generator/Recon_LPIPS", avg_lpips, step_counter
                        )
                        writer.add_scalar("TrainSteps/Generator/Adv", avg_adv, step_counter)
                        writer.add_scalar("TrainSteps/Generator/AdvRaw", avg_adv_raw, step_counter)
                        writer.add_scalar("TrainSteps/Generator/Loss", avg_step, step_counter)

                    # Scheduling
                    writer.add_scalar("Schedule/Generator/AdvWeight", lambda_eff, step_counter)
                    writer.add_scalar(
                        "Schedule/Generator/LR",
                        vae_optimizer.param_groups[0]["lr"],
                        step_counter,
                    )
                    writer.add_scalar(
                        "Schedule/Discriminator/LR",
                        disc_optimizer.param_groups[0]["lr"],
                        step_counter,
                    )
                    if len(disc_optimizer.param_groups) > 1:
                        writer.add_scalar(
                            "Schedule/Discriminator/BackboneLR",
                            disc_optimizer.param_groups[-1]["lr"],
                            step_counter,
                        )
                log.info(
                    f"\nStep {step_counter} | G={avg_step:.4f} (l1={avg_l1:.4f} lpips={avg_lpips:.4f} "
                    f"adv={avg_adv:.4f} raw={avg_adv_raw:.3f}) | D={avg_d:.4f} "
                    f"acc_r={acc_r:.2f} acc_f={acc_f:.2f} hf_r={hf_r:.2f} hf_f={hf_f:.2f} "
                    f"[last {cfg.training.log_every_n_steps} steps, {n_batches} batches]"
                )
                wd.reset()
                wg.reset()
                n_batches = 0

            if (
                writer is not None
                and cfg.training.log_images_every_n_steps > 0
                and step_counter % cfg.training.log_images_every_n_steps == 0
            ):
                _log_latent_samples(vae, fixed_latents, writer, device, step_counter)

        # Flush remaining partial gradient-accumulation window
        if micro_idx % accum != 0:
            _step_vae_optimizer()

        # Validation (recon L1 + MSE monitor, optional LPIPS)
        vae.eval()
        val_l1 = val_mse = val_lpips = 0.0
        n_val = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                posterior = vae.encode(x)
                z = posterior.latent_dist.mean
                recon = vae.decode(z).sample
                bs = x.size(0)
                val_l1 += F.l1_loss(recon, x).item() * bs
                val_mse += F.mse_loss(recon, x).item() * bs
                if use_lpips:
                    val_lpips += lpips_fn(x, recon).mean().item() * bs
                n_val += bs
        if writer is not None and fixed_val_images is not None:
            _log_real_vs_recon(vae, fixed_val_images, writer, device, step_counter)
        avg_val_l1 = val_l1 / n_val if n_val else float("inf")
        avg_val_mse = val_mse / n_val if n_val else float("inf")
        avg_val_lpips = val_lpips / n_val if n_val else float("inf")
        # Model selection tracks the perceptual objective the GAN term optimizes: the real
        # targets are 200px crops upscaled to 512, so pixel error is expected to trade off
        # against the sharpness the discriminator pushes for. L1/MSE stay logged as monitors.
        avg_val = avg_val_lpips if use_lpips else avg_val_l1
        avg_train = eg.mean("loss")
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_mse"].append(avg_val_mse)
        history["val_lpips"].append(avg_val_lpips)
        history["train_l1"].append(eg.mean("l1"))
        history["train_lpips"].append(eg.mean("lpips"))
        history["train_adv"].append(eg.mean("adv"))
        history["disc_loss"].append(ed.mean("loss"))

        vae_scheduler.step(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            best_mse_at_best = avg_val_mse
            patience_counter = 0
            _save_state(run_dir / "weights", save_disc=True)
        else:
            patience_counter += 1
        _save_state(run_dir / "weights_last", save_disc=False, merge=False)

        log.info(
            f"Epoch {epoch}/{cfg.training.epochs} | G={avg_train:.4f} "
            f"(l1={eg.mean('l1'):.4f} lpips={eg.mean('lpips'):.4f} adv={eg.mean('adv'):.4f}) "
            f"| D={ed.mean('loss'):.4f} acc_r={ed.mean('acc_r'):.2f} acc_f={ed.mean('acc_f'):.2f} "
            f"| val sel={avg_val:.4f} lpips={avg_val_lpips:.4f} l1={avg_val_l1:.4f} "
            f"mse={avg_val_mse:.4f} "
            f"| lr={vae_optimizer.param_groups[0]['lr']:.2e} patience={patience_counter}"
        )

        if writer is not None:
            writer.add_scalar("TrainEpoch/Generator/Loss", avg_train, epoch)
            writer.add_scalar("TrainEpoch/Discriminator/Loss", ed.mean("loss"), epoch)
            writer.add_scalar("TrainEpoch/Generator/Recon_L1", eg.mean("l1"), epoch)
            writer.add_scalar("TrainEpoch/Generator/Recon_LPIPS", eg.mean("lpips"), epoch)
            writer.add_scalar("TrainEpoch/Generator/Adv", eg.mean("adv"), epoch)
            writer.add_scalar("TrainEpoch/Discriminator/AccReal", ed.mean("acc_r"), epoch)
            writer.add_scalar("TrainEpoch/Discriminator/AccFake", ed.mean("acc_f"), epoch)
            writer.add_scalar("TrainEpoch/Discriminator/HFReal", ed.mean("hf_r"), epoch)
            writer.add_scalar("TrainEpoch/Discriminator/HFFake", ed.mean("hf_f"), epoch)
            writer.add_scalar("TrainEpoch/Generator/AdvRaw", eg.mean("adv_raw"), epoch)
            writer.add_scalar("ValidationEpoch/Generator/Selection", avg_val, epoch)
            writer.add_scalar("ValidationEpoch/Generator/Recon_L1", avg_val_l1, epoch)
            writer.add_scalar("ValidationEpoch/Generator/Recon_MSE", avg_val_mse, epoch)
            if use_lpips:
                writer.add_scalar("ValidationEpoch/Generator/Recon_LPIPS", avg_val_lpips, epoch)

        if patience_counter >= cfg.training.early_stopping_patience:
            log.info(f"Early stopping at epoch {epoch}.")
            break

    if writer is not None:
        writer.close()

    metrics = {
        "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "val_loss": best_val if best_val < float("inf") else None,
        "val_mse_at_best": best_mse_at_best,
        "selection_metric": "val_lpips" if use_lpips else "val_l1",
        "epochs_run": len(history["val_loss"]),
        "per_epoch": dict(history),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
