"""
Usage:
    uv run python scripts/train_sd_vae_adversarial.py --config experiments/sd_vae_adversarial/baseline/config.yaml
    uv run python scripts/train_sd_vae_adversarial.py --config experiments/sd_vae_adversarial/baseline/config.yaml --dry-run
"""

import argparse
import logging
from pathlib import Path

from allium_cepa_classifier.config.sd_vae_adversarial_config import SDVAEAdversarialExperimentConfig
from allium_cepa_classifier.training.sd_vae_adversarial_trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Build model and print param count without training"
    )
    args = parser.parse_args()

    cfg = SDVAEAdversarialExperimentConfig.from_yaml(args.config)
    run_dir = Path(args.config).parent
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "train.log"),
        ],
    )

    if args.dry_run:
        import torch
        from diffusers import AutoencoderKL
        from peft import LoraConfig, get_peft_model

        log = logging.getLogger(__name__)

        model = AutoencoderKL.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            subfolder=cfg.model.subfolder,
        )

        if cfg.model.lora.enabled:
            lora_config = LoraConfig(
                r=cfg.model.lora.r,
                lora_alpha=cfg.model.lora.alpha,
                target_modules=cfg.model.lora.target_modules,
                lora_dropout=cfg.model.lora.dropout,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            log.info(f"LoRA injected with r={cfg.model.lora.r}")

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
            for name, param in model.named_parameters():
                if "encoder" in name or (cfg.model.freeze_quant_conv and "quant_conv" in name):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Dry run OK. VAE trainable: {trainable:,} / {total:,}")
        print(f"Resolution: {cfg.model.resolution}x{cfg.model.resolution}")
        print(f"Decoder only: {cfg.model.decoder_only}")
        print(f"LoRA enabled: {cfg.model.lora.enabled}")
        if cfg.model.lora.enabled:
            print(f"LoRA r={cfg.model.lora.r}, alpha={cfg.model.lora.alpha}")
        print(f"Discriminator image_size: {cfg.discriminator.image_size}")
        print(
            f"Discriminator trainable: {cfg.discriminator.trainable} | "
            f"head lr: {cfg.discriminator.lr:.1e} | backbone lr: {cfg.discriminator.backbone_lr:.1e} | "
            f"label_smoothing: {cfg.discriminator.label_smoothing} | "
            f"freeze_norm_stats: {cfg.discriminator.freeze_norm_stats}"
        )
        print(
            f"Generator: l1={cfg.adversarial.weight_l1} (use={cfg.adversarial.use_recon_l1}) "
            f"lpips={cfg.adversarial.weight_lpips} lambda_adv={cfg.adversarial.lambda_adv} "
            f"clip={cfg.adversarial.grad_clip_norm}"
        )
        print(
            f"D warmup_steps: {cfg.discriminator.warmup_steps} | "
            f"lambda_ramp_steps: {cfg.adversarial.lambda_ramp_steps} | "
            f"grad_accum_steps: {cfg.adversarial.grad_accum_steps}"
        )
        print(f"Latent dataset: {cfg.adversarial.latent_dataset}")

        from allium_cepa_classifier.training.sd_vae_adversarial_trainer import (
            LatentParquetDataset,
            _disc_input_transform,
            _freeze_norm_stats,
            _hf_energy,
            _load_discriminator,
        )

        latent_ds = LatentParquetDataset(
            cfg.adversarial.latent_dataset, cfg.adversarial.latent_config
        )
        print(f"Latents: {len(latent_ds)} samples, shape={tuple(latent_ds.latents.shape[1:])}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        disc, disc_pp = _load_discriminator(cfg, device)
        norms = [m for m in disc.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        if norms:
            # Mirror the epoch-start sequence in run_training(): train() then re-pin.
            disc.train()
            if cfg.discriminator.freeze_norm_stats:
                _freeze_norm_stats(disc)
            pinned = sum(1 for m in norms if (not m.training) and m.momentum == 0.0)
            print(
                f"BatchNorm layers: {len(norms)} | pinned after disc.train(): {pinned}/{len(norms)}"
            )
        disc.eval()
        with torch.no_grad():
            gen = torch.Generator(device="cpu").manual_seed(0)
            probe = (
                torch.rand(
                    2,
                    3,
                    cfg.model.resolution,
                    cfg.model.resolution,
                    generator=gen,
                )
                .to(device)
                .mul_(2)
                .sub_(1)
            )
            disc_in = _disc_input_transform(probe, disc_pp)
            out = disc(disc_in)
            print(
                f"Discriminator output shape: {tuple(out.shape)} (expected (B, 1) real/fake logit)"
            )
            print(f"Discriminator norm stats: mean={disc_pp.mean} std={disc_pp.std}")
            print(f"Discriminator degrade_roundtrip_mid: {disc_pp.degrade_mid}")
            if disc_pp.degrade_mid:
                from dataclasses import replace

                undegraded = _disc_input_transform(probe, replace(disc_pp, degrade_mid=None))
                print(
                    f"HF energy on identical noise: degraded={_hf_energy(disc_in):.4f} "
                    f"undegraded={_hf_energy(undegraded):.4f} (must drop, both branches share it)"
                )
        return

    metrics = run_training(cfg, run_dir)
    val_loss = metrics["val_loss"]
    if val_loss is None:
        print("\nDone. No epoch completed, no checkpoint written.")
    else:
        print(
            f"\nDone. best {metrics['selection_metric']}={val_loss:.4f} "
            f"(val_mse_at_best={metrics['val_mse_at_best']:.4f})"
        )


if __name__ == "__main__":
    main()
