"""
Usage:
    uv run python scripts/train_sd_vae.py --config experiments/sd_vae_finetune/baseline/config.yaml
    uv run python scripts/train_sd_vae.py --config experiments/sd_vae_finetune/baseline/config.yaml --dry-run
"""

import argparse
import logging
from pathlib import Path

from allium_cepa_classifier.config.sd_vae_config import SDVAEExperimentConfig
from allium_cepa_classifier.training.sd_vae_trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build model and print param count without training",
    )
    args = parser.parse_args()

    cfg = SDVAEExperimentConfig.from_yaml(args.config)
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
        from diffusers import AutoencoderKL
        import torch
        
        model = AutoencoderKL.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            subfolder=cfg.model.subfolder,
        )
        
        if cfg.model.decoder_only:
            for name, param in model.named_parameters():
                if "encoder" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Dry run OK. Trainable: {trainable:,} / {total:,}")
        print(f"Resolution: {cfg.model.resolution}x{cfg.model.resolution}")
        print(f"Decoder only: {cfg.model.decoder_only}")
        return

    metrics = run_training(cfg, run_dir)
    print(f"\nDone. val_loss={metrics['val_loss']:.4f}")


if __name__ == "__main__":
    main()
