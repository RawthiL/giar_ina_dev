"""
Usage:
    uv run python scripts/train_vae.py --config experiments/vae/latent32_beta2/config.yaml
    uv run python scripts/train_vae.py --config experiments/vae/latent32_beta2/config.yaml --dry-run
"""

import argparse
import logging
from pathlib import Path

from allium_cepa_classifier.config.vae_config import VAEExperimentConfig
from allium_cepa_classifier.training.vae_trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build model and print param count without training",
    )
    args = parser.parse_args()

    cfg = VAEExperimentConfig.from_yaml(args.config)
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
        from allium_cepa_classifier.training.vae_model import VAE

        m = VAE(cfg.model, learnable_prior=cfg.training.learnable_prior)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        print(f"Dry run OK. Trainable: {trainable:,} / {total:,}")
        return

    metrics = run_training(cfg, run_dir)
    print(f"\nDone. val_loss={metrics['val_loss']:.4f}")


if __name__ == "__main__":
    main()
