"""
Re-runs evaluation plots for a completed VAE experiment without re-training.

Usage:
    uv run python scripts/evaluate_vae.py --config experiments/vae/latent32_beta05/config.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from allium_cepa_classifier.config.vae_config import VAEExperimentConfig
from allium_cepa_classifier.training.vae_evaluator import run_evaluation
from allium_cepa_classifier.training.vae_model import VAE
from allium_cepa_classifier.training.vae_trainer import _build_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = VAEExperimentConfig.from_yaml(args.config)
    run_dir = args.config.parent

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    weights_path = run_dir / "weights" / "vae.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"No weights found at {weights_path}. Run training first.")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.json found at {metrics_path}.")

    history = json.loads(metrics_path.read_text())["history"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    checkpoint = torch.load(weights_path, map_location=device)
    model = VAE(cfg.model, learnable_prior=cfg.training.learnable_prior).to(device)
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    with torch.no_grad():
        model.prior_mean.copy_(checkpoint["prior_mean"].to(device))
        model.prior_log_var.copy_(checkpoint["prior_log_var"].to(device))
    model.eval()
    log.info(f"Loaded weights from {weights_path}")

    _, val_loader = _build_loaders(cfg)

    run_evaluation(model, history, cfg, run_dir, val_loader, device)
    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
