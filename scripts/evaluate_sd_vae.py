"""
Re-runs evaluation plots for a completed SD VAE experiment without re-training.

Usage:
    uv run python scripts/evaluate_sd_vae.py --config experiments/sd_vae_finetune/baseline/config.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from allium_cepa_classifier.config.sd_vae_config import SDVAEExperimentConfig
from allium_cepa_classifier.training.sd_vae_evaluator import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = SDVAEExperimentConfig.from_yaml(args.config)
    run_dir = args.config.parent

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    weights_dir = run_dir / "weights"
    if not weights_dir.exists():
        raise FileNotFoundError(f"No weights found at {weights_dir}. Run training first.")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.json found at {metrics_path}.")

    history = json.loads(metrics_path.read_text())["history"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    from diffusers import AutoencoderKL
    # Load base model
    model = AutoencoderKL.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder=cfg.model.subfolder,
    ).to(device)

    # Load LoRA or full weights
    if cfg.model.lora.enabled:
        lora_dir = weights_dir / "lora"
        if not lora_dir.exists():
            raise FileNotFoundError(f"LoRA weights not found at {lora_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(lora_dir))
        log.info(f"Loaded LoRA weights from {lora_dir}")
    else:
        # Load fine-tuned diffusers weights
        model = AutoencoderKL.from_pretrained(str(weights_dir)).to(device)
        log.info(f"Loaded weights from {weights_dir}")

    model.eval()
    log.info("Model loaded")

    # For SD VAE evaluator, val_loader is unused but kept for API compatibility
    from torch.utils.data import DataLoader
    val_loader = DataLoader([], batch_size=1)

    run_evaluation(model, history, cfg, run_dir, val_loader, device)
    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
