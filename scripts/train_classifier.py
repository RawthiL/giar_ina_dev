"""
Usage:
    uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml
    uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml --dry-run
    uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml --no-calibrate
"""

import argparse
import logging
from pathlib import Path

from allium_cepa_classifier.config.experiment_config import ExperimentConfig
from allium_cepa_classifier.training.trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and build model only, do not train",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip temperature calibration after training",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
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
        from allium_cepa_classifier.training.model_builder import build_model

        # Infer the class count from the dataset when it is present; a dry run should
        # still work on a fresh clone with no data pulled, so fall back to binary.
        train_dir = cfg.data.resolved_crops_dir / "train"
        classes = (
            sorted(p.name for p in train_dir.iterdir() if p.is_dir()) if train_dir.is_dir() else []
        )
        num_classes = len(classes) or 2
        model = build_model(cfg.model, num_classes=num_classes)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Dry run OK. Run dir: {run_dir}")
        if classes:
            print(f"Classes ({num_classes}): {', '.join(classes)}")
        else:
            print(f"Classes: dataset not found at {train_dir}, assuming {num_classes}")
        print(f"Trainable params: {trainable:,} / {total:,}")
        return

    metrics = run_training(cfg, run_dir)
    print(f"\nDone. Artifacts in: {run_dir}")
    print(f"Test accuracy: {metrics['test_acc']:.4f}")

    if not args.no_calibrate:
        from allium_cepa_classifier.training.calibrator import run_calibration

        logging.info("\n--- Calibration ---")
        cal_metrics = run_calibration(run_dir)
        print(f"ECE before: {cal_metrics['ece_before']:.4f}  after: {cal_metrics['ece_after']:.4f}")
        print(f"Temperature: {cal_metrics['temperature']}")


if __name__ == "__main__":
    main()
