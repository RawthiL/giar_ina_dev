"""
Usage:
    uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml
    uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml --dry-run
    uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml --no-calibrate
"""
import argparse
import logging
import shutil
from datetime import datetime
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
    exp_path = Path(args.config).parent
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = exp_path / timestamp
    run_dir.mkdir(parents=True)
    (run_dir / "weights").mkdir()
    (run_dir / "plots").mkdir()

    shutil.copy(args.config, run_dir / "used_config.yaml")

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

        model = build_model(cfg.model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Dry run OK. Run dir: {run_dir}")
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
