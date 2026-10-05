"""
Usage:
    uv run python scripts/train_mitosis_classifier.py --config experiments/mitosis_classifier/real_only/config.yaml
    uv run python scripts/train_mitosis_classifier.py --config experiments/mitosis_classifier/real_plus_synth/config.yaml
    uv run python scripts/train_mitosis_classifier.py --config ... --dry-run
"""

import argparse
import logging
from pathlib import Path

from allium_cepa_classifier.config.mitosis_phase_config import MitosisPhaseConfig
from allium_cepa_classifier.training.mitosis_phase_trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and build model only, do not train",
    )
    args = parser.parse_args()

    cfg = MitosisPhaseConfig.from_yaml(args.config)
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

        model = build_model(cfg.model, num_classes=len(cfg.data.classes))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Dry run OK. Run dir: {run_dir}")
        print(f"Classes: {cfg.data.classes} (synthetic included: {cfg.data.include_synthetic})")
        print(f"Trainable params: {trainable:,} / {total:,}")
        return

    metrics = run_training(cfg, run_dir)
    print(f"\nDone. Artifacts in: {run_dir}")
    print(f"Test accuracy: {metrics['test_acc']:.4f}  macro-F1: {metrics['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
