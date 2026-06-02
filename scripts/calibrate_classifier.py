"""
Usage:
    uv run python scripts/calibrate.py --experiment experiments/20260502-153000_efficientnet_b1_baseline
"""

import argparse
import logging
from pathlib import Path

from allium_cepa_classifier.training.calibrator import run_calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Path to experiment directory, e.g. experiments/binary_classifier/efficientnet_b1",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.experiment / "calibrate.log"),
        ],
    )

    metrics = run_calibration(args.experiment)
    print(f"ECE before: {metrics['ece_before']:.4f}  after: {metrics['ece_after']:.4f}")
    print(f"Temperature: {metrics['temperature']}")


if __name__ == "__main__":
    main()
