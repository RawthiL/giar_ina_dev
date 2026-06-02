"""
Usage:
    uv run python scripts/sweep.py --configs experiments/binary_classifier/*/config.yaml
    uv run python scripts/sweep.py --configs experiments/binary_classifier/efficientnet_b1/config.yaml experiments/binary_classifier/resnet50/config.yaml
    uv run python scripts/sweep.py --configs experiments/binary_classifier/*/config.yaml --no-calibrate
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--no-calibrate", action="store_true", help="Skip calibration after training"
    )
    args = parser.parse_args()

    results = []
    for cfg_path in args.configs:
        print(f"\n{'=' * 60}")
        print(f"Starting: {cfg_path.parent.name}")
        print(f"{'=' * 60}")

        cmd = [sys.executable, "scripts/train_classifier.py", "--config", str(cfg_path)]
        if args.no_calibrate:
            cmd.append("--no-calibrate")

        result = subprocess.run(cmd, capture_output=False)
        status = "OK" if result.returncode == 0 else "FAILED"
        if not args.no_calibrate and status == "OK":
            status = "OK (with calibration)"
        results.append((cfg_path.parent.name, status))

    print("\n\nSweep summary:")
    for name, status in results:
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
