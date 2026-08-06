"""
Usage:
    uv run python scripts/train_detector.py --config experiments/yolo/yolo11n_200e/config.yaml
    uv run python scripts/train_detector.py --config experiments/yolo/yolo11n_200e/config.yaml --dry-run
    uv run python scripts/train_detector.py --config experiments/yolo/yolo11n_200e/config.yaml --no-calibrate
"""

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path

from allium_cepa_classifier.config.detector_config import DetectorConfig


def _extract_metrics(yolo_run_dir: Path) -> dict:
    results_csv = yolo_run_dir / "results.csv"
    if not results_csv.exists():
        return {}
    with results_csv.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = rows[-1]
    metrics = {}
    for k, v in last.items():
        k = k.strip()
        try:
            metrics[k] = float(v)
        except (ValueError, TypeError):
            metrics[k] = v
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train YOLO cell detector.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config only, do not train",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip confidence calibration after training",
    )
    args = parser.parse_args()

    cfg = DetectorConfig.from_yaml(args.config)
    run_dir = Path(args.config).resolve().parent
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
        print(f"Dry run OK. Run dir: {run_dir}")
        print(f"  weights: {cfg.weights}")
        print(f"  data:    {cfg.data}")
        print(f"  epochs:  {cfg.epochs}  imgsz: {cfg.imgsz}  device: {cfg.device}")
        return

    from ultralytics import YOLO, settings

    settings.update({"tensorboard": cfg.tensorboard})

    model = YOLO(str(cfg.weights))
    model.train(
        data=str(cfg.data),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        device=cfg.device,
        project=str(run_dir),
        name="yolo",
        exist_ok=True,
    )
    model.val(
        split="val",
        imgsz=cfg.imgsz,
        batch=16,
        conf=0.001,
        iou=0.7,
        plots=True,
        project=str(run_dir),
        name="yolo",
        exist_ok=True,
    )

    yolo_run_dir = run_dir / "yolo"

    best_pt = yolo_run_dir / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, run_dir / "weights" / "object_detection.pt")
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, cfg.out)

    for png in yolo_run_dir.glob("*.png"):
        shutil.copy(png, run_dir / "plots" / png.name)

    metrics = _extract_metrics(yolo_run_dir)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nDone. Artifacts in: {run_dir}")
    print(f"Detector weights copied to: {cfg.out}")

    if not args.no_calibrate:
        from allium_cepa_classifier.training.detector_calibrator import run_detection_calibration

        logging.info("\n--- Calibration ---")
        cal_metrics = run_detection_calibration(run_dir)
        print(f"ECE before: {cal_metrics['ece_before']:.4f}  after: {cal_metrics['ece_after']:.4f}")


if __name__ == "__main__":
    main()
