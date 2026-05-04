# Allium Cepa Automation

Computer vision pipeline for automated detection and mitosis classification in *Allium cepa* (onion root tip) microscopy images. Built at UTN / INA.

The pipeline detects cells using a YOLO-based detector, then classifies each detection as **mitosis** or **no mitosis** using an EfficientNet classifier (PyTorch / `timm`). Isotonic regression and vector-scaling calibration steps are available for both models to improve probability estimates.

---

## TensorFlow → PyTorch Migration

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Update dependencies (`timm`, `torchvision`; remove `tensorflow`) | ✅ Done |
| 2 | Rewrite classifier training (PyTorch training loop) | ✅ Done |
| 3 | Rewrite calibration (vector scaling, isotonic regression) | ✅ Done |
| 4 | Rewrite `AlliumCepaModel` inference | ✅ Done |
| 5 | Migrate training and calibration to experiment-based scripts | ✅ Done |
| 6 | Use DVC for dataset management | ⏳ Pending |

---

## Project Structure

```
allium_cepa_automation/
│
├── experiments/                        # Configs and results, co-located by experiment
│   ├── binary_classifier/
│   │   ├── efficientnet_b1/
│   │   │   ├── config.yaml             # Canonical config for this experiment
│   │   │   └── 20260503-161453/        # Timestamped run (one per training run)
│   │   │       ├── used_config.yaml
│   │   │       ├── train.log
│   │   │       ├── calibrate.log
│   │   │       ├── metrics.json
│   │   │       ├── weights/            # classifier.pt, classifier_calibrated.pt
│   │   │       └── plots/              # training_curves.png, reliability_diagram.png, ...
│   │   ├── efficientnet_b2/
│   │   ├── resnet50/
│   │   └── vgg19/
│   └── yolo/
│       └── yolo11n_200e/
│           ├── config.yaml
│           └── 20260503-211520/
│               ├── used_config.yaml
│               ├── train.log
│               ├── metrics.json
│               ├── weights/            # object_detection.pt, yolo_isotonic_calibrator.pkl
│               ├── plots/              # detector_reliability_diagram.png, ...
│               └── yolo/               # Raw Ultralytics output
│
├── notebooks/
│   ├── analysis/                       # compare_experiments.ipynb, eda.ipynb
│   ├── inference/                      # full_pipeline.ipynb
│   └── training/                       # VAE training notebooks
│
├── scripts/
│   ├── train_classifier.py             # Train binary classifier (with calibration)
│   ├── calibrate_classifier.py         # Re-run calibration on an existing run
│   ├── sweep.py                        # Run train_classifier over multiple configs
│   ├── train_detector.py               # Train YOLO detector (with calibration)
│   ├── calibrate_detector.py           # Re-run calibration on an existing detector run
│   └── utils/
│       ├── augment_crops.py            # Data augmentation for classifier crops
│       ├── classifier_dataset.py       # Prepare binary classifier dataset splits
│       ├── coco_to_yolo.py             # Convert COCO annotations to YOLO format
│       ├── download_hf_dataset.py      # Download dataset from Hugging Face
│       └── vae_dataset.py              # Prepare VAE dataset splits
│
├── src/
│   └── allium_cepa_classifier/
│       ├── config/
│       │   ├── allium_cepa_config.py   # Inference config (weights paths, image size, ...)
│       │   ├── detector_config.py      # YOLO training config
│       │   ├── experiment_config.py    # Classifier training config
│       │   └── training_config.py      # Dataset preparation paths
│       ├── data_models/
│       │   ├── allium_cepa_model.py    # Full inference pipeline (detect + classify)
│       │   └── allium_cepa_result.py   # Result dataclass with save/visualize helpers
│       ├── training/
│       │   ├── calibrator.py           # Vector-scaling calibration for classifier
│       │   ├── detector_calibrator.py  # Isotonic calibration for YOLO detector
│       │   ├── model_builder.py        # Build timm classifier from ExperimentConfig
│       │   └── trainer.py              # PyTorch training loop
│       └── weights/                    # Model weights — not tracked in git
│
├── thoughts/
│   └── shared/plans/                   # Implementation plans and migration notes
│
├── datasets/                           # Raw and processed datasets — not tracked in git
├── pyproject.toml
└── uv.lock
```

---

## Setup

```bash
uv sync --all-groups
```

> Model weights and datasets are not tracked in this repository. Place weights under `src/allium_cepa_classifier/weights/` and datasets under `datasets/`.

---

## Training

### Binary Classifier

```bash
# Train a single experiment (calibration runs automatically afterwards)
uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml

# Skip calibration
uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml --no-calibrate

# Sweep all experiments
uv run python scripts/sweep.py --configs experiments/binary_classifier/*/config.yaml
```

### YOLO Detector

```bash
# Train (calibration runs automatically afterwards)
uv run python scripts/train_detector.py --config experiments/yolo/yolo11n_200e/config.yaml

# Skip calibration
uv run python scripts/train_detector.py --config experiments/yolo/yolo11n_200e/config.yaml --no-calibrate
```

### Re-run Calibration Standalone

```bash
uv run python scripts/calibrate_classifier.py --experiment experiments/binary_classifier/efficientnet_b1/20260503-161453
uv run python scripts/calibrate_detector.py   --experiment experiments/yolo/yolo11n_200e/20260503-211520
```