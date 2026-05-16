# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computer vision pipeline for automated cell detection and mitosis classification in *Allium cepa* (onion root tip) microscopy images, developed at UTN/INA.

Two-stage pipeline:
1. **Detection**: YOLO model (Ultralytics) detects individual cells in full-FOV images.
2. **Classification**: EfficientNet/ResNet/VGG backbone (via `timm`) classifies each crop as *mitosis* or *no_mitosis*.

Both stages have post-hoc calibration: vector scaling (classifier) and isotonic regression (detector).

## Setup

```bash
uv sync --all-groups
uv run pre-commit install
```

Weights and datasets are not tracked in git. Place them at:
- Model weights → `src/allium_cepa_classifier/weights/`
- Datasets → `datasets/`

## Commands

### Linting & Formatting

```bash
uv run ruff check --fix .
uv run ruff format .
```

### Tests

```bash
uv run pytest
```

### Data Preparation (run once before training)

The dataset is pinned to a specific HuggingFace commit via `--rev` in the
`download_dataset` stage of `dvc.yaml`. To prepare data on a fresh clone:

```bash
dvc pull                                               # fetch cached outputs by content hash
dvc repro download_dataset coco_to_yolo prepare_crops  # or full dvc repro
```

To bump the pinned dataset version (intentional, rare):

```bash
# Get new SHA:
python -c "from huggingface_hub import repo_info; print(repo_info('GIAR-UTN/allium-cepa-dataset', repo_type='dataset').sha)"
# Edit dvc.yaml: change --rev <old-sha> → --rev <new-sha>
dvc repro download_dataset
git add dvc.yaml dvc.lock
git commit -m "chore: bump HF dataset to <new-sha>"
```

### Training

```bash
# Single experiment (calibration runs automatically after)
uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml

# Flags: --no-calibrate, --dry-run (builds model + prints param count, no training)
uv run python scripts/train_classifier.py --config ... --dry-run

# Sweep all classifier configs
uv run python scripts/sweep.py --configs experiments/binary_classifier/*/config.yaml

# YOLO detector
uv run python scripts/train_detector.py --config experiments/yolo/yolo11n_200e/config.yaml
```

### Standalone Calibration

```bash
uv run python scripts/calibrate_classifier.py --experiment experiments/binary_classifier/efficientnet_b1/20260503-161453
uv run python scripts/calibrate_detector.py --experiment experiments/yolo/yolo11n_200e/20260503-211520
```

## Architecture

### Experiment System

Each experiment has a canonical config at `experiments/<type>/<name>/config.yaml`. A training run creates a timestamped subdirectory:

```
experiments/binary_classifier/efficientnet_b1/
├── config.yaml
└── 20260503-161453/
    ├── used_config.yaml         ← exact config snapshot
    ├── metrics.json             ← acc + ECE before/after calibration
    ├── weights/
    │   ├── classifier.pt
    │   └── classifier_calibrated.pt
    └── plots/
```

### Config System

All configs are Pydantic v2 models extending `BaseConfig` (`src/allium_cepa_classifier/config/base_config.py`). `BaseConfig.from_yaml(path)` loads YAML into the model. `find_project_root()` walks up from `__file__` to `pyproject.toml` and is used to resolve absolute paths at import time.

| Config class | Purpose |
|---|---|
| `AlliumCepaConfig` | Inference: weights paths, image_size, batch_size |
| `ExperimentConfig` | Classifier training: model arch, hyperparams, data paths |
| `DetectorConfig` | YOLO training: weights, data.yaml, epochs, device |
| `TrainingConfig` | Dataset preparation: raw/processed paths |

### Model Architecture (`training/model_builder.py`)

`build_model(cfg)` produces a `BackboneWithHead`:
- **Backbone**: timm model with `num_classes=0` (feature extractor). Supported: `efficientnet_b1`, `efficientnet_b2`, `resnet50`, `vgg19`.
- **Head**: MLP `[in_features → 512 → 256 → 128 → 2]` with LeakyReLU(0.2) + Dropout. **No softmax** — outputs raw logits.
- **Stage freezing**: `freeze_model_stages(model, arch, n)` keeps only the last `n` backbone stages trainable. Architecture-specific stage groupings are hardcoded.

### Classifier Training Flow (`training/trainer.py`)

1. `ImageFolder` datasets from `datasets/crops/binary_classifier/{train,validation,test}/`
2. Class weights via sklearn `compute_class_weight("balanced")` with per-class multipliers from config (default: mitosis×2.0)
3. Adam + `ReduceLROnPlateau`, `CrossEntropyLoss` with class weights, early stopping on val loss
4. Saves `classifier.pt` with state dict + metadata (arch, image_size, class_to_idx, normalization)

### Calibration

**Classifier** (`training/calibrator.py`): Optimizes a per-class temperature vector (shape `[2]`) via L-BFGS-B on val set logits. `CalibratedClassifier` divides logits by the temperature vector before softmax. Bounds: `[0.01, 10.0]` per class.

**Detector** (`training/detector_calibrator.py`): Runs YOLO on val images at conf=0.01, matches predictions to ground truth via greedy IoU≥0.5, then fits `IsotonicRegression` (confidence → TP/FP). Saved as pickle.

### Inference (`data_models/allium_cepa_model.py`)

```python
from allium_cepa_classifier import AlliumCepaModel, AlliumCepaConfig

model = AlliumCepaModel(AlliumCepaConfig())
result = model.predict("path/to/image.png")   # or a directory
result.get_counts()         # {"total_cells": N, "mitotic_cells": M, "mitotic_index": float}
result.get_counts_with_ci() # includes mi, ci_lower, ci_upper, sigma_mi (+ all get_counts keys)
result.show_annotated()     # PIL image with bounding boxes
result.save_csv("out.csv")
```

Per-image: YOLO → isotonic calibration → crop cells → batch through classifier → temperature scaling → softmax → `AlliumCepaResult`.

The detector's isotonic calibrator (`yolo_isotonic_calibrator.pkl`) must sit beside `object_detection.pt` in `src/allium_cepa_classifier/weights/` for `get_counts_with_ci()` to produce calibrated CIs. If the pickle is absent, a warning is issued and raw YOLO confidence is used as `p_hat`.

### Dataset Conventions

- Raw: COCO-format at `datasets/allium_cepa_full_images_merged_v3/{split}/data/annotations.json`. `attributes.division == 1` → mitosis.
- Classifier crops: `datasets/crops/binary_classifier/{split}/{mitosis,no_mitosis}/`
- YOLO: `datasets/yolo_dataset/{split}/{images,labels}/` + `data.yaml`
- HuggingFace: `GIAR-UTN/allium-cepa-dataset` (parquet shards)

## Key Design Decisions

- **No softmax in model output**: `build_model()` returns raw logits; softmax is applied only at inference time. This lets calibration operate directly on logits.
- **Vector scaling over scalar temperature**: Per-class temperature `[2]` allows asymmetric calibration of the two classes.
- **Isotonic regression for detector**: Makes no assumptions about calibration function shape; handles non-monotonic confidence distributions.
- **Experiment isolation**: `used_config.yaml` is snapshot at run start so exact config is always co-located with artifacts.
- **Delta Method for MI confidence intervals**: `compute_mi_with_ci()` (`statistics/error_propagation.py`) propagates per-detection Bernoulli uncertainty through both detector and classifier via the Delta Method, giving a closed-form 95.45% CI (MI ± 2σ) without bootstrap. The covariance term `cov(N_mit, N_cel) = var_mit` follows from the same detections contributing to both sums under the independence assumption.

## Tech Stack

- Python 3.12 (strict), `uv` + `hatchling`, CUDA cu123
- PyTorch ≥ 2.3, `timm` ≥ 1.0, Ultralytics ≥ 8.3
- Ruff (lint + format), line length 100, rules: E, W, F, I, B, C4, UP
