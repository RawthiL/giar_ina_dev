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

```bash
uv run python scripts/utils/download_hf_dataset.py
uv run python scripts/utils/coco_to_yolo.py
uv run python scripts/utils/classifier_dataset.py
uv run python scripts/utils/augment_crops.py --ratio 1.0
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
result.get_counts()      # {"total_cells": N, "mitotic_cells": M, "mitotic_index": float}
result.show_annotated()  # PIL image with bounding boxes
result.save_csv("out.csv")
```

Per-image: YOLO → crop cells → batch through classifier → softmax → `AlliumCepaResult`.

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

## Tech Stack

- Python 3.12 (strict), `uv` + `hatchling`, CUDA cu123
- PyTorch ≥ 2.3, `timm` ≥ 1.0, Ultralytics ≥ 8.3
- Ruff (lint + format), line length 100, rules: E, W, F, I, B, C4, UP
