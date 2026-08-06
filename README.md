# Allium Cepa Automation

Computer vision pipeline for automated detection and mitosis classification in *Allium cepa* (onion root tip) microscopy images. Built at UTN / INA.

Two-stage pipeline:
1. **Detection** — YOLO detects individual cells in full-FOV images, calibrated with isotonic regression.
2. **Classification** — EfficientNet/ResNet/VGG backbone classifies each crop as *mitosis* / *no mitosis*, calibrated with vector scaling.

---

## Quick Start (fresh clone)

```bash
# 1. Install dependencies
uv sync --all-groups
uv run pre-commit install

# 2. Pull inference weights (detector + classifier + isotonic calibrator)
dvc pull src/allium_cepa_classifier/weights/*.dvc --remote weights

# 3. (Optional) Reproduce the full training pipeline from scratch
dvc repro download_dataset coco_to_yolo prepare_crops
dvc repro
```

> **Note:** Only the three files in `src/allium_cepa_classifier/weights/` are on the `weights` remote.
> Datasets and experiment artifacts live in the `hf` remote and are regenerated via `dvc repro`.

---

## Inference

The easiest way to run inference is `notebooks/inference/full_pipeline.ipynb`, which walks through the full detection + classification pipeline on any image or directory.

For programmatic use:

```python
from allium_cepa_classifier import AlliumCepaModel, AlliumCepaConfig

model = AlliumCepaModel(AlliumCepaConfig())
result = model.predict("path/to/image.png")   # or a directory

result.get_counts()          # {"total_cells": N, "mitotic_cells": M, "mitotic_index": float}
result.get_counts_with_ci()  # same + ci_lower, ci_upper, sigma_mi (Delta Method)
result.show_annotated()      # PIL image with bounding boxes
result.save_csv("out.csv")
```

**Inference weights** are downloaded to `src/allium_cepa_classifier/weights/` via step 2 above and are the only weights used at inference time — the `experiments/` directory is not involved.

| File | Purpose |
|---|---|
| `object_detection.pt` | Trained YOLO detector |
| `classifier_calibrated.pt` | Calibrated EfficientNet classifier |
| `yolo_isotonic_calibrator.pkl` | Isotonic calibrator for detector confidence |

---

## Training

> **Note:** the experiment configs in `experiments/` are set to **3 epochs** — enough for a quick end-to-end pipeline check, not for production-quality models. Increase `epochs` in the relevant `config.yaml` before a real training run.

```bash
# Single classifier experiment (calibration runs automatically)
uv run python scripts/train_classifier.py --config experiments/binary_classifier/efficientnet_b1/config.yaml

# Sweep all classifier configs
uv run python scripts/sweep.py --configs experiments/binary_classifier/*/config.yaml

# YOLO detector
uv run python scripts/train_detector.py --config experiments/yolo/yolo11n_200e/config.yaml

# VAE (unsupervised representation learning; no calibration stage)
#   - prepare data first: dvc repro prepare_vae_dataset
uv run python scripts/train_vae.py --config experiments/vae/latent32_beta2/config.yaml

# ControlNet synthetic-data generator (standalone; not part of inference)
#   - prepare data first: dvc repro prepare_controlnet_dataset
#   - set training.max_train_steps in the config for a quick smoke run (leave unset for full training)
#   - watch live: tensorboard --logdir experiments/controlnet/sd15_baseline/logs (IMAGES tab = validation samples)
uv run python scripts/train_controlnet.py --config experiments/controlnet/sd15_baseline/config.yaml
uv run python scripts/generate_controlnet_samples.py --config experiments/controlnet/sd15_baseline/config.yaml

# Re-run calibration only
uv run python scripts/calibrate_classifier.py --experiment experiments/binary_classifier/efficientnet_b1/<run-dir>
uv run python scripts/calibrate_detector.py   --experiment experiments/yolo/yolo11n_200e/<run-dir>
```

---

## Dataset Version Management

The dataset is pinned to a specific HuggingFace commit in `dvc.yaml`. To bump it:

```bash
python -c "from huggingface_hub import repo_info; print(repo_info('GIAR-UTN/allium-cepa-dataset', repo_type='dataset').sha)"
# Edit dvc.yaml: change --rev <old-sha> → --rev <new-sha>
dvc repro download_dataset
git add dvc.yaml dvc.lock && git commit -m "chore: bump HF dataset to <new-sha>"
```

To push updated inference weights to HF after retraining:

```bash
uv run python scripts/utils/push_weights_to_hf.py
```

---

## Project Structure

```
├── experiments/            # Configs and results per experiment
│   ├── binary_classifier/  # efficientnet_b1, efficientnet_b2, resnet50, vgg19
│   └── yolo/yolo11n_200e/
├── scripts/
│   ├── train_classifier.py
│   ├── train_detector.py
│   ├── train_controlnet.py        # wraps the vendored diffusers trainer
│   ├── generate_controlnet_samples.py
│   ├── calibrate_classifier.py
│   ├── calibrate_detector.py
│   ├── sweep.py
│   ├── vendor/            # Vendored third-party code (ruff-excluded)
│   └── utils/              # Data prep, augmentation, HF upload
├── src/allium_cepa_classifier/
│   ├── config/             # Pydantic config models
│   ├── data_models/        # AlliumCepaModel, AlliumCepaResult
│   ├── statistics/         # Delta Method CI (error_propagation.py)
│   ├── training/           # Trainer, calibrators, model builder
│   └── weights/            # Inference weights (DVC-tracked, not in git)
├── datasets/               # DVC-managed, not in git
├── notebooks/
├── dvc.yaml                # Pipeline definition
└── pyproject.toml
```

---

## Tech Stack

Python 3.12 · PyTorch ≥ 2.3 · `timm` · Ultralytics YOLO · Diffusers + Accelerate + Transformers (ControlNet, SD1.5) · TensorBoard · DVC · `uv` · Ruff