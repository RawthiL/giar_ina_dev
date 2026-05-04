# Notebook-to-Script Migration: Training & Calibration

## Overview

Migrate classifier training and calibration from Jupyter notebooks to CLI scripts
driven by per-experiment YAML configs. Adds multi-architecture sweep support
(EfficientNet B1/B2, ResNet50, VGG19), automatic per-run artifact output, and a
comparison notebook for cross-run analysis.

---

## Current State Analysis

- `notebooks/training/classifier.ipynb` and `calibrate_model.ipynb` are already
  pure-PyTorch (TF migration done). They can be read as a reference for exact
  training logic — no algorithmic changes are needed.
- `src/allium_cepa_classifier/config/base_config.py` already has `BaseConfig.from_yaml()`
  via pydantic. The new `ExperimentConfig` will extend this.
- `src/allium_cepa_classifier/config/training_config.py` exists but conflates
  filesystem paths with data hyperparameters. The new `DataConfig` absorbs both,
  keeping paths as optional overrides (defaults auto-detected via `find_project_root()`).
- `scripts/` only contains `run_inference.py`. Training scripts do not exist yet.
- No `configs/` directory or YAML experiment files exist.
- `notebooks/analysis/eda.ipynb` exists; a new `compare_experiments.ipynb` will
  be added alongside it.

---

## Desired End State

After this plan is complete:

- `scripts/train.py --config configs/experiments/efficientnet_b1.yaml` runs a full
  training job and writes all artifacts to `experiments/<run_id>/`.
- `scripts/calibrate.py --experiment experiments/<run_id>/` calibrates a trained
  model and appends calibration artifacts to the same run dir.
- `scripts/sweep.py --configs configs/experiments/*.yaml` trains + calibrates all
  architectures sequentially.
- 4 YAML files in `configs/experiments/` define the four architectures to compare.
- `notebooks/analysis/compare_experiments.ipynb` loads `metrics.json` from
  multiple run dirs and renders side-by-side comparison plots.
- Old training notebooks are moved to `notebooks/legacy/`.
- `experiments/` is gitignored except for `*/used_config.yaml` and `*/metrics.json`.

### Verification

```bash
uv run python -m pytest tests/ -q                    # no regressions
uv run python scripts/train.py --config configs/experiments/efficientnet_b1.yaml --dry-run
ls experiments/  # at least one run dir after a real run
```

---

## What We're NOT Doing

- Changing `AlliumCepaModel` / inference pipeline (separate concern).
- Adding MLflow, W&B, or any experiment tracking framework dependency.
- Writing unit tests for training logic (non-trivial, out of scope for this migration).
- Migrating detection or VAE training notebooks.
- Changing calibration algorithm — the same scipy L-BFGS-B vector scaling is kept.

---

## Experiment Directory Layout

```
experiments/
  20260502-153000_efficientnet_b1_baseline/
    used_config.yaml          ← verbatim copy of the input YAML
    weights/
      classifier.pt           ← best checkpoint (val_loss)
      classifier_calibrated.pt   ← added by calibrate.py
    metrics.json              ← all numerics (train/val/test acc, ECE, ...)
    plots/
      training_curves.png
      confusion_matrix.png
      classification_report.txt
      reliability_diagram.png    ← added by calibrate.py
    train.log
```

---

## Phase 1: ExperimentConfig + YAML files

### Overview

Create the pydantic `ExperimentConfig` with nested sub-configs that rolls together
the current `TrainingConfig` filesystem paths and all training hyperparameters.
Write 4 experiment YAML files.

### 1.1 New file: `src/allium_cepa_classifier/config/experiment_config.py`

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()


class HeadConfig(BaseModel):
    hidden_dims: list[int] = [512, 256, 128]
    dropouts: list[float] = [0.3, 0.2, 0.0]
    activation: Literal["leaky_relu", "relu", "gelu"] = "leaky_relu"


class ModelConfig(BaseModel):
    arch: Literal["efficientnet_b1", "efficientnet_b2", "resnet50", "vgg19"] = "efficientnet_b1"
    pretrained: bool = True
    freeze_stages: int = 3  # arch-specific; see ARCH_FREEZE_STAGES in model_builder.py
    head: HeadConfig = HeadConfig()


class LRSchedulerConfig(BaseModel):
    factor: float = 0.2
    patience: int = 5
    min_lr: float = 1e-6


class TrainingHPConfig(BaseModel):
    epochs: int = 30
    lr: float = 1e-5
    early_stopping_patience: int = 10
    class_weight_multipliers: dict[str, float] = {"mitosis": 2.0, "no_mitosis": 0.5}
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    augmentation: list[str] = ["hflip", "vflip", "color_jitter"]


class DataConfig(BaseModel):
    image_size: tuple[int, int] = (260, 260)
    batch_size: int = 32
    seed: int = 42
    # Path overrides — omit in YAML to use project-root-relative defaults
    binary_classifier_crops_dir: Path = _ROOT / "datasets/crops/binary_classifier"
    experiments_dir: Path = _ROOT / "experiments"


class ExperimentConfig(BaseConfig):
    experiment_name: str
    model: ModelConfig = ModelConfig()
    training: TrainingHPConfig = TrainingHPConfig()
    data: DataConfig = DataConfig()
```

### 1.2 Update `src/allium_cepa_classifier/config/__init__.py`

Add `ExperimentConfig` to exports:

```python
from .allium_cepa_config import AlliumCepaConfig
from .experiment_config import ExperimentConfig
from .training_config import TrainingConfig

__all__ = ["AlliumCepaConfig", "ExperimentConfig", "TrainingConfig"]
```

### 1.3 Create `configs/experiments/` with 4 YAML files

**`configs/experiments/efficientnet_b1.yaml`**
```yaml
experiment_name: efficientnet_b1_baseline

model:
  arch: efficientnet_b1
  pretrained: true
  freeze_stages: 2
  head:
    hidden_dims: [512, 256, 128]
    dropouts: [0.3, 0.2, 0.0]
    activation: leaky_relu

training:
  epochs: 20
  lr: 1.0e-5
  early_stopping_patience: 10
  class_weight_multipliers:
    mitosis: 2.0
    no_mitosis: 0.5
  lr_scheduler:
    factor: 0.2
    patience: 5
    min_lr: 1.0e-6
  augmentation: [hflip, vflip, color_jitter]

data:
  image_size: [260, 260]
  batch_size: 32
  seed: 42
```

**`configs/experiments/efficientnet_b2.yaml`** — same structure, `arch: efficientnet_b2`.

**`configs/experiments/resnet50.yaml`**
```yaml
experiment_name: resnet50_baseline

model:
  arch: resnet50
  pretrained: true
  freeze_stages: 2   # freezes conv1+bn1+layer1+layer2 (see ARCH_FREEZE_STAGES)
  head:
    hidden_dims: [512, 256, 128]
    dropouts: [0.3, 0.2, 0.0]
    activation: leaky_relu

training:
  epochs: 20
  lr: 1.0e-5
  early_stopping_patience: 10
  class_weight_multipliers:
    mitosis: 2.0
    no_mitosis: 0.5
  lr_scheduler:
    factor: 0.2
    patience: 5
    min_lr: 1.0e-6
  augmentation: [hflip, vflip, color_jitter]

data:
  image_size: [260, 260]
  batch_size: 32
  seed: 42
```

**`configs/experiments/vgg19.yaml`** — same as resnet50 but `arch: vgg19`,
`freeze_stages: 3` (freezes VGG feature blocks 0–2).

### Success Criteria — Phase 1

#### Automated
- [x] `uv run python -c "from allium_cepa_classifier.config import ExperimentConfig; ExperimentConfig.from_yaml('configs/experiments/efficientnet_b1.yaml')"` — no errors
- [x] All 4 YAML files parse without validation errors
- [x] Ruff passes: `uv run ruff check src/allium_cepa_classifier/config/experiment_config.py`

---

## Phase 2: Model Builder

### Overview

Create `src/allium_cepa_classifier/training/model_builder.py` that constructs
a `BackboneWithHead` for any of the 4 supported architectures and applies the
stage-based freeze logic.

The `BackboneWithHead` wrapper sidesteps the timm attribute inconsistency
(`model.classifier` vs `model.fc` vs `model.head`) by using `num_classes=0`
to get a pure feature extractor, then attaching the custom head.

### 2.1 New file: `src/allium_cepa_classifier/training/__init__.py`

Empty init to make `training` a proper subpackage.

### 2.2 New file: `src/allium_cepa_classifier/training/model_builder.py`

```python
"""
Builds BackboneWithHead for any supported architecture from an ExperimentConfig.

freeze_stages mapping (keeps last n stages trainable, freezes the rest):
  efficientnet_b1/b2  8 stages: [stem, blocks[0], blocks[1], ..., blocks[6]]
  resnet50            5 stages: [conv1+bn1, layer1, layer2, layer3, layer4]
  vgg19               5 stages: [features[0:5], features[5:10], features[10:19],
                                 features[19:28], features[28:37]]
"""
import torch.nn as nn
import timm

from allium_cepa_classifier.config.experiment_config import ModelConfig, HeadConfig


class BackboneWithHead(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def _build_head(cfg: HeadConfig, in_features: int, num_classes: int) -> nn.Sequential:
    ACT = {"leaky_relu": nn.LeakyReLU(0.2), "relu": nn.ReLU(), "gelu": nn.GELU()}
    layers = []
    dims = [in_features] + cfg.hidden_dims + [num_classes]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(type(ACT[cfg.activation])(**vars(ACT[cfg.activation])
                          if hasattr(ACT[cfg.activation], '__dict__') else {}))
            if i < len(cfg.dropouts) and cfg.dropouts[i] > 0:
                layers.append(nn.Dropout(cfg.dropouts[i]))
    return nn.Sequential(*layers)
```

> **Implementation note on `_build_head`:** The activation instantiation above
> is illustrative — simplify to a plain if/elif for the three activation types
> in the actual code. The key contract is: `hidden_dims[-1]` feeds into a final
> `Linear(..., num_classes)` with no activation or dropout.

**`freeze_stages` implementation:**

```python
def freeze_model_stages(model: BackboneWithHead, arch: str, n: int) -> None:
    """Freeze the first n stage groups of the backbone."""
    if n == 0:
        return
    backbone = model.backbone

    if arch.startswith("efficientnet"):
        # Stage 0: stem; stages 1-4: block groups
        stages = [
            [backbone.conv_stem, backbone.bn1],
            [backbone.blocks[0], backbone.blocks[1]],
            [backbone.blocks[2], backbone.blocks[3]],
            [backbone.blocks[4]],
            [backbone.blocks[5]],
        ]
    elif arch == "resnet50":
        stages = [
            [backbone.conv1, backbone.bn1],
            [backbone.layer1],
            [backbone.layer2],
            [backbone.layer3],
            [backbone.layer4],
        ]
    elif arch == "vgg19":
        # VGG features is a flat Sequential; freeze by conv-block boundaries
        boundaries = [5, 10, 19, 28, 37]
        stages = [list(backbone.features[boundaries[i-1] if i > 0 else 0 : boundaries[i]])
                  for i in range(len(boundaries))]
    else:
        raise ValueError(f"Unsupported arch for freezing: {arch}")

    for group in stages[:n]:
        for module in group:
            for param in module.parameters():
                param.requires_grad = False


def build_model(cfg: ModelConfig, num_classes: int = 2) -> BackboneWithHead:
    backbone = timm.create_model(cfg.arch, pretrained=cfg.pretrained, num_classes=0)
    in_features = backbone.num_features
    head = _build_head(cfg.head, in_features, num_classes)
    model = BackboneWithHead(backbone, head)
    freeze_model_stages(model, cfg.arch, cfg.freeze_stages)
    return model
```

### Success Criteria — Phase 2

#### Automated
- [x] `build_model` constructs without error for all 4 arches:
  ```bash
  uv run python -c "
  from allium_cepa_classifier.training.model_builder import build_model
  from allium_cepa_classifier.config.experiment_config import ModelConfig
  for arch in ['efficientnet_b1', 'efficientnet_b2', 'resnet50', 'vgg19']:
      m = build_model(ModelConfig(arch=arch))
      print(arch, sum(p.numel() for p in m.parameters() if p.requires_grad))
  "
  ```
- [x] Ruff passes on `model_builder.py`

#### Manual
- [x] Trainable parameter counts look reasonable for each arch (EfficientNets < 10M, ResNet50 ~25M total, VGG19 ~143M total — only the unfrozen layers should be much smaller)

---

## Phase 3: Training Script

### Overview

`scripts/train.py` reads an ExperimentConfig YAML, runs the training loop, evaluates
on the test split, and writes all artifacts to a timestamped experiment directory.

The training loop logic is extracted directly from `notebooks/training/classifier.ipynb`
(no algorithmic changes).

### 3.1 New file: `src/allium_cepa_classifier/training/trainer.py`

Contains the training loop, evaluation, and artifact-writing functions.
Key public functions:

```python
def run_training(cfg: ExperimentConfig, run_dir: Path) -> dict:
    """Full train + test evaluation. Returns metrics dict."""
    ...

def save_artifacts(history, y_true, y_pred, class_names, run_dir: Path, metrics: dict) -> None:
    """Writes training_curves.png, confusion_matrix.png, classification_report.txt, metrics.json."""
    ...
```

**Output files written by `trainer.py`:**
- `run_dir/weights/classifier.pt` — best checkpoint (same format as current notebook)
- `run_dir/metrics.json` — `{train_acc, val_acc, test_acc, best_val_loss, epochs_run}`
- `run_dir/plots/training_curves.png`
- `run_dir/plots/confusion_matrix.png`
- `run_dir/plots/classification_report.txt`
- `run_dir/train.log` — redirect stdout via `tee` in the CLI entry point

**Checkpoint format** (same as current notebook, adds `timm_model_name`):
```python
torch.save({
    "model_state_dict": best_state,
    "timm_model_name": cfg.model.arch,
    "num_classes": 2,
    "image_size": cfg.data.image_size,
    "class_to_idx": train_dataset.class_to_idx,
    "imagenet_mean": IMAGENET_MEAN,
    "imagenet_std": IMAGENET_STD,
}, run_dir / "weights" / "classifier.pt")
```

### 3.2 New file: `scripts/train.py`

```python
"""
Usage:
    uv run python scripts/train.py --config configs/experiments/efficientnet_b1.yaml
    uv run python scripts/train.py --config configs/experiments/efficientnet_b1.yaml --dry-run
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

from allium_cepa_classifier.config.experiment_config import ExperimentConfig
from allium_cepa_classifier.training.trainer import run_training, save_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse config and build model only, do not train")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.data.experiments_dir / f"{timestamp}_{cfg.experiment_name}"
    run_dir.mkdir(parents=True)
    (run_dir / "weights").mkdir()
    (run_dir / "plots").mkdir()

    # Save config snapshot immediately so the run is always reproducible
    shutil.copy(args.config, run_dir / "used_config.yaml")

    if args.dry_run:
        print(f"Dry run OK. Run dir would be: {run_dir}")
        return

    metrics = run_training(cfg, run_dir)
    print(f"\nDone. Artifacts in: {run_dir}")
    print(f"Test accuracy: {metrics['test_acc']:.4f}")


if __name__ == "__main__":
    main()
```

### Success Criteria — Phase 3

#### Automated
- [x] Dry run completes: `uv run python scripts/train.py --config configs/experiments/efficientnet_b1.yaml --dry-run`
- [x] `run_dir/used_config.yaml` is written on dry run
- [x] Ruff passes on `trainer.py` and `scripts/train.py`

#### Manual
- [ ] Full training run completes on `efficientnet_b1.yaml`, test accuracy comparable to notebook run (≥ 88%)
- [ ] All 5 artifact files appear in the run dir
- [ ] Training curves look healthy (no divergence)

**Pause for manual confirmation that a full training run completed and artifacts look correct before proceeding.**

---

## Phase 4: Calibration Script

### Overview

`scripts/calibrate.py` takes an existing experiment run dir, loads the trained
`classifier.pt`, runs the same scipy L-BFGS-B vector scaling from the current
`calibrate_model.ipynb`, and appends calibration artifacts to the run dir.

### 4.1 New file: `src/allium_cepa_classifier/training/calibrator.py`

Key public function:

```python
def run_calibration(run_dir: Path) -> dict:
    """
    Loads run_dir/weights/classifier.pt and run_dir/used_config.yaml.
    Writes classifier_calibrated.pt and reliability_diagram.png.
    Updates metrics.json with ece_before and ece_after.
    Returns calibration metrics.
    """
    ...
```

**Logic** (from `calibrate_model.ipynb`, no changes):
1. Load checkpoint + rebuild model architecture from `used_config.yaml`
2. Extract validation logits via DataLoader
3. `scipy.optimize.minimize` L-BFGS-B on vector scaling loss
4. Build `CalibratedClassifier` wrapper (same as notebook)
5. Compute ECE before/after
6. Save `run_dir/weights/classifier_calibrated.pt`
7. Save `run_dir/plots/reliability_diagram.png`
8. Update `run_dir/metrics.json` with `{ece_before, ece_after, temperature}`

**Loading the model inside calibrator** — use `ExperimentConfig.from_yaml(run_dir / "used_config.yaml")`
and `build_model(cfg.model)` + `load_state_dict` to rebuild. This avoids duplicating
the architecture definition.

### 4.2 New file: `scripts/calibrate.py`

```python
"""
Usage:
    uv run python scripts/calibrate.py --experiment experiments/20260502-153000_efficientnet_b1_baseline
"""
import argparse
from pathlib import Path

from allium_cepa_classifier.training.calibrator import run_calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path,
                        help="Path to a completed experiment run directory")
    args = parser.parse_args()

    metrics = run_calibration(args.experiment)
    print(f"ECE before: {metrics['ece_before']:.4f}  after: {metrics['ece_after']:.4f}")
    print(f"Temperature: {metrics['temperature']}")


if __name__ == "__main__":
    main()
```

### Success Criteria — Phase 4

#### Automated
- [x] `uv run python scripts/calibrate.py --experiment <run_dir>` completes without error
- [x] `run_dir/weights/classifier_calibrated.pt` exists and loads cleanly
- [x] `run_dir/metrics.json` contains `ece_before` and `ece_after` keys

#### Manual
- [ ] `ece_after < ece_before` on the validation set
- [ ] Reliability diagram shows calibrated curve closer to the diagonal

---

## Phase 5: Sweep Runner

### Overview

`scripts/sweep.py` runs `train.py` + `calibrate.py` sequentially for each of the
4 experiment YAMLs, catching and logging errors per run without aborting the sweep.

### New file: `scripts/sweep.py`

```python
"""
Usage:
    uv run python scripts/sweep.py --configs configs/experiments/*.yaml
    uv run python scripts/sweep.py --configs configs/experiments/efficientnet_b1.yaml configs/experiments/resnet50.yaml
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for cfg_path in args.configs:
        print(f"\n{'='*60}")
        print(f"Starting: {cfg_path.stem}")
        print(f"{'='*60}")

        # Train
        train_result = subprocess.run(
            [sys.executable, "scripts/train.py", "--config", str(cfg_path)],
            capture_output=False,
        )
        if train_result.returncode != 0:
            results.append((cfg_path.stem, "TRAIN FAILED"))
            continue

        # Find the run dir just created (most recent in experiments/)
        from allium_cepa_classifier.config.experiment_config import ExperimentConfig
        cfg = ExperimentConfig.from_yaml(cfg_path)
        exp_dirs = sorted(cfg.data.experiments_dir.glob(f"*_{cfg.experiment_name}"))
        if not exp_dirs:
            results.append((cfg_path.stem, "CALIBRATE SKIPPED (no run dir found)"))
            continue

        run_dir = exp_dirs[-1]

        # Calibrate
        cal_result = subprocess.run(
            [sys.executable, "scripts/calibrate.py", "--experiment", str(run_dir)],
            capture_output=False,
        )
        status = "OK" if cal_result.returncode == 0 else "CALIBRATE FAILED"
        results.append((cfg_path.stem, status))

    print("\n\nSweep summary:")
    for name, status in results:
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
```

### Success Criteria — Phase 5

#### Automated
- [x] `uv run python scripts/sweep.py --configs configs/experiments/efficientnet_b1.yaml configs/experiments/efficientnet_b2.yaml` runs both experiments end-to-end

#### Manual
- [ ] Both experiment dirs appear in `experiments/` with all expected artifacts
- [ ] Sweep summary prints "OK" for both

---

## Phase 6: Analysis Notebook + Archive

### Overview

Create `notebooks/analysis/compare_experiments.ipynb` for cross-run comparison.
Archive the two training notebooks to `notebooks/legacy/`.

### 6.1 New notebook: `notebooks/analysis/compare_experiments.ipynb`

Cells to implement:

1. **Imports + discovery** — glob `experiments/*/metrics.json`, load all into a list of dicts
2. **Accuracy table** — pandas DataFrame with `experiment_name`, `test_acc`, `val_acc`, `ece_before`, `ece_after`
3. **Bar chart** — test accuracy + ECE side-by-side for all runs
4. **Confusion matrix grid** — load `plots/confusion_matrix.png` from each run, display in a grid
5. **Training curves overlay** — reload `metrics.json` history fields per run (add `history` key to `metrics.json` in Phase 3)

> **Note:** Add a `history` key to `metrics.json` in `trainer.py` containing per-epoch
> `{train_loss, val_loss, train_acc, val_acc}` lists. This is needed for curve overlay.

### 6.2 Archive training notebooks

```bash
mkdir -p notebooks/legacy
git mv notebooks/training/classifier.ipynb notebooks/legacy/classifier.ipynb
git mv notebooks/training/calibrate_model.ipynb notebooks/legacy/calibrate_model.ipynb
```

### Success Criteria — Phase 6

#### Automated
- [x] `notebooks/legacy/classifier.ipynb` and `calibrate_model.ipynb` exist
- [x] `notebooks/training/` no longer contains those files

#### Manual
- [ ] `compare_experiments.ipynb` runs cell-by-cell and renders comparison bar chart
  after at least 2 experiment runs exist in `experiments/`

---

## Phase 7: Repository Hygiene

### 7.1 Update `.gitignore`

```gitignore
# Experiment outputs — keep only textual artifacts for tracking
experiments/
!experiments/
!experiments/*/
!experiments/*/used_config.yaml
!experiments/*/metrics.json
```

### 7.2 Ensure `training` subpackage is included in the wheel

**File:** `pyproject.toml`

The `[tool.hatch.build.targets.wheel]` entry already has
`packages = ["src/allium_cepa_classifier", "src/ui"]` — hatchling includes
subpackages automatically, so no change is needed as long as the new
`training/` directory has an `__init__.py`.

### Success Criteria — Phase 7

#### Automated
- [x] `git status` shows `experiments/` untracked but `experiments/*/metrics.json` tracked after a run
- [x] `uv build` succeeds and `training` subpackage appears in the wheel

---

## Testing Strategy

### Per-run automated verification (in `trainer.py`)
- Assert `class_to_idx == {"mitosis": 0, "no_mitosis": 1}` before training starts
- Assert saved checkpoint loads and has expected keys
- Assert `metrics.json` written with all required keys

### Manual testing — full sweep
1. Run the sweep over all 4 architectures
2. Open `compare_experiments.ipynb` — confirm comparison table renders
3. Pick the best model by `test_acc`; run `calibrate.py` on it if not already done
4. Confirm `classifier_calibrated.pt` can be loaded by `AlliumCepaModel` via
   `AlliumCepaConfig(classification_weights_path=<run_dir>/weights/classifier_calibrated.pt)`

---

## Implementation Order

Phases 1 → 2 → 3 → 4 → 5 → 6 → 7. Each phase has a clear verification gate
before the next begins. Phases 3 and 4 are the bulk of the work; Phases 1–2
are prerequisites with no training compute required.

---

## References

- Current training notebook: `notebooks/training/classifier.ipynb`
- Current calibration notebook: `notebooks/training/calibrate_model.ipynb`
- Existing config base: `src/allium_cepa_classifier/config/base_config.py`
- Existing pydantic configs: `src/allium_cepa_classifier/config/training_config.py`
- PyTorch migration plan: `thoughts/shared/plans/2026-04-18-tensorflow-to-pytorch-migration.md`
