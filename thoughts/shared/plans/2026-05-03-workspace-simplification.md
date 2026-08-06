# Workspace Simplification Plan

## Overview

Remove dead code, delete stale artifacts, convert two notebooks to scripts, relocate the `utils/` CLI scripts under `scripts/utils/`, slim down config classes, and fix a latent `.gitignore` inconsistency — without squishing anything together or breaking the inference/training pipeline.

## Current State Analysis

- `allium_cepa_model.py` at repo root: old TF/Keras version, untracked, replaced by `src/allium_cepa_classifier/models/allium_cepa_model.py`
- `notebooks/legacy/`: two notebooks archived after the notebook→script migration; no longer needed
- `src/ui/`: Streamlit app and CLI launcher; deleted per user decision
- `dist/`, `runs/`: local build/TF artifacts, gitignored but present on disk
- `scripts/run_inference.py`, `utils/format_roboflow.py`: hardcoded paths, no callers
- `notebooks/datasets/unify_datasets.ipynb`: one-shot, delete
- `notebooks/datasets/data_augmentation.ipynb`: broken (references non-existent `config.mitosis_crops_dir`, missing `cv2`/`scipy` deps) → fix and migrate to `scripts/utils/augment_crops.py`
- `notebooks/training/detection_model.ipynb`: thin YOLO train+val+save → migrate to `scripts/train_detector.py`
- `src/allium_cepa_classifier/utils/*.py`: four CLI scripts that live in `src/` but are never imported — logically belong in `scripts/utils/`
- `TrainingConfig`: has 8 unused fields (`image_height/width/size`, `confidence_threshold`, `batch_size`, `use_cpu`, `seed`, `model_path`) — only path fields + `splits` are ever read by callers
- `AlliumCepaConfig`: `confidence_threshold`, `image_height`, `image_width` are dead with UI deleted
- `src/allium_cepa_classifier/models/weights/`: weights stored inside the Python package; `models/` dir is gitignored globally which silently prevents new source files in that package from being tracked — weights should move to `src/allium_cepa_classifier/weights/`

## Desired End State

```
allium_cepa_automation/
├── configs/experiments/         # unchanged
├── experiments/                 # unchanged
├── notebooks/
│   ├── analysis/                # compare_experiments.ipynb, eda.ipynb
│   ├── datasets/                # (empty dir removed if nothing left)
│   ├── inference/               # full_pipeline.ipynb
│   └── training/                # detection_model gone (→ script), vae notebooks kept
├── scripts/
│   ├── calibrate.py
│   ├── sweep.py
│   ├── train.py
│   ├── train_detector.py        # NEW: replaces detection_model.ipynb
│   └── utils/
│       ├── augment_crops.py     # NEW: replaces data_augmentation.ipynb
│       ├── classifier_dataset.py
│       ├── coco_to_yolo.py
│       ├── download_hf_dataset.py
│       └── vae_dataset.py
├── src/allium_cepa_classifier/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── allium_cepa_config.py   # -confidence_threshold -image_height -image_width
│   │   ├── base_config.py
│   │   ├── experiment_config.py
│   │   └── training_config.py      # only path fields + splits remain
│   ├── data_models/             # unchanged
│   ├── models/                  # unchanged (allium_cepa_model.py stays here)
│   ├── training/                # unchanged
│   ├── weights/                 # NEW location for model weights (gitignored)
│   └── __init__.py
├── thoughts/
├── pyproject.toml               # remove src/ui, remove streamlit dep group
└── .gitignore                   # add src/allium_cepa_classifier/weights/, remove/fix models/ rule
```

## What We're NOT Doing

- Flattening `data_models/` or `models/` single-file subpackages (G: keep structure)
- Merging `TrainingConfig` into `ExperimentConfig` (different concerns)
- Touching VAE notebooks (kept as-is)
- Writing a README (separate task)
- Fixing the WIP `AlliumCepaModel._load_classification_model` architecture mismatch (separate bug)

---

## Phase 1: Delete dead files and artifacts

### Changes Required:

Delete the following (no code changes elsewhere needed):

- `allium_cepa_model.py` (root, TF version, untracked)
- `notebooks/legacy/calibrate_model.ipynb`
- `notebooks/legacy/classifier.ipynb`
- `notebooks/legacy/` directory
- `notebooks/datasets/unify_datasets.ipynb`
- `scripts/run_inference.py`
- `src/allium_cepa_classifier/utils/format_roboflow.py`
- `todo.txt`

Delete local-only artifacts (gitignored, no git rm needed):
- `dist/` directory
- `runs/` directory

### Success Criteria:

#### Automated Verification:
- [x] `git status` shows none of the deleted files as tracked
- [x] `uv run python -c "from allium_cepa_classifier import AlliumCepaModel, AlliumCepaConfig, AlliumCepaResult"` exits 0
- [x] `uv run ruff check src/ scripts/` exits 0 (pre-existing errors in `src/ui/app.py` resolved in Phase 2)

#### Manual Verification:
- [x] Confirm none of the deleted files are referenced in any tracked notebook or script

---

## Phase 2: Remove `src/ui/`

### Changes Required:

#### 1. Delete files
- `src/ui/__init__.py`
- `src/ui/app.py`
- `src/ui/cli.py`
- `src/ui/` directory

#### 2. `pyproject.toml`
Remove `"src/ui"` from wheel packages and drop the `ui` dependency group:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/allium_cepa_classifier"]

# Remove entirely:
# [dependency-groups]
# ui = [
#     "streamlit>=1.35.0",
# ]
```

Keep the `dev` dependency group as-is.

### Success Criteria:

#### Automated Verification:
- [x] `uv run python -c "from allium_cepa_classifier import AlliumCepaModel"` exits 0
- [x] `uv run ruff check src/ scripts/` exits 0
- [x] `uv sync --all-groups` succeeds

---

## Phase 3: Convert notebooks to scripts

### Changes Required:

#### 1. `scripts/train_detector.py` (new file, replaces `notebooks/training/detection_model.ipynb`)

```python
"""
Usage:
    uv run python scripts/train_detector.py
    uv run python scripts/train_detector.py --epochs 100 --device cpu
    uv run python scripts/train_detector.py --data datasets/yolo_dataset/data.yaml --weights src/allium_cepa_classifier/weights/yolo11n.pt --out src/allium_cepa_classifier/weights/object_detection.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO cell detector.")
    parser.add_argument("--weights", type=Path,
                        default=Path("src/allium_cepa_classifier/weights/yolo11n.pt"),
                        help="Pretrained YOLO weights to start from")
    parser.add_argument("--data", type=Path,
                        default=Path("datasets/yolo_dataset/data.yaml"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path,
                        default=Path("src/allium_cepa_classifier/weights/object_detection.pt"),
                        help="Where to save the trained weights")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        name="train",
    )
    model.val(split="val", imgsz=args.imgsz, batch=16, conf=0.001, iou=0.7, plots=True)
    model.save(str(args.out))
    print(f"Saved detector weights to: {args.out}")


if __name__ == "__main__":
    main()
```

Delete `notebooks/training/detection_model.ipynb`.

#### 2. `scripts/utils/augment_crops.py` (new file, replaces `notebooks/datasets/data_augmentation.ipynb`)

Fix the broken attribute reference (`mitosis_crops_dir` → `binary_classifier_crops_dir / "train" / "mitosis"`) and add missing deps note:

```python
"""
Augments mitosis crops in-place (train split only).

Usage:
    uv run python scripts/utils/augment_crops.py
    uv run python scripts/utils/augment_crops.py --config path/to/config.yaml --ratio 1.0

Requires: opencv-python-headless (already in deps), scipy (add to dev deps if missing).
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import gaussian_filter, map_coordinates

from allium_cepa_classifier.config.training_config import TrainingConfig

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def add_noise(image: Image.Image) -> Image.Image:
    arr = np.array(image)
    noise = np.random.normal(10, 25, arr.shape).astype(np.int32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def elastic_transform(image: Image.Image) -> Image.Image:
    arr = np.array(image)
    alpha = arr.shape[1] * 2
    sigma = arr.shape[1] * 0.08
    alpha_affine = arr.shape[1] * 0.08
    shape = arr.shape
    rng = np.random.RandomState(None)

    center = np.float32(shape[:2]) // 2
    sq = min(shape[:2]) // 3
    pts1 = np.float32([center + sq, [center[0] + sq, center[1] - sq], center - sq])
    pts2 = pts1 + rng.uniform(-alpha_affine, alpha_affine, pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    arr = cv2.warpAffine(arr, M, shape[:2][::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma) * alpha
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )
    return Image.fromarray(
        map_coordinates(arr, indices, order=1, mode="reflect").reshape(shape).astype(np.uint8)
    )


def augment(image: Image.Image) -> Image.Image:
    img = ImageOps.mirror(image)
    img = ImageOps.flip(img)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))
    if random.random() < 0.5:
        img = add_noise(img)
    return elastic_transform(img)


def augment_dir(src_dir: Path, ratio: float) -> int:
    originals = [p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    sample = random.sample(originals, int(len(originals) * ratio))
    for p in sample:
        aug = augment(Image.open(p))
        aug.save(src_dir / f"{p.stem}_aug{p.suffix}")
    return len(sample)


def main():
    parser = argparse.ArgumentParser(description="Augment mitosis crops in-place (train split).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Fraction of originals to augment (default: 1.0)")
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config, key="training") if args.config else TrainingConfig()
    src_dir = cfg.binary_classifier_crops_dir / "train" / "mitosis"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    print(f"Augmenting: {src_dir}")
    n = augment_dir(src_dir, args.ratio)
    print(f"Generated {n} augmented images.")


if __name__ == "__main__":
    main()
```

Also add `scipy>=1.13` to the `dev` dependency group in `pyproject.toml` (it's used here and not currently listed).

Delete `notebooks/datasets/data_augmentation.ipynb`.
If `notebooks/datasets/` is now empty, delete the directory.

### Success Criteria:

#### Automated Verification:
- [x] `uv run python scripts/train_detector.py --help` exits 0
- [x] `uv run python scripts/utils/augment_crops.py --help` exits 0
- [x] `uv run ruff check scripts/` exits 0

#### Manual Verification:
- [x] `scripts/train_detector.py` arguments match what the original notebook used

---

## Phase 4: Move `utils/` scripts to `scripts/utils/`

### Changes Required:

Move (using `git mv`):
- `src/allium_cepa_classifier/utils/coco_to_yolo.py` → `scripts/utils/coco_to_yolo.py`
- `src/allium_cepa_classifier/utils/classifier_dataset.py` → `scripts/utils/classifier_dataset.py`
- `src/allium_cepa_classifier/utils/download_hf_dataset.py` → `scripts/utils/download_hf_dataset.py`
- `src/allium_cepa_classifier/utils/vae_dataset.py` → `scripts/utils/vae_dataset.py`

Delete:
- `src/allium_cepa_classifier/utils/__init__.py`
- `src/allium_cepa_classifier/utils/` directory

Update module-level docstrings in the moved files to reflect new invocation:
- `python -m allium_cepa_classifier.utils.xxx` → `python scripts/utils/xxx.py`

Update `src/allium_cepa_classifier/config/__init__.py` if it re-exports anything from utils (it does not — confirm with grep).

No other Python files import from `utils/`, so no import updates needed.

### Success Criteria:

#### Automated Verification:
- [x] `uv run python scripts/utils/coco_to_yolo.py --help` exits 0
- [x] `uv run python scripts/utils/classifier_dataset.py --help` exits 0
- [x] `uv run python scripts/utils/download_hf_dataset.py --help` exits 0
- [x] `uv run python scripts/utils/vae_dataset.py --help` exits 0
- [x] `uv run ruff check src/ scripts/` exits 0
- [x] `uv run python -c "from allium_cepa_classifier import AlliumCepaModel"` exits 0

---

## Phase 5: Slim down config classes

### Changes Required:

#### 1. `src/allium_cepa_classifier/config/training_config.py`

Remove unused fields. Keep only what the dataset-prep scripts actually use:

```python
from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict

from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()


class TrainingConfig(BaseConfig):
    """Filesystem paths for dataset preparation scripts."""

    model_config = ConfigDict(frozen=True)

    raw_dataset_dir: Path = _ROOT / "datasets/allium_cepa_full_images_merged"
    yolo_dataset_dir: Path = _ROOT / "datasets/yolo_dataset"
    crops_dir: Path = _ROOT / "datasets/crops"
    binary_classifier_crops_dir: Path = crops_dir / "binary_classifier"
    vae_crops_dir: Path = crops_dir / "vae"
    splits: list[str] = ["train", "validation", "test"]
```

Removed: `image_height`, `image_width`, `image_size`, `confidence_threshold`, `batch_size`, `use_cpu`, `seed`, `model_path`.

#### 2. `src/allium_cepa_classifier/config/allium_cepa_config.py`

Remove `confidence_threshold`, `image_height`, `image_width` (only `image_size` needed as a tuple):

```python
from pathlib import Path

from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()


class AlliumCepaConfig(BaseConfig):
    """Configuration for AlliumCepaModel inference."""

    detection_weights_path: Path = (
        _ROOT / "src/allium_cepa_classifier/weights/object_detection.pt"
    )
    classification_weights_path: Path = (
        _ROOT / "src/allium_cepa_classifier/weights/classifier_efficientNetB1_20E.pt"
    )
    valid_image_extensions: list = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    image_size: tuple[int, int] = (200, 200)
    batch_size: int = 32
    use_cpu: bool = False
```

Note: `image_size` default changes from `(image_height, image_width)` → `(200, 200)` directly. The inference notebook may pass a config without this field and rely on the checkpoint value anyway (`ckpt.get("image_size", self.config.image_size)`).

### Success Criteria:

#### Automated Verification:
- [x] `uv run python -c "from allium_cepa_classifier.config import AlliumCepaConfig, TrainingConfig, ExperimentConfig; print('ok')"` exits 0
- [x] `uv run ruff check src/` exits 0

#### Manual Verification:
- [x] `notebooks/training/vae_32D_05B.ipynb` first cell (`from allium_cepa_classifier import TrainingConfig; TrainingConfig()`) still works

---

## Phase 6: Relocate weights and fix `.gitignore`

### Changes Required:

#### 1. Move weights directory

Move (on disk, not git — these files are gitignored):
```bash
mv src/allium_cepa_classifier/models/weights/ src/allium_cepa_classifier/weights/
```

#### 2. Update `AlliumCepaConfig` default paths (done in Phase 5 already — already points to `weights/`)

#### 3. Update `.gitignore`

The `models/` rule silently prevents any new Python source files from being tracked in `src/allium_cepa_classifier/models/`. Replace it with an explicit rule targeting only the weights directory:

```gitignore
# ML artifacts — large files, never commit
...
# Remove: models/
# Add:
src/allium_cepa_classifier/weights/
```

The existing `*.pt`, `*.keras`, `*.h5`, `*.pth` rules still catch any stray weight file that ends up elsewhere.

Also add `dist/` entry if missing (it's already there — confirm).

### Success Criteria:

#### Automated Verification:
- [x] `git check-ignore -v src/allium_cepa_classifier/models/allium_cepa_model.py` returns nothing (file is NOT ignored)
- [x] `git check-ignore -v src/allium_cepa_classifier/weights/object_detection.pt` returns a match (file IS ignored)
- [x] `uv run python -c "from allium_cepa_classifier import AlliumCepaConfig; print(AlliumCepaConfig().detection_weights_path)"` prints a path under `weights/`

#### Manual Verification:
- [ ] Confirm weight files physically exist at new location before running inference notebook

---

## References

- Notebook-to-script migration: `thoughts/shared/plans/2026-05-02-notebook-to-script-migration.md`
- TF→PyTorch migration: `thoughts/shared/plans/2026-04-18-tensorflow-to-pytorch-migration.md`
