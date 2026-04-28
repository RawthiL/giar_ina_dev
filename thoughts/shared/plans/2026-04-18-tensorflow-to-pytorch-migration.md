# TensorFlow → PyTorch Migration: Classifier + Calibration

## Overview

Migrate the EfficientNetB1 image classifier and its vector-scaling calibration pipeline from TensorFlow/Keras to PyTorch. This removes the `tensorflow==2.21.0` dependency (the only remaining TF dependency; detection already uses PyTorch via Ultralytics) and aligns the full stack on a single deep-learning framework.

**Scope:** `classifier.ipynb`, `calibrate_model.ipynb`, `allium_cepa_model.py`, `allium_cepa_config.py`, `pyproject.toml`.  
**Not in scope:** VAE notebooks, YOLO detection model, dataset preparation utilities.

---

## Current State Analysis

### Models that use TensorFlow
| File | TF usage |
|------|----------|
| `notebooks/training/classifier.ipynb` | Dataset pipeline, EfficientNetB1 via `tf.keras.applications`, training loop, save as `.keras` |
| `notebooks/training/calibrate_model.ipynb` | Loads `.keras` model, extracts logits through TF functional API, custom `VectorScalingLayer(tf.keras.layers.Layer)`, saves calibrated model as `.keras` |
| `src/allium_cepa_classifier/models/allium_cepa_model.py` | `keras.models.load_model()`, `tf.keras.utils.image_dataset_from_directory`, `tf.convert_to_tensor`, `tf.image.resize`, `tf.config.*` GPU management |
| `src/allium_cepa_classifier/config/allium_cepa_config.py` | `classification_weights_path` points to `.keras` file |

### Key constraints discovered
- `allium_cepa_model.py:182` — comment says EfficientNet normalizes internally, but `_classify_mitosis` still divides by 255. There is an inconsistency. The PyTorch version must normalize explicitly with ImageNet stats (mean/std) in **both** training and inference.
- Mitosis is **class index 0** in the current softmax output (`allium_cepa_model.py:193` — `argmax == 0`). Must be preserved.
- `allium_cepa_model.py:171,286` uses `tf.keras.utils.image_dataset_from_directory` for batched inference on temporary crops. Must be replaced with a pure-PyTorch `Dataset` + `DataLoader`.
- Calibration optimization (`calibrate_model.ipynb`) uses `scipy.optimize.minimize` — this is framework-agnostic and does not change.
- `torch>=2.3.0` and `torchvision` are already implicitly pulled in by `ultralytics`. `timm` needs to be added explicitly.

---

## Desired End State

After this plan is complete:
- `tensorflow` is removed from `pyproject.toml` and the environment.
- `timm` is added to `pyproject.toml`.
- `classifier.ipynb` trains an EfficientNetB1 with PyTorch, saves weights as `classifier_efficientNetB1_20E.pt` (state dict + metadata).
- `calibrate_model.ipynb` calibrates the PyTorch model and saves `classifier_calibrated.pt`.
- `allium_cepa_model.py` loads `.pt` weights and runs inference without any TF import.
- `allium_cepa_config.py` default `classification_weights_path` points to the new `.pt` file.
- The full inference pipeline (`full_pipeline.ipynb`, Streamlit app) continues to work unchanged from the user's perspective.

### Verification
- `uv run python -c "from allium_cepa_classifier.models.allium_cepa_model import AlliumCepaModel"` — no TF import errors.
- `uv run python -c "import tensorflow"` — should fail (TF removed).
- Running `full_pipeline.ipynb` on a test image produces a valid `AlliumCepaResult`.

---

## What We're NOT Doing

- Migrating VAE notebooks (`vae_32D_05B.ipynb`, `vae_32D_2B.ipynb`).
- Changing model architecture, hyperparameters, or training strategy.
- Converting existing `.keras` weights to PyTorch (re-training from scratch).
- Moving training code from notebooks to scripts.
- Adding unit tests.
- Changing the `AlliumCepaResult` data model or the Streamlit UI.

---

## Implementation Approach

Migrate in four sequential phases. Each phase is self-contained and can be verified before the next starts. Phases 2 and 3 (the notebooks) are the core work; phases 1 and 4 are dependency and wiring changes that bracket the notebook work.

**Framework choice — `timm`:** Both `timm` and `torchvision` have EfficientNetB1 with ImageNet weights. `timm` is chosen because (a) it is the academic CV standard for transfer learning experiments, (b) it exposes fine-grained block-level access needed to replicate the partial-freezing strategy from the Keras notebook, and (c) it is already common in research workflows like this one.

**Normalization:** timm's `efficientnet_b1` expects inputs normalized with ImageNet stats: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`, pixel values in `[0,1]`. This replaces the ad-hoc division by 255 in the current inference code.

**Partial freezing strategy:** The Keras notebook freezes all layers before `'block6a_expand_conv'`. In timm's EfficientNetB1, `model.blocks` is a `nn.Sequential` of 7 block groups (indices 0–6). `block6a` maps to `model.blocks[5]`. We freeze `model.conv_stem`, `model.bn1`, and `model.blocks[:5]`, leaving `model.blocks[5:]`, `model.conv_head`, `model.bn2`, and the custom classifier head trainable.

**Model save format:** `torch.save({'model_state_dict': model.state_dict(), 'num_classes': 2, 'image_size': 200}, path)`. Using a dict (not just the state dict) allows the loading code to verify shape expectations.

---

## Phase 1: Update Dependencies

### Overview
Add `timm` and `torchvision` to `pyproject.toml`; remove `tensorflow`. Sync the environment.

### Changes Required

#### 1. `pyproject.toml`
**File:** `pyproject.toml`  
**Changes:** In the `dependencies` list, replace `"tensorflow==2.21.0"` with `"timm>=1.0.0"` and add `"torchvision>=0.18.0"`.

```toml
dependencies = [
    # Deep learning
    "torch>=2.3.0",
    "torchvision>=0.18.0",
    "timm>=1.0.0",
    # Detection
    "ultralytics>=8.3.0",
    ...
]
```

#### 2. Sync environment
```bash
uv sync
```

### Success Criteria

#### Automated Verification
- [x] Environment syncs without error: `uv sync`
- [x] `timm` importable: `uv run python -c "import timm; print(timm.__version__)"`
- [x] `torchvision` importable: `uv run python -c "import torchvision; print(torchvision.__version__)"`
- [x] `tensorflow` is no longer installed: `uv run python -c "import tensorflow"` should raise `ModuleNotFoundError`

---

## Phase 2: Rewrite `classifier.ipynb`

### Overview
Rewrite the training notebook to use PyTorch end-to-end: `torchvision` + `timm` for the model, manual training loop with `ReduceLROnPlateau` and early stopping, save as `.pt`.

The notebook should be self-contained: run from top to bottom to reproduce the trained weights.

### Changes Required

#### Cell 1 — Imports
```python
import os
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import sys

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

#### Cell 2 — Config
```python
sys.path.insert(0, str(Path("../../src")))
from allium_cepa_classifier.config.training_config import TrainingConfig

config = TrainingConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config.batch_size          # 32
IMAGE_SIZE = config.image_size          # (200, 200)
EPOCHS     = 20
LR         = 1e-5
MODEL_NAME = "efficientNetB1"

print(f"Device: {DEVICE}")
```

#### Cell 3 — Transforms
```python
# ImageNet normalization stats (timm efficientnet_b1 expects these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

Note: The original notebook did not use augmentation during training (only `image_dataset_from_directory` with no augmentation). The augmentations above (`RandomHorizontalFlip`, `RandomVerticalFlip`, `ColorJitter`) are conservative additions appropriate for cell microscopy. If strict reproducibility with the original run is preferred, remove all augmentation from `train_transforms` and use only `Resize` + `ToTensor` + `Normalize`.

#### Cell 4 — Datasets and DataLoaders
```python
# torchvision ImageFolder expects: root/{class_name}/image.png
# Existing crops dir has exactly this layout:
#   datasets/crops/binary_classifier/{split}/{mitosis|no_mitosis}/
train_dir      = config.binary_classifier_crops_dir / "train"
validation_dir = config.binary_classifier_crops_dir / "validation"
test_dir       = config.binary_classifier_crops_dir / "test"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(validation_dir, transform=eval_transforms)
test_dataset  = datasets.ImageFolder(test_dir, transform=eval_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Class mapping: ImageFolder assigns indices alphabetically.
# 'mitosis' < 'no_mitosis' → class_to_idx = {'mitosis': 0, 'no_mitosis': 1}
# This matches the existing convention: argmax == 0 → mitosis.
print("Class mapping:", train_dataset.class_to_idx)
assert train_dataset.class_to_idx == {"mitosis": 0, "no_mitosis": 1}, (
    "Unexpected class order — check the crops directory structure."
)
```

#### Cell 5 — Class weights
```python
train_labels = [label for _, label in train_dataset.samples]
class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(train_labels))

# Same manual adjustment as original notebook:
class_weights[0] *= 2   # upweight mitosis
class_weights[1] *= 0.5 # downweight no_mitosis

weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"Class weights: mitosis={class_weights[0]:.3f}, no_mitosis={class_weights[1]:.3f}")
```

#### Cell 6 — Model definition
```python
def build_model(num_classes: int = 2) -> nn.Module:
    model = timm.create_model("efficientnet_b1", pretrained=True)

    # Freeze early blocks (equivalent to freezing before block6a in Keras):
    # timm blocks[0..4] ≈ Keras block1a..block5x
    # timm blocks[5..6] ≈ Keras block6a..block7a  ← kept trainable
    for param in model.conv_stem.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for i in range(5):   # freeze blocks 0–4
        for param in model.blocks[i].parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(128, num_classes),   # logits (no softmax here)
    )
    # Note: GlobalAveragePooling2D is already applied by timm before the classifier head.
    # The Dense(128) → LeakyReLU before the output in the Keras model is reproduced above.
    return model

model = build_model().to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}")
```

#### Cell 7 — Optimizer, scheduler, loss
```python
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.2, patience=5, min_lr=1e-6, threshold=0.01
)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
```

#### Cell 8 — Training loop
```python
best_val_loss = float("inf")
best_state    = None
patience_counter = 0
EARLY_STOPPING_PATIENCE = 10
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * images.size(0)
        train_correct += (logits.argmax(dim=1) == labels).sum().item()
        train_total   += images.size(0)

    # --- Validate ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss   = criterion(logits, labels)
            val_loss    += loss.item() * images.size(0)
            val_correct += (logits.argmax(dim=1) == labels).sum().item()
            val_total   += images.size(0)

    avg_train_loss = train_loss / train_total
    avg_val_loss   = val_loss   / val_total
    train_acc      = train_correct / train_total
    val_acc        = val_correct   / val_total

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    scheduler.step(avg_val_loss)

    # Early stopping (restore_best_weights=True equivalent)
    if avg_val_loss < best_val_loss:
        best_val_loss    = avg_val_loss
        best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train_loss={avg_train_loss:.4f} acc={train_acc:.4f} | "
        f"val_loss={avg_val_loss:.4f} acc={val_acc:.4f} | "
        f"lr={current_lr:.2e} | patience={patience_counter}"
    )

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

# Restore best weights
model.load_state_dict(best_state)
print(f"Restored best weights (val_loss={best_val_loss:.4f})")
```

#### Cell 9 — Test evaluation
```python
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        test_correct += (logits.argmax(dim=1) == labels).sum().item()
        test_total   += images.size(0)

print(f"Test accuracy: {test_correct / test_total:.4f}")
```

#### Cell 10 — Save model
```python
save_path = config.model_path / f"classifier_{MODEL_NAME}_{EPOCHS}E.pt"
torch.save({
    "model_state_dict": best_state,
    "num_classes": 2,
    "image_size": IMAGE_SIZE,
    "class_to_idx": train_dataset.class_to_idx,
    "imagenet_mean": IMAGENET_MEAN,
    "imagenet_std":  IMAGENET_STD,
}, save_path)
print(f"Saved: {save_path}")
```

### Success Criteria

#### Automated Verification
- [x] Notebook runs cell-by-cell without errors: kernel restart → run all
- [x] `classifier_efficientNetB1_20E.pt` appears in `src/allium_cepa_classifier/models/weights/`
- [x] Saved file loads correctly (built-in sanity check in the last cell):
  ```python
  ckpt = torch.load("...classifier_efficientNetB1_20E.pt")
  assert set(ckpt.keys()) == {"model_state_dict", "num_classes", "image_size", "class_to_idx", "imagenet_mean", "imagenet_std"}
  assert ckpt["class_to_idx"] == {"mitosis": 0, "no_mitosis": 1}
  ```

#### Manual Verification
- [ ] Test accuracy is comparable to the original Keras run (~91%)
- [ ] Training curves (loss/accuracy plots) look healthy — no divergence, no obvious overfitting
- [ ] Training converges before epoch 20 (early stopping fires or val_loss plateaus)

**Implementation Note:** After completing this phase and all automated verification passes, pause here for manual confirmation from the human that the test accuracy and training curves are acceptable before proceeding to Phase 3.

---

## Phase 3: Rewrite `calibrate_model.ipynb`

### Overview
Rewrite the calibration notebook to load the PyTorch classifier, extract logits for the validation set, run the same scipy L-BFGS-B vector scaling optimization, build a calibrated `nn.Module` that wraps the base model, and save it as `.pt`.

### Changes Required

#### Cell 1 — Imports
```python
import numpy as np
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path("../../src")))
from allium_cepa_classifier.config.training_config import TrainingConfig

config = TrainingConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### Cell 2 — Load classifier
```python
WEIGHTS_PATH = config.model_path / "classifier_efficientNetB1_20E.pt"

def load_classifier(weights_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(weights_path, map_location=device)

    model = timm.create_model("efficientnet_b1", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(128, ckpt["num_classes"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt

model, ckpt = load_classifier(WEIGHTS_PATH, DEVICE)
```

#### Cell 3 — Validation DataLoader
```python
IMAGE_SIZE   = ckpt["image_size"]
IMAGENET_MEAN = ckpt["imagenet_mean"]
IMAGENET_STD  = ckpt["imagenet_std"]

eval_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_dataset = datasets.ImageFolder(
    config.binary_classifier_crops_dir / "validation",
    transform=eval_transforms
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
```

#### Cell 4 — Extract logits from validation set
```python
# Build a model variant that returns logits (pre-softmax outputs)
# The last Linear(128, num_classes) in model.classifier is the logits layer.
# We create a feature extractor that stops just before the implicit softmax
# (note: CrossEntropyLoss was used in training, so model.classifier already
#  outputs raw logits with no softmax applied — no modification needed).

all_logits = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        logits = model(images)          # raw logits, shape (B, 2)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

all_logits = np.concatenate(all_logits, axis=0)   # (N, 2)
all_labels = np.concatenate(all_labels, axis=0)   # (N,)
print(f"Collected {len(all_labels)} validation samples")
```

#### Cell 5 — Vector scaling optimization
```python
NUM_CLASSES = all_logits.shape[1]

def vector_scale_loss(temp_vector: np.ndarray) -> float:
    scaled = all_logits / temp_vector
    # Numerically stable softmax
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_    = np.exp(shifted)
    probs   = exp_ / exp_.sum(axis=1, keepdims=True)
    return log_loss(all_labels, probs)

result = minimize(
    vector_scale_loss,
    x0=np.ones(NUM_CLASSES),
    bounds=[(0.01, 10.0)] * NUM_CLASSES,
    method="L-BFGS-B",
)
optimal_T = result.x
print(f"Optimal temperature vector: {optimal_T}")
print(f"  mitosis (class 0) T={optimal_T[0]:.4f}")
print(f"  no_mitosis (class 1) T={optimal_T[1]:.4f}")
```

#### Cell 6 — Calibrated model wrapper
```python
class CalibratedClassifier(nn.Module):
    """Wraps the base classifier, applying per-class temperature scaling on logits."""

    def __init__(self, base_model: nn.Module, temperature: np.ndarray):
        super().__init__()
        self.base_model  = base_model
        self.temperature = nn.Parameter(
            torch.tensor(temperature, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        scaled = logits / self.temperature
        return torch.softmax(scaled, dim=1)

    def get_temperature(self) -> np.ndarray:
        return self.temperature.cpu().numpy()


calibrated_model = CalibratedClassifier(model, optimal_T).to(DEVICE)
calibrated_model.eval()
```

#### Cell 7 — Calibration evaluation (ECE + reliability diagrams)
```python
def get_probs(mdl, loader, device):
    all_probs, all_true = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = mdl(images.to(device)).cpu().numpy()
            all_probs.append(probs)
            all_true.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_true)

def ece(probs, labels, n_bins=10):
    confidences = probs.max(axis=1)
    preds       = probs.argmax(axis=1)
    correct     = preds == labels
    bins        = np.linspace(0, 1, n_bins + 1)
    ece_val     = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece_val += mask.sum() / len(labels) * abs(acc - conf)
    return ece_val

# Evaluate original model (raw softmax)
class RawSoftmaxWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return torch.softmax(self.m(x), dim=1)

orig_probs, true_labels = get_probs(RawSoftmaxWrapper(model).to(DEVICE).eval(), val_loader, DEVICE)
cal_probs,  _           = get_probs(calibrated_model, val_loader, DEVICE)

print(f"Original ECE:   {ece(orig_probs, true_labels):.4f}")
print(f"Calibrated ECE: {ece(cal_probs,  true_labels):.4f}")

# Reliability diagrams per class
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for cls_idx, cls_name in enumerate(["mitosis", "no_mitosis"]):
    ax = axes[cls_idx]
    for probs, label in [(orig_probs, "original"), (cal_probs, "calibrated")]:
        binary_true  = (true_labels == cls_idx).astype(int)
        cls_probs    = probs[:, cls_idx]
        frac_pos, mean_pred = calibration_curve(binary_true, cls_probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_title(f"Reliability diagram — {cls_name}")
    ax.legend()
plt.tight_layout()
plt.show()
```

#### Cell 8 — Save calibrated model
```python
save_path = config.model_path / "classifier_calibrated.pt"
torch.save({
    "model_state_dict": calibrated_model.state_dict(),
    "temperature":      optimal_T.tolist(),
    "num_classes":      NUM_CLASSES,
    "image_size":       IMAGE_SIZE,
    "class_to_idx":     val_dataset.class_to_idx,
    "imagenet_mean":    IMAGENET_MEAN,
    "imagenet_std":     IMAGENET_STD,
}, save_path)
print(f"Saved: {save_path}")
```

### Success Criteria

#### Automated Verification
- [x] Notebook runs cell-by-cell without errors: kernel restart → run all
- [x] `classifier_calibrated.pt` appears in `src/allium_cepa_classifier/models/weights/`
- [x] Saved file loads:
  ```python
  ckpt = torch.load("...classifier_calibrated.pt")
  assert "temperature" in ckpt
  assert len(ckpt["temperature"]) == 2
  ```

#### Manual Verification
- [x] Calibrated ECE is lower than original ECE on the validation set
- [x] Reliability diagrams show calibrated curve closer to the diagonal

**Implementation Note:** Pause here for human review of the calibration plots before proceeding to Phase 4.

---

## Phase 4: Update `allium_cepa_model.py` and config

### Overview
Replace all TF-dependent inference code with pure PyTorch. Load `.pt` model, apply ImageNet transforms, run batched prediction via `DataLoader`.

### Changes Required

#### 1. `allium_cepa_config.py`
**File:** `src/allium_cepa_classifier/config/allium_cepa_config.py`  
**Change:** Update default `classification_weights_path` to point to the new `.pt` file.

```python
classification_weights_path: Path = _ROOT / "src/allium_cepa_classifier/models/weights/classifier_efficientNetB1_20E.pt"
```

#### 2. `allium_cepa_model.py`
**File:** `src/allium_cepa_classifier/models/allium_cepa_model.py`  
**Changes:** Full rewrite of imports and all methods. The public API (`predict`) and `AlliumCepaResult` shape are unchanged.

**New imports (replace TF imports):**
```python
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO

from allium_cepa_classifier.config import AlliumCepaConfig
from allium_cepa_classifier.data_models import AlliumCepaResult
```

**`_load_classification_model` replacement:**
```python
def _load_classification_model(self, weights_path: Path) -> nn.Module:
    if not weights_path.exists():
        raise FileNotFoundError(f"Classification model not found at: {weights_path}")

    ckpt = torch.load(weights_path, map_location=self._device)

    model = timm.create_model("efficientnet_b1", pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(128, ckpt["num_classes"]),
    )

    # If loading a calibrated model (CalibratedClassifier state dict),
    # the keys will be prefixed with 'base_model.' — detect and handle both.
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("base_model.") for k in state_dict):
        # Calibrated model: strip prefix and extract temperature
        state_dict = {k[len("base_model."):]: v for k, v in state_dict.items()
                      if k.startswith("base_model.")}
        temperature = torch.tensor(ckpt["temperature"], dtype=torch.float32)
        self._temperature = temperature.to(self._device)
    else:
        self._temperature = None

    model.load_state_dict(state_dict)
    model.to(self._device).eval()

    # Store normalization params from checkpoint
    self._imagenet_mean = ckpt.get("imagenet_mean", [0.485, 0.456, 0.406])
    self._imagenet_std  = ckpt.get("imagenet_std",  [0.229, 0.224, 0.225])

    return model
```

**Add `__init__` device setup (replace `_force_tensorflow_cpu`):**
```python
def __init__(self, config: AlliumCepaConfig):
    self.config = config
    if self.config.use_cpu:
        self._device = torch.device("cpu")
    else:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.detection_model = self._load_detection_model(self.config.detection_weights_path)
    self.classification_model = self._load_classification_model(self.config.classification_weights_path)
```

**New `_get_eval_transform` helper:**
```python
def _get_eval_transform(self) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(self.config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=self._imagenet_mean, std=self._imagenet_std),
    ])
```

**New `_run_classifier_on_crops` helper (replaces TF dataset + predict):**
```python
class _CropListDataset(Dataset):
    """Minimal Dataset wrapping a list of PIL crops with a transform."""
    def __init__(self, crops: List[Image.Image], transform):
        self.crops = crops
        self.transform = transform

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        return self.transform(self.crops[idx])


def _run_classifier_on_crops(self, crops: List[Image.Image]) -> np.ndarray:
    """Run batched inference on a list of PIL crops. Returns (N, 2) softmax probs."""
    dataset = _CropListDataset(crops, self._get_eval_transform())
    loader  = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(self._device)
            logits = self.classification_model(batch)   # (B, 2)
            if self._temperature is not None:
                logits = logits / self._temperature
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)   # (N, 2)
```

**Updated `_predict_single_image`:**  
Replace the section that creates `temp/crops/`, uses `tf.keras.utils.image_dataset_from_directory`, and calls `self.classification_model.predict(dataset)` with:

```python
# Crop all detections into PIL images (no disk I/O needed)
crops = []
for box in boxes:
    x_min_i, y_min_i, x_max_i, y_max_i = map(int, box)
    crops.append(image.crop((x_min_i, y_min_i, x_max_i, y_max_i)))

preds = self._run_classifier_on_crops(crops)   # (N, 2)
```

Also remove the `temp` directory creation, `shutil.rmtree`, and all `tf.*` calls. The crop PIL images are now kept in memory; no disk writes are needed.

**Updated `_predict_dir_image`:** Same refactor — collect crops in memory across all images, call `_run_classifier_on_crops` once with all crops, then attach predictions.

### Success Criteria

#### Automated Verification
- [x] Module imports without error: `uv run python -c "from allium_cepa_classifier.models.allium_cepa_model import AlliumCepaModel"`
- [x] No `tensorflow` or `keras` import anywhere in `src/`: `grep -r "tensorflow\|import keras" src/` → no output
- [x] `tensorflow` is absent from the environment: `uv run python -c "import tensorflow"` → `ModuleNotFoundError`

#### Manual Verification
- [ ] `full_pipeline.ipynb` runs end-to-end on a sample image and produces a non-empty `AlliumCepaResult`
- [ ] Streamlit app (`uv run streamlit run src/ui/app.py`) loads without error and classifies an uploaded image correctly
- [ ] Inference results (mitosis/no_mitosis labels and scores) are consistent with expected values from the original TF pipeline

**Implementation Note:** After completing this phase, run the full inference pipeline and compare a handful of predictions against the original model's outputs to confirm no regression was introduced.

---

## Testing Strategy

### Manual Testing Steps
1. Train the PyTorch classifier end-to-end in `classifier.ipynb`. Confirm test accuracy ≥ 88%.
2. Run `calibrate_model.ipynb`. Confirm calibrated ECE < original ECE.
3. Open `full_pipeline.ipynb`. Run on 2–3 sample images from `datasets/allium_cepa_full_images_merged_v3/test/`. Confirm detections and mitosis scores are produced.
4. Launch Streamlit: `uv run streamlit run src/ui/app.py`. Upload a test image. Confirm the annotated image and detections table render correctly.

---

## Migration Notes

- The `.keras` weight files in `src/allium_cepa_classifier/models/weights/` can be kept on disk for reference but are no longer used by any code after this migration.
- The `_CropListDataset` inner class avoids writing crops to disk during inference, which is a side-effect improvement — the current TF code writes PNG files to `temp/crops/` on every inference call.
- If `classifier_calibrated.pt` is used as the `classification_weights_path` in `AlliumCepaConfig`, the `_load_classification_model` method in Phase 4 automatically detects the calibrated format and applies temperature scaling at inference time.

---

## References

- Classifier training: `notebooks/training/classifier.ipynb`
- Calibration: `notebooks/training/calibrate_model.ipynb`
- Inference: `src/allium_cepa_classifier/models/allium_cepa_model.py`
- Config: `src/allium_cepa_classifier/config/allium_cepa_config.py`
- timm EfficientNetB1 docs: https://huggingface.co/timm/efficientnet_b1.ra4_e3600_r240_in1k
