# Allium Cepa Automation

> **[WIP] TensorFlow → PyTorch migration in progress.** See [Migration Plan](#tensorflow--pytorch-migration) below.

Computer vision pipeline for automated detection and mitosis classification in *Allium cepa* (onion root tip) microscopy images. Built at UTN / INA.

The pipeline detects cells using a YOLO-based detector, then classifies each detection as **mitosis** or **no mitosis** using an EfficientNetB1 classifier (PyTorch / `timm`). An optional vector-scaling calibration step improves probability estimates.

---

## TensorFlow → PyTorch Migration

**Status: Work In Progress**

Full plan: [`thoughts/shared/plans/2026-04-18-tensorflow-to-pytorch-migration.md`](thoughts/shared/plans/2026-04-18-tensorflow-to-pytorch-migration.md)

The goal is to remove the `tensorflow` dependency entirely and unify the full stack on PyTorch (the detector already uses PyTorch via Ultralytics).

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Update dependencies (`timm`, `torchvision`; remove `tensorflow`) | ✅ Done |
| 2 | Rewrite `classifier.ipynb` (PyTorch training loop) | ✅ Done |
| 3 | Rewrite `calibrate_model.ipynb` (PyTorch calibration) | ✅ Done |
| 4 | Rewrite `allium_cepa_model.py` inference + config update | 🔄 In progress |
| 4 | Use DVC for dataset management | ⏳ Pending |
| 4 | Migrate to experiment workflow instead of notebooks for training | ⏳ Pending |

---

## Project Structure

```
allium_cepa_automation/
├── datasets/               # Raw and processed image datasets (not tracked in git)
├── notebooks/
│   ├── training/           # classifier.ipynb, calibrate_model.ipynb, VAE notebooks
│   ├── analysis/
│   ├── inference/
│   └── datasets/
├── scripts/
├── src/
│   ├── allium_cepa_classifier/
│   │   ├── config/
│   │   ├── data_models/
│   │   ├── models/
│   │   │   ├── allium_cepa_model.py
│   │   │   └── weights/               # Not tracked in git
│   │   └── utils/
│   └── ui/
├── thoughts/
│   └── shared/plans/
├── pyproject.toml
└── uv.lock
```

---

## Setup

```bash
uv sync --all-groups
```

> Model weights and datasets are not tracked in this repository. Obtain them separately and place them under `src/allium_cepa_classifier/models/weights/` and `datasets/` respectively.
