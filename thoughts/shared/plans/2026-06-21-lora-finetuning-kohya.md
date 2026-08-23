# LoRA Fine-Tuning Experiment Architecture (kohya_ss) Implementation Plan

## Overview

Add a LoRA fine-tuning workflow for Stable Diffusion models, driven by kohya_ss
(`kohya-ss/sd-scripts`), integrated into the existing experiment + DVC system.
The workflow mirrors the ControlNet pattern: a Pydantic YAML config is the single
source of truth, a thin Python wrapper translates it into an
`accelerate launch <kohya entrypoint> <args>` subprocess, and a DVC stage makes
runs reproducible. The schema lets you sweep across **base SD models**
(SD1.5 / SD2.x / SD3.x) and **LoRA hyperparameters** (rank, alpha, learning
rates, network type incl. LyCORIS variants).

LoRA training here is a **standalone capability** (experimentation / synthetic
data / concept adapters). Like ControlNet and the VAE, the produced LoRA weights
are **not** loaded by `AlliumCepaModel` at inference time.

## Current State Analysis

The repo already has a well-established "wrap an external diffusion trainer"
pattern from ControlNet:

- **Wrapper**: [scripts/train_controlnet.py](../../../scripts/train_controlnet.py)
  builds an `accelerate launch <vendored script> <args>` command from a Pydantic
  config, streams stdout/stderr to console + `train.log`, supports `--dry-run`.
- **Config**: [src/allium_cepa_classifier/config/controlnet_config.py](../../../src/allium_cepa_classifier/config/controlnet_config.py)
  — `BaseConfig` subclass with nested `model` / `training` / `validation` / `data`
  Pydantic models; `find_project_root()` resolves absolute dataset paths.
- **Experiment dir**: `experiments/controlnet/<name>/{config.yaml, weights/, logs/, plots/, train.log}`.
- **DVC**: the `train_controlnet` stage in [dvc.yaml](../../../dvc.yaml) tracks the
  wrapper + vendored trainer + config + dataset as deps, `weights/` as out, and
  watches config blocks via `params`. A separate `prepare_controlnet_dataset`
  stage builds the dataset from the HF repo.
- **Vendoring convention**: third-party trainer code lives under `scripts/vendor/`,
  excluded from ruff (`pyproject.toml` `extend-exclude` + `force-exclude`),
  provenance documented in [scripts/vendor/README.md](../../../scripts/vendor/README.md).

**Key constraint discovered:** kohya_ss is **not** a single vendorable file like
the diffusers ControlNet script — it is a multi-module package (`train_network.py`,
`sdxl_train_network.py`, `sd3_train_network.py`, `library/`, `networks/`). So the
"copy one file into `scripts/vendor/`" approach does not apply; we use a **git
submodule** instead (user-confirmed choice).

**SD3 constraint (verified):** SD3/3.5 LoRA requires `sd3_train_network.py`, which
exists on the **`sd3` branch** of sd-scripts, not `main`. The `sd3` branch is a
superset that also carries `train_network.py` (SD1.x/2.x) and
`sdxl_train_network.py`. Pinning the submodule to a recent `sd3`-branch commit
therefore covers SD1.5, SD2.x, SDXL, and SD3.x with one checkout.

## Desired End State

A new experiment type that runs end-to-end via DVC:

```bash
# one-time, on a fresh clone:
git submodule update --init --recursive
dvc repro prepare_lora_dataset

# train a LoRA experiment:
uv run python scripts/train_lora.py --config experiments/lora/sd15_rank16/config.yaml
uv run python scripts/train_lora.py --config experiments/lora/sd15_rank16/config.yaml --dry-run

# or via DVC (foreach over configured experiments):
dvc repro train_lora

# generate 3 samples per mitotic phase (12 images total) as evidence:
uv run python scripts/generate_lora_samples.py --config experiments/lora/sd15_rank16/config.yaml
dvc repro generate_lora_samples
```

producing:

```
experiments/lora/sd15_rank16/
├── config.yaml
├── train.log
├── weights/
│   └── sd15_rank16.safetensors      ← the trained LoRA (DVC-tracked out)
├── logs/                            ← tensorboard event files
└── plots/
    └── lora_samples.png             ← 4×3 grid: phase column × 3 samples per phase
```

### Key Discoveries:
- ControlNet wrapper command-assembly pattern: [scripts/train_controlnet.py:34-73](../../../scripts/train_controlnet.py#L34-L73)
- Config nesting + path resolution pattern: [src/allium_cepa_classifier/config/controlnet_config.py:39-53](../../../src/allium_cepa_classifier/config/controlnet_config.py#L39-L53)
- DVC `foreach` experiment stage pattern (train_vae / train_classifier): [dvc.yaml](../../../dvc.yaml)
- HF dataset download helper used by all prepare stages: `scripts/utils/download_hf_dataset.py` (invoked with `--subfolder` / `--rev` / `--out`)
- Ruff vendor-exclusion: `pyproject.toml` lines 70-73

### kohya network types (recap of the design rationale)
- `network_module: networks.lora` (**default**) → standard LoRA; knobs: `network_dim` (rank), `network_alpha`.
- `network_module: lycoris.kohya` + `network_args: ["algo=locon", "conv_dim=8", "conv_alpha=4"]` → LoCon (adapts conv layers too).
- other LyCORIS algos via `algo=loha|lokr|dylora`.
- Schema is a superset: leaving `network_module`/`network_args` at defaults always yields plain standard LoRA. `lycoris_lora` is only needed when a LyCORIS algo is selected.

## What We're NOT Doing

- **Not** wiring LoRA into `AlliumCepaModel` inference (it stays standalone, like ControlNet/VAE).
- **Not** using the kohya_ss **GUI** (`bmaltais/kohya_ss`); we use the headless `sd-scripts` library directly.
- **Not** vendoring kohya source file-by-file (submodule instead).
- **Not** training SDXL in the first pass (the entrypoint is wired so it's easy to add later, but the example configs target SD1.5/SD2.x/SD3.x per the request).
- **Not** building a captioning model — captions use a fixed template `"micrograph of allium cepa root tip mitotic cell in {phase} phase"` derived from the folder-name phase label.
- **Not** including non-mitotic (interphase) images in the LoRA dataset.
- **Not** adding regularization images in the first pass.

## Implementation Approach

Five pieces, each mirroring an existing convention:

1. **Submodule** `scripts/vendor/sd-scripts/` pinned to an `sd3`-branch commit, excluded from ruff.
2. **Dependencies**: add a `lora` dependency group (accelerate already present; add kohya's runtime deps + optional `lycoris_lora`, `bitsandbytes`).
3. **Config model** `LoRAExperimentConfig` mirroring `ControlNetExperimentConfig`.
4. **Wrapper** `scripts/train_lora.py` selecting the kohya entrypoint by model family and assembling the CLI.
5. **DVC stages** `prepare_lora_dataset` (build kohya folder layout from HF) + `train_lora` (`foreach` experiments), plus one example experiment dir.

---

## Phase 1: Vendor kohya_ss as a submodule + dependencies

### Overview
Bring sd-scripts into the repo reproducibly and make its runtime importable.

### Changes Required:

#### 1. Git submodule
**Command**:
```bash
git submodule add -b sd3 https://github.com/kohya-ss/sd-scripts scripts/vendor/sd-scripts
cd scripts/vendor/sd-scripts && git checkout <pinned-sd3-commit> && cd -
git add .gitmodules scripts/vendor/sd-scripts
```
This creates `.gitmodules` (new file) pinning the exact commit.

#### 2. Ruff exclusion
**File**: `pyproject.toml`
**Changes**: extend the existing vendor exclusion to cover the submodule tree.
```toml
extend-exclude = ["scripts/vendor"]   # already covers scripts/vendor/sd-scripts
force-exclude = true                  # already set
```
Verify the submodule path is under the existing `scripts/vendor` glob (it is) — likely **no change needed**, just confirm.

#### 3. Dependencies
**File**: `pyproject.toml`
**Changes**: add a dependency group for LoRA training (kept optional so base installs stay lean). kohya needs a specific set of libs; pin compatibly with the existing `diffusers>=0.35.1,<0.36` / `accelerate>=0.31.0`.
```toml
[dependency-groups]
lora = [
  "lycoris-lora>=3.0.0",   # only used when network_module=lycoris.kohya
  "bitsandbytes>=0.43.0",  # for AdamW8bit / 8-bit optimizers
  # kohya's own requirements (safetensors, etc.) overlap with existing deps;
  # confirm against scripts/vendor/sd-scripts/requirements.txt during impl.
]
```
**Implementation note:** during implementation, read `scripts/vendor/sd-scripts/requirements.txt` and reconcile versions with the existing lockfile rather than guessing. Some kohya pins (e.g. a specific `diffusers`) may conflict with the repo's `<0.36` pin — resolve by relying on the repo's versions where the kohya scripts still run, and document any override.

#### 4. Vendor README
**File**: `scripts/vendor/README.md`
**Changes**: add an `sd-scripts/` section documenting source repo, branch, pinned commit, and re-sync instructions (`git submodule update --remote` is intentionally avoided; bumps are explicit `cd` + `checkout` + commit).

### Success Criteria:

#### Automated Verification:
- [x] Submodule resolves on clean clone: `git submodule update --init && test -f scripts/vendor/sd-scripts/train_network.py`
- [x] SD3 entrypoint present: `test -f scripts/vendor/sd-scripts/sd3_train_network.py`
- [x] Deps install: `uv sync --all-groups`
- [x] Ruff still clean and ignores submodule: `uv run ruff check .`

#### Manual Verification:
- [x] `accelerate launch scripts/vendor/sd-scripts/train_network.py --help` prints kohya's arg list without import errors.

**Implementation Note**: After completing this phase and all automated verification passes, pause for manual confirmation before proceeding.

---

## Phase 2: LoRA config model

### Overview
A Pydantic config mirroring `ControlNetExperimentConfig`, exposing base-model
selection, network type/hyperparams, training hyperparams, and dataset paths.

### Changes Required:

#### 1. New config module
**File**: `src/allium_cepa_classifier/config/lora_config.py`
**Changes**: new `LoRAExperimentConfig(BaseConfig)` with nested models.
```python
from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()

class LoRAModelConfig(BaseModel):
    pretrained_model_name_or_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    # selects the kohya entrypoint: train_network.py / sdxl_train_network.py / sd3_train_network.py
    model_family: Literal["sd15", "sd2", "sdxl", "sd3"] = "sd15"
    resolution: int = 512
    v2: bool = False                 # SD2.x
    v_parameterization: bool = False # SD2.x v-pred
    clip_skip: int | None = None

class LoRANetworkConfig(BaseModel):
    network_module: str = "networks.lora"   # or "lycoris.kohya"
    network_dim: int = 16                    # LoRA rank
    network_alpha: float = 8.0
    network_dropout: float | None = None
    network_args: list[str] = []             # e.g. ["algo=locon", "conv_dim=8", "conv_alpha=4"]
    network_train_unet_only: bool = False
    network_train_text_encoder_only: bool = False

class LoRATrainingConfig(BaseModel):
    train_batch_size: int = 2
    max_train_epochs: int = 10
    max_train_steps: int | None = None       # cap regardless of epochs; small for smoke tests
    learning_rate: float = 1e-4
    unet_lr: float | None = None
    text_encoder_lr: float | None = None
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 0
    optimizer_type: str = "AdamW8bit"
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"
    seed: int = 42
    enable_bucket: bool = True
    min_bucket_reso: int = 256
    max_bucket_reso: int = 1024
    cache_latents: bool = True
    save_every_n_epochs: int | None = None
    save_model_as: str = "safetensors"
    logging: bool = True                     # --log_with tensorboard

class LoRADataConfig(BaseModel):
    dataset_dir: Path = _ROOT / "datasets/crops/lora"
    train_data_dir: str = "img"              # kohya scans img/<repeats>_<concept>/
    caption_extension: str = ".txt"

class LoRAExperimentConfig(BaseConfig):
    experiment_name: str
    model: LoRAModelConfig = LoRAModelConfig()
    network: LoRANetworkConfig = LoRANetworkConfig()
    training: LoRATrainingConfig = LoRATrainingConfig()
    data: LoRADataConfig = LoRADataConfig()
```

#### 2. Register in config package
**File**: `src/allium_cepa_classifier/config/__init__.py`
**Changes**: export `LoRAExperimentConfig` alongside the others.

### Success Criteria:

#### Automated Verification:
- [x] Imports cleanly: `uv run python -c "from allium_cepa_classifier.config import LoRAExperimentConfig"`
- [x] Loads a minimal YAML: `uv run python -c "from allium_cepa_classifier.config import LoRAExperimentConfig; print(LoRAExperimentConfig.from_yaml('experiments/lora/sd15_rank16/config.yaml').network.network_dim)"`
- [x] Lint/format clean: `uv run ruff check . && uv run ruff format --check .`

#### Manual Verification:
- [x] Defaults match intended kohya semantics (rank/alpha, optimizer name spelled as kohya expects).

---

## Phase 3: Dataset prepare stage (kohya layout)

### Overview
A new DVC stage that builds the kohya folder layout
(`img/<repeats>_<concept>/*.png` + per-image `.txt` captions) from the existing
VAE tagged crops, applies the same offline augmentation as `augment_vae_crops.py`,
and writes RGB images (SD1.5 expects 3-channel input).

### Design decisions (all confirmed):

**Source**: `datasets/crops/vae/` tagged splits only — mitotic phases only, no untagged.
Collecting paths:
- `train/tagged/{prophase,metaphase,anaphase,telophase}/` — originals only (skip `_aug` stem suffix)
- `val/tagged/{prophase,metaphase,anaphase,telophase}/`
- `test/{prophase,metaphase,anaphase,telophase}/` — note: no `tagged/` layer at this level

**Augmentation**: apply `augment_vae_crops.py`'s transform set (`augment()`) to every
collected original at --ratio 1.0, producing one `_aug` sibling per image — then write
both original and augmented to the output dir. This gives consistent doubling across all
splits regardless of which split the source image came from.

**Color mode**: save as **RGB** (`image.convert("RGB")`) — the VAE saves grayscale but
SD models require 3-channel input. Both originals and augmented copies go out as RGB.

**Captions** (one `.txt` per image, same basename):
```
micrograph of allium cepa root tip mitotic cell in {phase} phase
```
where `{phase}` is one of: `prophase`, `metaphase`, `anaphase`, `telophase`.
Both the original and its `_aug` copy share the same caption text.

**kohya folder layout**:
```
datasets/crops/lora/
└── img/
    └── 10_allium mitosis/
        ├── <stem>.png
        ├── <stem>.txt
        ├── <stem>_aug.png
        ├── <stem>_aug.txt
        └── ...   (all 4 phases together, distinguished by caption)
```
All 4 phases go into **one concept folder** (`10_allium mitosis`) with phase encoded
in the per-image caption. The repeat count `10` means kohya will cycle through the
folder 10× per epoch; adjust via `--repeats` CLI arg or rename the folder in config.

**Source dependency**: the stage depends on `datasets/crops/vae` (already a DVC-tracked
output of `prepare_vae_dataset`), so `prepare_lora_dataset` must run after it.

### Changes Required:

#### 1. Prepare script
**File**: `scripts/utils/lora_dataset.py` (new)

```python
"""
Build a kohya-compatible LoRA dataset from the VAE tagged crops.

Collects original (non-aug) images from:
  datasets/crops/vae/train/tagged/{phase}/
  datasets/crops/vae/val/tagged/{phase}/
  datasets/crops/vae/test/{phase}/          ← no 'tagged/' layer here

For each image: copies as RGB, applies augment() once (also RGB), writes a
sibling .txt caption: "micrograph of allium cepa root tip mitotic cell in {phase} phase"

Output layout (kohya DreamBooth):
  datasets/crops/lora/img/10_allium mitosis/
      <stem>.png, <stem>.txt, <stem>_aug.png, <stem>_aug.txt, ...

Usage:
    uv run python scripts/utils/lora_dataset.py
    uv run python scripts/utils/lora_dataset.py --vae-dir datasets/crops/vae
                                                  --out datasets/crops/lora
                                                  --repeats 10
"""
import argparse, random, shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

CAPTION_TEMPLATE = "micrograph of allium cepa root tip mitotic cell in {phase} phase"


def augment(image: Image.Image) -> Image.Image:
    # Same transforms as augment_vae_crops.py, output stays RGB
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if random.random() < 0.5:
        image = ImageOps.flip(image)
    angle = random.uniform(-15.0, 15.0)
    image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=128)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    return image


def source_dirs(vae_dir: Path) -> list[tuple[Path, str]]:
    """Return (dir, phase) pairs for all tagged phase dirs across splits."""
    pairs = []
    for phase in PHASES:
        for split_path in [
            vae_dir / "train" / "tagged" / phase,
            vae_dir / "val"   / "tagged" / phase,
            vae_dir / "test"  / phase,            # no 'tagged/' layer in test
        ]:
            if split_path.exists():
                pairs.append((split_path, phase))
    return pairs


def collect_originals(phase_dir: Path) -> list[Path]:
    return [
        p for p in phase_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS and "_aug" not in p.stem
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-dir", type=Path, default=Path("datasets/crops/vae"))
    parser.add_argument("--out",     type=Path, default=Path("datasets/crops/lora"))
    parser.add_argument("--repeats", type=int,  default=10)
    args = parser.parse_args()

    concept_dir = args.out / "img" / f"{args.repeats}_allium mitosis"
    concept_dir.mkdir(parents=True, exist_ok=True)

    total_orig = total_aug = 0
    for phase_dir, phase in source_dirs(args.vae_dir):
        caption = CAPTION_TEMPLATE.format(phase=phase)
        for src in collect_originals(phase_dir):
            img = Image.open(src).convert("RGB")
            # original
            dst = concept_dir / src.name
            img.save(dst)
            (concept_dir / f"{src.stem}.txt").write_text(caption)
            total_orig += 1
            # augmented
            aug_stem = f"{src.stem}_aug"
            augment(img).save(concept_dir / f"{aug_stem}{src.suffix}")
            (concept_dir / f"{aug_stem}.txt").write_text(caption)
            total_aug += 1

    print(f"LoRA dataset: {total_orig} originals + {total_aug} augmented → {concept_dir}")


if __name__ == "__main__":
    main()
```

#### 2. DVC stage
**File**: `dvc.yaml`
**Changes**: add `prepare_lora_dataset` after `prepare_vae_dataset` (depends on its output).
```yaml
  prepare_lora_dataset:
    cmd: .venv/bin/python scripts/utils/lora_dataset.py
      --vae-dir datasets/crops/vae
      --out datasets/crops/lora
    deps:
    - scripts/utils/lora_dataset.py
    - datasets/crops/vae          # output of prepare_vae_dataset
    outs:
    - datasets/crops/lora
```

### Success Criteria:

#### Automated Verification:
- [x] Stage runs cleanly: `dvc repro prepare_lora_dataset`
- [x] Layout correct: `find datasets/crops/lora/img -maxdepth 2 -type d` shows one concept dir
- [x] Images and captions balanced: `python -c "from pathlib import Path; d=Path('datasets/crops/lora/img'); pngs={p.stem for p in d.rglob('*.png')}; txts={p.stem for p in d.rglob('*.txt')}; assert pngs==txts, f'mismatch: {pngs^txts}'; print(f'{len(pngs)} pairs OK')"`
- [x] RGB output: `python -c "from PIL import Image; img=next(Path('datasets/crops/lora/img').rglob('*.png')); print(Image.open(img).mode)"` → `RGB`
- [x] No `_aug` images from the source leak in (only freshly generated `_aug` copies): count originals vs `_aug` should be equal

#### Manual Verification:
- [ ] Spot-check 3–5 captions match the image content and phase.
- [ ] Images look correctly oriented and RGB (view a few with a viewer).
- [ ] Augmented copies show visible but mild transforms vs their originals.

---

## Phase 4: Training wrapper

### Overview
`scripts/train_lora.py` mirroring `train_controlnet.py`: load config, select the
kohya entrypoint by `model.model_family`, assemble `accelerate launch` args, stream
to console + `train.log`, support `--dry-run`.

### Changes Required:

#### 1. Wrapper script
**File**: `scripts/train_lora.py` (new)
**Changes**: entrypoint map + arg assembly.
```python
ENTRYPOINTS = {
    "sd15": "train_network.py",
    "sd2":  "train_network.py",
    "sdxl": "sdxl_train_network.py",
    "sd3":  "sd3_train_network.py",
}
SD_SCRIPTS = Path(__file__).resolve().parent / "vendor" / "sd-scripts"

def build_cmd(cfg, run_dir):
    entry = SD_SCRIPTS / ENTRYPOINTS[cfg.model.model_family]
    out_dir = run_dir / "weights"
    log_dir = run_dir / "logs"
    train_dir = (cfg.data.dataset_dir / cfg.data.train_data_dir).resolve()
    args = [
        "accelerate", "launch", str(entry),
        f"--pretrained_model_name_or_path={cfg.model.pretrained_model_name_or_path}",
        f"--train_data_dir={train_dir}",
        f"--output_dir={out_dir}",
        f"--output_name={cfg.experiment_name}",
        f"--logging_dir={log_dir}",
        "--log_with=tensorboard",
        f"--resolution={cfg.model.resolution}",
        f"--network_module={cfg.network.network_module}",
        f"--network_dim={cfg.network.network_dim}",
        f"--network_alpha={cfg.network.network_alpha}",
        f"--train_batch_size={cfg.training.train_batch_size}",
        f"--max_train_epochs={cfg.training.max_train_epochs}",
        f"--learning_rate={cfg.training.learning_rate}",
        f"--lr_scheduler={cfg.training.lr_scheduler}",
        f"--lr_warmup_steps={cfg.training.lr_warmup_steps}",
        f"--optimizer_type={cfg.training.optimizer_type}",
        f"--mixed_precision={cfg.training.mixed_precision}",
        f"--save_model_as={cfg.training.save_model_as}",
        f"--seed={cfg.training.seed}",
        f"--caption_extension={cfg.data.caption_extension}",
        f"--gradient_accumulation_steps={cfg.training.gradient_accumulation_steps}",
    ]
    # conditional flags
    if cfg.model.v2: args.append("--v2")
    if cfg.model.v_parameterization: args.append("--v_parameterization")
    if cfg.model.clip_skip is not None: args.append(f"--clip_skip={cfg.model.clip_skip}")
    if cfg.training.gradient_checkpointing: args.append("--gradient_checkpointing")
    if cfg.training.enable_bucket:
        args += [f"--enable_bucket",
                 f"--min_bucket_reso={cfg.training.min_bucket_reso}",
                 f"--max_bucket_reso={cfg.training.max_bucket_reso}"]
    if cfg.training.cache_latents: args.append("--cache_latents")
    if cfg.training.max_train_steps is not None:
        args.append(f"--max_train_steps={cfg.training.max_train_steps}")
    if cfg.training.unet_lr is not None: args.append(f"--unet_lr={cfg.training.unet_lr}")
    if cfg.training.text_encoder_lr is not None:
        args.append(f"--text_encoder_lr={cfg.training.text_encoder_lr}")
    if cfg.network.network_dropout is not None:
        args.append(f"--network_dropout={cfg.network.network_dropout}")
    if cfg.network.network_train_unet_only: args.append("--network_train_unet_only")
    if cfg.network.network_train_text_encoder_only: args.append("--network_train_text_encoder_only")
    for na in cfg.network.network_args:
        args.append(f"--network_args={na}")  # kohya accepts repeated/space-joined; confirm form
    if cfg.training.save_every_n_epochs is not None:
        args.append(f"--save_every_n_epochs={cfg.training.save_every_n_epochs}")
    return args
```
Reuse the `run()` streaming helper and `--dry-run` validation flow verbatim from
[scripts/train_controlnet.py:76-133](../../../scripts/train_controlnet.py#L76-L133)
(check submodule entrypoint exists + dataset prepared, print assembled command).

**Implementation note (network_args form):** kohya expects `--network_args`
followed by space-separated `key=value` tokens (e.g.
`--network_args "algo=locon" "conv_dim=8"`). Because we build a list and pass it
to `subprocess` without a shell, append each token as its own list element after a
single `--network_args` flag, not as repeated `--network_args=` pairs. Confirm
against `train_network.py --help` during implementation and adjust assembly
accordingly.

**Implementation note (SD3 args):** `sd3_train_network.py` takes extra required
args (e.g. CLIP/T5 text-encoder paths or a combined checkpoint, and SD3-specific
flags). The wrapper must branch on `model_family == "sd3"` to add these. Read
`scripts/vendor/sd-scripts/docs/sd3_train_network.md` during implementation and
extend `LoRAModelConfig` with the SD3-specific fields needed (e.g. `clip_l`,
`clip_g`, `t5xxl` paths). Keep them optional so SD1.5/2.x configs ignore them.

### Success Criteria:

#### Automated Verification:
- [x] Dry run prints a valid command for SD1.5: `uv run python scripts/train_lora.py --config experiments/lora/sd15_rank16/config.yaml --dry-run`
- [x] Dry run errors clearly when dataset missing / submodule absent.
- [x] Lint/format clean: `uv run ruff check . && uv run ruff format --check .`

#### Manual Verification:
- [ ] A capped smoke run (`max_train_steps: 5`) completes and writes `weights/<name>.safetensors`.
- [ ] TensorBoard shows loss curve under `experiments/lora/<name>/logs/`.
- [ ] A LyCORIS config (`network_module: lycoris.kohya`, `network_args: ["algo=locon"]`) also runs (validates the variant path).

**Implementation Note**: Pause for manual confirmation of a smoke run before Phase 5.

---

## Phase 5: DVC train stage + example experiments

### Overview
A `foreach` `train_lora` stage and at least one example experiment config.

### Changes Required:

#### 1. Example experiment(s)
**File**: `experiments/lora/sd15_rank16/config.yaml` (new), plus optionally
`experiments/lora/sd15_rank32/config.yaml` and an SD3 example.
```yaml
experiment_name: sd15_rank16
model:
  pretrained_model_name_or_path: stable-diffusion-v1-5/stable-diffusion-v1-5
  model_family: sd15
  resolution: 512
network:
  network_module: networks.lora
  network_dim: 16
  network_alpha: 8
training:
  train_batch_size: 2
  max_train_epochs: 10
  learning_rate: 1.0e-4
  lr_scheduler: cosine
  optimizer_type: AdamW8bit
  mixed_precision: fp16
  enable_bucket: true
data:
  train_data_dir: img
```

#### 2. DVC stage
**File**: `dvc.yaml`
**Changes**: add `train_lora` foreach (mirrors `train_vae`).
```yaml
  train_lora:
    foreach: [sd15_rank16]
    do:
      cmd: .venv/bin/python scripts/train_lora.py --config
        experiments/lora/${item}/config.yaml
      deps:
      - scripts/train_lora.py
      - scripts/vendor/sd-scripts
      - src/allium_cepa_classifier/config/lora_config.py
      - datasets/crops/lora
      - experiments/lora/${item}/config.yaml
      params:
      - experiments/lora/${item}/config.yaml:
        - model
        - network
        - training
        - data
      outs:
      - experiments/lora/${item}/weights
```

#### 3. Docs
**File**: `CLAUDE.md`
**Changes**: add a "LoRA (kohya_ss)" subsection under Training + an Architecture
note, mirroring the ControlNet documentation (commands, dataset layout, submodule
provenance, SD3-on-`sd3`-branch caveat, and the generate script).

### Success Criteria:

#### Automated Verification:
- [ ] `dvc repro train_lora` runs the stage (with a small `max_train_steps` for CI speed) and produces `experiments/lora/sd15_rank16/weights/sd15_rank16.safetensors`.
- [ ] `dvc status` is clean after a successful run.
- [x] Lint/format clean: `uv run ruff check . && uv run ruff format --check .`

#### Manual Verification:
- [ ] A full (uncapped) SD1.5 run produces a usable LoRA.
- [ ] Swapping `model_family`/`pretrained_model_name_or_path` to an SD2.x or SD3.x base trains via the correct entrypoint.
- [ ] Phase 6 generation script produces `plots/lora_samples.png` that visibly reflects the fine-tuned style.

---

## Phase 6: Sample generation script + DVC stage

### Overview
`scripts/generate_lora_samples.py` loads the trained LoRA into the appropriate
diffusers pipeline and generates **3 images per mitotic phase** (4 phases × 3 = 12
images), saved as a 4-column × 3-row grid at
`experiments/lora/<name>/plots/lora_samples.png`.

This mirrors `scripts/generate_controlnet_samples.py` but simpler: no conditioning
image is needed — the LoRA is loaded into a text-to-image pipeline via
`pipe.load_lora_weights()` and each phase caption drives generation.

The DVC `generate_lora_samples` stage depends on `weights/` (output of
`train_lora`) and produces `plots/lora_samples.png`, so the evidence grid is
automatically regenerated whenever the weights change.

### Changes Required:

#### 1. Generation script
**File**: `scripts/generate_lora_samples.py` (new)

```python
"""
Generate sample images from a trained LoRA checkpoint.

Loads the trained LoRA weights into the base SD pipeline (family-aware),
generates 3 images per mitotic phase (prophase, metaphase, anaphase, telophase),
and saves a 4×3 grid (phase columns × sample rows) to
experiments/lora/<name>/plots/lora_samples.png.

Usage:
    uv run python scripts/generate_lora_samples.py --config experiments/lora/sd15_rank16/config.yaml
    uv run python scripts/generate_lora_samples.py --config ... --samples 5 --seed 99
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
CAPTION_TEMPLATE = "micrograph of allium cepa root tip mitotic cell in {phase} phase"
NEGATIVE_PROMPT = "blurry, low quality, deformed, malformed, text, watermark, jpeg artifacts"

# Pipeline class per model family — imported lazily so missing deps don't break dry-runs.
PIPELINE_CLASSES = {
    "sd15": ("diffusers", "StableDiffusionPipeline"),
    "sd2":  ("diffusers", "StableDiffusionPipeline"),
    "sdxl": ("diffusers", "StableDiffusionXLPipeline"),
    "sd3":  ("diffusers", "StableDiffusion3Pipeline"),
}


def load_pipeline(cfg: LoRAExperimentConfig, lora_path: Path, device: str, dtype: torch.dtype):
    module_name, class_name = PIPELINE_CLASSES[cfg.model.model_family]
    import importlib
    PipelineClass = getattr(importlib.import_module(module_name), class_name)
    pipe = PipelineClass.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe.load_lora_weights(str(lora_path))
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=3, help="Images per phase (default 3).")
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = LoRAExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent
    lora_path = run_dir / "weights" / f"{cfg.experiment_name}.safetensors"
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}. Run training first.")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    print(f"Loading {cfg.model.model_family} pipeline from {cfg.model.pretrained_model_name_or_path} ...")
    pipe = load_pipeline(cfg, lora_path, args.device, dtype)

    n_phases = len(PHASES)
    fig, axes = plt.subplots(
        nrows=args.samples, ncols=n_phases,
        figsize=(4 * n_phases, 4 * args.samples),
        squeeze=False,
    )
    fig.suptitle(f"LoRA samples — {cfg.experiment_name}", fontsize=14)

    for col, phase in enumerate(PHASES):
        prompt = CAPTION_TEMPLATE.format(phase=phase)
        print(f"Phase: {phase}")
        for row in range(args.samples):
            seed = args.seed + row  # different seed per row, consistent across phases
            generator = torch.manual_seed(seed)
            img = pipe(
                prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).images[0]
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(phase, fontsize=11, fontweight="bold")

    fig.tight_layout()
    out_path = plots_dir / "lora_samples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
```

**Implementation notes:**
- SD3 (`StableDiffusion3Pipeline`) loads LoRA differently from SD1.x/SDXL — it may
  need text-encoder components specified. Verify `pipe.load_lora_weights()` works for
  SD3 in the version of diffusers pinned; fall back to `StableDiffusion3Pipeline`'s
  documented LoRA loading if the API differs.
- `seed + row` pattern ensures reproducible rows while producing variation across
  samples without extra config.
- Grid orientation — phases as columns (left→right: prophase→telophase), samples as
  rows (top→bottom) — makes phase comparison easy at a glance.

#### 2. DVC stage
**File**: `dvc.yaml`
**Changes**: add `generate_lora_samples` foreach, depending on `train_lora`'s
`weights/` output.
```yaml
  generate_lora_samples:
    foreach: [sd15_rank16]
    do:
      cmd: .venv/bin/python scripts/generate_lora_samples.py --config
        experiments/lora/${item}/config.yaml
      deps:
      - scripts/generate_lora_samples.py
      - src/allium_cepa_classifier/config/lora_config.py
      - experiments/lora/${item}/weights
      - experiments/lora/${item}/config.yaml
      outs:
      - experiments/lora/${item}/plots/lora_samples.png
```

### Success Criteria:

#### Automated Verification:
- [ ] `dvc repro generate_lora_samples` produces `experiments/lora/sd15_rank16/plots/lora_samples.png`.
- [ ] Image dimensions are non-zero and the file is a valid PNG: `python -c "from PIL import Image; img=Image.open('experiments/lora/sd15_rank16/plots/lora_samples.png'); print(img.size)"`.
- [x] Lint/format clean: `uv run ruff check . && uv run ruff format --check .`

#### Manual Verification:
- [ ] Grid shows 4 columns (one per phase) × 3 rows of generated images.
- [ ] Phase labels are legible in the column headers.
- [ ] Images visually resemble allium cepa micrograph style (not random noise), indicating the LoRA has learned from the dataset.

---

## Testing Strategy

### Unit-ish checks:
- Config round-trips from YAML; `network_args` list survives.
- `build_cmd` selects the right entrypoint per `model_family` and emits expected flags (assert on the arg list — no subprocess needed).

### Integration:
- Capped smoke run (`max_train_steps: 5`) for SD1.5 standard LoRA and one LyCORIS config.
- `dvc repro train_lora` end-to-end.

### Manual:
1. Smoke run → confirm `.safetensors` + TensorBoard loss curve.
2. Run `generate_lora_samples.py` → inspect `plots/lora_samples.png` for per-phase visual quality.
3. Repeat with an SD3.x base to exercise the `sd3_train_network.py` branch and `StableDiffusion3Pipeline` path.

## Performance Considerations

- LoRA is light vs full fine-tune, but base model load + latent caching dominates VRAM. Defaults (`train_batch_size: 2`, `gradient_checkpointing`, `cache_latents`, `AdamW8bit`, `fp16`) target a single consumer GPU; SDXL/SD3 need more VRAM and larger `resolution`.
- `cache_latents` speeds repeated epochs but disables caption/latent-altering augmentations.

## Migration Notes

- New submodule: existing clones must run `git submodule update --init --recursive` after pulling.
- `uv sync --all-groups` required to pick up the `lora` dependency group.
- Possible dependency-pin conflict between kohya's `requirements.txt` and the repo's `diffusers<0.36` — reconcile during Phase 1 (prefer repo pins where kohya still runs; document overrides).

## References

- ControlNet wrapper (template): [scripts/train_controlnet.py](../../../scripts/train_controlnet.py)
- ControlNet config (template): [src/allium_cepa_classifier/config/controlnet_config.py](../../../src/allium_cepa_classifier/config/controlnet_config.py)
- DVC stages: [dvc.yaml](../../../dvc.yaml)
- Vendor convention: [scripts/vendor/README.md](../../../scripts/vendor/README.md)
- kohya sd-scripts (`sd3` branch): https://github.com/kohya-ss/sd-scripts/tree/sd3
- SD3 LoRA docs: `scripts/vendor/sd-scripts/docs/sd3_train_network.md` (after submodule init)
- LyCORIS: https://github.com/KohakuBlueleaf/LyCORIS
