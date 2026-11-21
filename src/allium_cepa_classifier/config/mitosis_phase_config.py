from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .base_config import BaseConfig, find_project_root
from .experiment_config import HeadConfig, LRSchedulerConfig

_ROOT = find_project_root()

MITOSIS_CLASSES = ["prophase", "metaphase", "anaphase", "telophase"]


class MitosisModelConfig(BaseModel):
    arch: Literal["efficientnet_b1", "efficientnet_b2", "resnet50", "vgg19"] = "efficientnet_b2"
    pretrained: bool = True
    freeze_stages: int = 2
    head: HeadConfig = HeadConfig()


class MitosisTrainingConfig(BaseModel):
    epochs: int = 30
    lr: float = 1e-4
    early_stopping_patience: int = 10
    use_balanced_class_weights: bool = True
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    augmentation: list[str] = ["hflip", "vflip", "color_jitter"]
    tensorboard: bool = True


class MitosisDataConfig(BaseModel):
    batch_size: int = 32
    seed: int = 42
    normalize_mean: list[float] = [0.485, 0.456, 0.406]
    normalize_std: list[float] = [0.229, 0.224, 0.225]
    classes: list[str] = MITOSIS_CLASSES
    # ImageFolder root with {train,validation,test}/{phase}/ subdirs (real-only).
    # Native 256x256 crops (hardlinked from datasets/crops/vae_256); no resize is applied.
    crops_dir: Path = _ROOT / "datasets/crops/mitosis_phase"
    # When True, train adds images from synthetic_dir; validation/test stay real-only
    # so the synthetic benefit is never confounded by synthetic eval data.
    include_synthetic: bool = False
    synthetic_dir: Path = _ROOT / "datasets/crops/mitosis_phase_synth"
    # Low-data budgets (data-budget sweep): deterministic per-class caps applied at
    # train time. None = keep the whole pool. Validation/test are never capped.
    train_max_per_class: int | None = None  # cap REAL train crops per class (all classes)
    synthetic_max_per_class: int | None = None  # cap SYNTH train images per class
    budget_seed: int = 1234  # RNG for the subsample (separate from data.seed)
    # Per-class factors applied on top of sklearn "balanced" weights (missing class = 1.0).
    class_weight_multipliers: dict[str, float] | None = None
    experiments_dir: Path = _ROOT / "experiments"
    # When True, copy test images whose predicted class differs from the true class to
    # <run_dir>/misclassifications/<true_class>/<predicted_class>/.
    save_misclassifications: bool = False


class MitosisPhaseConfig(BaseConfig):
    experiment_name: str
    model: MitosisModelConfig = MitosisModelConfig()
    training: MitosisTrainingConfig = MitosisTrainingConfig()
    data: MitosisDataConfig = MitosisDataConfig()
