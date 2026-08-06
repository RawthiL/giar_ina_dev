from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()


class ControlNetModelConfig(BaseModel):
    pretrained_model_name_or_path: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    resolution: int = 512


class ControlNetTrainingConfig(BaseModel):
    train_batch_size: int = 16
    num_train_epochs: int = 20
    max_train_steps: int | None = None  # set small for smoke tests; overrides epochs when set
    learning_rate: float = 1e-5
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    seed: int = 42
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    enable_xformers: bool = False  # requires the optional xformers dependency
    checkpointing_steps: int = 2500
    validation_steps: int = 500
    report_to: str = "tensorboard"  # NOT wandb


class ControlNetValidationConfig(BaseModel):
    prompt: str = "a micrograph of an allium cepa root tip mitotic cell"
    # Relative to the train split dir (e.g. datasets/crops/controlnet/train/).
    image: str = "blurred_upscaled/004_00130_51_blurred_512x512.png"


class ControlNetDataConfig(BaseModel):
    dataset_dir: Path = _ROOT / "datasets/crops/controlnet"
    train_split: str = "train"
    test_split: str = "test"
    image_column: str = "image"
    conditioning_image_column: str = "conditioning_image"
    caption_column: str = "text"


class ControlNetExperimentConfig(BaseConfig):
    experiment_name: str
    model: ControlNetModelConfig = ControlNetModelConfig()
    training: ControlNetTrainingConfig = ControlNetTrainingConfig()
    validation: ControlNetValidationConfig = ControlNetValidationConfig()
    data: ControlNetDataConfig = ControlNetDataConfig()
