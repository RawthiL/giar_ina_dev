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
    vae_256_crops_dir: Path = crops_dir / "vae_256"
    crop_size: tuple[int, int] = (256, 256)
    crop_margin: float = 0.0
    splits: list[str] = ["train", "validation", "test"]
