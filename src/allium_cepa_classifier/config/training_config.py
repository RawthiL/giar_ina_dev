from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict

from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()


class TrainingConfig(BaseConfig):
    """
    Filesystem paths for the Allium Cepa datasets.

    Defaults point to the standard project layout. Override via YAML
    when the dataset lives elsewhere (different machine, external drive, etc.).
    """

    model_config = ConfigDict(frozen=True)  # makes all fields immutable

    raw_dataset_dir: Path = _ROOT / "datasets/allium_cepa_full_images_merged_v3"
    yolo_dataset_dir: Path = _ROOT / "datasets/yolo_dataset"
    crops_dir: Path = _ROOT / "datasets/crops"
    binary_classifier_crops_dir: Path = crops_dir / "binary_classifier"
    vae_crops_dir: Path = crops_dir / "vae"
    model_path: Path = _ROOT / "src/allium_cepa_classifier/models/weights"
    splits: list[str] = ["train", "validation", "test"]
    image_height: int = 260
    image_width: int = 260
    image_size: tuple[int, int] = (image_height, image_width)
    confidence_threshold: float = 0.5
    batch_size: int = 32
    use_cpu: bool = False
    seed: int = 42
