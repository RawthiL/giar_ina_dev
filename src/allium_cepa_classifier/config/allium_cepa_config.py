from pathlib import Path

from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()


class AlliumCepaConfig(BaseConfig):
    """Configuration for AlliumCepaModel inference."""

    detection_weights_path: Path = _ROOT / "src/allium_cepa_classifier/weights/object_detection.pt"
    detection_calibrator_path: Path = (
        _ROOT / "src/allium_cepa_classifier/weights/yolo_isotonic_calibrator.pkl"
    )
    classification_weights_path: Path = (
        _ROOT / "src/allium_cepa_classifier/weights/classifier_calibrated.pt"
    )
    valid_image_extensions: list = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    image_size: tuple[int, int] = (200, 200)
    batch_size: int = 32
    use_cpu: bool = False
