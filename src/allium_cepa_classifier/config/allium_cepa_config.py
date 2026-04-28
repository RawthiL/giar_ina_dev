from pathlib import Path

from .base_config import BaseConfig, find_project_root

_ROOT = find_project_root()


class AlliumCepaConfig(BaseConfig):
    """
    Configuration for AlliumCepaModel.

    Holds the paths to the detection and classification model weights.
    Defaults point to some reasonable project-local paths (adjust as needed).
    """
    detection_weights_path: Path = _ROOT / "src/allium_cepa_classifier/models/weights/object_detection_v1.pt"
    classification_weights_path: Path = _ROOT / "src/allium_cepa_classifier/models/weights/classifier_efficientNetB1_20E.pt"
    valid_image_extensions: list = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    image_height: int = 200
    image_width: int = 200
    image_size: tuple[int, int] = (image_height, image_width)
    confidence_threshold: float = 0.5
    batch_size: int = 32
    use_cpu: bool = False
