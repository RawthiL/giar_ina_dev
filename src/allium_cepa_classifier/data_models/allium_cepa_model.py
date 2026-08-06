import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO

from allium_cepa_classifier.config import AlliumCepaConfig
from allium_cepa_classifier.training.detector_calibrator import ObjectDetectionCalibrator

from .allium_cepa_result import AlliumCepaResult

_EMPTY_COLUMNS = [
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "confidence",
    "p_hat",
    "class_id",
    "class_name",
    "image",
    "mitosis",
    "mitosis_score",
    "q_interphase",
    "q_mitosis",
]


class _CropListDataset(Dataset):
    """Minimal Dataset wrapping a list of PIL crops with a transform."""

    def __init__(self, crops: list[Image.Image], transform):
        self.crops = crops
        self.transform = transform

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        return self.transform(self.crops[idx])


class AlliumCepaModel:
    def __init__(self, config: AlliumCepaConfig):
        self.config = config

        if self.config.use_cpu:
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.detection_model = self._load_detection_model(self.config.detection_weights_path)
        self._detection_calibrator = self._load_detection_calibrator(
            self.config.detection_calibrator_path
        )
        self.classification_model = self._load_classification_model(
            self.config.classification_weights_path
        )

    def _load_detection_model(self, weights_path: Path):
        if not weights_path.exists():
            raise FileNotFoundError(f"Detection model not found at: {weights_path}")
        return YOLO(weights_path)

    def _load_detection_calibrator(self, path: Path):
        if not path.exists():
            warnings.warn(
                f"Detection calibrator not found at {path}; raw YOLO confidence will be used "
                "as p_hat. Confidence intervals from get_counts_with_ci() will be biased.",
                stacklevel=3,
            )
            return None
        cal = ObjectDetectionCalibrator()
        cal.load(path)
        return cal

    def _calibrate_confidences(self, conf: np.ndarray) -> np.ndarray:
        if self._detection_calibrator is None:
            return conf.astype(np.float64)
        return np.clip(self._detection_calibrator.predict(conf), 0.0, 1.0)

    def _load_classification_model(self, weights_path: Path) -> nn.Module:
        if not weights_path.exists():
            raise FileNotFoundError(f"Classification model not found at: {weights_path}")

        ckpt = torch.load(weights_path, map_location=self._device)

        timm_model_name = ckpt.get("timm_model_name", "efficientnet_b2")
        model = timm.create_model(timm_model_name, pretrained=False)
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

        state_dict = ckpt["model_state_dict"]
        if any(k.startswith("base_model.") for k in state_dict):
            # Calibrated checkpoint: strip prefix and store temperature
            base_state = {
                k[len("base_model.") :]: v
                for k, v in state_dict.items()
                if k.startswith("base_model.")
            }
            self._temperature = torch.tensor(ckpt["temperature"], dtype=torch.float32).to(
                self._device
            )
            model.load_state_dict(base_state)
        else:
            self._temperature = None
            model.load_state_dict(state_dict)

        self._image_size = tuple(ckpt.get("image_size", self.config.image_size))
        self._imagenet_mean = ckpt.get("imagenet_mean", [0.485, 0.456, 0.406])
        self._imagenet_std = ckpt.get("imagenet_std", [0.229, 0.224, 0.225])

        model.to(self._device).eval()
        return model

    def _get_eval_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(self._image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self._imagenet_mean, std=self._imagenet_std),
            ]
        )

    def _run_classifier_on_crops(self, crops: list[Image.Image]) -> np.ndarray:
        """Run batched inference on a list of PIL crops. Returns (N, 2) softmax probs."""
        dataset = _CropListDataset(crops, self._get_eval_transform())
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                logits = self.classification_model(batch)
                if self._temperature is not None:
                    logits = logits / self._temperature
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)  # (N, 2)

    def _predict_single_image(self, image_path: Path) -> AlliumCepaResult:
        image_name = Path(image_path).name
        t_detect_start = time.perf_counter()

        image = Image.open(image_path)
        yolo_results = self.detection_model(image)[0]

        if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
            detection_s = time.perf_counter() - t_detect_start
            timing = {
                "detection_per_image_s": {image_name: detection_s},
                "detection_total_s": detection_s,
                "classification_total_s": 0.0,
                "total_s": detection_s,
            }
            return AlliumCepaResult(
                dir=Path(image_path).parent,
                detections=pd.DataFrame(columns=_EMPTY_COLUMNS),
                timing=timing,
            )

        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        confidences = yolo_results.boxes.conf.cpu().numpy()
        p_hats = self._calibrate_confidences(confidences)
        class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
        class_names_map = self.detection_model.names

        rows = []
        crops = []
        for box, conf, p_hat_i, cls_id in zip(boxes, confidences, p_hats, class_ids, strict=False):
            x_min_i, y_min_i, x_max_i, y_max_i = map(int, box)
            crops.append(image.crop((x_min_i, y_min_i, x_max_i, y_max_i)))
            rows.append(
                {
                    "x_min": x_min_i,
                    "y_min": y_min_i,
                    "x_max": x_max_i,
                    "y_max": y_max_i,
                    "confidence": float(conf),
                    "p_hat": float(p_hat_i),
                    "class_id": int(cls_id),
                    "class_name": class_names_map.get(int(cls_id), str(cls_id)),
                    "image": image_name,
                    "mitosis": None,
                    "mitosis_score": None,
                    "q_interphase": None,
                    "q_mitosis": None,
                }
            )

        detection_s = time.perf_counter() - t_detect_start
        t_classify_start = time.perf_counter()

        preds = self._run_classifier_on_crops(crops)  # (N, 2)

        for i, row in enumerate(rows):
            class_probs = preds[i]
            row["mitosis"] = bool(int(np.argmax(class_probs)) == 0)
            row["mitosis_score"] = float(class_probs[0])
            # Classifier internal ordering: index 0 = mitosis, index 1 = interphase.
            # q_hat convention (per compute_mi_with_ci): index 0 = interphase, index 1 = mitosis.
            row["q_mitosis"] = float(class_probs[0])
            row["q_interphase"] = float(class_probs[1])

        classification_s = time.perf_counter() - t_classify_start
        timing = {
            "detection_per_image_s": {image_name: detection_s},
            "detection_total_s": detection_s,
            "classification_total_s": classification_s,
            "total_s": detection_s + classification_s,
        }

        return AlliumCepaResult(
            dir=Path(image_path).parent, detections=pd.DataFrame(rows), timing=timing
        )

    def _predict_dir_image(self, images_paths: list) -> AlliumCepaResult:
        rows = []
        crops = []
        detection_per_image_s = {}

        for image_path in images_paths:
            image_name = Path(image_path).name
            t_img_detect_start = time.perf_counter()

            image = Image.open(image_path)
            yolo_results = self.detection_model(image)[0]

            if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
                detection_per_image_s[image_name] = time.perf_counter() - t_img_detect_start
                continue

            boxes = yolo_results.boxes.xyxy.cpu().numpy()
            confidences = yolo_results.boxes.conf.cpu().numpy()
            p_hats = self._calibrate_confidences(confidences)
            class_ids = yolo_results.boxes.cls.cpu().numpy().astype(int)
            class_names_map = self.detection_model.names

            for box, conf, p_hat_i, cls_id in zip(
                boxes, confidences, p_hats, class_ids, strict=False
            ):
                x_min_i, y_min_i, x_max_i, y_max_i = map(int, box)
                crops.append(image.crop((x_min_i, y_min_i, x_max_i, y_max_i)))
                rows.append(
                    {
                        "x_min": x_min_i,
                        "y_min": y_min_i,
                        "x_max": x_max_i,
                        "y_max": y_max_i,
                        "confidence": float(conf),
                        "p_hat": float(p_hat_i),
                        "class_id": int(cls_id),
                        "class_name": class_names_map.get(int(cls_id), str(cls_id)),
                        "image": image_name,
                        "mitosis": None,
                        "mitosis_score": None,
                        "q_interphase": None,
                        "q_mitosis": None,
                    }
                )

            detection_per_image_s[image_name] = time.perf_counter() - t_img_detect_start

        detection_total_s = sum(detection_per_image_s.values())

        if not rows:
            timing = {
                "detection_per_image_s": detection_per_image_s,
                "detection_total_s": detection_total_s,
                "classification_total_s": 0.0,
                "total_s": detection_total_s,
            }
            return AlliumCepaResult(
                dir=Path(images_paths[0]).parent,
                detections=pd.DataFrame(columns=_EMPTY_COLUMNS),
                timing=timing,
            )

        t_classify_start = time.perf_counter()

        preds = self._run_classifier_on_crops(crops)  # (N, 2)

        for i, row in enumerate(rows):
            class_probs = preds[i]
            row["mitosis"] = bool(int(np.argmax(class_probs)) == 0)
            row["mitosis_score"] = float(class_probs[0])
            # Classifier internal ordering: index 0 = mitosis, index 1 = interphase.
            # q_hat convention (per compute_mi_with_ci): index 0 = interphase, index 1 = mitosis.
            row["q_mitosis"] = float(class_probs[0])
            row["q_interphase"] = float(class_probs[1])

        classification_s = time.perf_counter() - t_classify_start
        timing = {
            "detection_per_image_s": detection_per_image_s,
            "detection_total_s": detection_total_s,
            "classification_total_s": classification_s,
            "total_s": detection_total_s + classification_s,
        }

        return AlliumCepaResult(
            dir=Path(images_paths[0]).parent, detections=pd.DataFrame(rows), timing=timing
        )

    def predict(self, image_path: str | Path) -> AlliumCepaResult | list[AlliumCepaResult]:
        """
        Run detection and classification on a single image or a directory of images.

        Args:
            image_path: Path to a single image file or a directory of images.

        Returns:
            - An AlliumCepaResult for a single image.
            - A list of AlliumCepaResult objects for a directory of images.
        """
        input_path = Path(image_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        if input_path.is_file():
            return self._predict_single_image(input_path)
        elif input_path.is_dir():
            image_paths = list(input_path.glob("*"))

            for path in image_paths:
                if path.is_file() and path.suffix.lower() not in self.config.valid_image_extensions:
                    raise ValueError(f"Directory contains a non-image file: {path}")

            return self._predict_dir_image(image_paths)
        else:
            raise ValueError(f"Input path is not a valid file or directory: {input_path}")
