from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from allium_cepa_classifier.config.experiment_config import ExperimentConfig
from allium_cepa_classifier.training.model_builder import build_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)


class CalibratedClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, temperature: np.ndarray):
        super().__init__()
        self.base_model = base_model
        self.temperature = nn.Parameter(
            torch.tensor(temperature, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        return torch.softmax(logits / self.temperature, dim=1)

    def get_temperature(self) -> np.ndarray:
        return self.temperature.cpu().numpy()


class _SoftmaxWrapper(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.m(x), dim=1)


def _get_probs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = model(images.to(device)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = preds == labels
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece_val += mask.sum() / len(labels) * abs(acc - conf)
    return float(ece_val)


def run_calibration(run_dir: Path) -> dict:
    """
    Loads run_dir/weights/classifier.pt and run_dir/used_config.yaml.
    Writes classifier_calibrated.pt and reliability_diagram.png.
    Updates metrics.json with ece_before and ece_after.
    Returns calibration metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    ckpt_path = run_dir / "weights" / "classifier.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ExperimentConfig.from_yaml(run_dir / "config.yaml")
    model = build_model(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Loaded model from {ckpt_path}")

    image_size = ckpt["image_size"]
    normalize_mean = ckpt["normalize_mean"]
    normalize_std = ckpt["normalize_std"]

    eval_tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )

    val_ds = datasets.ImageFolder(
        cfg.data.binary_classifier_crops_dir / "validation",
        transform=eval_tfm,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    log.info(f"Validation samples: {len(val_ds)}")

    all_logits, all_labels_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels_list.append(labels.numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels_list)

    num_classes = all_logits.shape[1]

    def vector_scale_loss(temp_vector: np.ndarray) -> float:
        scaled = all_logits / temp_vector
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_ = np.exp(shifted)
        probs = exp_ / exp_.sum(axis=1, keepdims=True)
        return log_loss(all_labels, probs)

    result = minimize(
        vector_scale_loss,
        x0=np.ones(num_classes),
        bounds=[(0.01, 10.0)] * num_classes,
        method="L-BFGS-B",
    )
    optimal_T = result.x
    log.info(f"Optimal temperature vector: {optimal_T}")

    calibrated_model = CalibratedClassifier(model, optimal_T).to(device)
    calibrated_model.eval()

    orig_probs, true_labels = _get_probs(
        _SoftmaxWrapper(model).to(device).eval(), val_loader, device
    )
    cal_probs, _ = _get_probs(calibrated_model, val_loader, device)

    ece_before = _ece(orig_probs, true_labels)
    ece_after = _ece(cal_probs, true_labels)
    log.info(f"ECE before: {ece_before:.4f}  after: {ece_after:.4f}")

    class_names = [k for k, v in sorted(ckpt["class_to_idx"].items(), key=lambda x: x[1])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cls_idx, cls_name in enumerate(class_names):
        ax = axes[cls_idx]
        for probs, label in [(orig_probs, "original"), (cal_probs, "calibrated")]:
            binary_true = (true_labels == cls_idx).astype(int)
            cls_probs = probs[:, cls_idx]
            frac_pos, mean_pred = calibration_curve(binary_true, cls_probs, n_bins=10)
            ax.plot(mean_pred, frac_pos, marker="o", label=label)
        ax.plot([0, 1], [0, 1], "k--", label="perfect")
        ax.set_title(f"Reliability diagram — {cls_name}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "plots" / "reliability_diagram.png", dpi=150)
    plt.close(fig)

    cal_ckpt_path = run_dir / "weights" / "classifier_calibrated.pt"
    torch.save(
        {
            "model_state_dict": calibrated_model.state_dict(),
            "temperature": optimal_T.tolist(),
            "num_classes": num_classes,
            "image_size": image_size,
            "class_to_idx": ckpt["class_to_idx"],
            "normalize_mean": normalize_mean,
            "normalize_std": normalize_std,
        },
        cal_ckpt_path,
    )
    log.info(f"Saved calibrated checkpoint: {cal_ckpt_path}")

    cal_metrics = {
        "ece_before": ece_before,
        "ece_after": ece_after,
        "temperature": optimal_T.tolist(),
    }
    (run_dir / "calibration_metrics.json").write_text(json.dumps(cal_metrics, indent=2))

    return cal_metrics
