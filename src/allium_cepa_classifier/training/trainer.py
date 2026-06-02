from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from allium_cepa_classifier.config.experiment_config import ExperimentConfig
from allium_cepa_classifier.training.model_builder import build_model

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except Exception:
    _SummaryWriter = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)


def _build_transforms(cfg: ExperimentConfig):
    aug = set(cfg.training.augmentation)
    train_ops: list = [transforms.Resize(cfg.data.image_size)]
    if "hflip" in aug:
        train_ops.append(transforms.RandomHorizontalFlip())
    if "vflip" in aug:
        train_ops.append(transforms.RandomVerticalFlip())
    if "color_jitter" in aug:
        train_ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.normalize_mean, std=cfg.data.normalize_std),
    ]
    eval_ops = [
        transforms.Resize(cfg.data.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.normalize_mean, std=cfg.data.normalize_std),
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def _build_loaders(cfg: ExperimentConfig, train_tfm, eval_tfm):
    crops_dir = cfg.data.binary_classifier_crops_dir
    train_ds = datasets.ImageFolder(crops_dir / "train", transform=train_tfm)
    val_ds = datasets.ImageFolder(crops_dir / "validation", transform=eval_tfm)
    test_ds = datasets.ImageFolder(crops_dir / "test", transform=eval_tfm)

    kw = {"batch_size": cfg.data.batch_size, "num_workers": 4, "pin_memory": True}
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        train_ds,
    )


def _compute_class_weights(
    train_ds, multipliers: dict[str, float], device: torch.device
) -> torch.Tensor:
    labels = [lbl for _, lbl in train_ds.samples]
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(labels))
    for cls_name, idx in train_ds.class_to_idx.items():
        if cls_name in multipliers:
            weights[idx] *= multipliers[cls_name]
    return torch.tensor(weights, dtype=torch.float32).to(device)


def run_training(cfg: ExperimentConfig, run_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    torch.manual_seed(cfg.data.seed)
    np.random.seed(cfg.data.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_tfm, eval_tfm = _build_transforms(cfg)
    train_loader, val_loader, test_loader, train_ds = _build_loaders(cfg, train_tfm, eval_tfm)
    log.info(f"Class mapping: {train_ds.class_to_idx}")

    weights_tensor = _compute_class_weights(train_ds, cfg.training.class_weight_multipliers, device)
    log.info(f"Class weights: mitosis={weights_tensor[0]:.3f}, no_mitosis={weights_tensor[1]:.3f}")

    model = build_model(cfg.model).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable params: {trainable:,} / {total:,}")

    sched_cfg = cfg.training.lr_scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.training.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg.factor,
        patience=sched_cfg.patience,
        min_lr=sched_cfg.min_lr,
        threshold=0.01,
    )
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    best_val_loss = float("inf")
    best_state: dict | None = None
    patience_counter = 0
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    writer = None
    if cfg.training.tensorboard and _SummaryWriter is not None:
        try:
            writer = _SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        except Exception as e:
            log.warning(f"TensorBoard writer failed to open, logging disabled: {e}")

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += images.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)

        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch {epoch:02d}/{cfg.training.epochs} | "
            f"time={time.time() - t0:.1f}s | "
            f"train_loss={avg_train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={avg_val_loss:.4f} acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | patience={patience_counter}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LR", current_lr, epoch)

        if patience_counter >= cfg.training.early_stopping_patience:
            log.info(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    log.info(f"Restored best weights (val_loss={best_val_loss:.4f})")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.numpy())

    test_acc = sum(p == t for p, t in zip(y_pred, y_true, strict=True)) / len(y_true)
    log.info(f"Test accuracy: {test_acc:.4f}")

    if writer is not None:
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        writer.add_hparams(
            {"arch": cfg.model.arch, "lr": cfg.training.lr, "batch_size": cfg.data.batch_size},
            {"hparam/val_loss": best_val_loss, "hparam/test_acc": test_acc},
        )
        writer.close()

    torch.save(
        {
            "model_state_dict": best_state,
            "timm_model_name": cfg.model.arch,
            "num_classes": 2,
            "image_size": cfg.data.image_size,
            "class_to_idx": train_ds.class_to_idx,
            "normalize_mean": cfg.data.normalize_mean,
            "normalize_std": cfg.data.normalize_std,
        },
        run_dir / "weights" / "classifier.pt",
    )

    class_names = list(train_ds.class_to_idx.keys())
    metrics = {
        "train_acc": history["train_acc"][-1],
        "val_acc": history["val_acc"][-1],
        "test_acc": test_acc,
        "best_val_loss": best_val_loss,
        "epochs_run": len(history["train_loss"]),
        "history": history,
    }
    save_artifacts(history, y_true, y_pred, class_names, run_dir, metrics)

    return metrics


def save_artifacts(
    history: dict,
    y_true: list,
    y_pred: list,
    class_names: list[str],
    run_dir: Path,
    metrics: dict,
) -> None:
    epochs_range = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_range, history["train_acc"], label="Training Accuracy")
    axes[0].plot(epochs_range, history["val_acc"], label="Validation Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Training and Validation Accuracy")
    axes[1].plot(epochs_range, history["train_loss"], label="Training Loss")
    axes[1].plot(epochs_range, history["val_loss"], label="Validation Loss")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Training and Validation Loss")
    plt.tight_layout()
    plt.savefig(run_dir / "plots" / "training_curves.png", dpi=150)
    plt.close(fig)

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Normalized Confusion Matrix (Recall per row)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(run_dir / "plots" / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    report = classification_report(y_true, y_pred, target_names=class_names)
    (run_dir / "plots" / "classification_report.txt").write_text(report)

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
