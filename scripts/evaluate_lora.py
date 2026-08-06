"""
Extensible LoRA evaluator that writes metrics.json into the experiment directory.

Metrics:
  loss             — final/min/avg loss from kohya TensorBoard event files
  fid              — Fréchet Inception Distance between generated crops and real
                     mitotic test crops (requires generate_lora_samples to have run first)
  classifier_judge — run the binary mitosis classifier on generated crops; reports
                     mitosis_rate and mean_mitosis_prob (requires classifier weights)

Usage:
    uv run python scripts/evaluate_lora.py --config experiments/lora/sd15_rank16/config.yaml
    uv run python scripts/evaluate_lora.py --config ... --metrics loss fid classifier_judge
"""

import argparse
import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from scipy.linalg import sqrtm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3

from allium_cepa_classifier.config.base_config import find_project_root
from allium_cepa_classifier.config.lora_config import LoRAExperimentConfig

_ROOT = find_project_root()

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

MetricFn = Callable[[LoRAExperimentConfig, Path], dict[str, Any]]
METRICS: dict[str, MetricFn] = {}


def register(name: str) -> Callable[[MetricFn], MetricFn]:
    def decorator(fn: MetricFn) -> MetricFn:
        METRICS[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_tb_event_dir(log_dir: Path) -> Path | None:
    """Kohya writes events into log_dir/<timestamp>/network_train/. Find the latest one."""
    for depth in range(4):
        pattern = "/".join(["*"] * depth) + ("/" if depth else "") + "events.out.*"
        hits = sorted(log_dir.glob(pattern))
        if hits:
            return hits[-1].parent
    return None


def _load_images(paths: list[Path], transform) -> torch.Tensor:
    tensors = [transform(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(tensors)


def _extract_inception_features(paths: list[Path], device: str, batch_size: int = 32) -> np.ndarray:
    """Extract 2048-dim InceptionV3 pool features from a list of image paths."""
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.fc = nn.Identity()
    model.eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]
    )

    all_features: list[np.ndarray] = []
    for i in range(0, len(paths), batch_size):
        batch = _load_images(paths[i : i + batch_size], transform).to(device)
        with torch.no_grad():
            features = model(batch)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def _frechet_distance(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
    """Fréchet distance between two sets of feature vectors."""
    mu_a, mu_b = feat_a.mean(axis=0), feat_b.mean(axis=0)
    sigma_a = np.cov(feat_a, rowvar=False)
    sigma_b = np.cov(feat_b, rowvar=False)

    diff = mu_a - mu_b
    # Matrix square root — may produce tiny imaginary parts due to floating point
    covmean = sqrtm(sigma_a @ sigma_b)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma_a + sigma_b - 2.0 * covmean))


def _load_classifier(
    weights_path: Path, device: str
) -> tuple[nn.Module, tuple[int, int], list[float], list[float], torch.Tensor | None]:
    """Load the binary mitosis classifier. Returns (model, image_size, mean, std, temperature)."""
    ckpt = torch.load(weights_path, map_location=device)

    timm_name = ckpt.get("timm_model_name", "efficientnet_b2")
    model = timm.create_model(timm_name, pretrained=False)
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

    temperature: torch.Tensor | None = None
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("base_model.") for k in state_dict):
        base_state = {
            k[len("base_model.") :]: v for k, v in state_dict.items() if k.startswith("base_model.")
        }
        temperature = torch.tensor(ckpt["temperature"], dtype=torch.float32).to(device)
        model.load_state_dict(base_state)
    else:
        model.load_state_dict(state_dict)

    image_size = tuple(ckpt.get("image_size", (224, 224)))
    mean = ckpt.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = ckpt.get("imagenet_std", [0.229, 0.224, 0.225])

    model.to(device).eval()
    return model, image_size, mean, std, temperature


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------


@register("loss")
def loss_from_tb(cfg: LoRAExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """Read final/min/avg loss from the kohya TensorBoard event files in <run_dir>/logs/."""
    log_dir = run_dir / "logs"
    if not log_dir.exists():
        print(f"[evaluate_lora] WARNING: no logs dir at {log_dir} — loss metrics unavailable.")
        return {"final_loss": None, "min_loss": None, "avg_loss": None}

    event_dir = _find_tb_event_dir(log_dir)
    if event_dir is None:
        print(f"[evaluate_lora] WARNING: no TensorBoard event files found under {log_dir}.")
        return {"final_loss": None, "min_loss": None, "avg_loss": None}

    ea = EventAccumulator(str(event_dir))
    ea.Reload()

    available = ea.Tags().get("scalars", [])
    loss_tags = [t for t in available if "loss" in t.lower()]
    if not loss_tags:
        print(f"[evaluate_lora] WARNING: no loss scalar tags found. Available: {available}")
        return {"final_loss": None, "min_loss": None, "avg_loss": None}

    tag = next((t for t in loss_tags if "current" in t), loss_tags[0])
    events = ea.Scalars(tag)
    if not events:
        return {"final_loss": None, "min_loss": None, "avg_loss": None}

    values = [e.value for e in events]
    return {
        "final_loss": round(values[-1], 6),
        "min_loss": round(min(values), 6),
        "avg_loss": round(sum(values) / len(values), 6),
        "loss_tag": tag,
    }


@register("fid")
def fid_from_generated(cfg: LoRAExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """
    Fréchet Inception Distance between generated crops and real mitotic test crops.

    NOTE: FID requires a large number of samples (ideally ≥2048) for a reliable estimate.
    With the default 52 generated images, the covariance matrix is severely under-determined
    (2048 features, 52 samples), so the absolute FID value is very noisy. Use it for
    relative comparisons between experiments with identical sample counts, not as an
    absolute quality indicator.
    """
    generated_dir = run_dir / "plots" / "generated"
    if not generated_dir.exists():
        print(
            "[evaluate_lora] WARNING: plots/generated/ not found — run generate_lora_samples first."
        )
        return {"fid": None, "fid_n_generated": None, "fid_n_real": None}

    generated_paths = sorted(generated_dir.glob("**/*.png"))
    if not generated_paths:
        print("[evaluate_lora] WARNING: no generated images found in plots/generated/.")
        return {"fid": None, "fid_n_generated": 0, "fid_n_real": None}

    real_dir = _ROOT / "datasets/crops/binary_classifier/test/mitosis"
    real_paths = sorted(real_dir.glob("*.png")) + sorted(real_dir.glob("*.jpg"))
    if not real_paths:
        print(
            f"[evaluate_lora] WARNING: no real mitosis crops found at {real_dir} — FID unavailable."
        )
        return {"fid": None, "fid_n_generated": len(generated_paths), "fid_n_real": 0}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[evaluate_lora] FID: extracting features from {len(generated_paths)} generated + {len(real_paths)} real images ..."
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat_gen = _extract_inception_features(generated_paths, device)
        feat_real = _extract_inception_features(real_paths, device)

    fid_value = _frechet_distance(feat_gen, feat_real)
    n_gen = len(generated_paths)
    noisy = n_gen < 2048

    return {
        "fid": round(fid_value, 3),
        "fid_n_generated": n_gen,
        "fid_n_real": len(real_paths),
        "fid_noisy_estimate": noisy,
    }


@register("classifier_judge")
def classifier_judge(cfg: LoRAExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """
    Run the binary mitosis classifier on generated crops.

    Reports:
      mitosis_rate      — fraction of generated images classified as mitotic (threshold 0.5)
      mean_mitosis_prob — mean predicted probability of being mitotic
      judge_n_images    — number of images evaluated

    Requires classifier weights at src/allium_cepa_classifier/weights/. Skips gracefully
    if weights are absent.
    """
    generated_dir = run_dir / "plots" / "generated"
    if not generated_dir.exists():
        print(
            "[evaluate_lora] WARNING: plots/generated/ not found — run generate_lora_samples first."
        )
        return {"mitosis_rate": None, "mean_mitosis_prob": None, "judge_n_images": 0}

    generated_paths = sorted(generated_dir.glob("**/*.png"))
    if not generated_paths:
        return {"mitosis_rate": None, "mean_mitosis_prob": None, "judge_n_images": 0}

    weights_dir = _ROOT / "src" / "allium_cepa_classifier" / "weights"
    weights_path = weights_dir / "classifier_calibrated.pt"
    if not weights_path.exists():
        weights_path = weights_dir / "classifier.pt"
    if not weights_path.exists():
        print(
            f"[evaluate_lora] WARNING: no classifier weights found at {weights_dir} — classifier_judge unavailable."
        )
        return {"mitosis_rate": None, "mean_mitosis_prob": None, "judge_n_images": 0}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[evaluate_lora] classifier_judge: scoring {len(generated_paths)} images ...")
    model, image_size, mean, std, temperature = _load_classifier(weights_path, device)

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    mitosis_probs: list[float] = []
    batch_size = 32
    for i in range(0, len(generated_paths), batch_size):
        batch_paths = generated_paths[i : i + batch_size]
        batch = _load_images(batch_paths, transform).to(device)
        with torch.no_grad():
            logits = model(batch)
            if temperature is not None:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=1)
        # class index 0 = mitosis (matches AlliumCepaModel convention)
        mitosis_probs.extend(probs[:, 0].cpu().tolist())

    mitosis_rate = sum(p > 0.5 for p in mitosis_probs) / len(mitosis_probs)
    mean_prob = sum(mitosis_probs) / len(mitosis_probs)

    return {
        "mitosis_rate": round(mitosis_rate, 4),
        "mean_mitosis_prob": round(mean_prob, 4),
        "judge_n_images": len(mitosis_probs),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained LoRA experiment.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(METRICS),
        choices=list(METRICS),
        help="Which metrics to compute (default: all).",
    )
    args = parser.parse_args()

    if not args.config.exists():
        sys.exit(f"Config not found: {args.config}")

    cfg = LoRAExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent

    out: dict[str, Any] = {"experiment_name": cfg.experiment_name}
    for name in args.metrics:
        out.update(METRICS[name](cfg, run_dir))

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"[evaluate_lora] wrote {metrics_path}")
    for k, v in sorted(out.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
