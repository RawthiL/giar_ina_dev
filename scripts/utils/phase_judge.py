"""
Loader for the trained 4-class mitotic-phase classifier, used as:

  * the *judge* for phase-conditioning diagnostics and the `phase_consistency` metric
  * the *feature extractor* for domain KID (backbone penultimate features)

Handles both checkpoint flavours written by the training pipeline:
`classifier.pt` (plain `BackboneWithHead`) and `classifier_calibrated.pt`
(`CalibratedClassifier`, whose state dict is prefixed `base_model.` and which carries
a per-class temperature vector).

Note this deliberately rebuilds the model via `build_model()` from the experiment config
rather than hand-assembling a timm head — the saved state dict uses `backbone.*` / `head.*`
keys, which only `BackboneWithHead` matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from allium_cepa_classifier.config.experiment_config import ExperimentConfig
from allium_cepa_classifier.training.model_builder import BackboneWithHead, build_model

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass
class PhaseJudge:
    model: BackboneWithHead
    transform: transforms.Compose
    class_names: list[str]
    temperature: torch.Tensor | None
    device: str

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.class_names)}

    @torch.no_grad()
    def probs(self, paths: list[Path], batch_size: int = 32) -> np.ndarray:
        """Calibrated class probabilities, shape (N, num_classes)."""
        out: list[np.ndarray] = []
        for i in range(0, len(paths), batch_size):
            batch = self._batch(paths[i : i + batch_size])
            logits = self.model(batch)
            if self.temperature is not None:
                logits = logits / self.temperature
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, len(self.class_names)))

    @torch.no_grad()
    def features(self, paths: list[Path], batch_size: int = 32) -> np.ndarray:
        """Backbone penultimate features, shape (N, in_features) — the KID feature space."""
        out: list[np.ndarray] = []
        for i in range(0, len(paths), batch_size):
            batch = self._batch(paths[i : i + batch_size])
            out.append(self.model.backbone(batch).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, 0))

    def _batch(self, paths: list[Path]) -> torch.Tensor:
        tensors = [self.transform(Image.open(p).convert("RGB")) for p in paths]
        return torch.stack(tensors).to(self.device)


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def load_phase_judge(
    run_dir: Path,
    device: str | None = None,
    prefer_calibrated: bool = True,
) -> PhaseJudge:
    """Load the phase classifier from an experiment run dir (e.g. experiments/phase_classifier/efficientnet_b2)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    weights_dir = run_dir / "weights"
    candidates = ["classifier_calibrated.pt", "classifier.pt"]
    if not prefer_calibrated:
        candidates.reverse()
    ckpt_path = next((weights_dir / c for c in candidates if (weights_dir / c).exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No classifier checkpoint in {weights_dir}. Train the phase classifier first."
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ExperimentConfig.from_yaml(run_dir / "config.yaml")

    num_classes = ckpt["num_classes"]
    model = build_model(cfg.model, num_classes=num_classes)

    state_dict = ckpt["model_state_dict"]
    temperature: torch.Tensor | None = None
    if any(k.startswith("base_model.") for k in state_dict):
        state_dict = {
            k[len("base_model.") :]: v for k, v in state_dict.items() if k.startswith("base_model.")
        }
        temperature = torch.tensor(ckpt["temperature"], dtype=torch.float32, device=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize(tuple(ckpt["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=ckpt["normalize_mean"], std=ckpt["normalize_std"]),
        ]
    )
    class_names = [k for k, _ in sorted(ckpt["class_to_idx"].items(), key=lambda kv: kv[1])]

    return PhaseJudge(
        model=model,
        transform=transform,
        class_names=class_names,
        temperature=temperature,
        device=device,
    )
