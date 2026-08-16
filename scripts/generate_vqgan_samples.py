"""
Generate sample reconstructions from a trained VQGAN checkpoint.

Loads the trained VQModel, runs it over test images from each mitotic phase,
and saves a grid (originals vs reconstructions) to
`experiments/vqgan/<name>/plots/vqgan_samples.png`.

VQGAN is a standalone compressor; this script is not part of the
AlliumCepaModel inference pipeline.

Usage:
    uv run python scripts/generate_vqgan_samples.py --config experiments/vqgan/vqgan_baseline/config.yaml
    uv run python scripts/generate_vqgan_samples.py --config ... --images img1.png img2.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffusers import VQModel
from PIL import Image
from torchvision import transforms

from allium_cepa_classifier.config.vqgan_config import VQGANExperimentConfig

PHASES = ["anaphase", "metaphase", "prophase", "telophase"]
SAMPLES_PER_PHASE = 2


def resolve_test_images(cfg: VQGANExperimentConfig, paths: list[str]) -> list[tuple[Path, str]]:
    """Resolve test images per phase, or use explicit paths.

    Returns list of (image_path, label) tuples.
    """
    if paths:
        return [(Path(p), p) for p in paths]

    test_dir = cfg.data.dataset_dir.parent / "test"
    images: list[tuple[Path, str]] = []
    for phase in PHASES:
        phase_dir = test_dir / phase
        if not phase_dir.exists():
            continue
        pngs = sorted(phase_dir.glob("*.png"))
        selected = pngs[:SAMPLES_PER_PHASE]
        images.extend((p, phase) for p in selected)
    if not images:
        raise FileNotFoundError(f"No test images found in {test_dir}/{PHASES}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="Explicit image paths (default: 2 per phase from test set).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = VQGANExperimentConfig.from_yaml(args.config)
    run_dir = args.config.resolve().parent
    weights_dir = run_dir / "weights" / "vqmodel"
    if not weights_dir.exists():
        raise FileNotFoundError(f"VQModel weights not found: {weights_dir}. Run training first.")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    image_specs = resolve_test_images(cfg, args.images)

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    torch.manual_seed(args.seed)

    print(f"Loading VQModel from {weights_dir} on {args.device} ({dtype}) ...")
    model = VQModel.from_pretrained(weights_dir, torch_dtype=dtype).to(args.device)
    model.eval()

    resolution = cfg.model.resolution
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )

    originals, reconstructions, labels = [], [], []
    with torch.no_grad():
        for img_path, label in image_specs:
            print(f"  {img_path.name}  ({label})")
            pil_img = Image.open(img_path).convert("RGB")
            tensor = transform(pil_img).unsqueeze(0).to(args.device, dtype=dtype)
            reconstructed = model(tensor).sample
            originals.append(tensor)
            reconstructions.append(reconstructed)
            labels.append(label)

    n = len(originals)
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(4 * n, 8), squeeze=False)
    fig.suptitle("VQGAN Reconstructions", fontsize=16)

    for i in range(n):
        orig = originals[i].squeeze(0).cpu().float().clamp(0, 1)
        recon = reconstructions[i].squeeze(0).cpu().float().clamp(0, 1)

        axes[0, i].imshow(orig.permute(1, 2, 0))
        axes[0, i].set_title(f"Original\n({labels[i]})", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(recon.permute(1, 2, 0))
        axes[1, i].set_title("Reconstruction", fontsize=10)
        axes[1, i].axis("off")

    fig.tight_layout()
    out_path = plots_dir / "vqgan_samples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
