"""
Augments VAE training images in-place (train split only).

Uses mild, structure-preserving transforms suitable for reconstruction tasks:
flips, a small (±5°) mirror-padded rotation, brightness/contrast. Rotation corners are
filled by reflecting the edge (not a constant value) so no artificial letterbox is learned.
Elastic transforms and noise are intentionally omitted — they corrupt pixel structure and
inflate reconstruction loss.

Usage:
    uv run python scripts/utils/augment_vae_crops.py
    uv run python scripts/utils/augment_vae_crops.py --ratio 1.0
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def _rotate_reflect(image: Image.Image, angle: float) -> Image.Image:
    """Rotate about the center, filling new corners with mirrored edge pixels.

    Uses numpy reflect padding instead of a flat ``fillcolor``, so augmented crops never
    gain an artificial constant-value letterbox. ``np.pad(mode="reflect")`` re-tiles for
    any pad size, so small crops are safe. Works for both grayscale (L) and RGB images.
    """
    if abs(angle) < 0.5:
        return image
    w, h = image.size
    pad = int(math.ceil(math.hypot(w / 2, h / 2) - min(w, h) / 2)) + 2
    arr = np.asarray(image)
    pad_width = [(pad, pad), (pad, pad)] + ([(0, 0)] if arr.ndim == 3 else [])
    arr = np.pad(arr, pad_width, mode="reflect")
    rotated = Image.fromarray(arr).rotate(angle, resample=Image.BILINEAR, expand=False)
    return rotated.crop((pad, pad, pad + w, pad + h))


def augment(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if random.random() < 0.5:
        image = ImageOps.flip(image)
    angle = random.uniform(-5.0, 5.0)
    image = _rotate_reflect(image, angle)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    return image


def augment_dir(src: Path, ratio: float) -> int:
    originals = [p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS and "_aug" not in p.stem]
    sample = random.sample(originals, int(len(originals) * ratio))
    for p in sample:
        aug = augment(Image.open(p).convert("L"))
        aug.save(src / f"{p.stem}_aug{p.suffix}")
    return len(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment VAE training images in-place.")
    parser.add_argument(
        "--vae-dir",
        type=Path,
        default=Path("datasets/crops/vae"),
        help="Root VAE dataset directory (default: datasets/crops/vae)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Fraction of originals to augment per directory (default: 1.0)",
    )
    args = parser.parse_args()

    train_dir = args.vae_dir / "train"
    total = 0

    tagged = train_dir / "tagged"
    if tagged.exists():
        for phase_dir in sorted(p for p in tagged.iterdir() if p.is_dir()):
            n = augment_dir(phase_dir, args.ratio)
            print(f"  tagged/{phase_dir.name}: +{n} augmented images")
            total += n

    untagged = train_dir / "untagged"
    if untagged.exists():
        n = augment_dir(untagged, args.ratio)
        print(f"  untagged: +{n} augmented images")
        total += n

    print(f"\nTotal augmented: {total}")


if __name__ == "__main__":
    main()
