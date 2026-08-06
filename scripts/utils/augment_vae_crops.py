"""
Augments VAE training images in-place (train split only).

Uses mild, structure-preserving transforms suitable for reconstruction tasks:
flips, small rotation, brightness/contrast. Elastic transforms and noise are
intentionally omitted — they corrupt pixel structure and inflate reconstruction loss.

Usage:
    uv run python scripts/utils/augment_vae_crops.py
    uv run python scripts/utils/augment_vae_crops.py --ratio 1.0
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def augment(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if random.random() < 0.5:
        image = ImageOps.flip(image)
    angle = random.uniform(-15.0, 15.0)
    image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=128)
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
