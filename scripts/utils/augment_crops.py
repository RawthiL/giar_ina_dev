"""
Augments mitosis crops in-place (train split only).

Usage:
    uv run python scripts/utils/augment_crops.py
    uv run python scripts/utils/augment_crops.py --config path/to/config.yaml --ratio 1.0

Requires: opencv-python-headless (already in deps), scipy (add to dev deps if missing).
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import gaussian_filter, map_coordinates

from allium_cepa_classifier.config.training_config import TrainingConfig

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def add_noise(image: Image.Image) -> Image.Image:
    arr = np.array(image)
    noise = np.random.normal(10, 25, arr.shape).astype(np.int32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def elastic_transform(image: Image.Image) -> Image.Image:
    arr = np.array(image)
    alpha = arr.shape[1] * 2
    sigma = arr.shape[1] * 0.08
    alpha_affine = arr.shape[1] * 0.08
    shape = arr.shape
    rng = np.random.RandomState(None)

    center = np.float32(shape[:2]) // 2
    sq = min(shape[:2]) // 3
    pts1 = np.float32([center + sq, [center[0] + sq, center[1] - sq], center - sq])
    pts2 = pts1 + rng.uniform(-alpha_affine, alpha_affine, pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    arr = cv2.warpAffine(arr, M, shape[:2][::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rng.rand(*shape) * 2 - 1), sigma) * alpha
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )
    return Image.fromarray(
        map_coordinates(arr, indices, order=1, mode="reflect").reshape(shape).astype(np.uint8)
    )


def augment(image: Image.Image) -> Image.Image:
    img = ImageOps.mirror(image)
    img = ImageOps.flip(img)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))
    if random.random() < 0.5:
        img = add_noise(img)
    return elastic_transform(img)


def augment_dir(src_dir: Path, ratio: float) -> int:
    originals = [p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    sample = random.sample(originals, int(len(originals) * ratio))
    for p in sample:
        aug = augment(Image.open(p))
        aug.save(src_dir / f"{p.stem}_aug{p.suffix}")
    return len(sample)


def main():
    parser = argparse.ArgumentParser(description="Augment mitosis crops in-place (train split).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help="Fraction of originals to augment (default: 1.0)",
    )
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config, key="training") if args.config else TrainingConfig()
    src_dir = cfg.binary_classifier_crops_dir / "train" / "mitosis"

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {src_dir}")

    print(f"Augmenting: {src_dir}")
    n = augment_dir(src_dir, args.ratio)
    print(f"Generated {n} augmented images.")


if __name__ == "__main__":
    main()
