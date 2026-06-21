"""
Build a kohya-compatible LoRA dataset from the VAE tagged crops.

Collects original (non-aug) images from:
  datasets/crops/vae/train/tagged/{phase}/
  datasets/crops/vae/val/tagged/{phase}/
  datasets/crops/vae/test/{phase}/          ← no 'tagged/' layer here

For each image: copies as RGB, applies augment() once (also RGB), writes a
sibling .txt caption: "micrograph of allium cepa root tip mitotic cell in {phase} phase"

Output layout (kohya DreamBooth):
  datasets/crops/lora/img/10_allium mitosis/
      <stem>.png, <stem>.txt, <stem>_aug.png, <stem>_aug.txt, ...

Usage:
    uv run python scripts/utils/lora_dataset.py
    uv run python scripts/utils/lora_dataset.py --vae-dir datasets/crops/vae
                                                  --out datasets/crops/lora
                                                  --repeats 10
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

CAPTION_TEMPLATE = "micrograph of allium cepa root tip mitotic cell in {phase} phase"


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


def source_dirs(vae_dir: Path) -> list[tuple[Path, str, str]]:
    """Return (dir, phase, split_prefix) tuples for all tagged phase dirs across splits."""
    pairs = []
    for phase in PHASES:
        for split_name, split_path in [
            ("train", vae_dir / "train" / "tagged" / phase),
            ("val", vae_dir / "val" / "tagged" / phase),
            ("test", vae_dir / "test" / phase),  # no 'tagged/' layer in test
        ]:
            if split_path.exists():
                pairs.append((split_path, phase, split_name))
    return pairs


def collect_originals(phase_dir: Path) -> list[Path]:
    return [p for p in phase_dir.iterdir() if p.suffix.lower() in IMG_EXTS and "_aug" not in p.stem]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-dir", type=Path, default=Path("datasets/crops/vae"))
    parser.add_argument("--out", type=Path, default=Path("datasets/crops/lora"))
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    concept_dir = args.out / "img" / f"{args.repeats}_allium mitosis"
    concept_dir.mkdir(parents=True, exist_ok=True)

    total_orig = total_aug = 0
    for phase_dir, phase, split_prefix in source_dirs(args.vae_dir):
        caption = CAPTION_TEMPLATE.format(phase=phase)
        for src in collect_originals(phase_dir):
            img = Image.open(src).convert("RGB")
            # Prefix with split name to avoid collisions between splits
            stem = f"{split_prefix}_{src.stem}"
            # original
            img.save(concept_dir / f"{stem}{src.suffix}")
            (concept_dir / f"{stem}.txt").write_text(caption)
            total_orig += 1
            # augmented
            aug_stem = f"{stem}_aug"
            augment(img).save(concept_dir / f"{aug_stem}{src.suffix}")
            (concept_dir / f"{aug_stem}.txt").write_text(caption)
            total_aug += 1

    print(f"LoRA dataset: {total_orig} originals + {total_aug} augmented → {concept_dir}")


if __name__ == "__main__":
    main()
