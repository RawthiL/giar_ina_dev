"""
Build a kohya-compatible LoRA dataset from the VAE tagged crops.

Collects original (non-aug) images from:
  datasets/crops/vae/train/tagged/{phase}/
  datasets/crops/vae/val/tagged/{phase}/
  datasets/crops/vae/test/{phase}/          ← no 'tagged/' layer here

For each image: copies as RGB, applies augment() N times (--copies), writes a
sibling .txt caption: "micrograph of allium cepa root tip mitotic cell in {phase} phase"

Output layout (kohya DreamBooth):
  datasets/crops/lora/<version>/img/10_allium mitosis/
      <stem>.png, <stem>.txt
      <stem>_aug01.png, <stem>_aug01.txt   ← first aug copy (if copies >= 1)
      <stem>_aug02.png, <stem>_aug02.txt   ← second aug copy (if copies >= 2)
      ...

Named version defaults (applied when --copies / --aug-strength are omitted):
  no_aug    → --copies 0  --aug-strength mild
  baseline  → --copies 1  --aug-strength mild
  aug2x     → --copies 2  --aug-strength mild
  heavy_aug → --copies 1  --aug-strength heavy

Usage:
    uv run python scripts/utils/lora_dataset.py --version baseline
    uv run python scripts/utils/lora_dataset.py --version no_aug
    uv run python scripts/utils/lora_dataset.py --vae-dir datasets/crops/vae
                                                  --out datasets/crops/lora
                                                  --version heavy_aug
                                                  --repeats 10
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

CAPTION_TEMPLATE = "micrograph of allium cepa root tip mitotic cell in {phase} phase"

# Per-version defaults: version_name → (copies, aug_strength)
_VERSION_DEFAULTS: dict[str, tuple[int, str]] = {
    "no_aug": (0, "mild"),
    "baseline": (1, "mild"),
    "aug2x": (2, "mild"),
    "heavy_aug": (1, "heavy"),
}


def augment_mild(image: Image.Image) -> Image.Image:
    """Light augmentation: ±15° rotation, moderate brightness/contrast jitter."""
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if random.random() < 0.5:
        image = ImageOps.flip(image)
    angle = random.uniform(-15.0, 15.0)
    image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=128)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    return image


def augment_heavy(image: Image.Image) -> Image.Image:
    """Strong augmentation: ±45° rotation, wide brightness/contrast/sharpness jitter."""
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if random.random() < 0.5:
        image = ImageOps.flip(image)
    angle = random.uniform(-45.0, 45.0)
    image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=128)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.5, 1.5))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.5, 1.5))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.5, 1.5))
    # Simulate staining variation: occasional grayscale→RGB
    if random.random() < 0.1:
        image = ImageOps.grayscale(image).convert("RGB")
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
    parser.add_argument(
        "--version", default="baseline", help="Named dataset version subfolder under --out."
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--copies",
        type=int,
        default=None,
        help="Augmented copies per original (0 = no augmentation). Defaults from --version.",
    )
    parser.add_argument(
        "--aug-strength",
        choices=["mild", "heavy"],
        default=None,
        help="Augmentation intensity. Defaults from --version.",
    )
    args = parser.parse_args()

    # Apply per-version defaults for flags not explicitly set
    ver_copies, ver_strength = _VERSION_DEFAULTS.get(args.version, (1, "mild"))
    copies = args.copies if args.copies is not None else ver_copies
    aug_strength = args.aug_strength if args.aug_strength is not None else ver_strength
    augment_fn = augment_mild if aug_strength == "mild" else augment_heavy

    concept_dir = args.out / args.version / "img" / f"{args.repeats}_allium mitosis"
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
            # augmented copies
            for i in range(copies):
                aug_stem = f"{stem}_aug{i + 1:02d}"
                augment_fn(img).save(concept_dir / f"{aug_stem}{src.suffix}")
                (concept_dir / f"{aug_stem}.txt").write_text(caption)
                total_aug += 1

    print(
        f"LoRA dataset [{args.version}] (copies={copies}, strength={aug_strength}): "
        f"{total_orig} originals + {total_aug} augmented → {concept_dir}"
    )


if __name__ == "__main__":
    main()
