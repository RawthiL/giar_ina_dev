"""
Build a kohya-compatible LoRA dataset from the 256px cell-crop tagged sets.

Collects original (non-aug) images from:
  datasets/crops/vae_256/train/tagged/{phase}/
  datasets/crops/vae_256/val/tagged/{phase}/
  datasets/crops/vae_256/test/{phase}/        ← no 'tagged/' layer here
across all five generatable classes (4 mitotic phases + interphase). The interphase pool is
~26k crops (vs ~300/phase mitotic), so it is randomly subsampled to --interphase-cap (seeded,
deterministic) to keep the concept folder balanced instead of interphase-dominated.

For each image: copies as RGB, applies augment() N times (--copies), writes a
sibling .txt caption from name_types.caption_for(): "micrograph of allium cepa root tip mitotic
cell in {phase} phase" for mitotic phases, "micrograph of allium cepa root tip cell in interphase"
for interphase. With --add-source-cue (default on for the 'typecue' version) the caption gains a
provenance source cue read straight from the crop's ``<source>_`` filename prefix,
e.g. "... and source {ina|onion|roboflow}".

Output layout (kohya DreamBooth):
  datasets/crops/lora/<version>/img/10_allium mitosis/
      <stem>.png, <stem>.txt
      <stem>_aug01.png, <stem>_aug01.txt   ← first aug copy (if copies >= 1)
      <stem>_aug02.png, <stem>_aug02.txt   ← second aug copy (if copies >= 2)
      ...

Named version defaults (applied when --copies / --aug-strength / --add-source-cue are omitted):
  no_aug    → --copies 0  --aug-strength mild
  baseline  → --copies 1  --aug-strength mild
  aug2x     → --copies 2  --aug-strength mild
  heavy_aug → --copies 1  --aug-strength heavy
  typecue   → --copies 1  --aug-strength mild  --add-source-cue

Usage:
    uv run python scripts/utils/lora_dataset.py --version baseline
    uv run python scripts/utils/lora_dataset.py --version no_aug
    uv run python scripts/utils/lora_dataset.py --vae-dir datasets/crops/vae_256
                                                  --out datasets/crops/lora
                                                  --version heavy_aug
                                                  --repeats 10
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# Make name_types importable whether this runs via cwd or an absolute DVC path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from name_types import INTERPHASE, PHASES, caption_for, source_from_name  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# Per-version defaults: version_name -> (copies, aug_strength, add_source_cue)
_VERSION_DEFAULTS: dict[str, tuple[int, str, bool]] = {
    "no_aug": (0, "mild", False),
    "baseline": (1, "mild", False),
    "aug2x": (2, "mild", False),
    "heavy_aug": (1, "heavy", False),
    "typecue": (1, "mild", True),
}


def _rotate_reflect(image: Image.Image, angle: float) -> Image.Image:
    """Rotate about the center, filling new corners with mirrored edge pixels.

    Uses numpy reflect padding instead of a flat ``fillcolor``, so augmented crops
    never gain artificial constant-value letterboxes (which the LoRA otherwise learns
    to reproduce). ``np.pad(mode="reflect")`` re-tiles for any pad size, so small
    crops are safe.
    """
    if abs(angle) < 0.5:
        return image
    w, h = image.size
    pad = int(math.ceil(math.hypot(w / 2, h / 2) - min(w, h) / 2)) + 2
    arr = np.asarray(image)
    arr = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    rotated = Image.fromarray(arr).rotate(angle, resample=Image.BILINEAR, expand=False)
    return rotated.crop((pad, pad, pad + w, pad + h))


def augment_mild(image: Image.Image) -> Image.Image:
    """Light augmentation: ±5° mirror-padded rotation, brightness/contrast jitter."""
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if random.random() < 0.5:
        image = ImageOps.flip(image)
    angle = random.uniform(-5.0, 5.0)
    image = _rotate_reflect(image, angle)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    return image


def augment_heavy(image: Image.Image) -> Image.Image:
    """Strong aug: ±12° mirror-padded rotation + wide brightness/contrast/sharpness jitter."""
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if random.random() < 0.5:
        image = ImageOps.flip(image)
    angle = random.uniform(-12.0, 12.0)
    image = _rotate_reflect(image, angle)
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


def subsample_pool(
    items: list[tuple[Path, str, str]], cap: int, seed: int
) -> list[tuple[Path, str, str]]:
    """Deterministically cap a (path, phase, split) pool: sort, seeded sample, re-sort.

    ``cap < 0`` keeps everything. Sorting before/after makes the selection independent of
    filesystem iteration order, so DVC rebuilds are reproducible.
    """
    if cap < 0 or len(items) <= cap:
        return items
    ordered = sorted(items, key=lambda e: str(e[0]))
    return sorted(random.Random(seed).sample(ordered, cap), key=lambda e: str(e[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-dir", type=Path, default=Path("datasets/crops/vae_256"))
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
    parser.add_argument(
        "--add-source-cue",
        action="store_true",
        default=None,
        help="Append 'and source <cue>' to each caption, cue read from the crop filename's "
        "leading <source>_ token. Defaults from --version (off, except 'typecue').",
    )
    parser.add_argument(
        "--interphase-cap",
        type=int,
        default=300,
        help="Max interphase originals kept across all splits (seeded subsample; the raw "
        "pool is ~26k vs ~300 per mitotic phase). Negative keeps everything.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Deterministic seed for the interphase subsample."
    )
    args = parser.parse_args()

    # Apply per-version defaults for flags not explicitly set
    ver_copies, ver_strength, ver_source_cue = _VERSION_DEFAULTS.get(
        args.version, (1, "mild", False)
    )
    copies = args.copies if args.copies is not None else ver_copies
    aug_strength = args.aug_strength if args.aug_strength is not None else ver_strength
    add_source_cue = args.add_source_cue if args.add_source_cue is not None else ver_source_cue
    augment_fn = augment_mild if aug_strength == "mild" else augment_heavy

    concept_dir = args.out / args.version / "img" / f"{args.repeats}_allium mitosis"
    concept_dir.mkdir(parents=True, exist_ok=True)

    pool: list[tuple[Path, str, str]] = [
        (src, phase, split_prefix)
        for phase_dir, phase, split_prefix in source_dirs(args.vae_dir)
        for src in collect_originals(phase_dir)
    ]
    mitotic = [e for e in pool if e[1] != INTERPHASE]
    inter_all = [e for e in pool if e[1] == INTERPHASE]
    interphase = subsample_pool(inter_all, args.interphase_cap, args.seed)

    total_orig = total_aug = 0
    for src, phase, split_prefix in mitotic + interphase:
        source = source_from_name(src.name) if add_source_cue else None
        caption = caption_for(phase, source)
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
        f"LoRA dataset [{args.version}] (copies={copies}, strength={aug_strength}, "
        f"source_cue={add_source_cue}, interphase={len(interphase)}/{len(inter_all)}, "
        f"cap={args.interphase_cap}, seed={args.seed}): "
        f"{total_orig} originals + {total_aug} augmented → {concept_dir}"
    )


if __name__ == "__main__":
    main()
