"""
Build a kohya-compatible LoRA dataset from the VAE tagged crops.

Collects original (non-aug) images from:
  datasets/crops/vae/train/tagged/{phase}/
  datasets/crops/vae/val/tagged/{phase}/
  datasets/crops/vae/test/{phase}/          ← no 'tagged/' layer here

For each image: copies as RGB, applies augment() N times (--copies), writes a
sibling .txt caption: "micrograph of allium cepa root tip mitotic cell in {phase} phase"

Output layout (kohya DreamBooth), `--layout single`:
  datasets/crops/lora/<version>/img/10_allium mitosis/
      <stem>.png, <stem>.txt
      <stem>_aug01.png, <stem>_aug01.txt   ← first aug copy (if copies >= 1)
      <stem>_aug02.png, <stem>_aug02.txt   ← second aug copy (if copies >= 2)
      ...

Output layout, `--layout per-phase` — one concept folder per mitotic phase, with per-phase
repeats computed to give every phase roughly `--target-per-phase` samples per epoch:
  datasets/crops/lora/<version>/img/3_prophase allium mitosis/
  datasets/crops/lora/<version>/img/5_metaphase allium mitosis/
  datasets/crops/lora/<version>/img/7_anaphase allium mitosis/
  datasets/crops/lora/<version>/img/6_telophase allium mitosis/

The per-phase layout exists because the single-folder layout leaves the four phases competing
for one concept: the 2026-08-09 conditioning diagnostic measured a 70.5% mean diagonal, but with
the two scarcest phases collapsing into their nearest visual neighbour (anaphase 51% -> metaphase,
telophase 57% -> prophase). Separate folders plus balancing repeats give each phase its own
concept and equal exposure.

`--dedup` drops exact duplicate crops by content hash. The upstream VAE splits contain 247
byte-identical duplicates among 1441 files, which would otherwise be over-weighted 2x.

Named version defaults (applied when the corresponding flag is omitted):
  no_aug    → --copies 0  --aug-strength mild   --layout single
  baseline  → --copies 1  --aug-strength mild   --layout single
  aug2x     → --copies 2  --aug-strength mild   --layout single
  heavy_aug → --copies 1  --aug-strength heavy  --layout single
  per_phase → --copies 1  --aug-strength mild   --layout per-phase  --dedup
  per_phase_nofill → same as per_phase, rebuilt after the rotation-fill fix
  per_phase_noaug  → --copies 0                     --layout per-phase  --dedup
  per_phase_d4x3   → --copies 3  --aug-strength d4  --layout per-phase  --dedup  (lossless)

Usage:
    uv run python scripts/utils/lora_dataset.py --version baseline
    uv run python scripts/utils/lora_dataset.py --version per_phase
    uv run python scripts/utils/lora_dataset.py --vae-dir datasets/crops/vae
                                                  --out datasets/crops/lora
                                                  --version heavy_aug
                                                  --repeats 10
"""

import argparse
import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

CAPTION_TEMPLATE = "micrograph of allium cepa root tip mitotic cell in {phase} phase"

# Per-version defaults: version_name → (copies, aug_strength, layout, dedup)
_VERSION_DEFAULTS: dict[str, tuple[int, str, str, bool]] = {
    "no_aug": (0, "mild", "single", False),
    "baseline": (1, "mild", "single", False),
    "aug2x": (2, "mild", "single", False),
    "heavy_aug": (1, "heavy", "single", False),
    "per_phase": (1, "mild", "per-phase", True),
    # Same recipe as per_phase, rebuilt after the rotation-fill fix. Kept as a separate version
    # so p3_per_phase's inputs stay reproducible while the two can be trained head to head.
    "per_phase_nofill": (1, "mild", "per-phase", True),
    # A: null hypothesis — does offline augmentation earn its place at all?
    "per_phase_noaug": (0, "mild", "per-phase", True),
    # B3: maximum orientation diversity with zero resampling. copies=3 draws three *distinct*
    # D4 elements per original, so 1194 -> 4776 images, all pixel-exact. `compute_repeats` drops
    # the repeat counts to hold ~3000 samples/phase, so exposure is unchanged.
    "per_phase_d4x3": (3, "d4", "per-phase", True),
    # Arm D: D4 orientation with the colour jitter removed. The 3-arm ablation showed the two
    # augmentation defects are separable — resampling caused the oversmoothing (recon_ratio
    # 0.444 -> 0.653 once removed), the 0.7-1.3 brightness/contrast jitter caused the
    # distributional mismatch (kid_classifier only fell 1.273 -> 1.243 with jitter still on,
    # vs 0.739 with no augmentation at all). This arm keeps the orientation and drops the jitter.
    "per_phase_d4x3_nojitter": (3, "d4_nojitter", "per-phase", True),
}

# Per-version override for --target-per-phase. Repeats are integers, so a version with many
# images per phase gets small repeat counts where rounding is coarse: per_phase_d4x3 at the
# default 3000 landed on repeats 2-3 and a 34.6% phase spread, under-exposing anaphase — the
# weakest phase — by 14% relative to per_phase_nofill, which would have confounded the very
# comparison the version exists to make. A higher target buys finer granularity for free:
# exposure per unique image is 30000/n_unique regardless, so only the phase *ratios* change.
_VERSION_TARGET_PER_PHASE: dict[str, int] = {
    "per_phase_d4x3": 12000,
    "per_phase_d4x3_nojitter": 12000,
}


def rotate_without_fill(image: Image.Image, angle: float) -> Image.Image:
    """
    Rotate, then centre-crop to the largest fill-free region and restore the original size.

    A plain `.rotate(angle, fillcolor=...)` leaves constant-coloured wedges in the corners.
    With `--copies 1` half of every augmented set carried them, and the LoRA learned the tilted
    frame *as part of the concept*: samples from `p3_per_phase` come out on rotated canvases with
    filled corners. Cropping to the inscribed rectangle keeps the rotation and drops the wedges.

    For a w×h image rotated by θ, the centred axis-aligned rectangle of the same aspect ratio
    that stays inside the rotated image scales by 1/(cos|θ| + sin|θ|). Exact for squares (all
    crops here are square); mildly conservative otherwise, which errs the safe way.
    """
    w, h = image.size
    rotated = image.rotate(angle, resample=Image.BILINEAR)
    rad = math.radians(min(abs(angle), 45.0))
    scale = 1.0 / (math.cos(rad) + math.sin(rad))
    cw, ch = max(1, int(w * scale)), max(1, int(h * scale))
    left, top = (w - cw) // 2, (h - ch) // 2
    return rotated.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


# The dihedral group of the square: the 8 orientations reachable without resampling a single
# pixel. Identity is excluded — the unmodified original is already in the dataset.
D4_TRANSPOSES: list[tuple[int, ...]] = [
    (Image.Transpose.ROTATE_90,),
    (Image.Transpose.ROTATE_180,),
    (Image.Transpose.ROTATE_270,),
    (Image.Transpose.FLIP_LEFT_RIGHT,),
    (Image.Transpose.FLIP_TOP_BOTTOM,),
    (Image.Transpose.TRANSPOSE,),
    (Image.Transpose.TRANSVERSE,),
]


def augment_d4_nojitter(image: Image.Image, variant: int | None = None) -> Image.Image:
    """Pure D4 transpose: orientation only, no photometric change whatsoever."""
    ops = (
        D4_TRANSPOSES[variant % len(D4_TRANSPOSES)]
        if variant is not None
        else random.choice(D4_TRANSPOSES)
    )
    for op in ops:
        image = image.transpose(op)
    return image


def augment_d4(image: Image.Image, variant: int | None = None) -> Image.Image:
    """
    Lossless orientation augmentation: one D4 transpose plus brightness/contrast jitter.

    Motivated by a measurement. `rotate_without_fill` (the fix for the red-wedge trap) rotates
    with BILINEAR, crops to the inscribed rectangle, then resizes back up — two resamplings and
    a 1.22x upscale at +-15deg, applied to half the dataset. `p3_per_phase_nofill` shows the
    cost: vqgan_recon_ratio fell 0.737 -> 0.444, i.e. generations markedly smoother than real
    crops (~1.0 is the target in both directions).

    Every transpose here is a pure pixel permutation, so orientation diversity costs nothing in
    sharpness. The trade is angular coverage: 8 discrete orientations instead of a continuum.
    That is likely acceptable because cells in root-tip squashes have no canonical orientation,
    so the 1194 real crops already span continuous rotation on their own — the augmentation was
    synthesising diversity the data already had, and paying interpolation blur for it.

    `variant` selects a specific D4 element so a caller making several copies of one image can
    draw distinct orientations instead of sampling with replacement.
    """
    ops = (
        D4_TRANSPOSES[variant % len(D4_TRANSPOSES)]
        if variant is not None
        else random.choice(D4_TRANSPOSES)
    )
    for op in ops:
        image = image.transpose(op)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    return image


def augment_mild(image: Image.Image) -> Image.Image:
    """Light augmentation: ±15° rotation, moderate brightness/contrast jitter."""
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if random.random() < 0.5:
        image = ImageOps.flip(image)
    image = rotate_without_fill(image, random.uniform(-15.0, 15.0))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    return image


def augment_heavy(image: Image.Image) -> Image.Image:
    """Strong augmentation: ±45° rotation, wide brightness/contrast/sharpness jitter."""
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if random.random() < 0.5:
        image = ImageOps.flip(image)
    image = rotate_without_fill(image, random.uniform(-45.0, 45.0))
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
    return sorted(
        p for p in phase_dir.iterdir() if p.suffix.lower() in IMG_EXTS and "_aug" not in p.stem
    )


def collect_by_phase(vae_dir: Path, dedup: bool) -> dict[str, list[tuple[Path, str]]]:
    """phase → [(path, split_prefix)], optionally deduplicated by content hash."""
    by_phase: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    seen: set[str] = set()
    dropped = 0

    for phase_dir, phase, split_prefix in source_dirs(vae_dir):
        for src in collect_originals(phase_dir):
            if dedup:
                digest = hashlib.md5(src.read_bytes()).hexdigest()
                if digest in seen:
                    dropped += 1
                    continue
                seen.add(digest)
            by_phase[phase].append((src, split_prefix))

    if dedup:
        print(f"Deduplication: dropped {dropped} exact duplicate crops.")
    return by_phase


def compute_repeats(n_images: int, target: int) -> int:
    """Repeats that bring a concept folder closest to `target` samples per epoch."""
    return max(1, round(target / n_images)) if n_images else 1


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
        choices=["mild", "heavy", "d4", "d4_nojitter"],
        default=None,
        help="Augmentation intensity. `d4` is lossless (transposes only). Defaults from --version.",
    )
    parser.add_argument(
        "--layout",
        choices=["single", "per-phase"],
        default=None,
        help="One concept folder for all phases, or one per phase. Defaults from --version.",
    )
    parser.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Drop exact duplicate crops by content hash. Defaults from --version.",
    )
    parser.add_argument(
        "--target-per-phase",
        type=int,
        default=None,
        help="per-phase layout: samples per epoch each phase should get, via its repeat count.",
    )
    args = parser.parse_args()

    # Apply per-version defaults for flags not explicitly set
    ver_copies, ver_strength, ver_layout, ver_dedup = _VERSION_DEFAULTS.get(
        args.version, (1, "mild", "single", False)
    )
    copies = args.copies if args.copies is not None else ver_copies
    aug_strength = args.aug_strength if args.aug_strength is not None else ver_strength
    layout = args.layout if args.layout is not None else ver_layout
    dedup = args.dedup if args.dedup is not None else ver_dedup
    augment_fn = {
        "mild": augment_mild,
        "heavy": augment_heavy,
        "d4": augment_d4,
        "d4_nojitter": augment_d4_nojitter,
    }[aug_strength]
    target_per_phase = (
        args.target_per_phase
        if args.target_per_phase is not None
        else _VERSION_TARGET_PER_PHASE.get(args.version, 3000)
    )

    by_phase = collect_by_phase(args.vae_dir, dedup)
    img_root = args.out / args.version / "img"

    # Resolve each phase to its concept folder and repeat count.
    if layout == "per-phase":
        # Each image contributes 1 original + `copies` augmented variants.
        concepts = {
            phase: (
                compute_repeats(len(items) * (1 + copies), target_per_phase),
                f"{phase} allium mitosis",
            )
            for phase, items in by_phase.items()
        }
    else:
        concepts = dict.fromkeys(by_phase, (args.repeats, "allium mitosis"))

    rows: list[tuple[str, int, int, int, int]] = []
    for phase, items in by_phase.items():
        repeats, concept_name = concepts[phase]
        concept_dir = img_root / f"{repeats}_{concept_name}"
        concept_dir.mkdir(parents=True, exist_ok=True)
        caption = CAPTION_TEMPLATE.format(phase=phase)

        n_orig = n_aug = 0
        for src, split_prefix in items:
            img = Image.open(src).convert("RGB")
            # Prefix with split name to avoid collisions between splits
            stem = f"{split_prefix}_{src.stem}"
            img.save(concept_dir / f"{stem}{src.suffix}")
            (concept_dir / f"{stem}.txt").write_text(caption)
            n_orig += 1
            # d4 draws distinct orientations per original; the resampling augmenters are
            # continuous, so sampling with replacement is already effectively distinct.
            variants = (
                random.sample(range(len(D4_TRANSPOSES)), k=min(copies, len(D4_TRANSPOSES)))
                if aug_strength.startswith("d4")
                else [None] * copies
            )
            for i in range(copies):
                aug_stem = f"{stem}_aug{i + 1:02d}"
                variant = variants[i] if i < len(variants) else None
                aug = augment_fn(img, variant) if aug_strength.startswith("d4") else augment_fn(img)
                aug.save(concept_dir / f"{aug_stem}{src.suffix}")
                (concept_dir / f"{aug_stem}.txt").write_text(caption)
                n_aug += 1

        rows.append((phase, n_orig, n_aug, repeats, (n_orig + n_aug) * repeats))

    print(
        f"\nLoRA dataset [{args.version}] "
        f"(layout={layout}, copies={copies}, strength={aug_strength}, dedup={dedup})"
    )
    header = f"{'phase':<12}{'orig':>8}{'aug':>8}{'images':>9}{'repeats':>9}{'per epoch':>11}"
    print(header)
    print("-" * len(header))
    for phase, n_orig, n_aug, repeats, per_epoch in sorted(rows, key=lambda r: -r[4]):
        print(f"{phase:<12}{n_orig:>8}{n_aug:>8}{n_orig + n_aug:>9}{repeats:>9}{per_epoch:>11}")
    print("-" * len(header))
    print(
        f"{'TOTAL':<12}{sum(r[1] for r in rows):>8}{sum(r[2] for r in rows):>8}"
        f"{sum(r[1] + r[2] for r in rows):>9}{'':>9}{sum(r[4] for r in rows):>11}"
    )
    print(f"\n→ {img_root}")


if __name__ == "__main__":
    main()
