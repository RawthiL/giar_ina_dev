"""
Build ImageFolder datasets for the phase classifier (4-class mitosis or 5-class cell-phase).

Real split (always produced):
  <out>/{train,validation,test}/{class}/
from datasets/crops/vae_256 (native 256px; the untagged folders are ignored):
  train/tagged/{class}/  -> train/{class}/
  val/tagged/{class}/    -> validation/{class}/
  test/{class}/          -> test/{class}/          (no 'tagged/' layer in test)

--classes selects the vocabulary: the default is the 4 mitotic phases; add `interphase`
for the 5-class cell-phase dataset. The interphase pool is ~26k train crops (vs ~300 per
mitotic phase), so it is subsampled to --interphase-cap (seeded, deterministic) on
train + validation; the test split always keeps every crop for honest final metrics.
Run against an empty --out whenever the cap/seed changes (existing files are never removed).

Synthetic split (only with --synthetic-images-dir, i.e. output of
scripts/generate_diffuser_latent_dataset.py --save-images):
  <synthetic-out>/{class}/{basename}
  Filenames look like '00042_prophase_seed3791.png' (or with an _augXX suffix); the class
  token is parsed from the name; images of classes outside --classes are skipped.

Usage:
    uv run python scripts/utils/prepare_mitosis_crops.py
    uv run python scripts/utils/prepare_mitosis_crops.py \
        --synthetic-images-dir datasets/mitosis_synth/vae256_adv/images
    uv run python scripts/utils/prepare_mitosis_crops.py \
        --classes prophase metaphase anaphase telophase interphase \
        --out datasets/crops/mitosis_phase5 --interphase-cap 500 \
        --synthetic-images-dir datasets/mitosis_synth/vae256_adv5/images \
        --synthetic-out datasets/crops/mitosis_phase5_synth
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

MITOTIC_CLASSES = ["prophase", "metaphase", "anaphase", "telophase"]
INTERPHASE = "interphase"
ALL_CLASSES = [*MITOTIC_CLASSES, INTERPHASE]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink when possible (cheap for ~4k small crops), else copy."""
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _collect_images(src_dir: Path) -> list[Path]:
    return [
        p
        for p in sorted(src_dir.iterdir())
        if p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
    ]


def subsample_pool(paths: list[Path], cap: int, seed: int) -> list[Path]:
    """Deterministically cap a path pool: sort, seeded sample, re-sort.

    ``cap < 0`` or a small pool keeps everything. Sorting before/after makes the result
    independent of filesystem iteration order, so DVC rebuilds are reproducible.
    """
    if cap < 0 or len(paths) <= cap:
        return paths
    ordered = sorted(paths, key=str)
    return sorted(random.Random(seed).sample(ordered, cap), key=str)


def build_real(
    vae_dir: Path,
    out_dir: Path,
    classes: list[str],
    interphase_cap: int = 500,
    seed: int = 42,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    splits = {
        "train": [(c, vae_dir / "train" / "tagged" / c) for c in classes],
        "validation": [(c, vae_dir / "val" / "tagged" / c) for c in classes],
        "test": [(c, vae_dir / "test" / c) for c in classes],
    }
    for split, pairs in splits.items():
        for cls, src in pairs:
            if not src.is_dir():
                raise FileNotFoundError(f"Missing source dir: {src}")
            images = _collect_images(src)
            # Cap interphase on train/validation only; the test split stays complete.
            if cls == INTERPHASE and split != "test":
                images = subsample_pool(images, interphase_cap, seed)
            dst_dir = out_dir / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                _link_or_copy(img, dst_dir / img.name)
        counts[split] = sum(len(_collect_images(out_dir / split / c)) for c in classes)
    return counts


def _phase_from_name(stem: str, classes: list[str]) -> str | None:
    """Parse the class token from a generated sample stem, e.g. '00042_prophase_seed..._aug01'."""
    for part in stem.split("_"):
        if part in classes:
            return part
    return None


def build_synthetic(images_dir: Path, out_dir: Path, classes: list[str]) -> dict[str, int]:
    counts = dict.fromkeys(classes, 0)
    imgs = _collect_images(images_dir)
    if not imgs:
        raise FileNotFoundError(f"No images found under {images_dir}")
    for img in imgs:
        cls = _phase_from_name(img.stem, classes)
        if cls is None:
            print(f"⚠️ Skipping {img.name}: no class token of {classes} in filename")
            continue
        dst_dir = out_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(img, dst_dir / img.name)
        counts[cls] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-dir", type=Path, default=Path("datasets/crops/vae_256"))
    parser.add_argument("--out", type=Path, default=Path("datasets/crops/mitosis_phase"))
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=ALL_CLASSES,
        default=MITOTIC_CLASSES,
        metavar="CLASS",
        help="Classes to build (default: the 4 mitotic phases). "
        "Add 'interphase' for the 5-class cell-phase dataset.",
    )
    parser.add_argument(
        "--interphase-cap",
        type=int,
        default=500,
        help="Max interphase crops kept per train/validation split (seeded subsample; the "
        "raw pool is ~26k vs ~300 per mitotic phase). Negative keeps everything. "
        "The test split is never capped.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Deterministic seed for the interphase subsample."
    )
    parser.add_argument(
        "--synthetic-images-dir",
        type=Path,
        default=None,
        help="Dir of generated sample PNGs (out/images from generate_diffuser_latent_dataset.py). "
        "Omit to build the real dataset only.",
    )
    parser.add_argument(
        "--synthetic-out",
        type=Path,
        default=Path("datasets/crops/mitosis_phase_synth"),
    )
    args = parser.parse_args()

    classes = list(dict.fromkeys(args.classes))

    real_counts = build_real(args.vae_dir, args.out, classes, args.interphase_cap, args.seed)
    print(
        f"Real cell-phase dataset [{'+'.join(classes)}] → "
        + ", ".join(f"{k}={v}" for k, v in real_counts.items())
        + f" (interphase cap={args.interphase_cap}, seed={args.seed}) ({args.out})"
    )

    if args.synthetic_images_dir:
        synth_counts = build_synthetic(args.synthetic_images_dir, args.synthetic_out, classes)
        print(
            "Synthetic dataset → "
            + ", ".join(f"{k}={v}" for k, v in synth_counts.items())
            + f" total={sum(synth_counts.values())} ({args.synthetic_out})"
        )


if __name__ == "__main__":
    sys.exit(main())
