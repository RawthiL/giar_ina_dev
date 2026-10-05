"""
Build ImageFolder datasets for the 4-class mitosis-phase classifier.

Real split (always produced):
  datasets/crops/mitosis_phase/{train,validation,test}/{phase}/
from datasets/crops/vae:
  train/tagged/{phase}/  -> train/{phase}/
  val/tagged/{phase}/    -> validation/{phase}/
  test/{phase}/          -> test/{phase}/          (no 'tagged/' layer in test)

Synthetic split (only with --synthetic-images-dir, i.e. output of
scripts/generate_diffuser_latent_dataset.py --save-images):
  datasets/crops/mitosis_phase_synth/{phase}/{basename}
  Filenames look like '00042_prophase_seed3791.png' (or with an _augXX suffix); the
  phase token is parsed from the name.

Usage:
    uv run python scripts/utils/prepare_mitosis_crops.py
    uv run python scripts/utils/prepare_mitosis_crops.py \
        --synthetic-images-dir datasets/mitosis_synth/vae256_adv/images
"""

import argparse
import shutil
import sys
from pathlib import Path

PHASES = ["prophase", "metaphase", "anaphase", "telophase"]
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


def build_real(vae_dir: Path, out_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    splits = {
        "train": [(p, vae_dir / "train" / "tagged" / p) for p in PHASES],
        "validation": [(p, vae_dir / "val" / "tagged" / p) for p in PHASES],
        "test": [(p, vae_dir / "test" / p) for p in PHASES],
    }
    for split, pairs in splits.items():
        for phase, src in pairs:
            if not src.is_dir():
                raise FileNotFoundError(f"Missing source dir: {src}")
            dst_dir = out_dir / split / phase
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img in _collect_images(src):
                _link_or_copy(img, dst_dir / img.name)
        counts[split] = sum(len(_collect_images(out_dir / split / ph)) for ph in PHASES)
    return counts


def _phase_from_name(stem: str) -> str | None:
    """Parse the phase token from a generated sample stem, e.g. '00042_prophase_seed..._aug01'."""
    for part in stem.split("_"):
        if part in PHASES:
            return part
    return None


def build_synthetic(images_dir: Path, out_dir: Path) -> dict[str, int]:
    counts = dict.fromkeys(PHASES, 0)
    imgs = _collect_images(images_dir)
    if not imgs:
        raise FileNotFoundError(f"No images found under {images_dir}")
    for img in imgs:
        phase = _phase_from_name(img.stem)
        if phase is None:
            print(f"⚠️ Skipping {img.name}: no phase token in filename")
            continue
        dst_dir = out_dir / phase
        dst_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(img, dst_dir / img.name)
        counts[phase] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-dir", type=Path, default=Path("datasets/crops/vae"))
    parser.add_argument("--out", type=Path, default=Path("datasets/crops/mitosis_phase"))
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

    real_counts = build_real(args.vae_dir, args.out)
    print(
        "Real mitosis-phase dataset → "
        + ", ".join(f"{k}={v}" for k, v in real_counts.items())
        + f" ({args.out})"
    )

    if args.synthetic_images_dir:
        synth_counts = build_synthetic(args.synthetic_images_dir, args.synthetic_out)
        print(
            "Synthetic dataset → "
            + ", ".join(f"{k}={v}" for k, v in synth_counts.items())
            + f" total={sum(synth_counts.values())} ({args.synthetic_out})"
        )


if __name__ == "__main__":
    sys.exit(main())
