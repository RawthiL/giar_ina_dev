"""
One-time dataset preparation: carves a val split out of the VAE training data.
Run locally before packing shards and uploading the updated dataset to HuggingFace.

Usage:
    uv run python scripts/utils/split_vae_val.py
    uv run python scripts/utils/split_vae_val.py --val-split 0.2 --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def _split_dir(src: Path, dst: Path, val_split: float, rng: random.Random) -> tuple[int, int]:
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    n_val = max(1, round(len(images) * val_split))
    val_set = rng.sample(images, n_val)

    dst.mkdir(parents=True, exist_ok=True)
    for p in val_set:
        shutil.move(str(p), dst / p.name)

    return len(images) - n_val, n_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Split VAE train data into train/val.")
    parser.add_argument("--vae-dir", type=Path, default=Path("datasets/crops/vae"))
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vae_dir: Path = args.vae_dir
    train_dir = vae_dir / "train"
    val_dir = vae_dir / "val"

    if val_dir.exists():
        print(f"WARNING: {val_dir} already exists — skipping to avoid data loss.")
        print("Delete it manually if you want to re-split.")
        return

    rng = random.Random(args.seed)
    total_train = total_val = 0

    tagged_train = train_dir / "tagged"
    if tagged_train.exists():
        for phase_dir in sorted(p for p in tagged_train.iterdir() if p.is_dir()):
            n_tr, n_va = _split_dir(
                phase_dir,
                val_dir / "tagged" / phase_dir.name,
                args.val_split,
                rng,
            )
            print(f"  tagged/{phase_dir.name}: {n_tr} train, {n_va} val")
            total_train += n_tr
            total_val += n_va

    untagged_train = train_dir / "untagged"
    if untagged_train.exists():
        n_tr, n_va = _split_dir(untagged_train, val_dir / "untagged", args.val_split, rng)
        print(f"  untagged: {n_tr} train, {n_va} val")
        total_train += n_tr
        total_val += n_va

    print(f"\nTotal: {total_train} train, {total_val} val")
    print(f"Val written to: {val_dir.resolve()}")
    print("\nNext: uv run python scripts/utils/pack_vae_for_hf.py")


if __name__ == "__main__":
    main()
