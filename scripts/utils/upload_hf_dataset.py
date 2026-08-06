"""
Pack a COCO-structured dataset into parquet shards ready for HuggingFace upload.

Expected local structure:
    {dataset_dir}/{split}/images/*.jpg
    {dataset_dir}/{split}/data/annotations.json

Output structure (mirrors HF subfolder layout):
    {out_dir}/{split}/images/data_shard-XXXXX.parquet
    {out_dir}/{split}/data/annotations.json

Upload manually with:
    huggingface-cli upload-large-folder <repo_id> <out_dir> \\
        --repo-type dataset --path-in-repo <subfolder>

Usage:
    uv run python scripts/utils/upload_hf_dataset.py \\
        --dataset datasets/allium_cepa_full_images_merged \\
        --out /tmp/hf_upload
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SHARD_SIZE_MB = 100
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _PROJECT_ROOT / "tmp" / "hf_packed"


def pack_split(images_dir: Path, out_dir: Path, shard_size_mb: int) -> int:
    """Pack images from images_dir into parquet shards in out_dir. Returns shard count."""
    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_size_bytes = shard_size_mb * 1024 * 1024
    records: list[dict] = []
    total_bytes = 0
    shard_idx = 0

    def flush() -> None:
        nonlocal shard_idx, records, total_bytes
        df = pd.DataFrame(records)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, out_dir / f"data_shard-{shard_idx:05d}.parquet")
        shard_idx += 1
        records = []
        total_bytes = 0

    for img_path in tqdm(images, desc="    packing", unit="img", leave=False):
        raw = img_path.read_bytes()
        records.append({"image": raw, "filename": img_path.name})
        total_bytes += len(raw)
        if total_bytes >= shard_size_bytes:
            flush()

    if records:
        flush()

    return shard_idx


def pack_dataset(
    dataset_dir: Path,
    out_dir: Path,
    subfolder: str,
    splits: list[str],
    shard_size_mb: int,
) -> None:
    dest = out_dir / subfolder
    dest.mkdir(parents=True, exist_ok=True)

    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"  ⚠  Split '{split}' not found, skipping.")
            continue

        print(f"\n[{split}] Packing images ...")
        n_shards = pack_split(
            images_dir=split_dir / "images",
            out_dir=dest / split / "images",
            shard_size_mb=shard_size_mb,
        )
        print(f"  → {n_shards} shard(s)")

        ann_src = split_dir / "data" / "annotations.json"
        if ann_src.exists():
            ann_dst = dest / split / "data" / "annotations.json"
            ann_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ann_src, ann_dst)

    for fname in ["full_annotations.json", "README.txt"]:
        src = dataset_dir / fname
        if src.exists():
            shutil.copy2(src, dest / fname)

    print(f"\nDone. Packed dataset at: {dest.resolve()}")
    print("\nTo upload manually:")
    print(
        f"  huggingface-cli upload-large-folder GIAR-UTN/allium-cepa-dataset"
        f" {dest.resolve()} --repo-type dataset --path-in-repo {subfolder}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack a COCO-structured dataset into parquet shards for HuggingFace upload."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Local dataset directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory for packed files (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="allium_cepa_full_images_merged",
        help="Subfolder name inside the HF repo",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation", "test"], help="Splits to pack"
    )
    parser.add_argument(
        "--shard-size-mb", type=int, default=DEFAULT_SHARD_SIZE_MB, help="Shard size in MB"
    )
    args = parser.parse_args()

    pack_dataset(
        dataset_dir=args.dataset,
        out_dir=args.out,
        subfolder=args.subfolder,
        splits=args.splits,
        shard_size_mb=args.shard_size_mb,
    )


if __name__ == "__main__":
    main()
