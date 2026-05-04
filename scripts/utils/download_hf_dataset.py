"""
Download and extract parquet image shards from a Hugging Face dataset.

Usage:
    python scripts/utils/download_hf_dataset.py
    python scripts/utils/download_hf_dataset.py --subfolder full_fov/ina/images --out datasets/full_fov/ina/images
"""

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

REPO_ID = "GIAR-UTN/allium-cepa-dataset"
DEFAULT_SUBFOLDER = "full_fov"
DEFAULT_OUT = Path(__file__).resolve().parents[5] / "datasets" / "full_fov"


def _detect_image_column(df: pd.DataFrame) -> str:
    """
    Return the name of the column that holds image data.
    HuggingFace stores images as dicts {"bytes": b"...", "path": "..."} or as raw bytes.
    Falls back to the first binary/object column.
    """
    for col in df.columns:
        sample = df[col].iloc[0]
        if isinstance(sample, dict) and "bytes" in sample:
            return col
        if isinstance(sample, bytes):
            return col
    raise ValueError(f"No image column detected. Columns found: {list(df.columns)}")


def _save_row(row, image_col: str, out_dir: Path) -> bool:
    """Write a single row's image bytes to disk. Returns True if a new file was written."""
    cell = row[image_col]

    if isinstance(cell, dict):
        raw: bytes = cell["bytes"]
        filename: str = cell.get("path") or f"image_{row.name}.jpg"
    elif isinstance(cell, bytes):
        raw = cell
        filename = f"image_{row.name}.jpg"
    else:
        return False

    dest = out_dir / Path(filename).name
    if dest.exists():
        return False

    dest.write_bytes(raw)
    return True


def extract_shard(shard_path: Path, out_dir: Path) -> tuple[int, int]:
    """
    Read a parquet shard and write each image to out_dir.
    Returns (written, skipped) counts.
    """
    df = pd.read_parquet(shard_path)
    image_col = _detect_image_column(df)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    for _, row in df.iterrows():
        if _save_row(row, image_col, out_dir):
            written += 1
        else:
            skipped += 1
    return written, skipped


def download_and_extract(
    repo_id: str = REPO_ID,
    subfolder: str = DEFAULT_SUBFOLDER,
    out_dir: Path = DEFAULT_OUT,
    keep_parquet: bool = False,
) -> None:
    print(f"Listing shards in {repo_id}/{subfolder} ...")
    all_files = list(list_repo_files(repo_id, repo_type="dataset"))
    shards = sorted(f for f in all_files if f.startswith(subfolder) and f.endswith(".parquet"))

    if not shards:
        raise FileNotFoundError(f"No .parquet files found under '{subfolder}' in {repo_id}")

    print(f"Found {len(shards)} shard(s). Output -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_cache = out_dir / "_parquet_cache"
    parquet_cache.mkdir(exist_ok=True)

    total_written = total_skipped = 0

    for shard_remote_path in tqdm(shards, desc="Shards", unit="shard"):
        shard_name = Path(shard_remote_path).name
        local_parquet = parquet_cache / shard_name

        if not local_parquet.exists():
            tqdm.write(f"  Downloading {shard_name} ...")
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=shard_remote_path,
                repo_type="dataset",
                local_dir=str(parquet_cache),
                local_dir_use_symlinks=False,
            )
            local_parquet = Path(downloaded)

        tqdm.write(f"  Extracting {shard_name} ...")
        written, skipped = extract_shard(local_parquet, out_dir)
        total_written += written
        total_skipped += skipped
        tqdm.write(f"    -> {written} written, {skipped} skipped (already exist)")

        if not keep_parquet:
            local_parquet.unlink()

    if not keep_parquet and parquet_cache.exists():
        try:
            parquet_cache.rmdir()  # only removes if empty
        except OSError:
            pass

    print("---")
    print(f"Done. {total_written} images written, {total_skipped} already existed.")
    print(f"Output dir: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract parquet image shards from Hugging Face."
    )
    parser.add_argument(
        "--repo",
        default=REPO_ID,
        help=f"HuggingFace dataset repo id (default: {REPO_ID})",
    )
    parser.add_argument(
        "--subfolder",
        default=DEFAULT_SUBFOLDER,
        help=f"Path prefix inside the repo to filter shards (default: {DEFAULT_SUBFOLDER})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory for extracted images (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--keep-parquet",
        action="store_true",
        help="Keep downloaded .parquet files after extraction (default: delete them)",
    )
    args = parser.parse_args()

    download_and_extract(
        repo_id=args.repo,
        subfolder=args.subfolder,
        out_dir=args.out,
        keep_parquet=args.keep_parquet,
    )


if __name__ == "__main__":
    main()
