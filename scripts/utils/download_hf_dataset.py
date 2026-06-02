"""
Download a COCO-structured dataset from HuggingFace Hub.

Reconstructs:
    {out_dir}/{split}/images/*.jpg
    {out_dir}/{split}/data/annotations.json

Usage:
    uv run python scripts/utils/download_hf_dataset.py
"""

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

REPO_ID = "GIAR-UTN/allium-cepa-dataset"
SUBFOLDER = "allium_cepa_full_images_merged"
DEFAULT_OUT = Path("datasets/allium_cepa_full_images_merged")


def download_dataset(
    repo_id: str = REPO_ID,
    subfolder: str = SUBFOLDER,
    out_dir: Path = DEFAULT_OUT,
    rev: str | None = None,
) -> None:
    print(f"Listing files in {repo_id}/{subfolder} ...")
    all_files = list(list_repo_files(repo_id, repo_type="dataset", revision=rev))
    relevant = sorted(f for f in all_files if f.startswith(subfolder + "/"))

    parquet_files = [f for f in relevant if f.endswith(".parquet")]
    other_files = [f for f in relevant if not f.endswith(".parquet")]

    print(f"Found {len(parquet_files)} shard(s), {len(other_files)} metadata file(s)")
    out_dir.mkdir(parents=True, exist_ok=True)

    for remote_path in tqdm(parquet_files, desc="Image shards"):
        rel = Path(remote_path).relative_to(subfolder)
        images_dir = out_dir / rel.parent
        images_dir.mkdir(parents=True, exist_ok=True)

        cached = Path(
            hf_hub_download(
                repo_id=repo_id, filename=remote_path, repo_type="dataset", revision=rev
            )
        )
        df = pd.read_parquet(cached)

        written = skipped = 0
        for _, row in df.iterrows():
            dest = images_dir / row["filename"]
            if dest.exists():
                skipped += 1
                continue
            dest.write_bytes(row["image"])
            written += 1
        tqdm.write(f"  {rel.name}: {written} written, {skipped} skipped")

    for remote_path in tqdm(other_files, desc="Metadata files"):
        rel = Path(remote_path).relative_to(subfolder)
        local_dest = out_dir / rel
        if local_dest.exists():
            continue
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        cached = Path(
            hf_hub_download(
                repo_id=repo_id, filename=remote_path, repo_type="dataset", revision=rev
            )
        )
        local_dest.write_bytes(cached.read_bytes())
        tqdm.write(f"  {rel}: downloaded")

    print(f"\nDone. Output: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download COCO-structured dataset from HuggingFace Hub."
    )
    parser.add_argument("--repo", default=REPO_ID, help=f"HF dataset repo id (default: {REPO_ID})")
    parser.add_argument(
        "--subfolder", default=SUBFOLDER, help=f"Subfolder inside the repo (default: {SUBFOLDER})"
    )
    parser.add_argument(
        "--rev",
        default=None,
        help="HF commit SHA, branch, or tag to pin (default: latest)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    download_dataset(repo_id=args.repo, subfolder=args.subfolder, out_dir=args.out, rev=args.rev)


if __name__ == "__main__":
    main()
