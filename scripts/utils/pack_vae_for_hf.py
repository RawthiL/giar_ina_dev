"""
Packs VAE dataset directories into parquet shards ready for HuggingFace upload.

Output structure mirrors the HF subfolder layout so files can be uploaded directly:
    hf_staging/cropped/vae/{split}/{subdir}/data_shard-NNNNN.parquet

The downloader reconstructs:
    datasets/crops/vae/{split}/{subdir}/*.png

Usage:
    # Pack only the new val split (most common after split_vae_val.py)
    uv run python scripts/utils/pack_vae_for_hf.py

    # Pack everything (train + val + test)
    uv run python scripts/utils/pack_vae_for_hf.py --splits train val test

After running, upload with:
    huggingface-cli upload GIAR-UTN/allium-cepa-dataset hf_staging/cropped/vae cropped/vae --repo-type dataset
Then get the new SHA:
    python -c "from huggingface_hub import repo_info; print(repo_info('GIAR-UTN/allium-cepa-dataset', repo_type='dataset').sha)"
"""

import argparse
import io
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
SHARD_SIZE_MB = 100
REPO_ID = "GIAR-UTN/allium-cepa-dataset"


def _to_png_bytes(path: Path) -> bytes:
    buf = io.BytesIO()
    Image.open(path).save(buf, format="PNG")
    return buf.getvalue()


def _flush(records: list[dict], out_dir: Path, idx: int) -> None:
    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_dir / f"data_shard-{idx:05d}.parquet")


def pack_dir(src: Path, out_dir: Path, label: str, desc: str) -> int:
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = total_bytes = 0
    records: list[dict] = []
    limit = SHARD_SIZE_MB * 1024 * 1024

    for p in tqdm(images, desc=desc, leave=False):
        img_bytes = _to_png_bytes(p)
        records.append({"image": img_bytes, "label": label, "filename": p.name})
        total_bytes += len(img_bytes)

        if total_bytes >= limit:
            _flush(records, out_dir, shard_idx)
            shard_idx += 1
            records = []
            total_bytes = 0

    if records:
        _flush(records, out_dir, shard_idx)

    return len(images)


def pack_split(split_dir: Path, out_base: Path, split: str) -> None:
    print(f"\nPacking {split}/")

    tagged = split_dir / "tagged"
    if tagged.exists():
        for phase_dir in sorted(p for p in tagged.iterdir() if p.is_dir()):
            n = pack_dir(
                phase_dir,
                out_base / split / "tagged" / phase_dir.name,
                label=phase_dir.name,
                desc=f"{split}/tagged/{phase_dir.name}",
            )
            print(f"  tagged/{phase_dir.name}: {n} images")

    untagged = split_dir / "untagged"
    if untagged.exists():
        n = pack_dir(
            untagged, out_base / split / "untagged", label="untagged", desc=f"{split}/untagged"
        )
        print(f"  untagged: {n} images")

    # test split uses plain phase dirs (no tagged/untagged wrapper)
    for phase_dir in sorted(
        p for p in split_dir.iterdir() if p.is_dir() and p.name not in ("tagged", "untagged")
    ):
        n = pack_dir(
            phase_dir,
            out_base / split / phase_dir.name,
            label=phase_dir.name,
            desc=f"{split}/{phase_dir.name}",
        )
        print(f"  {phase_dir.name}: {n} images")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack VAE dataset dirs into HF-ready parquet shards."
    )
    parser.add_argument("--vae-dir", type=Path, default=Path("datasets/crops/vae"))
    parser.add_argument("--out", type=Path, default=Path("hf_staging/cropped/vae"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val"],
        help="Which splits to pack (default: val). Use 'train val test' to pack all.",
    )
    args = parser.parse_args()

    vae_dir: Path = args.vae_dir
    out_base: Path = args.out

    for split in args.splits:
        split_dir = vae_dir / split
        if not split_dir.exists():
            print(f"  SKIP {split}: {split_dir} not found")
            continue
        pack_split(split_dir, out_base, split)

    print(f"\nStaging area: {out_base.resolve()}")
    print("\nUpload to HuggingFace:")
    print(f"  huggingface-cli upload {REPO_ID} {args.out} cropped/vae --repo-type dataset")
    print("\nGet new SHA after upload:")
    print(
        f"  python -c \"from huggingface_hub import repo_info; print(repo_info('{REPO_ID}', repo_type='dataset').sha)\""
    )


if __name__ == "__main__":
    main()
