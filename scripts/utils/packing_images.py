import argparse
import glob
import os

import cv2 as cv
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def images_to_parquet_sharded(
    image_dir, output_path, output_prefix="data_shard", shard_size_mb=100
):
    """
    Converts a folder of images into multiple Parquet shard files (~shard_size_mb each).

    Args:
        image_dir (str): Path to dataset folder. If images are grouped in
                         subfolders, the folder name will be used as label.
        output_path (str): Path to processed dataset folder.
        output_prefix (str): Prefix for output parquet files (e.g., "data_shard").
        shard_size_mb (int): Approximate shard size in MB.
    """
    shard_size_bytes = shard_size_mb * 1024 * 1024
    shard_index = 0
    records = []
    all_images_count = 0
    total_written = 0

    image_paths = glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True)

    for path in tqdm(image_paths, desc="Packing images"):
        ext = os.path.splitext(path)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
            continue

        label = os.path.basename(os.path.dirname(path))

        img_bytes = cv.imread(path)
        img_bytes = cv.imencode(".png", img_bytes)[1].tobytes()

        records.append({"image": img_bytes, "label": label, "filename": os.path.basename(path)})
        total_written += len(img_bytes)

        # When current shard size exceeds limit, flush to disk
        if total_written >= shard_size_bytes:
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df)
            out_path = os.path.join(output_path, f"{output_prefix}-{shard_index:05d}.parquet")
            pq.write_table(table, out_path)
            all_images_count += len(records)
            # Reset for next shard
            shard_index += 1
            records = []
            total_written = 0

    # Write leftover records
    if records:
        df = pd.DataFrame(records)
        table = pa.Table.from_pandas(df)
        out_path = os.path.join(output_path, f"{output_prefix}-{shard_index:05d}.parquet")
        pq.write_table(table, out_path)
        all_images_count += len(records)

    print(f"✅ Wrote {all_images_count} images to {output_path}")


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="This script takes an images folder and convert them into parquet files."
    )

    # Add arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for the resulting dataset.",
    )
    parser.add_argument(
        "--output_prefix",
        "-op",
        type=str,
        required=False,
        default="data_shard",
        help="Output prefix for the resulting dataset.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input path for the dataset to convert.",
    )
    parser.add_argument(
        "--parquet_shard_size_mb",
        "-ps",
        type=int,
        default=100,
        required=False,
        help="Size of the parquet shard in MB.",
    )

    args = parser.parse_args()

    # Example usage
    images_to_parquet_sharded(
        args.input,
        args.output,
        output_prefix=args.output_prefix,
        shard_size_mb=args.parquet_shard_size_mb,
    )
