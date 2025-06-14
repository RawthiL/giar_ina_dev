import os
from typing import Optional
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024 "


def main():
    # Create an ArgumentParser objectf
    parser = argparse.ArgumentParser(
        description="This script uses a SAM model to perform an initial segmentation of potential cells in a full-fov image."
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
        "--seed",
        "-s",
        type=int,
        required=False,
        default=0,
        help="Random seed to be used in the run.",
    )
    parser.add_argument(
        "--reference_csv_path",
        "-rp",
        type=str,
        required=True,
        help="Path to the reference CSV files.",
    )
    parser.add_argument(
        "--ajust_csv_path",
        "-ap",
        type=str,
        required=True,
        help="Path to the CSV files to adjust.",
    )
    parser.add_argument(
        "--bin_start",
        "-bs",
        type=int,
        required=True,
        help="Start of bins for area analysis.",
    )
    parser.add_argument(
        "--bin_end",
        "-be",
        type=int,
        required=True,
        help="End of bins for area analysis.",
    )
    parser.add_argument(
        "--bin_width",
        "-bw",
        type=int,
        required=True,
        help="Width of bins for area analysis.",
    )
    parser.add_argument(
        "--min_area_threshold",
        "-minathrs",
        type=int,
        required=True,
        help="Minimum area for an area to be used.",
    )
    parser.add_argument(
        "--max_area_threshold",
        "-maxathrs",
        type=int,
        required=True,
        help="Maximum area for an area to be used.",
    )

    args = parser.parse_args()

    SEED = int(args.seed)
    OUTPUT_PATH = args.output
    REF_CSV_PATH = args.reference_csv_path
    ADJUST_CSV_PATH = args.ajust_csv_path

    BINS_START = args.bin_start
    BINS_END = args.bin_end
    BINS_WIDTH = args.bin_width
    AREA_MIN = args.min_area_threshold
    AREA_MAX = args.max_area_threshold

    # Import and set random seeds
    import random

    random.seed(SEED)

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    import glob
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from dvclive import Live
    from dataclasses import dataclass, asdict
    import json

    if not os.path.exists(os.path.join(OUTPUT_PATH, "histogram")):
        os.makedirs(os.path.join(OUTPUT_PATH, "histogram"))

    @dataclass
    class AreaData:
        area_promedio: int
        lado_cuadrado: float
        diff_area: Optional[float] = None
        dif_lado: Optional[float] = None

    with Live() as live:
        partitions_data = {}

        # Get al lreference files
        all_files = glob.glob(os.path.join(REF_CSV_PATH, "*.csv"))
        dfs = []
        # Process all referecnces
        for file in all_files:
            df = pd.read_csv(file)

            # Extract image_section and image_id from filename (e.g., "001_0001.csv")
            filename = os.path.splitext(os.path.basename(file))[0]
            sample, image_id = filename.split("_")

            df["image_id"] = image_id
            df["sample"] = sample

            # Keep only the required columns
            df = df[["image_id", "cell_id", "bbox_area", "sample"]]
            dfs.append(df)

        # Combine all into a single DataFrame
        df = pd.concat(dfs, ignore_index=True)

        # Set fixed bins and xticks
        bin_start = BINS_START
        bin_end = BINS_END
        bin_width = BINS_WIDTH
        bins = np.arange(bin_start, bin_end + bin_width, bin_width)
        xticks = np.arange(bin_start, bin_end + bin_width * 3, bin_width * 3)

        # Get unique samples
        samples = sorted(df["sample"].unique())

        # Plot histograms for each class
        for cls in samples:
            class_df = df[df["sample"] == cls]

            fig = plt.figure(figsize=(20, 6))
            plt.hist(
                class_df["bbox_area"],
                bins=bins,
                color="cornflowerblue",
                edgecolor="black",
            )
            plt.title(f"Area Histogram for Class {cls}")
            plt.xlabel("Area (pixels)")
            plt.ylabel("Frequency")
            plt.xticks(xticks, rotation=45)
            plt.xlim(bin_start, bin_end)
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()

            # Save and log the image
            filename = os.path.join(OUTPUT_PATH, "histogram", f"hist_class_{cls}.png")
            # plt.savefig(filename)
            plt.draw()
            live.log_image(filename, fig)
            plt.close()

        # Filtered data
        df_filtered = df[df["bbox_area"] > AREA_MIN]

        # Bins y xticks
        bins = np.arange(bin_start, bin_end + bin_width, bin_width)
        xticks = np.arange(
            bin_start, bin_end + bin_width * 10, bin_width * 10
        )  # menos ticks

        # Clases
        classes = sorted(df["sample"].unique())

        # -------------------------------
        # Plot 1: Normalized histograms (All Data)
        fig = plt.figure(figsize=(22, 8))
        for cls in classes:
            class_df = df[df["sample"] == cls]
            plt.hist(
                class_df["bbox_area"],
                bins=bins,
                alpha=0.5,
                label=f"Partición {cls}",
                edgecolor="black",
                density=True,
            )

        plt.xlabel("Área (pixeles²)", fontsize=22)
        plt.ylabel("Frecuencia", fontsize=22)
        plt.xticks(xticks, rotation=45, fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlim(bin_start, bin_end)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend(fontsize=22, loc="upper right")
        plt.tight_layout()

        # Save and log the image
        filename = os.path.join(OUTPUT_PATH, "histogram", "hist_compiled.png")
        # plt.savefig(filename)
        plt.draw()
        live.log_image(filename, fig)
        plt.close()

        # -------------------------------
        # Plot 2: Normalized histograms (Filtered data > AREA_MIN) con moda y media
        fig = plt.figure(figsize=(22, 8))

        for cls in classes:
            class_df = df_filtered[df_filtered["sample"] == cls]
            if len(class_df) < 5:
                continue

            # Histograma
            plt.hist(
                class_df["bbox_area"],
                bins=bins,
                alpha=0.4,
                label=f"Partición {cls}",
                edgecolor="black",
                density=True,
            )

            # Moda
            counts, edges = np.histogram(class_df["bbox_area"], bins=bins, density=True)
            max_bin_idx = np.argmax(counts)
            mode_center = (edges[max_bin_idx] + edges[max_bin_idx + 1]) / 2
            plt.axvline(
                x=mode_center,
                linestyle="--",
                color="red",
                linewidth=2,
                label=f"Mode {cls}: {int(mode_center)} px",
            )

            # Media
            mean_val = class_df["bbox_area"].mean()
            plt.axvline(
                x=mean_val,
                linestyle="-",
                color="blue",
                linewidth=2,
                label=f"Mean {cls}: {int(mean_val)} px",
            )

        plt.xlabel("Área (pixeles²)", fontsize=18)
        plt.ylabel("Frecuencia", fontsize=18)
        plt.xticks(xticks, rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(bin_start, bin_end)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend(fontsize=14, loc="upper right")
        plt.tight_layout()

        # Save and log the image
        filename = os.path.join(
            OUTPUT_PATH, "histogram", "hist_compiled_thresholded_low.png"
        )
        # plt.savefig(filename)
        plt.draw()
        live.log_image(filename, fig)
        plt.close()

        # Filtered data
        df_filtered = df[(df["bbox_area"] >= AREA_MIN) & (df["bbox_area"] <= AREA_MAX)]

        # Set fixed bins and xticks
        bins = np.arange(bin_start, bin_end + bin_width, bin_width)
        xticks = np.arange(bin_start, bin_end + bin_width * 3, bin_width * 3)

        # Track means
        class_means = []

        # Plot: Normalized histograms (filtered data)
        fig = plt.figure(figsize=(20, 6))

        for cls in classes:
            class_df = df_filtered[df_filtered["sample"] == cls]
            if len(class_df) < 5:
                continue

            # Plot histogram
            plt.hist(
                class_df["bbox_area"],
                bins=bins,
                alpha=0.4,
                label=f"Partition {cls}",
                edgecolor="black",
                density=True,
            )

            # Mode (most frequent bin center)
            counts, edges = np.histogram(class_df["bbox_area"], bins=bins, density=True)
            max_bin_idx = np.argmax(counts)
            mode_center = (edges[max_bin_idx] + edges[max_bin_idx + 1]) / 2
            plt.axvline(
                x=mode_center,
                linestyle="--",
                color="red",
                linewidth=2,
                label=f"Mode {cls}: {int(mode_center)} px",
            )

            # Mean
            mean_val = class_df["bbox_area"].mean()
            class_means.append(mean_val)
            plt.axvline(
                x=mean_val,
                linestyle="-",
                color="blue",
                linewidth=2,
                label=f"Mean {cls}: {int(mean_val)} px",
            )

        # Mean of class means
        if class_means:
            mean_of_means = np.mean(class_means)
            plt.axvline(
                x=mean_of_means,
                linestyle="-",
                color="green",
                linewidth=3,
                label=f"Mean of Means: {int(mean_of_means)} px",
            )

        plt.title(
            "Normalized Histogram by Partition (Area 6000–30000) with Mode, Mean, and Overall Mean"
        )
        plt.xlabel("Area (pixels^2)")
        plt.ylabel("Density")
        plt.xticks(xticks, rotation=45)
        plt.xlim(bin_start, bin_end)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        # Save and log the image
        filename = os.path.join(
            OUTPUT_PATH, "histogram", "hist_compiled_thresholded.png"
        )
        # plt.savefig(filename)
        plt.draw()
        live.log_image(filename, fig)
        plt.close()

        # Add result data
        partitions_data["ina"] = dict()
        partitions_data["ina"]["Abril2023"] = asdict(
            AreaData(
                area_promedio=mean_of_means,
                lado_cuadrado=np.sqrt(mean_of_means),
                diff_area=None,
                dif_lado=None,
            )
        )
        live.log_metric("area_promedio", mean_of_means.mean())
        live.log_metric("lado_cuadrado", np.sqrt(mean_of_means))

        # Analyze other sources
        all_files = glob.glob(os.path.join(ADJUST_CSV_PATH, "*.csv"))
        dfs = []

        for file in all_files:
            df_r = pd.read_csv(file)

            # Extract image_section and image_id from filename (e.g., "001_0001.csv")
            filename = os.path.splitext(os.path.basename(file))[0]
            sample, image_id = filename.split("_")

            df_r["image_id"] = image_id
            df_r["sample"] = sample

            # Keep only the required columns
            df_r = df_r[["image_id", "cell_id", "bbox_area", "sample"]]
            dfs.append(df_r)

        # Combine all into a single DataFrame
        df_r = pd.concat(dfs, ignore_index=True)

        # Set fixed bins and xticks
        bins = np.arange(bin_start, bin_end + bin_width, bin_width)
        xticks = np.arange(bin_start, bin_end + bin_width * 3, bin_width * 3)

        partitions_data["onion_cell_merged"] = {}

        # Get unique classes
        classes = sorted(df_r["sample"].unique())

        # Plot histograms for each class
        for cls in classes:
            class_df_r = df_r[df_r["sample"] == cls]

            # plt.figure(figsize=(20, 6))
            # plt.hist(class_df_r['bbox_area'], bins=bins, color='cornflowerblue', edgecolor='black')

            # Compute and plot mean
            mean_val = class_df_r["bbox_area"].mean()

            partitions_data["onion_cell_merged"][cls] = asdict(
                AreaData(
                    area_promedio=mean_val,
                    lado_cuadrado=np.sqrt(mean_val),
                    diff_area=partitions_data["ina"]["Abril2023"]["area_promedio"]
                    - mean_val,
                    dif_lado=partitions_data["ina"]["Abril2023"]["lado_cuadrado"]
                    - np.sqrt(mean_val),
                )
            )

            # plt.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean: {int(mean_val)} px")

            # plt.title(f"Area Histogram for Class {cls}")
            # plt.xlabel("Area (pixels)")
            # plt.ylabel("Frequency")
            # plt.xticks(xticks, rotation=45)
            # plt.xlim(bin_start, bin_end)
            # plt.grid(axis='y', linestyle='--', alpha=0.6)
            # plt.legend()
            # plt.tight_layout()
            # plt.show()

        # Ensure folder exists
        # os.makedirs(os.path.dirname(TEMP_PATH), exist_ok=True)

        # Write JSON, converting np.float64 to float inline
        with open(os.path.join(OUTPUT_PATH, "datasets_area_data.json"), "w") as f:
            json.dump(
                partitions_data,
                f,
                default=lambda x: x.item() if isinstance(x, np.generic) else x,
                indent=2,
            )

        return


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
