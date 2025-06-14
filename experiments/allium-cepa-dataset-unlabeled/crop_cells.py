import sys
import os
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
        "--target_img_side_px",
        "-ims",
        type=int,
        required=True,
        help="Target image side in px.",
    )
    parser.add_argument(
        "--cell_dataset_name",
        "-cdn",
        type=str,
        required=True,
        help="Name of the dataset to process.",
    )
    parser.add_argument(
        "--cell_dataset_section",
        "-cds",
        type=str,
        default="",
        required=False,
        help="Section of the dataset to process, leave blank for INA.",
    )

    args = parser.parse_args()

    SEED = int(args.seed)
    OUTPUT_PATH = args.output
    DATASET = args.cell_dataset_name
    DATASET_SECTION = args.cell_dataset_section
    IMG_TARGET_SIDE = int(args.target_img_side_px)

    # Import and set random seeds
    import numpy as np
    import random
    import tensorflow as tf

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    sys.path.insert(0, "../../packages/python")
    from models import cell_segmentation as segmentators

    sys.path.insert(0, "../../")
    from config import DATASETS_PATH
    import json
    from tqdm import tqdm
    import cv2 as cv
    import pandas as pd

    # Input paths
    IMAGES_PATH = os.path.join(
        DATASETS_PATH, "full_fov", DATASET, "images", DATASET_SECTION
    )
    CSV_PATH = os.path.join(OUTPUT_PATH, "cropped", DATASET, "data", DATASET_SECTION)
    JSON_PATH = os.path.join(OUTPUT_PATH, "cropped", "datasets_area_data.json")

    # Output paths
    CROPS_PATH = os.path.join(
        OUTPUT_PATH, "cropped", DATASET, "images", DATASET_SECTION
    )

    # List of elements to use
    # csvs = sorted(
    #     os.listdir(CSV_PATH)
    # )  # Paths to the csv of SAM detections of each image
    images = sorted(
        os.listdir(IMAGES_PATH)
    )  # full_images from where the crops are made
    with open(
        JSON_PATH, "r"
    ) as f:  # json with the information of the filename of the images
        area_data = json.load(f)

    # Dataset Genertation
    if not os.path.exists(CROPS_PATH):
        os.makedirs(CROPS_PATH)

    # This is the reference
    resize_factor = IMG_TARGET_SIDE / area_data["ina"]["Abril2023"]["lado_cuadrado"]

    for image in tqdm(images):
        image_name, image_type = image.split(".")
        if DATASET == "ina":
            image_group = "Abril2023"
        elif DATASET == "onion_cell_merged":
            image_group = image_name[0]
        else:
            raise ValueError(f"Unknown dataset {DATASET}")
        image_side = area_data[DATASET][image_group]["lado_cuadrado"]
        image_resize_factor = int(resize_factor * image_side)

        img = cv.imread(os.path.join(IMAGES_PATH, image))

        df = pd.read_csv(os.path.join(CSV_PATH, f"{image_name}.csv"))
        df_bbox = df[df["image"] == image_name][["x", "y", "w", "h", "cell_id"]]

        for _, row in df_bbox.iterrows():
            cell_id = row["cell_id"]
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]
            x, y, w, h = segmentators.CellMaskGenerator.adjust_bbox(
                segmentators.CellMaskGenerator,
                x,
                y,
                w,
                h,
                image_resize_factor * image_resize_factor,
                img.shape[1],
                img.shape[0],
            )

            crop = cv.resize(
                img[y : y + h, x : x + w], (IMG_TARGET_SIDE, IMG_TARGET_SIDE)
            )
            output_path = os.path.join(CROPS_PATH, f"{image_name}_{cell_id}.png")
            cv.imwrite(output_path, crop)


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
