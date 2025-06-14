import sys
import os

import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024 "


def main():
    # Create an ArgumentParser object
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
        "--sam_model",
        "-sm",
        type=str,
        required=True,
        help="Path to the SAM pth file.",
    )
    parser.add_argument(
        "--gpus_list",
        "-gpus",
        type=str,
        required=False,
        default="0",
        help="List of available GPUs to use. PCIe order.",
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
        "--device",
        "-d",
        type=str,
        required=False,
        default="cuda",
        help="Device to use, cpu or cuda.",
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

    SAM_PATH = args.sam_model
    DEVICE_USE = args.device
    SEED = int(args.seed)
    DATASET = args.cell_dataset_name
    DATASET_SECTION = args.cell_dataset_section
    OUTPUT_PATH = args.output

    # Configure the GPU backend to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_list
    # Import and set random seeds
    import numpy as np
    import random
    import tensorflow as tf

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Do the rest of imports
    sys.path.insert(0, "../../")
    from config import MODELS_PATH, DATASETS_PATH

    sys.path.insert(0, "../../packages/python")
    from data import utils as data_utils
    from models import cell_segmentation as segmentators

    import tensorflow as tf

    print("GPU Available:", tf.config.list_physical_devices("GPU"))
    print("cuDNN Enabled:", tf.test.is_built_with_cuda())

    # Load SAM model
    SAM_CHECKPOINT_PATH = os.path.join(MODELS_PATH, SAM_PATH)
    cmg = segmentators.SAMCellMaskGenerator(
        SAM_CHECKPOINT_PATH, model_type="vit_h", device=DEVICE_USE
    )

    # Set paths
    IMAGE_PATH = os.path.join(
        DATASETS_PATH, "full_fov", DATASET, "images", DATASET_SECTION
    )
    CSV_PATH = os.path.join(OUTPUT_PATH, "cropped", DATASET, "data", DATASET_SECTION)

    # Apply segmentation to the whole
    data_utils.dataset_cell_segmentation(cmg, IMAGE_PATH, CSV_PATH)

    return


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
