from Datasets.Augmentation.nucleai_segmentation.cell_mask_generator import CellMaskGenerator
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    # Load the .env file
    load_dotenv()

    # Get the variables from the environment
    HOME = os.getenv('HOME')
    IMAGE_PATH = os.getenv('IMAGE_PATH')
    SAM_CHECKPOINT_PATH = os.getenv('SAM_CHECKPOINT_PATH')
    CSV_PATH = os.getenv('CSV_PATH')

    print("HOME:", HOME)
    print("IMAGE_PATH:", IMAGE_PATH)
    print("SAM_CHECKPOINT_PATH:", SAM_CHECKPOINT_PATH)
    print("CSV_PATH:", CSV_PATH)

    cmg = CellMaskGenerator(SAM_CHECKPOINT_PATH)
    cmg.generate_all_masks(IMAGE_PATH, CSV_PATH)