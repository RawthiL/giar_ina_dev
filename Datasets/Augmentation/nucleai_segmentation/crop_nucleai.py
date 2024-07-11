from cell_mask_generator import CellMaskGenerator
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
    OUTPUT = os.getenv('OUTPUT')

    print("HOME:", HOME)
    print("IMAGE_PATH:", IMAGE_PATH)
    print("SAM_CHECKPOINT_PATH:", SAM_CHECKPOINT_PATH)
    print("CSV_PATH:", CSV_PATH)
    print("OUTPUT:", OUTPUT)

    cmg = CellMaskGenerator(SAM_CHECKPOINT_PATH)

    for file in os.listdir(IMAGE_PATH):
        image_name = os.fsdecode(file)
        image = os.path.join(IMAGE_PATH, image_name)
        cmg.crop_cells(image, CSV_PATH, OUTPUT)