import pandas as pd
import os
import cv2 as cv
from pathlib import Path
import numpy as np

from data import augmentation as data_augmentation
from data import utils as data_utils


def cut_images(paths, height, width, overlapX, overlapY, metadata):
    """
    brief: Given a dictionary containing the path to the parent folder, and the
    names of subfolders with a list of images in each one it cuts the images
    and creates a csv file where it stores metadata of the cutted images
    input: paths    - dictionary - data structure that contains all paths to images
    input: height   - number     - pixels of the subimage to cut corresponding to the height
    input: width    - number     - pixels of the subimage to cut corresponding to the width
    input: overlapX - number     - pixels of overlap in X to use when cutting
    input: overlapY - number     - pixels of overlap in Y to use when cutting
    input: metadata - bool       - Indicates whether to create metadata or not
    output: null
    """


    #Create folder path to store cutted images
    path = paths.pop('path')
    path = path[:-1] if path.endswith('/') else path # If it ends with / delete it
    new_path = path + f'_{width}x{height}(ov{overlapX}x{overlapY})'

    classes = list(data_utils.dict_attributes.values())
    metadata_dir = 'target'
    subimages_classes = []

    for subfolder in paths:
        output_dir = Path(new_path + '/' + subfolder)
        output_dir.mkdir(parents=True, exist_ok=True) # I create all folderes if they do not exist
        row_index = 0
    print(f"Cutting images in: {output_dir}")

    # Loop through and cut each image
    for idx, image_file in enumerate(paths[subfolder]):

        # Open image and get data from it
        print(f"Image: {idx + 1}/{len(paths[subfolder])}", end='\r')
        image = cv.imread(image_file)
        image_name = os.path.splitext(os.path.basename(image_file))[0]

        # Cut image
        cuts_list = data_augmentation.cut_image(image, height, width, overlapX, overlapY)

        # Save to disk
        for idx, img_cut in enumerate(cuts_list):

            subimage_name = f"{image_name}_{idx}.png"

            # Get classes from target images
            if subfolder == metadata_dir and metadata:
                subimages_classes.append(data_utils.get_image_metadata(subimage))
                row_index += 1

            # Save cutted image
            subimage_path = os.path.join(output_dir, subimage_name)
            subimage = cv.cvtColor(subimage, cv.COLOR_BGR2GRAY)
            cv.imwrite(subimage_path, subimage)

    # When finished with the target folder, create metadata of its images
    if subfolder == metadata_dir and metadata:
        print("Writing metadata")
        df = pd.DataFrame(subimages_classes, columns=data_utils.get_image_metadata_columns())
        df.to_csv(f"{new_path}/metadata.csv", index=True)
