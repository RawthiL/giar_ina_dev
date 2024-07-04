from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import cv2 as cv
from pathlib import Path
import argparse
import numpy as np
from typing import Tuple, List

def cut_image(image : Image.Image, height : int, width : int, overlapX : int, overlapY: int ) -> List:
    """
    brief: receives an image and creates a list of cuts. The cuts are taken consecutivelly and with a given overlap.
    """

    # Get size
    image_width, image_height = image.size 
    # Convert to numpy
    image = np.array(image)

    subimage_list = list()

    # Cut images with given height and width
    # The overlap will be the height and width divided by 2
    for i in range(0, image_height - overlapY, overlapY):
        for j in range(0, image_width - overlapX, overlapX):

            subimage = image[i:i+height, j:j+width]
            # If subimage dimensions are not the achieved discard the cut,
            # This might occur if the given w or h are not divisible by the orginal shape
            h, w = subimage.shape[:2]
            if h!=height or w!=width:
                continue

            subimage_list.append(subimage)

    return subimage_list


def change_image_basic(image : Image.Image, target_image : Image.Image) -> Tuple[Image.Image, Image.Image]:
    """
    brief: Given an image and its target mask, returns the modified image in terms of brightness, contrast, saturation and flip.
    It also applies the flipping to to the target mask.
    """

    # Get the height and width of the loaded image
    new_width, new_height = image.size 

    #Transformation
    common_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomResizedCrop((new_height, new_width), scale=(0.7, 1.0)),  # Random zoom with a scaling range
    ])  # Define common transformations for both images: input and target


    transform_image = transforms.Compose([
        common_transform,
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # Random adjustment of brightness, contrast, and saturation for transform_image only
    ])# Specific transformations for each type of image input

    transform_target = common_transform   #Specific transformation for target

    # Apply the respective transformations to  input and target
    augmented_image = transform_image(image)
    augmented_target_image = transform_target(target_image)

    return augmented_image, augmented_target_image
