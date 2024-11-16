import os
import cv2
import numpy as np
import albumentations as A

from PIL import Image
from .utils import find_files
from typing import Tuple, List
from torchvision import transforms

def cut_image(
    image: Image.Image, height: int, width: int, overlapX: int, overlapY: int
) -> List:
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
            subimage = image[i : i + height, j : j + width]
            # If subimage dimensions are not the achieved discard the cut,
            # This might occur if the given w or h are not divisible by the orginal shape
            h, w = subimage.shape[:2]
            if h != height or w != width:
                continue

            subimage_list.append(subimage)

    return subimage_list


def change_image_basic(
    image: Image.Image, target_image: Image.Image
) -> Tuple[Image.Image, Image.Image]:
    """
    brief: Given an image and its target mask, returns the modified image in terms of brightness, contrast, saturation and flip.
    It also applies the flipping to to the target mask.
    """

    # Get the height and width of the loaded image
    new_width, new_height = image.size

    # Transformation
    common_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomResizedCrop(
                (new_height, new_width), scale=(0.7, 1.0)
            ),  # Random zoom with a scaling range
        ]
    )  # Define common transformations for both images: input and target

    transform_image = transforms.Compose(
        [
            common_transform,
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5
            ),  # Random adjustment of brightness, contrast, and saturation for transform_image only
        ]
    )  # Specific transformations for each type of image input

    transform_target = common_transform  # Specific transformation for target

    # Apply the respective transformations to  input and target
    augmented_image = transform_image(image)
    augmented_target_image = transform_target(target_image)

    return augmented_image, augmented_target_image


def augment_folder(path, ext='.png'):
    #Get a list of all images in folder
    files = find_files(path, ext)
    images_paths = files[list(files.keys())[1]]

    # Define the augmentation pipeline
    transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomBrightnessContrast(p=0.4),
        A.RandomGamma(p=1, gamma_limit=(60, 110)),
        A.GaussNoise(p=0.2)
    ])

    for idx, image_path in enumerate(images_paths):
        print(f"Image: {idx + 1}/{len(image_path)}", end='\r')
        
        # Load an image and apply the transformation
        image_name = os.path.basename(image_path)
        path = os.path.dirname(image_path)
        base, ext = os.path.splitext(image_name)
        
        image = cv2.imread(image_path)
        augmented_image = transform(image=image)['image']

        new_name = f"{base}_augmented{ext}"
        if os.path.exists(f"{path}/{new_name}"):
            new_name = f"{base}_augmented_1{ext}"

        # Save the augmented image
        cv2.imwrite(f"{path}/{new_name}", augmented_image)