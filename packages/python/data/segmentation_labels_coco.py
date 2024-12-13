import json
import argparse
import cv2 as cv
import os
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils
import csv

"""
brief: Counts the number of connected components (blobs) in a binary mask crop.
input: crop - np.array - A binary mask crop where blobs are to be counted.
output: int - The number of detected blobs in the mask.
"""
def blob_quantity_in_crop(crop):
    binary_mask = (crop > 0).astype(np.uint8)
    _, labels = cv.connectedComponents(binary_mask)
    return labels.max()

"""
brief: Removes blobs touching the edges from a binary image.
input: image - np.array - Binary image from which edge-touching blobs are removed.
output: np.array - Image with edge-touching blobs removed.
"""
def remove_incomplete_blobs(image):
    
    # Create a copy of the image to work on without altering the original
    processed_image = image.copy()

    # Find contours of blobs in the binary image
    contours, _ = cv.findContours(processed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Loop over each contour to check if it touches the edge
    for contour in contours:
        # Get bounding rectangle of the contour
        x, y, w, h = cv.boundingRect(contour)

        # Check if the contour touches the edge of the image
        if x <= 0 or y <= 0 or x + w >= image.shape[1] or y + h >= image.shape[0]:
            # If it touches the edge, remove only the precise contour with black (not the bounding box)
            cv.drawContours(processed_image, [contour], -1, (0, 0, 0), thickness=cv.FILLED)

    return processed_image

"""
brief: Processes a crop to remove incomplete blobs and determine if it requires review.
input: crop (np.array): The binary mask crop to process.
       crop_name (str): Name identifier for the crop, used for logging.
output: crop_copy (np.array): The processed crop, or the original if review is needed.
        review_required (bool): Indicates whether the crop should be marked for review.
"""
def transform_crop(crop, crop_name):
    crop_copy = remove_incomplete_blobs(crop)
    review_required = False
    
    if blob_quantity_in_crop(crop_copy) == 0: # Check if the crop contains any blob
        review_required = True
        print(f"Crop {crop_name} marked for review (the blob is bigger than the image).")
        crop_copy = crop
    else:
        if blob_quantity_in_crop(crop_copy) > 1: # Check if the crop contains more than one blob
            review_required = True
            print(f"Crop {crop_name} marked for review (more than one blob in the image).")
            crop_copy = crop
    
    return crop_copy, review_required

"""
brief: Applies morphological erosion to binary cell masks.
input: binary_mask - np.array - Binary mask of the cells.
       kernel_size - int - Size of the structuring element for erosion.
       iterations - int - Number of times erosion is applied.
output: np.array - The eroded binary mask.
"""
def erode_cell_edges(binary_mask, kernel_size=5, iterations=1):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    eroded_mask = cv.erode(binary_mask, kernel, iterations=iterations)
    return eroded_mask

"""
brief: Filters out cells whose area exceeds a defined percentage of the average cell area.
input: annotations - list - List of annotation dictionaries containing cell data.
       area_threshold_percentage - float - Percentage above the average area to exclude cells.
output: list - List of annotations for cells within the area threshold.
"""
def filter_cells_by_area(annotations, area_threshold_percentage):
    areas = [annotation['area'] for annotation in annotations]

    print(f"Filtering cells based on area threshold of {area_threshold_percentage}%...")

    if not areas:
        print("No cells found to filter. Skipping.")
        return annotations

    average_area = sum(areas) / len(areas)
    threshold = average_area * (1 + area_threshold_percentage / 100)

    filtered_annotations = [
        annotation for annotation in annotations if annotation['area'] <= threshold
    ]

    print(f"Filtered out {len(annotations) - len(filtered_annotations)} cells exceeding the threshold of {threshold:.2f}.")
    return filtered_annotations

"""
brief: Saves cropped images and their corresponding masks to the output directory.
input: output_path (Path): Directory where the cropped images and masks will be saved.
       cropped_mask (np.array): The binary mask of the cropped region.
       cropped_image (np.array): The image of the cropped region.
       target_crop_name (str): Name for the cropped file to be saved.
"""
def save_cropped_images(output_path, cropped_mask, cropped_image, target_crop_name):
    image_output_path = output_path / "input"
    target_output_path = output_path / "target"
    image_output_path.mkdir(parents=True, exist_ok=True)
    target_output_path.mkdir(parents=True, exist_ok=True)
    
    target_crop_path = output_path / "target" / target_crop_name
    image_crop_path = output_path / "input" / target_crop_name

    cv.imwrite(str(target_crop_path), cropped_mask)
    cv.imwrite(str(image_crop_path), cropped_image)
    #print(f"Saved cropped image and mask for {target_crop_name}.")

"""
brief: Crops an image and its mask based on the bounding box of a cell mask.
input: annotation - dict - Annotation dictionary containing cell mask, image, and metadata.
output: tuple - Cropped mask, cropped image, and target crop name.
"""
def crop_image_from_annotation(annotation):
    cell_mask = annotation['cell_mask']
    image = annotation['image']
    annotation_id = annotation['annotation_id']
    image_file = annotation['image_file']
    padding = annotation['padding']

    rle = mask_utils.encode(np.asfortranarray(cell_mask))
    bbox = mask_utils.toBbox(rle).tolist()

    x, y, w, h = map(int, bbox)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    cropped_mask = cell_mask[y:y+h, x:x+w]
    cropped_image = image[y:y+h, x:x+w]
    target_crop_name = f"{image_file.stem}_cell_{annotation_id}.png"

    return cropped_mask, cropped_image, target_crop_name

"""
brief: Saves annotation data in JSON and CSV formats.
input: annotations - list - List of annotation dictionaries to save.
       output_path - Path - Directory where files are saved.
"""
def save_json_and_csv(annotations, output_path):
    json_serializable_annotations = []
    csv_data = []

    for annotation in annotations:
        json_serializable_annotation = {
            "label_value": int(annotation["label_value"]),
            "image_id": int(annotation["image_id"]),
            "annotation_id": int(annotation["annotation_id"]),
            "image_file": str(annotation["image_file"]),
            "padding": int(annotation["padding"]),
            "area": float(annotation["area"]),
            "review_required": bool(annotation["review_required"])
        }

        json_serializable_annotations.append(json_serializable_annotation)

        csv_data.append({
            "file_name": f"{(annotation['image_file']).stem}_cell_{int(annotation['annotation_id'])}.png",
            "cell_size": float(annotation["area"]),
            "cell_class": int(annotation["label_value"]),
            "review_required": bool(annotation["review_required"])
        })

    output_json = output_path / "annotations.json"
    csv_file = output_path / "cell_data.csv"

    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as json_file:
        json.dump(json_serializable_annotations, json_file, indent=4)
    print(f"COCO annotations saved to {output_json}")

    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ["file_name", "cell_size", "cell_class", "review_required"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"Cell data saved to {csv_file}")

"""
brief: Converts non-zero pixels of images in a directory to white (255).
input: image_path (str): Directory path containing images to be processed.
"""
def transform_images_to_white(image_path):
    if not os.path.isdir(image_path):
        print(f"Error: {image_path} is not a valid directory.")
        return

    for filename in os.listdir(image_path):
        file_path = os.path.join(image_path, filename)
        
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        
        image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not read {file_path}. Skipping.")
            continue
        
        image[image > 0] = 255
        
        cv.imwrite(file_path, image)
        #print(f"Processed and saved: {file_path}")

"""
brief: Generates annotations for an image and its mask.
input: image - np.array - The image to generate annotations for.
       mask - np.array - The mask to generate annotations for.
       image_file - Path - Path to the image file.
       image_id - int - Unique identifier for the image.
       annotation_id - int - Unique identifier for the annotations.
       padding - int - Padding to add around the cell mask.
output: tuple - List of annotations and the updated annotation ID.
"""
def generate_annotations_for_image(image, mask, image_file, image_id, annotation_id, padding): 
    annotations = []

    for label_value in np.unique(mask):
        if label_value == 0:  # Skip background
            continue

        binary_mask = (mask == label_value).astype(np.uint8)
        num_labels, labels = cv.connectedComponents(binary_mask)

        for cell_label in range(1, num_labels):
            cell_mask = (labels == cell_label).astype(np.uint8)
            annotations.append({
                "label_value": label_value,
                "cell_mask": cell_mask,
                "image": image,
                "image_id": image_id,
                "annotation_id": annotation_id,
                "image_file": image_file,
                "padding": padding,
                "area": float(mask_utils.area(mask_utils.encode(np.asfortranarray(cell_mask)))),
                "review_required": False
            })
            annotation_id += 1

    return annotations, annotation_id

"""
brief: Loads an image and its mask from files.
input: image_file - Path - Path to the image file.
       mask_file - Path - Path to the mask file.
output: tuple - The image and mask loaded from the files.
"""
def get_image_and_mask(image_file, mask_file):
    image = cv.imread(str(image_file))
    mask = cv.imread(str(mask_file), cv.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"ERROR: Failed to load {image_file.name} or its mask. Skipping.")
        return None, None

    return image, mask
