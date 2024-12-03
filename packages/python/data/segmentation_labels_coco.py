import json
import argparse
import cv2 as cv
import os
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils
import csv
from segmentation_labels import remove_incomplete_blobs

"""
brief: Counts the number of blobs in a cropped binary mask.
input: crop - np.array - The binary mask crop to analyze.
output: int - The number of blobs detected in the crop.
"""
def blob_quantity_in_crop(crop):
    _, labels = cv.connectedComponents(crop)
    return labels.max()


"""
brief: Transforms a crop by removing incomplete blobs and flags it for review if needed.
input: crop - np.array - The binary mask crop to process.
       crop_name - str - Name of the crop for logging purposes.
output: tuple - Processed mask and bool indicating if review is required.
"""
def transform_crop(crop, crop_name):
    processed_crop = remove_incomplete_blobs(crop)
    blob_count = blob_quantity_in_crop(processed_crop)
    review_required = blob_count != 1
    if review_required:
        print(f"Crop {crop_name} marked for review (blob count: {blob_count}).")
    return processed_crop, review_required


"""
brief: Annotates a single cell and saves its cropped image and mask.
input: label_value - int - The class label of the cell.
       cell_mask - np.array - Binary mask of the cell.
       image - np.array - The original image.
       image_id - int - The ID of the image in the dataset.
       annotation_id - int - The unique ID for the annotation.
       csv_data - list - List to store CSV data rows.
       coco_data - dict - COCO annotations data structure.
       segmented_input - Path - Directory to save cropped input images.
       segmented_target - Path - Directory to save cropped target masks.
       image_file - Path - Original image file path.
       margin - int - Margin to add around the cropped region.
output: int - Updated annotation ID.
"""
def annotate_and_save_cell(label_value, cell_mask, image, image_id, annotation_id, csv_data, coco_data, segmented_input, segmented_target, image_file, margin):
    contours, _ = cv.findContours(cell_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
    segmentation = [contour.flatten().tolist() for contour in contours if len(contour.flatten()) >= 6]
    rle = mask_utils.encode(np.asfortranarray(cell_mask))
    area = float(mask_utils.area(rle))
    bbox = mask_utils.toBbox(rle).tolist()

    # Expand the bounding box with margin
    x, y, w, h = map(int, bbox)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)

    # Add annotation to COCO
    coco_data["annotations"].append({
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": int(label_value),
        "segmentation": segmentation,
        "area": float(area),
        "bbox": [float(x), float(y), float(w), float(h)],
        "iscrowd": int(0),
    })

    # Prepare crop for saving
    cropped_mask = cell_mask[y:y+h, x:x+w]
    cropped_image = image[y:y+h, x:x+w]
    cropped_mask, review_required = transform_crop(cropped_mask, f"{image_file.stem}_cell_{annotation_id}")

    # Save cropped data
    target_crop_name = f"{image_file.stem}_cell_{annotation_id}.png"
    target_crop_path = segmented_target / target_crop_name
    input_crop_path = segmented_input / target_crop_name

    cv.imwrite(str(target_crop_path), cropped_mask)
    cv.imwrite(str(input_crop_path), cropped_image)

    # Add to CSV
    csv_data.append({
        "file_name": target_crop_name,
        "cell_size": area,
        "cell_class": int(label_value),
        "review_required": review_required,
    })

    return annotation_id + 1


"""
brief: Processes a dataset to generate COCO annotations and crop images.
input: input_path - Path - Directory containing input images.
       target_path - Path - Directory containing target masks.
       output_path - Path - Directory to save results.
       margin - int - Margin to add around cropped regions.
       filter_threshold - float - Threshold to filter large cells.
output: None
"""
def generate_coco_annotations(input_path, target_path, output_path, margin):
    coco_data = {"images": [], "annotations": [], "categories": []}
    csv_data = []
    category_set = set()
    annotation_id = 1

    segmented_input = output_path / "input"
    segmented_target = output_path / "target"
    segmented_input.mkdir(parents=True, exist_ok=True)
    segmented_target.mkdir(parents=True, exist_ok=True)

    for image_id, image_file in enumerate(input_path.iterdir(), start=1):
        if image_file.suffix not in [".png", ".jpg", ".jpeg"]:
            continue

        mask_file = target_path / image_file.name
        if not mask_file.exists():
            print(f"WARNING: Mask for {image_file.name} not found. Skipping.")
            continue

        image = cv.imread(str(image_file))
        mask = cv.imread(str(mask_file), cv.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"ERROR: Failed to load {image_file.name} or its mask. Skipping.")
            continue

        for label_value in np.unique(mask):
            if label_value == 0:  # Skip background
                continue

            binary_mask = (mask == label_value).astype(np.uint8)
            num_labels, labels = cv.connectedComponents(binary_mask)

            for cell_label in range(1, num_labels):
                cell_mask = (labels == cell_label).astype(np.uint8)
                annotation_id = annotate_and_save_cell(
                    label_value, cell_mask, image, image_id, annotation_id,
                    csv_data, coco_data, segmented_input, segmented_target, image_file, margin
                )

        category_set.add(label_value)

    for category_id in sorted(category_set):
        coco_data["categories"].append({
            "id": int(category_id),
            "name": f"Class_{category_id}",
            "supercategory": "cell",
        })

    save_coco_and_csv(coco_data, csv_data, output_path)


"""
brief: Save COCO annotations and cell data to files.
input: coco_data - dict - COCO annotations data structure.
       csv_data - list - List of rows for the CSV.
       output_path - Path - Directory to save the files.
output: None
"""
def save_coco_and_csv(coco_data, csv_data, output_path):
    output_json = output_path / "annotations.json"
    csv_file = output_path / "cell_data.csv"

    with open(output_json, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)
    print(f"COCO annotations saved to {output_json}")

    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ["file_name", "cell_size", "cell_class", "review_required"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    print(f"Cell data saved to {csv_file}")


"""
brief: Processes all images in the given directory and modifies all non-zero pixels to white (255).
input: 
        image_path (str): The path to the directory containing images to be processed.
output:
        None: The images are saved back to their original paths after transformation.
"""
def transform_images_to_white(image_path):
    
    # Ensure the provided path is a directory
    if not os.path.isdir(image_path):
        print(f"Error: {image_path} is not a valid directory.")
        return

    # Loop over all files in the given directory
    for filename in os.listdir(image_path):
        file_path = os.path.join(image_path, filename)
        
        # Skip non-image files (you can filter by extensions like .png, .jpg, etc.)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Read the image
        image = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: Could not read {file_path}. Skipping.")
            continue
        
        # Modify all non-zero pixels to white (255)
        image[image > 0] = 255
        
        # Save the processed image back to the same path
        cv.imwrite(file_path, image)
        print(f"Processed and saved: {file_path}")


"""
brief: main function with the logic of the script
input: the arguments passed by command line
"""
def main(args):
    root_path = Path(args.root_path)
    input_dir = root_path / "input"
    target_dir = root_path / "target"
    output_dir = root_path / "segmented"

    if not input_dir.exists() or not target_dir.exists():
        print("ERROR: 'input' and 'target' directories are required in the root path.")
    else:
        generate_coco_annotations(input_dir, target_dir, output_dir, args.margin)
        transform_images_to_white(output_dir / "target")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate COCO annotations, crop cells, and save cell data.")
    parser.add_argument("--root_path", required=True, help="Root directory containing 'input' and 'target' folders.")
    parser.add_argument("--margin", type=int, default=10, help="Margin to add around each crop.")
    parser.add_argument("--filter_threshold", type=float, default=1.5, help="Threshold to filter large cells.")

    args = parser.parse_args()
    main(args)