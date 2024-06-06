import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import os


"""
brief: Finds the centers and sizes of the blobs in a binary image.
input: image - np.array - The input image in numpy array format.
output: tuple - Lists of blob centers and sizes detected.
"""
def find_blob_centers_and_sizes(image):
    print("Detecting blobs in the image...")
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray_image, (5, 5), 0)
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    centers = []
    sizes = []
    
    for contour in contours:
        M = cv.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            size = cv.contourArea(contour)
            centers.append((cx, cy))
            sizes.append(size)
    
    print(f"Detected {len(centers)} blobs.")
    return centers, sizes


"""
brief: Cuts a square subimage around a central point with a given size and padding.
input: image - np.array - The input image in numpy array format.
       center - tuple - Coordinates (x, y) of the blob center.
       size - int - Size of the crop.
       padding - int - Additional padding around the crop.
output: np.array - The cropped image.
"""
def cut_image_around_center(image, center, size, padding):
    cx, cy = center
    half_size = size // 2
    x_start = max(cx - half_size - padding, 0)
    y_start = max(cy - half_size - padding, 0)
    x_end = min(cx + half_size + padding, image.shape[1])
    y_end = min(cy + half_size + padding, image.shape[0])
    
    return image[y_start:y_end, x_start:x_end]


"""
brief: Filters out blobs that exceed an average size by a given threshold.
input: centers - list - List of blob centers' coordinates.
       sizes - list - List of blob sizes.
       threshold - float - Threshold to filter out large blobs.
output: list - List of filtered tuples of centers and sizes.
"""
def filter_large_blobs(centers, sizes, threshold):
    print("Filtering large blobs...")
    average_size = np.mean(sizes)
    filtered = [(center, size) for center, size in zip(centers, sizes) if size <= average_size * (1 + threshold / 100)]
    print(f"{len(filtered)} blobs remaining after filtering.")
    return filtered


"""
brief: Saves the metadata to a CSV file.
input: metadata - dict - Dictionary with blob information.
       output_dir - Path - Directory where the CSV file will be saved.
output: None
"""
def save_metadata(metadata, output_dir):
    print("Saving metadata to CSV file...")
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_dir / 'metadata.csv', index=False)
    print("Metadata successfully saved.")


"""
brief: Processes an image: detects blobs, cuts them, and saves relevant data.
input: image_file - str - Name of the image file.
       args - argparse.Namespace - Script arguments.
       output_dir - Path - Output directory.
       metadata - dict - Dictionary to store blob information.
output: None
"""
def process_image(image_file, args, output_dir, metadata):
    print(f"Processing image {image_file}...")
    image = cv.imread(os.path.join(args.path, "target", image_file))
    centers, sizes = find_blob_centers_and_sizes(image)
    base_name = os.path.splitext(image_file)[0]
    
    filtered_centers_sizes = filter_large_blobs(centers, sizes, args.threshold)

    for i, (center, size) in enumerate(filtered_centers_sizes):
        patch_target = cut_image_around_center(image, center, 100, args.padding)  # Define the crop size (adjust as needed)

        # Save data for each blob in target
        metadata['filename'].append(f"{base_name}_{i}")
        metadata['blob_size'].append(size)
        metadata['class'].append(np.random.randint(0, 7))  # Assign a random class from 0 to 7

        # Crop target images based on these blobs
        target_output_dir = output_dir / 'target' / base_name
        target_output_dir.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(target_output_dir / f"{base_name}_p_{i}_m.png"), patch_target)

        # Crop input images based on these blobs
        input_image_path = os.path.join(args.path, "input", f"{base_name}.png")
        if os.path.exists(input_image_path):
            input_image = cv.imread(input_image_path)
            patch_input = cut_image_around_center(input_image, center, 100, args.padding)
            
            input_output_dir = output_dir / 'input' / base_name
            input_output_dir.mkdir(parents=True, exist_ok=True)
            cv.imwrite(str(input_output_dir / f"{base_name}_p_{i}.png"), patch_input)
    print(f"Processing of image {image_file} completed.")


"""
brief: main function with the logic of the script
input: the arguments passed by command line
"""
def main(args):
    if not os.path.exists(args.path):
        print("ERROR: The provided path is not valid")
        return

    output_dir = Path(args.path) / 'segmented'
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {'filename': [], 'blob_size': [], 'class': []}

    # Find blobs in target
    target_folder = os.path.join(args.path, "target")
    target_image_files = [f for f in os.listdir(target_folder) if f.endswith(args.type)]

    for image_file in target_image_files:
        process_image(image_file, args, output_dir, metadata)

    save_metadata(metadata, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for segmenting cells in images")
    parser.add_argument("--type", type=str, default='.png', help="File extension to search for, default is .png")
    parser.add_argument("--padding", type=int, default=100, help="Padding to add around the detected blob")
    parser.add_argument("--threshold", type=float, default=50.0, help="Percentage threshold to eliminate large blobs")
    parser.add_argument("path", help="Main directory where the images are located")

    args = parser.parse_args()
    main(args)
