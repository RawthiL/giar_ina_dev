import pandas as pd
import numpy as np
import cv2 as cv
import argparse
import os

def find_files(path, extention):
    """
    Finds all files with the given extension within a directory and its subdirectories.

    Args:
        path (str): Path to the directory to search.
        extension (str): File extension to search for (e.g., '.png').

    Returns:
        list: List of full paths to all matching files.
    """

    dir_list = []
    for root, _, files in os.walk(path):
        for file in sorted(files):
            if file.endswith(extention):
                png_file_path = os.path.join(root, file)
                dir_list.append(png_file_path)
    return dir_list

def get_og_area_of_eroded_shape(image, cnt, kernel):
    """
    Gets the area of the selected countour, dilates it with the kernel thar eroded it 
    and calculates the area. It has an error less than 1%

    Args:
        image: Eroded image (NumPy array).
        cnt: Contour (NumPy array).
        kernel: Kernel with which the image was eroded (NumPy array)

    Returns:
        Number: area of the original shape
    """

    # Get new image only with one shape
    mask = np.zeros(image.shape[:2], np.uint8)  # Create a mask with zeros
    cv.drawContours(mask, cnt, -1, (255, 255, 255), -1) # Fill mask with white shape
    extracted_shape = cv.bitwise_and(image, image, mask=mask) # Apply mask to image
    dilated_img = cv.dilate(extracted_shape, kernel) 
    contours, _ = cv.findContours(dilated_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # Get the countour to calculate area
    x,y,w,h = cv.boundingRect(contours[0])

    return cv.contourArea(contours[0]), x, y, w, h

def complete_cells_detection(paths, output, kernel, cell_analysis):
    """
    Analyzes a list of images to count complete cells (not touching edges).

    Args:
        image_paths (list): List of full paths to the images.
        output_path (str): Path to save the CSV file containing results.
    """
        
    complete_cells_per_image = []
    complete_cells_data = []

    for i, image_path in enumerate(paths):

        print(f"Scanning image: {i + 1}/{len(paths)} ({int((i + 1)*100/len(paths))}%)", end='\r')

        image = cv.imread(image_path)

        # Convert the image to grayscale if it is not already
        try:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        except:
            pass

        _, thresh_image = cv.threshold(image, 0, 255, cv.THRESH_BINARY)
        thresh_image = cv.erode(thresh_image, kernel)

        # Geting contours from the thresholded image
        contours, _ = cv.findContours(thresh_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        # Iterate contours to check if it's bounding rect touches the border, if it does not is a complete cell
        complete_cells = 0
        for i, contour in enumerate(contours):
            x,y,w,h = cv.boundingRect(contour)
            if not (x == 0 or y == 0 or x + w == image.shape[1] or y + h == image.shape[0]):
                complete_cells += 1
                if cell_analysis:
                    a, x, y, w, h = get_og_area_of_eroded_shape(thresh_image, contour, kernel)
                    complete_cells_data.append([a, x, y, w, h, w*h, os.path.basename(image_path)])
        complete_cells_per_image.append([os.path.basename(image_path), complete_cells]) # Save with the name of the analyzed folder

    # Create CSV with complete cells per image (with the name of the analyzed folder)
    cells_per_image_df = pd.DataFrame(complete_cells_per_image, columns=['image','complete_cells'])
    csv_path = f"{output}/complete_cells_per_image_in_{os.path.basename(os.path.dirname(image_path))}.csv"
    cells_per_image_df.to_csv(csv_path, index=True)
    print(f"CSV file created: {csv_path}")

    if cell_analysis:
        # Create CSV with data of all complete cells
        complete_cells_data_df = pd.DataFrame(complete_cells_data, columns=["area","x","y","w","h","bbox_area","image"])
        csv_path = f"{output}/complete_cells_data_in_{os.path.basename(os.path.dirname(image_path))}.csv"
        complete_cells_data_df.to_csv(csv_path, index=True)
        print(f"CSV file created: {csv_path}")

        areas   = np.array(complete_cells_data_df['area'])
        widths  = np.array(complete_cells_data_df['w'])
        heights = np.array(complete_cells_data_df['h'])

        print(f"AREA   - Average: {round(np.average(areas),2)} - Min: {np.min(areas)} - Max: {np.max(areas)}")
        print(f"WIDTH  - Average: {round(np.average(widths),2)} - Min: {np.min(widths)} - Max: {np.max(widths)}")
        print(f"HEIGHT - Average: {round(np.average(heights),2)} - Min: {np.min(heights)} - Max: {np.max(heights)}")

def main(args):
    """
    Main function with the logic of the script
    Args:
        args (list): the arguments passed by command line
    """

    paths = find_files(args.path, args.type)
    kernel = np.ones((args.kernel, args.kernel), np.int8)
    # eroded = if args.eroded 
    if len(paths) > 0:
        complete_cells_detection(paths, args.outputDir, kernel, args.cellAnalysis)
        pass
    else: 
      print("ERROR: No images found on given directory")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for analizing cell images and determine how many complete cells they have")
    parser.add_argument("--outputDir",   type=str,  default="",     help="Directory where to store the csv output, default is parent of PATH")
    parser.add_argument("--type",        type=str,  default='.png', help="File extention to look for, default is .png")
    parser.add_argument("--kernel",      type=int,  default=27,     help="Define the NxN kernel to use, where N is the input parameter")
    parser.add_argument("--cellAnalysis",type=bool, default=False,  help="Decides whether to include the analysis of complete cells or not")
    parser.add_argument("path", help="Directory where the images are")

    # Argument processing
    args = parser.parse_args()
    args.path = args.path[:-1] if args.path.endswith('/') else args.path # If it ends with / delete it
    args.type = '.' + args.type if not args.type.startswith('.') else args.type # If it does not start with . add it

    if args.outputDir == "":
        args.outputDir = os.path.dirname(args.path)

    if os.path.exists(args.path):
        if os.path.exists(args.outputDir):
                main(args) # Only if both paths exits execute main function
                pass
        else:
            print("ERROR: Output path is not valid")
    else:
        print("ERROR: Path is not valid")