from pathlib import Path
import pandas as pd
import numpy as np
import cv2 as cv
import argparse
import os

"""
brief: Given a path it searches all the files with given extention
input: path - string - path to directory to analize, it must not end in "/"
output: dict containing the path searched, and then all the subfloders with
a list each containing all matched files in said folder 
"""
def find_files(path, extention):
  dir_dict = {}  # Dictionary to store images in each sudirectory in path
  dir_dict['path'] = path # Add the path to the dict

  for root, _, files in os.walk(path):
      
      dir = os.path.relpath(root, path) #Get actual folder in loop
      paths = [] # List to store image paths

      # Process files in the current subdir
      for file in sorted(files):
          if file.endswith(extention):
              png_file_path = os.path.join(root, file)
              paths.append(png_file_path)

      # If the folder contains given files, i store them in the dict
      if (dir not in dir_dict) and (len(paths) > 0):
          dir_dict[dir] = paths

  return dir_dict

"""
brief: Given a dictionary containing the path to the parent folder, and the
names of subfolders with a list of images in each one it cuts the images
and creates a csv file where it stores metadata of the cutted images
input: paths - dictionary - data structure that contains all paths to images
input: height - number - pixels of the subimage to cut corresponding to the height
input: width - number - pixels of the subimage to cut corresponding to the width
input: metadata - bool - Indicates whether to create metadata or not
output: null
"""
def cut_images(paths, height, width, metadata):

  #Create folder path to store cutted images
  path = paths.pop('path')
  path = path[:-1] if path.endswith('/') else path # If it ends with / delete it
  new_path = path + f'_{height}x{width}'

  classes = [0, 1, 2, 3, 4, 5, 6, 7]
  metadata_dir = 'target'
  subimages_classes = []

  for subfolder in paths:
    output_dir = Path(new_path + '/' + subfolder)
    output_dir.mkdir(parents=True, exist_ok=True) # I create all folderes if they do not exist
    print(f"Cutting images in: {output_dir}")

    # Loop through and cut each image
    for idx, image_file in enumerate(paths[subfolder]):

        # Open image and get data from it
        print(f"Image: {idx + 1}/{len(paths[subfolder])}", end='\r')
        image = cv.imread(image_file)
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        image_height, image_width = image.shape[:2]

        # Cut images with given height and width
        # The overlap will be the height and width divided by 2
        for i in range(0, image_height - int(height/2), int(height/2)):
            for j in range(0, image_width - int(width/2), int(width/2)):

                subimage = image[i:i+height, j:j+width]
                # If subimage dimensions are not the provided discard the cut,
                # This might occur if the given w or h are not divisible by the orginal shape
                h, w = subimage.shape[:2]
                if h!=height or w!=width:
                  continue

                subimage_name = f"{image_name}_{i}_{j}.png"

                # Get classes from target images
                if subfolder == metadata_dir and metadata:
                  subimage_classes = np.unique(subimage)
                  subimages_classes.append([new_path] + [subimage_name] + [cls in subimage_classes for cls in classes])

                # Save cutted image
                subimage_path = os.path.join(output_dir, subimage_name)
                cv.imwrite(subimage_path, subimage)

    # When finished with the target folder, create metadata of its images
    if subfolder == metadata_dir and metadata:
      print("Writing metadata")
      df = pd.DataFrame(subimages_classes, columns=['path', 'image_name', 0, 1, 2, 3, 4, 5, 6, 7])
      df.to_csv(f"{new_path}/metadata.csv", index=False)

"""
brief: main function with the logic of the script
input: the arguments passed by command line
"""
def main(args):
  if os.path.exists(args.path):
    ext = '.' + args.type if not args.type.startswith('.') else args.type
    paths = find_files(args.path, ext)
    if len(paths.keys()) > 1:
      cut_images(paths, args.height, args.width, args.metadata)
    else:
      print("ERROR: No images found on given directory")
  else:
    print("ERROR: Given path is not valid")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Script for cutting images into smaller parts for data augmentation")
  parser.add_argument("--width",    type=int,  default=512, help="Width to use when cutting images, default is 512")
  parser.add_argument("--height",   type=int,  default=512, help="Height to use when cutting images, default is 512")
  parser.add_argument("--type",     type=str,  default='.png', help="File extention to look for")
  parser.add_argument("--metadata", type=bool, default=True, help="Create csv metadata of image classes")
  parser.add_argument("path", help="Parent directory where the images are")

  args = parser.parse_args()

  main(args)