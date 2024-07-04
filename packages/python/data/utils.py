from pycocotools.coco import COCO
from typing import Tuple
import os, shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

# Cell clases codes
dict_attributes = dict()
dict_attributes['Interfase'] = 1
dict_attributes['Profase'] = 2
dict_attributes['Metafase'] = 3
dict_attributes['Anafase'] = 4
dict_attributes['Telofase'] = 5
dict_attributes['Desconocido'] = 6


def find_files(path, extention):
  """
    brief: Given a path it searches all the files with given extention
    input: path      - string - path to directory to analize, it must not end in "/"
    input: extention - string - extention of files to look for 
    output: dict containing the path searched, and then all the subfloders with
    a list each containing all matched files in said folder 
    """
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



def process_dataset_annotations(images_path : str, annotation_file : str, output_path: str= "") -> Tuple[np.array, np.array]:
    """
    brief: Reads an annotation file and creates a dataset using the given images 
    input: images_path      - string - path to images that were annotated
    input: annotation_file - string - file containing the annotations in COCO format 
    input: output_path - string - path were the dataset will be written
    output: a tuple containing all the processed images and their respective masks
    """

    
    if len(output_path) > 0:
        # Set output path three
        if os.path.exists(output_path):
            print("Deleting output path.")
            shutil.rmtree(output_path)

        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, "input"))
        os.mkdir(os.path.join(output_path, "target"))

    # Read annotations
    coco = COCO(annotation_file)

    cat_ids = coco.getCatIds()
    annotated_ids = list()
    for id in range(len(coco.imgs)):
        if len(coco.getAnnIds(imgIds=id, catIds=cat_ids, iscrowd=None)) > 0:
            annotated_ids.append(id)
    count = len(annotated_ids)

    print("Found %d annotations"%count)

    # Loop over samples
    for idx, image_id in tqdm(enumerate(annotated_ids), total=len(annotated_ids)):
        # Get image info
        img = coco.imgs[image_id]
        # Load image
        image = np.asarray(Image.open(os.path.join(images_path,img['file_name'])).convert('L'))
        if idx == 0:
            # Reserve memory
            images = np.zeros((len(annotated_ids), image.shape[0], image.shape[1], 1), dtype=np.uint8)
            masks = np.zeros_like(images)
        images[idx, :, :, 0] = image
        # Get labels
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        # Construct mask
        mask_aux = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask_aux[coco.annToMask(anns[i]) != 0] = dict_attributes[anns[i]['attributes']['Fase']]
        masks[idx, :, :, 0] = mask_aux

    if len(output_path) > 0:
        # Save dataset
        for idx, image_id in tqdm(enumerate(annotated_ids), total=len(annotated_ids)):
            # Save
            im = Image.fromarray(images[idx, :, :, 0] )
            im.save(os.path.join(os.path.join(output_path, "input"), "%d.png"%image_id))
            im = Image.fromarray(masks[idx, :, :, 0])
            im.save(os.path.join(os.path.join(output_path, "target"), "%d.png"%image_id))

    return images, masks
