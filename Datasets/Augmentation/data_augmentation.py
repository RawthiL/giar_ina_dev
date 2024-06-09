from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import cv2 as cv
from pathlib import Path
import argparse
import numpy as np

"""
brief: Given a path of the image and the target, it returns the modified image in terms of brightness, contrast, saturation and flip and target in terms of flip
input: paths - string - paths of the image and the target
output: modified images
"""

def change_image(path_image, path_target):

     # Load images
    image = Image.open(path_image)
    target_image = Image.open(path_target)
    new_height, new_width = image.size # Get the height and width of the loaded image

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

"""
brief: Given a path of the balanced metadata, it returns a dataframe with new images and the updated metadata file.
input: path - string - path to directory of the balanced metadata to analize
output:-
"""

def augmententation(path_metadata):

    #Open metadata of the balanced dataset
    df = pd.read_csv(path_metadata+"/metadata_balanced.csv")

    #check stadistics of each type of cell
    dict_len_categories = {}
    for col2 in df.columns[3:]:  # Exclude the columns 'path' ,'image_name', 0 (background)
        dict_len_categories[col2] = df[col2].sum() #Total observations per cell type

    valor_max=max(dict_len_categories.values()) # Total number of the cell type with the highest number of samples
    cat_max = max(dict_len_categories, key=dict_len_categories.get) #Cell type with the highest number of samples
    sorted_cat = sorted(dict_len_categories, key=dict_len_categories.get) # Cell types ordered from least to most samples


    print(f"total by category : 1={dict_len_categories['1']}, 2={dict_len_categories['2']} , 3={dict_len_categories['3']}, 4={dict_len_categories['4']}, 5={dict_len_categories['5']}, 6={dict_len_categories['6']}")

    #Iterate through and analyze each of the cell types
    for col in sorted_cat: #
        df_to_merge=pd.DataFrame(columns=['path', 'image_name', '0', '1', '2', '3', '4', '5', '6']) #empty dataframe to merge later in each iteration in the metadata file with the new samples
        df_tmp=df[(df[col]== True)]
        i=0
        while len(df_tmp)<valor_max: #Check in each loop that the data augmentation does not exceed the maximum number of samples of the cell type with the highest number of samples
            df_tmp2=df[(df[col]== True) & (df[cat_max]== False) ] #exclude type of cell with the highest number of samples
            i+=1
            f = df_tmp2.sample(n=1) # Choose a random sample
            #input: select the path of the image - path_image_input and select the new path for the transformed image - path_input
            path_image_input=Path(f["path"].iloc[0]+f"/{f['image_name'].iloc[0]}")
            path_input = os.path.join(f["path"].iloc[0], f["image_name"].iloc[0].split(".")[0] + f"_{i}." + f["image_name"].iloc[0].split(".")[1])

            #target: select the path of the image - path_image_target and select the new path for the transformed image - path_target
            path_image_target=Path(f["path"].iloc[0]+f"/{f['image_name'].iloc[0]}")
            path_target = os.path.join(f["path"].iloc[0].replace("input", "target"), f["image_name"].iloc[0].split(".")[0]+f"_{i}."+f["image_name"].iloc[0].split(".")[1])

            # transform input image and the respective target
            augmented_image, augmented_target_image = change_image(path_image_input,path_image_target)

            #Save images
            augmented_image.save(path_input)
            augmented_target_image.save(path_target)

            #Update the metadata file with the new samples
            f['path'] = f["path"].iloc[0]
            f['image_name'] = f["image_name"].iloc[0].split(".")[0]+f"_{i}." + f["image_name"].iloc[0].split(".")[1]

            #Concatenate the row at the end of the DataFrame
            df_tmp=pd.concat([df_tmp, f], ignore_index=True) # Necessary to check condition in the while loop "len(df_tmp)<valor_max"
            df_to_merge = pd.concat([df_to_merge, f], ignore_index=True)
        df = pd.concat([df,df_to_merge], ignore_index=True) #Add all the new images of the category in the metadata file

    #Balance the dataset

    dict_len_categories = {}
    for col2 in df.columns[3:]:  # Exclude the columns 'path' ,'image_name', 0 (background) ; recalcula los totales en cada iteración
        dict_len_categories[col2] = df[col2].sum() #Total observations per cell type
    sorted_cat = sorted(dict_len_categories, key=dict_len_categories.get,reverse=True) # Cell types ordered from most to least samples
    for col in sorted_cat:
        cat=["1","2","3","4","5","6"]
        cat.remove(col)
        df_tmp2=df[(df[col]== True)]
        nro_total_drop= len(df_tmp2) - valor_max #Total number of samples to drop of the type of cell taking in considaration the type of cell with the highest number samples of the first input metadata
        mask = (df[col]== True) & (df.loc[:,cat].eq(False).all(axis=1))
        df_tmp=df[mask]

        if nro_total_drop > 0:
            if len(df_tmp)>=nro_total_drop:
                indices_to_drop = np.random.choice(df_tmp.index, nro_total_drop, replace=False)
                df=df.drop(indices_to_drop)
            else:
                indices_to_drop = np.random.choice(df_tmp.index, len(df_tmp), replace=False)
                df=df.drop(indices_to_drop)
                df_tmp2=df[df[col]== True]
                indices_to_drop = np.random.choice(df_tmp2.index, nro_total_drop - len(df_tmp), replace=False)
                df=df.drop(indices_to_drop)

    dict_len_categories = {}
    for col2 in df.columns[3:]:  # Exclude the columns 'path' ,'image_name', 0 (background)
        dict_len_categories[col2] = df[col2].sum() #Total observations per cell type
    print(f"updated total by category : 1={dict_len_categories['1']}, 2={dict_len_categories['2']} , 3={dict_len_categories['3']}, 4={dict_len_categories['4']}, 5={dict_len_categories['5']}, 6={dict_len_categories['6']}")
    df.to_csv(f"{path_metadata}/metadata_balanced.csv", index=False)

def main(args):
  if os.path.exists(args.path_metadata):
    df = augmententation(args.path_metadata)
  else:
    print("ERROR: Given path is not valid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for data augmentation")
    parser.add_argument("path_metadata", help="Parent directory where the balanced metadata file is located")
    args = parser.parse_args()
    main(args)
