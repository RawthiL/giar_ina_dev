from pycocotools.coco import COCO
from typing import Tuple
import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2 as cv
from typing import List


# Image clases codes
dict_attributes = dict()
dict_attributes["Fondo"] = 0
dict_attributes["Interfase"] = 1
dict_attributes["Profase"] = 2
dict_attributes["Metafase"] = 3
dict_attributes["Anafase"] = 4
dict_attributes["Telofase"] = 5
dict_attributes["Desconocido"] = 6


def get_image_metadata_columns():
    return ["image"] + list(dict_attributes.keys()) + ["split"]


def get_image_metadata(image_name, img, split="None"):
    """
    brief: Given an image, returns the metadata.
    input: img      - np.array - image
    output: a list of metadata fields
    """

    # Currently this only returns a list of booleas wheter the image cointains each class
    image_classes = np.unique(img)

    classes = list(dict_attributes.values())

    return [image_name] + [cls in image_classes for cls in classes] + [split]


def find_files(path, extention):
    """
    brief: Given a path it searches all the files with given extention
    input: path      - string - path to directory to analize, it must not end in "/"
    input: extention - string - extention of files to look for
    output: dict containing the path searched, and then all the subfloders with
    a list each containing all matched files in said folder
    """
    dir_dict = {}  # Dictionary to store images in each sudirectory in path
    dir_dict["path"] = path  # Add the path to the dict

    for root, _, files in os.walk(path):
        dir = os.path.relpath(root, path)  # Get actual folder in loop
        paths = []  # List to store image paths

        # Process files in the current subdir
        for file in sorted(files):
            if file.endswith(extention):
                png_file_path = os.path.join(root, file)
                paths.append(png_file_path)

        # If the folder contains given files, i store them in the dict
        if (dir not in dir_dict) and (len(paths) > 0):
            dir_dict[dir] = paths

    return dir_dict


def process_dataset_annotations(
    images_path: str, annotation_file: str, output_path: str = ""
) -> Tuple[np.array, np.array]:
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

    print("Found %d annotations" % count)

    metadata = list()
    # Loop over samples
    for idx, image_id in tqdm(enumerate(annotated_ids), total=len(annotated_ids)):
        # Get image info
        img = coco.imgs[image_id]
        # Load image
        image = np.asarray(
            Image.open(os.path.join(images_path, img["file_name"])).convert("L")
        )
        if idx == 0:
            # Reserve memory
            images = np.zeros(
                (len(annotated_ids), image.shape[0], image.shape[1], 1), dtype=np.uint8
            )
            masks = np.zeros_like(images)
        images[idx, :, :, 0] = image
        # Get labels
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img["id"], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        # Construct mask
        mask_aux = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask_aux[coco.annToMask(anns[i]) != 0] = dict_attributes[
                anns[i]["attributes"]["Fase"]
            ]
        masks[idx, :, :, 0] = mask_aux
        img_name = "%d.png" % image_id
        metadata.append(get_image_metadata(img_name, mask_aux))
    metadata_df = pd.DataFrame(metadata, columns=get_image_metadata_columns())

    if len(output_path) > 0:
        # Save dataset
        for idx, image_id in tqdm(enumerate(annotated_ids), total=len(annotated_ids)):
            img_name = "%d.png" % image_id
            # Save
            im = Image.fromarray(images[idx, :, :, 0])
            im.save(os.path.join(os.path.join(output_path, "input"), img_name))
            im = Image.fromarray(masks[idx, :, :, 0])
            im.save(os.path.join(os.path.join(output_path, "target"), img_name))

        metadata_df.to_csv(os.path.join(output_path, "metadata.csv"), index=True)

    return images, masks, metadata_df


def metadata_balanced(metadata_df):
    """
    brief: Given a dataset metadata, it returns the a more balanced dataframe
    input: path - string - path to directory of the metadata to analize
    output: balanced dataframe
    """

    mask = (metadata_df["Fondo"]) & metadata_df.loc[:, "Interfase":"Desconocido"].eq(
        False
    ).all(axis=1)
    metadata_df = metadata_df.drop(
        index=metadata_df[mask].index
    )  # discart images that only contain background
    # Analyze if the dataset is balanced

    dict_len_categories = {}
    skip_cols = 2  # Exclude the columns'image_name', "Fondo"
    for col in metadata_df.columns[skip_cols : len(dict_attributes) + 1]:
        dict_len_categories[col] = metadata_df[
            col
        ].sum()  # Total observations per cell type

    total_dict = sum(dict_len_categories.values())
    percentages = (
        np.array(list(dict_len_categories.values())) / total_dict
    )  # percentage of each category from 1 to 6
    percentages_dict = {
        key: percentage
        for key, percentage in zip(dict_len_categories.keys(), percentages)
    }

    mean_percentage = np.mean(percentages)
    std_percentage = np.std(percentages)
    cv_percentage = (
        std_percentage / mean_percentage
    )  # coefficient of variation CV ,statistical measure used to assess the relative variability of a dataset compared to its mean

    cv_max = 0.4  # Maximum acceptable coefficient of variation (CV)
    per_max = 0.2  # Maximum acceptable percentage per category

    check_imbalance = cv_percentage > cv_max or np.any(
        percentages > per_max
    )  # True for unbalanced, false for balanced.
    print("total by category :")
    for key in dict_len_categories.keys():
        print(f"\t{key} = {dict_len_categories[key]}")

    # Balancing dataset

    if check_imbalance:
        min_cat = min(
            percentages_dict, key=percentages_dict.get
        )  # Feature with the least number of observations.
        min_perc = percentages_dict[min_cat]  # Percentage of that feature.
        for col in metadata_df.columns[skip_cols : len(dict_attributes) + 1]:
            perc = (
                metadata_df[col].sum() / total_dict
            )  # Calculate the updated percentage after discarding rows for each feature in each cycle of the for loop
            if (
                col != min_cat and perc > per_max
            ):  # Exclude the feature with the lowest percentage and check the maximum percentage
                perc_ajustada = (
                    perc - (min_perc + (perc - min_perc) * 0.5)
                )  # Adjustment percentage of the feature decreases as it approaches the minimum value.
                nro_total_drop = int(
                    perc_ajustada * total_dict
                )  # Total number of rows to remove per category.

                cat = list(dict_attributes.keys())
                cat.remove(col)
                mask = (
                    (metadata_df[col])
                    & (1 - metadata_df[min_cat])
                    & (metadata_df[col])
                    & (metadata_df.loc[:, cat].eq(False).all(axis=1))
                )
                df_tmp = metadata_df[mask]

                if len(df_tmp) >= nro_total_drop:
                    indices_to_drop = np.random.choice(
                        df_tmp.index, nro_total_drop, replace=False
                    )
                    metadata_df = metadata_df.drop(indices_to_drop)
                else:
                    indices_to_drop = np.random.choice(
                        df_tmp.index, len(df_tmp), replace=False
                    )
                    metadata_df = metadata_df.drop(indices_to_drop)
                    df_tmp2 = metadata_df[
                        (metadata_df[col]) & (1 - metadata_df[min_cat])
                    ]
                    indices_to_drop = np.random.choice(
                        df_tmp2.index, nro_total_drop - len(df_tmp), replace=False
                    )
                    metadata_df = metadata_df.drop(indices_to_drop)

                dict_len_categories = {}  # Update dictionary after the drop operation
                for col in metadata_df.columns[skip_cols : len(dict_attributes) + 1]:
                    dict_len_categories[col] = metadata_df[col].sum()
                total_dict = sum(dict_len_categories.values())
    print("updated total by category  :")
    for key in dict_len_categories.keys():
        print(f"\t{key} = {dict_len_categories[key]}")

    return metadata_df


def create_dataset(dataset_path, dataset_name, metadata_df, output_path=None):
    """
    brief: Given a metadata dataframe, it creates the directory and subfolders of train and validation datasets
    input: path - string - path to directory of the dataset
    input: metadata_df - metadata dataframe
    output: null
    """

    # split the dataset in train and validation
    metadata_df["prefix"] = metadata_df["image"].apply(
        lambda x: "_".join(x.split("_")[:2])
    )
    unique_prefixes = metadata_df["prefix"].unique()
    train_prefixes, val_prefixes = train_test_split(
        unique_prefixes, test_size=0.2, random_state=42
    )
    train_df = metadata_df[metadata_df["prefix"].isin(train_prefixes)]
    val_df = metadata_df[metadata_df["prefix"].isin(val_prefixes)]

    if output_path is not None:
        # Create folder path to store cutted images
        new_path = os.path.join(output_path, dataset_name)
        # create directories
        input_train_dir = Path(os.path.join(new_path, "input", "train"))
        input_train_dir.mkdir(parents=True, exist_ok=True)
        input_val_dir = Path(os.path.join(new_path, "input", "validation"))
        input_val_dir.mkdir(parents=True, exist_ok=True)
        target_train_dir = Path(os.path.join(new_path, "target", "train"))
        target_train_dir.mkdir(parents=True, exist_ok=True)
        target_val_dir = Path(os.path.join(new_path, "target", "validation"))
        target_val_dir.mkdir(parents=True, exist_ok=True)

    # Updated metadata
    # Crear un DataFrame vacío para almacenar el metadata nueva
    metadata_new = pd.DataFrame(columns=get_image_metadata_columns())

    # Listas para almacenar las filas temporales
    rows_train = []
    rows_val = []

    # Iterar sobre las filas de train_df
    count = 0
    for i, v in train_df.iterrows():
        # Leer y guardar la imagen de entrada
        image_input_train = np.asarray(
            Image.open(os.path.join(dataset_path, "input", v.iloc[0])).convert("L")
        )
        if output_path is not None:
            path_input_train = os.path.join(input_train_dir, v.iloc[0])
            cv.imwrite(path_input_train, image_input_train)

        # Leer y guardar la imagen de destino
        image_target_train = np.asarray(
            Image.open(os.path.join(dataset_path, "target", v.iloc[0])).convert("L")
        )
        if output_path is not None:
            path_target_train = os.path.join(target_train_dir, v.iloc[0])
            cv.imwrite(path_target_train, image_target_train)

        # Crear un diccionario con los valores a agregar
        new_row = dict()
        for idx, column in enumerate(get_image_metadata_columns()):
            if column == "split":
                new_row[column] = "Train"
            else:
                new_row[column] = v.iloc[idx]

        # Agregar el diccionario a la lista de filas de entrenamiento
        rows_train.append(new_row)

        if count == 0:
            train_images = np.zeros(
                (
                    len(train_df),
                    image_input_train.shape[0],
                    image_input_train.shape[1],
                    1,
                ),
                dtype=np.uint8,
            )
            train_target = np.zeros_like(train_images)
        train_images[count, :, :, 0] = image_input_train
        train_target[count, :, :, 0] = image_target_train
        count += 1

    # Iterar sobre las filas de val_df
    count = 0
    for i, v in val_df.iterrows():
        # Leer y guardar la imagen de entrada
        image_input_val = np.asarray(
            Image.open(os.path.join(dataset_path, "input", v.iloc[0])).convert("L")
        )
        if output_path is not None:
            path_input_val = os.path.join(input_val_dir, v.iloc[0])
            cv.imwrite(path_input_val, image_input_val)

        # Leer y guardar la imagen de destino
        image_target_val = np.asarray(
            Image.open(os.path.join(dataset_path, "target", v.iloc[0])).convert("L")
        )
        if output_path is not None:
            path_target_val = os.path.join(target_val_dir, v.iloc[0])
            cv.imwrite(path_target_val, image_target_val)

        # Crear un diccionario con los valores a agregar
        new_row = dict()
        for idx, column in enumerate(get_image_metadata_columns()):
            if column == "split":
                new_row[column] = "Validation"
            else:
                new_row[column] = v.iloc[idx]

        # Agregar el diccionario a la lista de filas de validación
        rows_val.append(new_row)

        if count == 0:
            validation_images = np.zeros(
                (len(val_df), image_input_val.shape[0], image_input_val.shape[1], 1),
                dtype=np.uint8,
            )
            validation_target = np.zeros_like(validation_images)
        validation_images[count, :, :, 0] = image_input_val
        validation_target[count, :, :, 0] = image_target_val
        count += 1

    # Convertir las listas de filas a DataFrames y concatenarlas con metadata_new
    metadata_new = pd.concat(
        [metadata_new, pd.DataFrame(rows_train), pd.DataFrame(rows_val)],
        ignore_index=True,
    )
    if output_path is not None:
        metadata_new.to_csv(f"{new_path}/metadata.csv", index=False)

    return (
        train_images,
        train_target,
        validation_images,
        validation_target,
        metadata_new,
    )


def find_blob_centers_and_sizes(image, blur_kernel=(5, 5), verbose=False):
    """
    brief: Finds the centers and sizes of the blobs in a binary image.
    input: image - np.array - The input image in numpy array format.
    output: tuple - Lists of blob centers and sizes detected.
    """
    if verbose:
        print("Detecting blobs in the image...")

    blurred = cv.GaussianBlur(image, blur_kernel, 0)
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    centers = []
    sizes = []

    for contour in contours:
        M = cv.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            size = cv.contourArea(contour)
            centers.append((cx, cy))
            sizes.append(size)
    if verbose:
        print(f"Detected {len(centers)} blobs.")
    return centers, sizes


def filter_large_blobs(centers, sizes, threshold_sigma=2, verbose=False):
    """
    brief: Filters out blobs that exceed an average size by a given threshold.
    input: centers - list - List of blob centers' coordinates.
        sizes - list - List of blob sizes.
        threshold - float - Threshold to filter out large blobs.
    output: list - List of filtered tuples of centers and sizes.
    """
    average_size = np.mean(sizes)
    std_size = np.std(sizes, ddof=1)
    filtered = [
        (center, size)
        for center, size in zip(centers, sizes)
        if size <= average_size + threshold_sigma * std_size
    ]
    if verbose:
        print(f"{len(filtered)} blobs remaining after filtering.")
    return filtered


def cut_image_around_center(image, center, size, padding):
    """
    brief: Cuts a square subimage around a central point with a given size and padding.
    input: image - np.array - The input image in numpy array format.
        center - tuple - Coordinates (x, y) of the blob center.
        size - int - Size of the crop.
        padding - int - Additional padding around the crop.
    output: np.array - The cropped image.
    """
    image = np.asarray(image)
    cx, cy = center
    half_size = size // 2
    x_start = max(cx - half_size - padding, 0)
    y_start = max(cy - half_size - padding, 0)
    x_end = min(cx + half_size + padding, image.shape[1])
    y_end = min(cy + half_size + padding, image.shape[0])

    return image[y_start:y_end, x_start:x_end]


def centred_cells_dataset(
    images: List[Image],
    masks: List[Image],
    target_size,
    padding=0,
    blur_kernel=(5, 5),
    threshold_sigma=2,
    verbose=True,
):
    """
    brief: Extracts all the cells from a list of images and masks.
    input:
        images: List of images to cut.
        masks: List of masks asociated to images.
        target_size: Size of the cropped cell images.
        padding : Additional padding around the crop.
    output: A pair of lists containing the images and masks of the cells.
    """

    images_out = list()
    masks_out = list()

    # For each image
    for image_use, mask_use in zip(images, masks):
        # Convert mask to binary
        mask_cell = ((np.asarray(mask_use) > 0) * 255).astype(np.uint8)

        # Find blobs
        centers, sizes = find_blob_centers_and_sizes(
            mask_cell, blur_kernel=blur_kernel, verbose=verbose
        )
        # Keep large one
        filtered_centers_sizes = filter_large_blobs(
            centers, sizes, threshold_sigma=threshold_sigma, verbose=verbose
        )

        # Extract all cells
        for i, (center, size) in enumerate(filtered_centers_sizes):
            # Cut cell
            image_cell = cut_image_around_center(
                image_use, center, target_size, padding
            )  # Define the crop size (adjust as needed)
            mask_cell = cut_image_around_center(
                mask_use, center, target_size, padding
            )  # Define the crop size (adjust as needed)

            images_out.append(image_cell)
            masks_out.append(mask_cell)

    return images_out, masks_out


def dataset_cell_segmentation(segment_model, images_path, output_csv):
    for file in os.listdir(images_path):
        # Load image in the correct format
        image = segment_model.load_image(os.path.join(images_path, file))
        # Get the mask metadata
        image_name = os.fsdecode(file)
        df = segment_model.get_mask_metadata(image, image_name=image_name)

        # append to CSV
        if not os.path.isfile(output_csv):
            df.to_csv(output_csv, mode="w", header=True, index=False)
        else:
            df.to_csv(output_csv, mode="a", header=False, index=False)
