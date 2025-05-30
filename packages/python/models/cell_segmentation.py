import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2 as cv
import os
import supervision as sv
import numpy as np
import pandas as pd
from tensorflow import keras
from typing import List, Any, Dict, Tuple, Optional


class CellMaskGenerator:
    """
    Base class
    """

    def generate(self, image: np.array) -> Tuple[np.array, pd.DataFrame]:
        """
        Takes a grayscale image and produces a binary mask of valid cells.
        It also produces a dataframe containing all the data of the segmentation process
        in the form of a dataframe with columns:
        - "area" : Number of pixels in the segmented area
        - "x" : X coordinate of the center of the segmented area
        - "y" : Y coordinate of the center of the segmented area
        - "w" : Width (maximum length in the X axis) of the segmented area
        - "h" : Height (maximum length in the Y axis) of the segmented area
        - "bbox_area" : Number of pixels of the bounding box (w*h)
        - "image" : Name of the processed image
        - "cell_id" : Instance number of the segmented area.
        """
        raise ValueError("Method not implemented.")

    def load_image(self, image_path: str) -> np.array:
        """
        Loads an image and prepares it to be processed by the model
        """
        raise ValueError("Method not implemented.")

    def adjust_bbox(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        target_area: int,
        max_width: Optional[int] = 3072,
        max_height: Optional[int] = 2048,
    ) -> Tuple[int, int, int, int]:
        """
        Adjusts the bounding box to match the target area while keeping the center of the original box.

        Parameters:
        - x: int : The x-coordinate of the top-left corner of the bounding box.
        - y: int : The y-coordinate of the top-left corner of the bounding box.
        - w: int : The width of the bounding box.
        - h: int : The height of the bounding box.
        - target_area: int : The target area for the bounding box.

        Returns:
        - Tuple[int, int, int, int] : The adjusted bounding box coordinates and dimensions.
        """
        side = int(np.sqrt(target_area))

        w_dif = abs(w - side)
        h_dif = abs(h - side)

        if w < side:
            x -= int(w_dif / 2)
        else:
            x += int(w_dif / 2)

        if h < side:
            y -= int(h_dif / 2)
        else:
            y += int(h_dif / 2)

        x = max(0, x)
        y = max(0, y)

        x = min(x, max_width - w)
        y = min(y, max_height - h)

        w = side
        h = side

        return x, y, w, h

    def crop_cells(
        self,
        image_path: str,
        masks_path: str,
        output_dir: str,
        bbox_area: Optional[int] = 200 * 200,
    ) -> None:
        """
        Crops the image based on the given bounding boxes and saves the cropped images into the specified directory.

        Parameters:
        - image_path: str : The path to the input image.
        - masks_path: str : The path to the mask CSV file generated.
        - output_dir: str : The directory where cropped images will be saved.
        - bbox_area: Optional[int] : The target area for each bounding box. Default is 200*200.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Read the image
        image = cv.imread(image_path)
        image_name = os.path.basename(image_path)
        image_base_name, image_ext = os.path.splitext(image_name)

        df = pd.read_csv(masks_path)
        df_bbox = df[df["image"] == image_name][["x", "y", "w", "h", "cell_id"]]

        # Iterate over the bounding boxes and crop the image
        for _, row in df_bbox.iterrows():
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]
            cell_id = row["cell_id"]
            crop = image[y : y + h, x : x + w]
            crop_name = image_name.replace(image_ext, f"_{cell_id}{image_ext}")
            output_path = os.path.join(output_dir, crop_name)
            cv.imwrite(output_path, crop)

    def bbox_applier(
        model_path: str,
        csv_path: str,
        cells_path: str,
        images_path: str,
        encoder_path=None,
    ) -> None:
        """
        Adds bounding boxes to the cells of the original image by filtering out the noise with a ml model

        Parameters:
        - model_path:  str : path to the model to use
        - csv_path:    str : path to the csv with the data of the masks
        - cells_path:  str : path to the individual cells images
        - images_path: str : path to the full images

        Outputs:
        If it does not exists, creates the detected_cells folder where it stores the full images with
        the new bounding boxed added
        """

        OUTPUT_PATH = "../detected_cells"
        _, model_extension = os.path.splitext(model_path)
        df = pd.read_csv(csv_path)
        imgs = sorted(os.listdir(cells_path))
        os.makedirs(OUTPUT_PATH, exist_ok=True)

        prv_image = ""

        for idx, file in enumerate(imgs):
            print(f"Image: {idx + 1}/{len(imgs)}", end="\r")

            # Load image
            image_name = os.fsdecode(file)
            image_path = cells_path + image_name
            img = cv.imread(image_path)  # , cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (128, 128))
            cell_id = image_name.split("_")[-1].split(".")[0]
            img_nbr = image_name.split("_")[0]
            og_image = f'{image_name[:image_name.rfind('_')]}.png'  # Take the image name until the last '_'

            # Load the full image only when there is an image change
            if prv_image != og_image:
                full_image_path = os.path.join(images_path, og_image)
                full_image = cv.imread(full_image_path)
            prv_image = og_image

            # if model_extension != '.keras':
            #     kmeans = joblib.load(model_path)

            #     if encoder_path != None: # Use clustering with encoder embeddings for detection
            #         img = np.expand_dims(img, axis=(0, -1))
            #         encoder = keras.models.load_model(encoder_path)
            #         feature = encoder.predict(img, verbose=0).astype(float)
            #         prediction = kmeans.predict(feature)[0]

            #     else: # Use clustering with color histogram for detection
            #         hist_predict = cv.calcHist(img, [0], None, [8], [0, 256]).flatten()
            #         hist_predict = np.array(hist_predict).reshape(1, -1)
            #         hist_predict = hist_predict/255
            #         prediction = kmeans.predict(hist_predict)[0]
            # else:

            model = keras.models.load_model(model_path)
            img = np.array(img)
            img = img / 255

            prediction = model.predict(np.expand_dims(img, 0), verbose=0)[0][0]

            is_cell = True if prediction >= 0.5 else False

            # If there is a cell, draw a rectangle
            if is_cell:
                row = df.loc[
                    (df["image"] == og_image) & (df["cell_id"] == int(cell_id))
                ].to_dict("records")[0]

                x, y, w, h = row["x"], row["y"], row["w"], row["h"]
                cv.rectangle(full_image, (x, y), (x + w, y + h), 255, 10)

            # Save the new image when there is an image change or is the last file
            if (prv_image != og_image) or (idx + 1 == len(imgs)):
                cv.imwrite(os.path.join(OUTPUT_PATH, f"{img_nbr}.png"), full_image)


class SAMCellMaskGenerator(CellMaskGenerator):
    """
    A class to generate cell masks using a SAM model
    """

    def __init__(self, checkpoint: str, model_type: str = "vit_h", device: str = "cpu"):
        """
        Initialize the CellMaskGenerator with a specific model and device.

        Parameters:
        - model_type: str : The type of model to use. Default is 'vit_h'.
        - device: str : The device to run the model on. Default is 'cpu'.
        """
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint).to(
            device=torch.device(device)
        )
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.mask_anotator = sv.MaskAnnotator()

    def get_mask_metadata(self, image: np.array, image_name="iamge") -> pd.DataFrame:
        """
        Takes a grayscale image and produces a dataframe containing all the data of the segmentation process
        "area", "x", "y", "w", "h", "bbox_area", "image", "cell_id"
        """

        sam_result = self.mask_generator.generate(image)
        filtered_sam_result = self._filter_masks(sam_result)
        masks_df = self._masks_to_df(filtered_sam_result, image_name)

        return masks_df

    def load_image(self, image_path: str) -> np.array:
        """
        Loads an image and prepares it to be processed by the model
        """
        image_bgr = cv.imread(image_path)
        return cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

    def _filter_masks(self, result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter masks to remove outliers based on their area.

        Parameters:
        - result: List[Dict[str, Any]] : The list of masks to filter.

        Returns:
        - List[Dict[str, Any]]: The filtered list of masks.
        """
        mask_areas = np.array([m["area"] for m in result])

        # Calculate the quartiles and IQR
        q1 = np.percentile(mask_areas, 25)
        q3 = np.percentile(mask_areas, 75)
        iqr = q3 - q1

        # Determine the outlier bounds
        lower_bound = q1 - 0.5 * iqr
        upper_bound = q3 + 0.5 * iqr

        # Filter out the outliers
        filtered_sam_result = [
            entry for entry in result if lower_bound <= entry["area"] <= upper_bound
        ]

        return filtered_sam_result

    def _masks_to_df(
        self, masks: List[Dict[str, Any]], image_name: str
    ) -> pd.DataFrame:
        """
        Convert masks to a DataFrame.

        Parameters:
        - masks: List[Dict[str, Any]] : The list of masks.
        - image_name: str : The name of the image.

        Returns:
        - pd.DataFrame: A DataFrame containing the mask data.
        """
        rows = []
        # Iterate over filtered_sam_result and add rows to the DataFrame
        for i, s in enumerate(masks):
            area = s["area"]
            bbox = s["bbox"]  # bbox is in [x, y, w, h] format
            x, y, w, h = bbox
            bbox_area = w * h

            # Create a new row as a dictionary
            new_row = {
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "bbox_area": bbox_area,
                "image": image_name,
                "cell_id": i,
            }

            # Append the new row to the DataFrame
            rows.append(new_row)
        df = pd.DataFrame(
            rows, columns=["area", "x", "y", "w", "h", "bbox_area", "image", "cell_id"]
        )
        return df
