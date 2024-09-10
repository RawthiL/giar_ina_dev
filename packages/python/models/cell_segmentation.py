import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2 as cv
import supervision as sv
import numpy as np
import pandas as pd
from typing import List, Any, Dict, Tuple

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
