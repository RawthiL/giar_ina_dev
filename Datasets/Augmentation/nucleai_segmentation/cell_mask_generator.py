import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import cv2
import supervision as sv
import numpy as np
import pandas as pd
import glob
import re
from typing import Optional, List, Any, Dict, Tuple

class CellMaskGenerator:
    """
    A class to generate cell masks using a specified model and save the results in a CSV file.
    """

    def __init__(self, checkpoint: str , model_type: str = 'vit_h', device: str = 'cpu'):
        """
        Initialize the CellMaskGenerator with a specific model and device.

        Parameters:
        - model_type: str : The type of model to use. Default is 'vit_h'.
        - device: str : The device to run the model on. Default is 'cpu'.
        """
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device=torch.device(device))
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.mask_anotator = sv.MaskAnnotator()

    def generate_all_masks(self, images_path: str, output_path: str) -> None:
        """
        Generate masks for all images in the specified directory and save the results to a CSV file.

        Parameters:
        - images_path: str : The path to the directory containing images.
        - output_path: str : The path to the output CSV file.
        """
        for file in os.listdir(images_path):
            image_name = os.fsdecode(file)
            image = images_path + image_name
            masks = self._generate_masks(image, image_name)
            self._append_to_csv(masks, output_path)

    def _generate_masks(self, images_path: str, image_name: str) -> pd.DataFrame:
        """
        Generate masks for a single image.

        Parameters:
        - images_path: str : The path to the image.

        Returns:
        - pd.DataFrame: A DataFrame containing the mask data.
        """
        image_bgr = cv2.imread(images_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam_result = self.mask_generator.generate(image_rgb)
        filtered_sam_result = self._filter_masks(sam_result)
        masks_df = self._masks_to_df(filtered_sam_result, image_name)
        return masks_df
        
    def _filter_masks(self, result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter masks to remove outliers based on their area.

        Parameters:
        - result: List[Dict[str, Any]] : The list of masks to filter.

        Returns:
        - List[Dict[str, Any]]: The filtered list of masks.
        """
        mask_areas = np.array([m['area'] for m in result])

        # Calculate the quartiles and IQR
        q1 = np.percentile(mask_areas, 25)
        q3 = np.percentile(mask_areas, 75)
        iqr = q3 - q1

        # Determine the outlier bounds
        lower_bound = q1 - 0.5 * iqr
        upper_bound = q3 + 0.5 * iqr

        # Filter out the outliers
        filtered_sam_result = [entry for entry in result if lower_bound <= entry['area'] <= upper_bound]
        
        return filtered_sam_result
    
    def _masks_to_df(self, masks: List[Dict[str, Any]], image_name: str) -> pd.DataFrame:
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
            area = s['area']
            bbox = s['bbox']  # bbox is in [x, y, w, h] format
            x, y, w, h = bbox
            bbox_area = w * h

            # Create a new row as a dictionary
            new_row = {
                'area': area,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'bbox_area': bbox_area,
                'image': image_name,
                'cell_id': i
            }

            # Append the new row to the DataFrame
            rows.append(new_row)
        df = pd.DataFrame(rows, columns=['area', 'x', 'y', 'w', 'h', 'bbox_area', 'image', 'cell_id'])
        return df
    
    def _append_to_csv(self, df: pd.DataFrame, output_csv: str) -> None:
        """
        Append the DataFrame to a CSV file.

        Parameters:
        - df: pd.DataFrame : The DataFrame to append.
        - output_csv: str : The path to the output CSV file.
        """
        # Check if file exists to write header
        if not os.path.isfile(output_csv):
            df.to_csv(output_csv, mode='w', header=True, index=False)
        else:
            df.to_csv(output_csv, mode='a', header=False, index=False)

    def crop_cells(self, image_path: str, masks_path: str, output_dir: str, bbox_area: Optional[int] = 200 * 200) -> None:
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
            image = cv2.imread(image_path)
            image_name = os.path.basename(image_path)

            df = pd.read_csv(masks_path)
            df_bbox = df[df['image'] == image_name][['x', 'y', 'w', 'h', 'cell_id']]       
            
            # Iterate over the bounding boxes and crop the image
            for _, row in df_bbox.iterrows():
                x, y, w, h = self._adjust_bbox(row['x'], row['y'], row['w'], row['h'], bbox_area)
                cell_id = row['cell_id']
                crop = image[y:y+h, x:x+w]
                crop_name = image_name.replace('.png', f'_{cell_id}.png')
                output_path = os.path.join(output_dir, crop_name)
                cv2.imwrite(output_path, crop)

    def _adjust_bbox(self, x: int, y: int, w: int, h: int, target_area: int) -> Tuple[int, int, int, int]:
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

            x = min(x, 2048 - w)
            y = min(y, 3072 - h)

            w = side
            h = side

            return x, y, w, h