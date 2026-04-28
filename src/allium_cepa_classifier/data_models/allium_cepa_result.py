from __future__ import annotations

from pathlib import Path
from typing import Union
from dataclasses import dataclass

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

@dataclass
class AlliumCepaResult:
    """
    Result of running the AlliumCepaModel on a single image.

    Attributes
    ----------
    image : PIL.Image.Image
        Original input image.
    detections : pd.DataFrame
        DataFrame with one row per detected instance, including:
        - x_min, y_min, x_max, y_max
        - confidence
        - class_id
        - class_name
        - mitosis (classification result)
        - mitosis_score (optional probability)
    """
    def __init__(self, image: Image.Image, detections: pd.DataFrame) -> None:
        self.image = image
        self.detections = detections
        
    def save_csv(self, output_path: Union[str, Path]) -> None:
        """
        Save the detections DataFrame as a CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.detections.to_csv(output_path, index=False)

    def show_annotated(self, image_name: str = "", font_size: int = 10, line_width: int = 7) -> Image.Image:

        """
        Return a copy of the image with bounding boxes and labels drawn on it.
        """
        unique_images = self.detections["image"].unique()

        if not image_name:
            if len(unique_images) == 1:
                image_name = unique_images[0]
            elif len(unique_images) > 1:
                raise ValueError(
                    f"Multiple images found in results. Please specify one using the 'image_name' parameter. Available images: {list(unique_images)}"
                )
            # If len is 0, it will be handled later.

        elif image_name not in unique_images:
            raise ValueError(f"Image '{image_name}' not found. Available images are: {list(unique_images)}")

        if not image_name: # This handles the case of zero detections
            raise ValueError("Cannot show annotations as there are no detections.")

        image_to_show = Image.open(self.dir / image_name)
        detections_for_image = self.detections[self.detections["image"] == image_name]

        annotated = image_to_show.copy()
        draw = ImageDraw.Draw(annotated)
 
        # Try to use a default font; if that fails, fall back silently
        try:
            # If the default font is used it may not support different sizes
            font = ImageFont.truetype("DejaVuSans.ttf", size=font_size)
        except Exception:
            font = ImageFont.load_default()

        for _, row in detections_for_image.iterrows():
            x_min = int(row["x_min"])
            y_min = int(row["y_min"])
            x_max = int(row["x_max"])
            y_max = int(row["y_max"])
            color = "green" if row.get("mitosis", True) else "red"

            # Draw rectangle
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=line_width)

        # --- Draw Legend ---
        legend_items = {
            "Mitosis": "green",
            "No Mitosis": "red"
        }
        
        start_x = 15
        start_y = 15
        box_size = font_size
        padding = 10
        text_x_offset = box_size + 10

        # Create a separate image for the legend with a transparent background
        legend_img = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
        legend_draw = ImageDraw.Draw(legend_img)

        # Draw semi-transparent background for the legend
        legend_height = (box_size + padding) * len(legend_items) + padding
        legend_width = 200 # A fixed width should be sufficient
        legend_draw.rectangle([start_x, start_y, start_x + legend_width, start_y + legend_height], fill=(0, 0, 0, 128))

        current_y = start_y + padding
        for label, color in legend_items.items():
            legend_draw.rectangle([start_x + padding, current_y, start_x + padding + box_size, current_y + box_size], fill=color)
            legend_draw.text((start_x + padding + text_x_offset, current_y), label, fill="white", font=font)
            current_y += box_size + padding
        
        annotated.paste(legend_img, (0, 0), legend_img)

        # annotated.show()
        return annotated
     
    def get_counts(self) -> dict[str, Union[int, float]]:
        """
        Calculate and return cell counts and the mitotic index.

        Returns:
            A dictionary containing:
            - 'total_cells': int
            - 'mitotic_cells': int
            - 'non_mitotic_cells': int
            - 'mitotic_index': float
        """
        total_cells = len(self.detections)
        if total_cells == 0:
            return {
                "total_cells": 0,
                "mitotic_cells": 0,
                "non_mitotic_cells": 0,
                "mitotic_index": 0.0,
            }

        mitotic_cells = int(self.detections["mitosis"].sum())
        non_mitotic_cells = total_cells - mitotic_cells
        mitotic_index = float(mitotic_cells / total_cells)

        return {
            "total_cells": total_cells,
            "mitotic_cells": mitotic_cells,
            "non_mitotic_cells": non_mitotic_cells,
            "mitotic_index": mitotic_index,
        }
    
    
    @property
    def mitotic_index(self) -> float:
        """
        Mitotic index as a percentage:
        (number of mitotic cells / total number of cells) * 100

        Returns
        -------
        float
            Mitotic index in percent. Returns 0.0 if no cells are present.
        """
        if "mitosis" not in self.detections.columns:
            return 0.0

        total_cells = len(self.detections)
        if total_cells == 0:
            return 0.0

        mitosis_series = self.detections["mitosis"]

        mitotic_cells = mitosis_series.apply(
            lambda x: (
                x is True
                or (isinstance(x, str) and x.lower() in {"mitosis", "mitotic", "m"})
            )
        ).sum()

        return (mitotic_cells / total_cells) * 100.0
