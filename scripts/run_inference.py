import os
import json
from pathlib import Path

import torch
from PIL import Image

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "src/allium_cepa_classifier/models/weights/object_detection_v1.pt"
IMAGE_DIR = Path("datasets/allium_cepa_full_images_merged_for_tagging")  # root to search for images
OUTPUT_JSON = Path("annotations.json")       # write next to where you run the script
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

CONF_THRESHOLD = 0.25   # ignore detections below this
IOU_NMS = None          # leave None to use model defaults; or set e.g. 0.5

# -----------------------------
# Utilities
# -----------------------------
def list_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])

def xyxy_to_xywh(x1, y1, x2, y2):
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    return [float(x1), float(y1), w, h]

def load_image_size(path: Path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        return im.width, im.height

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Model loader (Ultralytics first; fallback to raw torch / YOLOv5)
# -----------------------------
class InferenceAdapter:
    """
    Provides a unified `predict(image_path)` -> list[dict] API returning:
      [{ "xyxy": [x1,y1,x2,y2], "conf": float, "cls": int }]
    """
    def __init__(self, model_path):
        self.backend = None
        self.model = None

        # Try Ultralytics YOLO (v8/v9)
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.backend = "ultralytics"
            return
        except Exception:
            pass

        # Try torch.hub YOLOv5 (if weights work there)
        try:
            # NOTE: This requires internet for the repo the first time; if offline, skip.
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, verbose=False)
            self.model.to(device())
            self.backend = "yolov5_hub"
            return
        except Exception:
            pass

        # Fallback: raw torch.load (must implement __call__/forward appropriately)
        mdl = torch.load(model_path, map_location=device())
        if hasattr(mdl, "eval"):
            mdl.eval()
        self.model = mdl
        self.backend = "raw"

    def predict(self, image_path: Path):
        if self.backend == "ultralytics":
            # Ultralytics: returns list of Results
            kwargs = {}
            if CONF_THRESHOLD is not None:
                kwargs["conf"] = CONF_THRESHOLD
            if IOU_NMS is not None:
                kwargs["iou"] = IOU_NMS
            res = self.model(str(image_path), **kwargs)
            r0 = res[0]
            dets = []
            if hasattr(r0, "boxes") and r0.boxes is not None:
                boxes = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                clss = r0.boxes.cls.cpu().numpy()
                for (x1, y1, x2, y2), cf, cl in zip(boxes, confs, clss):
                    if cf < CONF_THRESHOLD:
                        continue
                    dets.append({"xyxy": [float(x1), float(y1), float(x2), float(y2)],
                                 "conf": float(cf), "cls": int(cl)})
            return dets

        if self.backend == "yolov5_hub":
            self.model.conf = CONF_THRESHOLD if CONF_THRESHOLD is not None else self.model.conf
            if IOU_NMS is not None:
                self.model.iou = IOU_NMS
            out = self.model(str(image_path))
            # xyxy tensor: [x1,y1,x2,y2,conf,cls]
            xyxy = out.xyxy[0].cpu().numpy() if hasattr(out, "xyxy") else []
            dets = []
            for row in xyxy:
                x1, y1, x2, y2, cf, cl = row[:6]
                if cf < CONF_THRESHOLD:
                    continue
                dets.append({"xyxy": [float(x1), float(y1), float(x2), float(y2)],
                             "conf": float(cf), "cls": int(cl)})
            return dets

        # Raw model fallback — you may need to adapt this part to your custom forward
        # Expecting a return shaped like YOLOv5: list of tensors, first per-image tensor with columns xyxy+conf+cls
        with torch.no_grad():
            # implement simple PIL -> tensor if your model needs it; here we assume it accepts paths
            out = self.model(str(image_path))
        dets = []
        # Try common patterns
        try:
            # out[0] tensor
            arr = out[0].detach().cpu().numpy()
            for row in arr:
                x1, y1, x2, y2, cf, cl = row[:6]
                if cf < CONF_THRESHOLD:
                    continue
                dets.append({"xyxy": [float(x1), float(y1), float(x2), float(y2)],
                             "conf": float(cf), "cls": int(cl)})
            return dets
        except Exception:
            # As a last resort, return no detections
            return []

# -----------------------------
# Build COCO dict
# -----------------------------
def main():
    images = []
    annotations = []
    categories = [
        {
            "id": 1,
            "name": "cell",
            "supercategory": "cell"
        }
    ]

    adapter = InferenceAdapter(MODEL_PATH)

    img_id = 1
    ann_id = 1

    img_paths = list_images(IMAGE_DIR)
    if not img_paths:
        print(f"No images found under: {IMAGE_DIR.resolve()}")
        return

    for p in img_paths:
        try:
            width, height = load_image_size(p)
        except Exception as e:
            print(f"Skipping unreadable image {p}: {e}")
            continue

        images.append({
            "id": img_id,
            "file_name": str(p.as_posix()),
            "width": width,
            "height": height
        })

        dets = adapter.predict(p)
        for det in dets:
            x1, y1, x2, y2 = det["xyxy"]
            bbox = xyxy_to_xywh(x1, y1, x2, y2)
            area = bbox[2] * bbox[3]

            # Force every detection to category_id=1 ("cell")
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [round(b, 3) for b in bbox],
                "area": round(float(area), 3),
                "iscrowd": 0,
                "segmentation": [],  # no masks available from plain boxes
                # Optional: include score for traceability (not standard for GT, but harmless)
                "score": float(det.get("conf", 1.0))
            }
            annotations.append(ann)
            ann_id += 1

        img_id += 1

    coco = {
        "info": {
            "description": "Auto-generated detections labeled as 'cell'",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote COCO annotations to: {OUTPUT_JSON.resolve()}")
    print(f"Images: {len(images)} | Annotations: {len(annotations)}")

if __name__ == "__main__":
    main()
