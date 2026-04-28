import argparse
import json
import os

from PIL import Image

from allium_cepa_classifier.config.training_config import TrainingConfig


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def crop_and_save(
    image_path: str, bbox: list, save_path: str, ann_id: int, image_name: str
) -> None:
    """Crop an image using bbox [x, y, w, h] and save it, handling empty crops gracefully."""
    try:
        with Image.open(image_path) as img:
            x, y, w, h = bbox
            # Handle invalid or zero-size bboxes
            if w <= 0 or h <= 0:
                print(f"⚠️ Skipping annotation {ann_id} ({image_name}): invalid bbox {bbox}")
                return

            crop = img.crop((x, y, x + w, y + h))

            if crop.size[0] == 0 or crop.size[1] == 0:
                print(
                    f"⚠️ Skipping annotation {ann_id} ({image_name}): empty crop after bbox {bbox}"
                )
                return

            crop.save(save_path)

    except Exception as e:
        print(f"❌ Failed to process annotation {ann_id} in image '{image_name}' with bbox {bbox}")
        print(f"   Error: {type(e).__name__}: {e}")


def process_split(split_name: str, cfg: TrainingConfig, limit: int = None) -> None:
    """Process a single dataset split (train/valid/test) with optional crop limit."""
    print(f"Processing {split_name}...")

    annotation_path = cfg.raw_dataset_dir / split_name / "data" / "annotations.json"

    with open(annotation_path) as f:
        data = json.load(f)

    images = {img["id"]: img["file_name"] for img in data.get("images", [])}
    annotations = data.get("annotations", [])

    if limit is not None:
        annotations = annotations[:limit]
        print(f"🔹 Limiting to first {limit} annotations for {split_name}")

    for ann in annotations:
        image_id = ann["image_id"]
        bbox = ann["bbox"]
        attributes = ann.get("attributes", {})
        division = attributes.get("division", 0)
        category = "mitosis" if division == 1 else "no_mitosis"

        image_name = os.path.splitext(images.get(image_id, f"missing_{image_id}"))[0]
        image_path = str(cfg.raw_dataset_dir / split_name / "images" / images.get(image_id, ""))
        save_dir = str(cfg.binary_classifier_crops_dir / split_name / category)
        ensure_dir(save_dir)

        save_path = os.path.join(save_dir, f"{image_name}_{ann['id']}.jpg")

        if not os.path.exists(image_path):
            print(f"⚠️ Image not found for annotation {ann['id']}: {image_path}")
            continue
        crop_and_save(image_path, bbox, save_path, ann["id"], image_name)

    print(f"✅ Done processing {split_name}. ({len(annotations)} crops processed)\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Crop images from dataset annotations.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of annotations to crop per split.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Uses TrainingConfig defaults if omitted.",
    )
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config, key="training") if args.config else TrainingConfig()

    for split in cfg.splits:
        process_split(split, cfg, limit=args.limit)


if __name__ == "__main__":
    main()
