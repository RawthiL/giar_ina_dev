import json
import shutil
from pathlib import Path

import yaml

from allium_cepa_classifier.config.training_config import TrainingConfig

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
YOLO_CLASS_ID_FOR_CELL = 0


def find_all_images(img_dir: Path) -> dict[str, Path]:
    """
    Recursively find all images under img_dir and return a mapping:
        filename (string, case-sensitive) -> full path
    """
    mapping = {}
    if not img_dir.exists():
        return mapping
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            mapping[p.name] = p
    return mapping


def load_coco(annot_path: Path):
    with open(annot_path, encoding="utf-8") as f:
        coco = json.load(f)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    return images, annotations, categories


def build_image_index(images: list[dict]) -> dict[int, dict]:
    """
    Map image_id -> {file_name, width, height}
    """
    idx = {}
    for im in images:
        idx[im["id"]] = {
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
        }
    return idx


def build_annotations_by_image(annotations: list[dict]) -> dict[int, list[dict]]:
    """
    Map image_id -> list of annotation dicts
    """
    by_img = {}
    for ann in annotations:
        by_img.setdefault(ann["image_id"], []).append(ann)
    return by_img


def coco_bbox_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """
    COCO bbox: [x_min, y_min, width, height] in pixels
    YOLO bbox: (x_center, y_center, width, height) normalized to [0,1]
    """
    x, y, w, h = bbox
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return x_c, y_c, w_n, h_n


def ensure_dirs(base: Path):
    (base / "images").mkdir(parents=True, exist_ok=True)
    (base / "labels").mkdir(parents=True, exist_ok=True)


def process_split(split: str, cfg: TrainingConfig):
    split_root = cfg.raw_dataset_dir / split
    annot_path = split_root / "data" / "annotations.json"
    img_root = split_root / "images"

    if not annot_path.exists():
        print(f"⚠️  No annotations for split '{split}': {annot_path} (skipping)")
        return (0, 0, 0)

    # Load COCO for this split
    images, annotations, categories = load_coco(annot_path)
    image_idx = build_image_index(images)
    ann_by_img = build_annotations_by_image(annotations)

    # Map of filename -> full path for this split
    filemap = find_all_images(img_root)

    # Prepare output structure
    out_split_root = cfg.yolo_dataset_dir / split
    ensure_dirs(out_split_root)
    out_imgs_dir = out_split_root / "images"
    out_labels_dir = out_split_root / "labels"

    missing_files = 0
    converted = 0
    copied_imgs = 0

    # In many COCO exports, file_name may include subfolders. We only match by basename.
    # Build a case-insensitive helper map as a fallback.
    lower_map = {k.lower(): v for k, v in filemap.items()}

    for img_id, meta in image_idx.items():
        fname = Path(meta["file_name"]).name
        img_w = meta["width"]
        img_h = meta["height"]

        src_path = filemap.get(fname)
        if src_path is None:
            src_path = lower_map.get(fname.lower())

        if src_path is None or not src_path.exists():
            missing_files += 1
            print(f"⚠️  [{split}] Missing image referenced in JSON: {fname} (image_id={img_id})")
            continue

        # Destination paths
        out_img_path = out_imgs_dir / fname
        out_lbl_path = out_labels_dir / (Path(fname).stem + ".txt")

        # Copy image
        if not out_img_path.exists():
            shutil.copy2(src_path, out_img_path)
            copied_imgs += 1

        # Build YOLO label lines (only category_id == 1 -> class 1)
        lines = []
        for ann in ann_by_img.get(img_id, []):
            if ann.get("category_id") != 1:
                continue
            x_c, y_c, w_n, h_n = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            # Clamp to [0,1]
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w_n = max(0.0, min(1.0, w_n))
            h_n = max(0.0, min(1.0, h_n))
            lines.append(f"{YOLO_CLASS_ID_FOR_CELL} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        out_labels_dir.mkdir(parents=True, exist_ok=True)
        with open(out_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        converted += 1

    return converted, copied_imgs, missing_files


def create_data_yaml(class_names: list[str], cfg: TrainingConfig):
    """
    Write data.yaml into yolo_dataset_dir using absolute paths for the machine
    running this script. YOLO requires absolute paths to avoid working-directory
    issues when training from different locations.
    """
    root = cfg.yolo_dataset_dir.resolve()

    # YOLO convention: key is 'val', not 'validation'
    split_key_map = {"train": "train", "validation": "val", "test": "test"}

    data = {"path": str(root)}
    for split in cfg.splits:
        key = split_key_map.get(split, split)
        data[key] = f"{split}/images"
    data["names"] = dict(enumerate(class_names))

    yaml_path = cfg.yolo_dataset_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f" data.yaml  -> {yaml_path.resolve()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Uses TrainingConfig defaults if omitted.",
    )
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config, key="training") if args.config else TrainingConfig()

    total_converted = 0
    total_copied = 0
    total_missing = 0

    for split in cfg.splits:
        converted, copied, missing = process_split(split, cfg)
        total_converted += converted
        total_copied += copied
        total_missing += missing

    create_data_yaml(class_names=["cell"], cfg=cfg)

    print("--------------------------------------------------")
    print("✅ Done.")
    print(f" Total images processed (all splits): {total_converted}")
    print(f" Total images copied                 : {total_copied}")
    print(f" Total missing image files           : {total_missing}")
    print(f" Output root: {cfg.yolo_dataset_dir.resolve()}")
    print("--------------------------------------------------")
    for split in cfg.splits:
        print(f" {split}:")
        print(f"   images -> {(cfg.yolo_dataset_dir / split / 'images').resolve()}")
        print(f"   labels -> {(cfg.yolo_dataset_dir / split / 'labels').resolve()}")


if __name__ == "__main__":
    main()
