import json

# Constants
ANNOTATIONS_TRAIN = "datasets/onion_cell_merged/images/annotations_coco_test.json"
ANNOTATIONS_VALID = "datasets/onion_cell_merged/images/annotations_coco_valid.json"
ANNOTATIONS_TEST = "datasets/onion_cell_merged/images/annotations_coco_test.json"

DIVISION = 1
NO_DIVISION = 0


def process_annotations(path: str) -> None:
    """Load, modify, and overwrite a COCO-style annotation JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    # Replace categories with one fixed class
    data["categories"] = [{"id": 1, "name": "cell", "supercategory": ""}]

    # Modify each annotation
    for annotation in data.get("annotations", []):
        division = DIVISION if annotation.get("category_id", 1) == 1 else NO_DIVISION
        annotation["attributes"] = {
            "division": division,
            "mitosis_stage": 0,
            "occluded": False,
            "rotation": 0,
        }
        annotation["category_id"] = 1

    # Overwrite the same file
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Updated file: {path}")


def main() -> None:
    """Process all annotation JSON files."""
    for path in [ANNOTATIONS_TRAIN, ANNOTATIONS_VALID, ANNOTATIONS_TEST]:
        process_annotations(path)


if __name__ == "__main__":
    main()
