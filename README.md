# Dataset Processing Scripts

This repository contains scripts to process and prepare the datasets for use.

The main workflow is divided into two primary scripts:

1.  **`dataset_preprocess`**: Standardizes the organization of the original datasets.
2.  **`annotated_dataset`**: Crops the standardized images based on their original annotations to create a classified dataset.

## 1. dataset_preprocess

This notebook ingests the original raw datasets and reorganizes them into a standard format.

### Before you run

You must first merge the three original dataset folders (`ina`, `onion_cell_merged`, and `roboflow_datasets`) under a single parent folder named `original`.

The required input structure is:

```
original/
├── ina/
├── onion_cell_merged/
└── roboflow_datasets/
```

### What it does

The script will generate a new folder named `processed`. Inside this folder, it will create a subfolder for each dataset. Each of these subfolders will contain an `images` folder with all the images and a single `annotations_coco.json` file.

The output structure will be:

```
processed/
├── ina/
│   ├── images/
│   └── annotations_coco.json
├── onion_cell_merged/
│   ├── images/
│   └── annotations_coco.json
└── roboflow_datasets/
    ├── images/
    └── annotations_coco.json
```

## 2. annotated_dataset

This script takes the standardized output from the previous notebook and crops the images according to their bounding box annotations.

### What it does

The script will generate a new folder named `annotated`. Inside this folder, it will create a subfolder for each dataset, which in turn will contain a separate subfolder for each annotated class.

The output structure will be:

```
annotated/
├── ina/
│   ├── class_1/ (e.g., prophase)
│   ├── class_2/ (e.g., metaphase)
│   └── ...
└── onion_cell_merged/
    ├── class_1/
    ├── class_2/
    └── ...
```

## ⚠️ Important Note

Please be aware that some images from the `roboflow_datasets` were found to be **misclassified in the original source**. These incorrect labels may carry over into the final cropped dataset.
