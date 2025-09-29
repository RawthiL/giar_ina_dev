# Branch for experimentatation of VAE and difussion

## Dataset Creation for VAE Training

This guide outlines the semi-automated process for creating the dataset required to run `notebooks/train/vae.ipynb`.

### Step 1: Prerequisites

Before you begin, ensure the following data sources are correctly placed:

- **Tagged INA Images:** `media/images/ina/tagged_images/` (with COCO annotations)
- **Roboflow Datasets:** `media/images/roboflow_datasets/` (with COCO annotations)
- **Zooniverse Data:**
  - `media/zooniverse_tagged_images.csv`
  - `media/cropped_images/ina/images` (containing SAM-cropped images)
  - `media/cropped_images/onion_cell_merged/images/*` (containing SAM-cropped images)
- **Onion Cell Merged:** `media/images/onion_cell_merged/images/` (with COCO annotations)

### Step 2: Automated Cropping

Run the notebook: `notebooks/dataset_generation/vae_dataset.ipynb`

This script will process all prerequisite data, extract the annotated cell crops, and organize them into initial directories. It will also automatically create a `test` set from the generated `train` set after a manual clean up step.

### Step 3: Manual Curation & Consolidation

This manual step is crucial for ensuring the quality of the training data. The automated script creates a temporary structure to make this process easier.

**A. Consolidate Tagged Mitosis Cells:**

1.  **Inspect:** Review the cropped images inside the newly created source folders (e.g., `media/cropped_images/vae/train/ina_tagged/`, `.../roboflow/`, etc.). Correct any misclassifications.
2.  **Consolidate:** Move all correctly classified images from their source folders into the final, combined class folders located at:
    `media/cropped_images/vae/train/tagged/[class_name]/`
    (e.g., move all correct prophase images to `.../tagged/prophase/`)

**B. Prepare Untagged Division Cells:**

1.  **Move:** Take all images from the `media/cropped_images/vae/train/onion_cell_merged/d/` folder (which contains all cells in division) and move them to:
    `media/cropped_images/vae/train/untagged/`

### Step 4: Data Augmentation

Run the notebook: `notebooks/dataset_generation/data_augmentation.ipynb`

You will need to run this notebook **twice** for each of the final training directories (`.../tagged/*` and `.../untagged/`).

---

### Final VAE Dataset Structure

After completing these steps, your dataset in `media/cropped_images/vae/` will be ready for training and should have the following structure:

media/cropped_images/vae/
├── test
│   ├── anaphase
│   ├── metaphase
│   ├── prophase
│   └── telophase
└── train
├── tagged
│   ├── anaphase
│   ├── metaphase
│   ├── prophase
│   └── telophase
└── untagged

## Dataset Creation for Diffusion Training

This guide explains how to generate the paired dataset required to run the ControlNet fine-tuning notebook: `notebooks/train/fine_tune_controlnet.ipynb`.

### Step 1: Prerequisites

Before you begin, ensure the following files and directories are in place:

- **VAE Dataset:** The complete dataset located at `cropped_images/vae/` (generated from the previous VAE training step).
- **VAE Encoder:** The trained model file at `models/vae/encoder.keras`.
- **VAE Decoder:** The trained model file at `models/vae/decoder.keras`.

### Step 2: Clean Augmented Data (Optional)

For the highest quality results, it is recommended to train the diffusion model only on the original, non-augmented images.

To do this, search for any files containing `_aug` within the `cropped_images/vae/` directory and delete them before proceeding to the next step.

### Step 3: Automated Dataset Generation

Run the notebook: `notebooks/dataset_generation/diffusion_dataset.ipynb`

This script uses the trained VAE to create the paired dataset. For each original image (the target, or **`sharp_upscaled`**), it generates a corresponding VAE-processed version (the input, or **`blurred_upscaled`**). These input/target pairs are essential for training the diffusion model.

The script will create a new dataset at `cropped_images/control_net/`, containing `train` and `test` splits, the paired image folders, and a `metadata.json` file that links each image pair.

---

### Final Dataset Structure

After running the generation script, your final dataset will be organized as follows:

media/cropped_images/control_net
├── test
│   ├── blurred_upscaled (Input images for the model)
| | ├── B_img1.png
| | ├── B_img2.png
│   ├── sharp_upscaled (Ground truth/target images)
| | ├── S_img1.png
| | ├── S_img2.png
│   └── metadata.json (Links blurred to sharp images)
└── train
├── blurred_upscaled
├── sharp_upscaled
   └── metadata.json
