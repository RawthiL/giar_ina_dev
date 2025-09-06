# Cell Segmentation and Classification - Allium Protocol

This repository contains the development code for a system designed for the segmentation and classification of plant cells, specifically those used in the Allium protocol. It includes a series of Jupyter notebooks that facilitate cell segmentation, analysis, and classification using various machine learning techniques. 
The repository also supports DVC experiments to produce the datasets and models, all of them in the `./experiments` folder.

## Requierements

1. poetry install
2. dvc pull


## Table of Contents

1. [Notebooks](#notebooks)
2. [Experiments](#experiments)
4. [Future Developments: Classification](#future-developments-classification)

## Notebooks

The following notebooks are responsible for generating datasets and augmenting them for further analysis:

- **[detection_dataset.ipynb](notebooks/dataset_generation/detection_dataset.ipynb)**  
  This notebook utilizes the model output from the clustering notebook to create a labeled dataset of cells and noise.

These notebooks analyze the results of the segmentation and classification processes:

- **[clasiffy_df.ipynb](notebooks/result_analysis/clasiffy_df.ipynb)**  
  This notebook processes all the images from the selected dataset, searches for all the crops belonging to these images, and uses all the models to predict whether each crop is noise or a cell. The results are stored in the corresponding CSV file for each image.

- **[results_comparisson.ipynb](notebooks/result_analysis/results_comparisson.ipynb)**  
  This notebook uses the output from `clasiffy_df.ipynb` to create tables and plots for comparing model performance.

- **[onion_cell_merged_section_performance.ipynb](notebooks/result_analysis/onion_cell_merged_section_performance.ipynb)**  
  This notebook creates confusion matrices to compare the performance of a model across all sections of the onion cell merged dataset.

- **[ina_section_performance.ipynb](notebooks/result_analysis/ina_section_performance.ipynb)**  
  This notebook creates confusion matrices to compare the performance of a model across all sections of the ina dataset.

The following notebooks are responsible for training machine learning models:

## Experiments

The following experiments are responsible for:

- **[allium-cepa-dataset-unlabeled](experiments/allium-cepa-dataset-unlabeled)**  
  This experiment analyzes the base dataset using SAM, produces the detection data (bounding boxes) of each found object, calculates the average area of the cell objects and produces a normalized (in pixel size) dataset containing crops of each of the detected objects.

- **[cell-detection-encoders-training](experiments/cell-detection-encoders-training)**  
  Trains an encoder on the cropped images data. This encoder is later used to generate the clusters used in the suppervised training experiment.

- **[cell-detection-encoders-training](experiments/cell-detection-encoders-training)**  
  Using the clusters defined manually (with the help of the previous encoder), this experiment construct an agumented training dataset with labels `cell` and `not` to perform supervised training. The result is a classification model that will determine if the entities detected by SAM are cells or not.


## Future Developments: Classification

In this section, we will explore various methods for classifying cells. The inputs for these methods will be the outputs from the segmentation
