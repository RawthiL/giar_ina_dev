import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
MODELS_PATH = os.path.join(BASE_DIR, 'models')
DATASETS_PATH = os.path.join(BASE_DIR, 'datasets')
IMG_SHAPE = (200, 200)
SEED = 42
# TEMP_PATH = os.path.join(BASE_DIR, 'temp')
# RESULTS_PATH = os.path.join(BASE_DIR, 'results')
# IMAGES_PATH = os.path.join(DATASETS_PATH, 'images')
# CROPPED_PATH = os.path.join(DATASETS_PATH, 'cropped_images')