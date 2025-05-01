import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
MODELS_PATH = os.path.join(BASE_DIR, 'models')
MEDIA_PATH = os.path.join(BASE_DIR, 'media')
TEMP_PATH = os.path.join(BASE_DIR, 'temp')
IMAGES_PATH = os.path.join(MEDIA_PATH, 'images')
CROPPED_PATH = os.path.join(MEDIA_PATH, 'cropped_images')