import sys
import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024 "


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="This script performs the training of a clustering method for the cropped datasets."
    )

    # Add arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for the resulting dataset.",
    )
    parser.add_argument(
        "--gpus_list",
        "-gpus",
        type=str,
        required=False,
        default="0",
        help="List of available GPUs to use. PCIe order.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        required=False,
        default=0,
        help="Random seed to be used in the run.",
    )
    parser.add_argument(
        "--input_shape",
        "-is",
        type=str,
        required=True,
        help="Input shape to the encoder.",
    )
    parser.add_argument(
        "--cluster_num",
        "-cn",
        type=int,
        required=True,
        help="Number of clusters to create.",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        required=True,
        help="Agglomeration method. Choose from: 'hdbscan', 'kmeans' or 'agglomerative'",
    )
    parser.add_argument(
        "--encoding_model",
        "-em",
        type=str,
        required=True,
        help="Base (pre-trained) encoding model. Chose from: VGG16, VGG19, ResNet50, InceptionV3, DenseNet121, MobileNetV2.",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        required=True,
        help="Batch size to use.",
    )

    args = parser.parse_args()

    SEED = int(args.seed)
    INPUT_SHAPE = [int(a) for a in args.input_shape.split(",")]
    NUM_CLUSTERS = args.cluster_num
    METHOD = args.method
    MODEL = args.encoding_model
    BATCH_SIZE = args.batch_size
    OUTPUT_PATH = args.output

    # Configure the GPU backend to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_list
    # Import and set random seeds
    import numpy as np
    import random
    import tensorflow as tf

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Do the rest of imports
    from tqdm.contrib.concurrent import process_map
    import cv2
    import joblib
    from sklearn.cluster import KMeans
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.cluster import HDBSCAN
    from tensorflow.keras.applications import (
        VGG16,
        VGG19,
        ResNet50,
        InceptionV3,
        DenseNet121,
        MobileNetV2,
    )
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model
    import keras

    sys.path.insert(0, "../../packages/python")
    from data import utils as data_utils

    sys.path.insert(0, "../../")
    from config import DATASETS_PATH

    CROPPED_PATH = os.path.join(DATASETS_PATH, "cropped", "ina", "images")

    # Load pre-trained models
    if MODEL == "VGG16":
        preprocess_input = keras.applications.vgg16.preprocess_input
        encoder = VGG16(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    elif MODEL == "VGG19":
        preprocess_input = keras.applications.vgg19.preprocess_input
        VGG19(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    elif MODEL == "ResNet50":
        preprocess_input = tf.keras.layers.Identity
        ResNet50(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    elif MODEL == "InceptionV3":
        preprocess_input = keras.applications.inception_v3.preprocess_input
        InceptionV3(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    elif MODEL == "DenseNet121":
        preprocess_input = tf.keras.layers.Identity
        DenseNet121(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    elif MODEL == "MobileNetV2":
        preprocess_input = tf.keras.layers.Identity
        MobileNetV2(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    else:
        raise ValueError("Selected model not supported")

    inp = keras.Input(shape=INPUT_SHAPE)
    x = preprocess_input(inp)
    x = encoder(x)
    x = Flatten()(x)
    encoder = Model(inputs=inp, outputs=x)

    # Load the images
    CROPPED_PATHs = sorted(data_utils.get_relative_file_paths(CROPPED_PATH))
    images = process_map(
        cv2.imread,
        CROPPED_PATHs,
        total=len(CROPPED_PATHs),
        max_workers=16,
        chunksize=32,
    )

    # Generate encoder embeddings

    # Transform input images for encoder input
    resized_images = [cv2.resize(image, INPUT_SHAPE[0:2]) for image in images]
    resized_images = np.array(resized_images)

    # Extract features from encoder
    enc_features_array = np.zeros((resized_images.shape[0], encoder.output_shape[-1]))

    ini = 0
    while True:
        start = ini * BATCH_SIZE
        end = start + BATCH_SIZE

        if start >= resized_images.shape[0]:
            break

        if end >= resized_images.shape[0]:
            end = resized_images.shape[0] - 1

        this_batch = resized_images[start:end]

        enc_features_array[start:end] = encoder.predict(this_batch, verbose=0)

        ini += 1

    # enc_features_array_norm = [a / (np.linalg.norm(a) + 1e-16) for a in enc_features_array]
    enc_features_array_norm = enc_features_array

    # Train Clustering

    if METHOD == "kmeans":
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
        clustering = kmeans.fit(enc_features_array_norm)

        centroids = kmeans.cluster_centers_
        seleccted_class = -np.ones((len(enc_features_array_norm)), dtype=int)
        for idx, feature in enumerate(enc_features_array_norm):
            dist = 1e99
            for cluster in range(NUM_CLUSTERS):
                t_dist = np.linalg.norm(feature - centroids[cluster])
                if dist > t_dist:
                    dist = t_dist
                    seleccted_class[idx] = cluster
    elif METHOD == "agglomerative":
        # Aca si o si normalizamos
        enc_features_array_norm = [
            a / (np.linalg.norm(a) + 1e-16) for a in enc_features_array
        ]

        # ag_clustering = AgglomerativeClustering
        #     n_clusters = None,
        #     metric = 'euclidean',
        #     linkage = 'ward',
        #     distance_threshold = 1.0,
        #     compute_full_tree = True,
        # )

        ag_clustering = AgglomerativeClustering(
            n_clusters=NUM_CLUSTERS,
            linkage="ward",
        )

        clustering = ag_clustering.fit(enc_features_array_norm)
        seleccted_class = clustering.labels_

    elif METHOD == "hdbscan":
        hdb = HDBSCAN()
        clustering = hdb.fit(enc_features_array_norm)
        seleccted_class = clustering.labels_

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    joblib.dump(clustering, os.path.join(OUTPUT_PATH, "clustering.pkl"))

    return


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
