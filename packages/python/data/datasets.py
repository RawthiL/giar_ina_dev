import tensorflow as tf
import numpy as np
import cv2


def decode_example_to_cv2(example):
    nparr = np.frombuffer(example["image"], np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def decode_example_to_tensorflow(example, input_shape):
    image = tf.io.decode_image(example["image"], channels=1)  # Escala de grises
    image = tf.image.resize(image, (input_shape[0], input_shape[1]))
    return image


def decode_labeled_example_to_tensorflow(example, input_shape):
    image = tf.io.decode_image(example["image"], channels=1)  # Escala de grises
    image = tf.image.resize(image, (input_shape[0], input_shape[1]))
    if example["label"] == "cells":
        onehot = tf.squeeze(tf.one_hot([0], 2))
    else:
        onehot = tf.squeeze(tf.one_hot([1], 2))
    return image, onehot


def augment_tensorflow(image, input_shape):
    # Augmentación

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    scales = tf.random.uniform([], 0.8, 1.0)
    crop_size = tf.cast(scales * input_shape[0], tf.int32)
    image = tf.image.random_crop(image, size=[crop_size, crop_size, 1])
    image = tf.image.resize(image, (input_shape[0], input_shape[1]))
    # Normalize [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    return image, image
