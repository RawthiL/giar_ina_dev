import tensorflow as tf

# Load & augmentation
def load_image(path, input_shape):

    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)  # Escala de grises
    image = tf.image.resize(image, (input_shape[0], input_shape[1]))
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
    return image,image