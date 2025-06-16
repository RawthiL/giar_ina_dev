import sys
import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024 "


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="This script uses a SAM model to perform an initial segmentation of potential cells in a full-fov image."
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
        "--batch_size",
        "-bs",
        type=int,
        required=True,
        help="Batch size to use.",
    )
    parser.add_argument(
        "--input_shape",
        "-is",
        type=str,
        required=True,
        help="Input shape to the encoder.",
    )
    parser.add_argument(
        "--val_split",
        "-vs",
        type=float,
        required=True,
        help="Validation split proportion.",
    )
    parser.add_argument(
        "--kernel_size",
        "-ks",
        type=str,
        required=True,
        help="Convolution kernel size.",
    )
    parser.add_argument(
        "--filter_size",
        "-fs",
        type=int,
        required=True,
        help="Number of filters per layer.",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        required=True,
        help="Number of epochs to train.",
    )

    args = parser.parse_args()

    SEED = int(args.seed)
    OUTPUT_PATH = args.output
    INPUT_SHAPE = [int(a) for a in args.input_shape.split(",")]
    BATCH_SIZE = args.batch_size
    VALIDITAION_SPLIT = args.val_split
    KERNEL_SIZE = [int(a) for a in args.kernel_size.split(",")]
    FILTER_SIZE = args.filter_size
    EPOCHS = args.epochs

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
    import random
    import itertools

    import numpy as np
    import matplotlib.pyplot as plt

    import tensorflow as tf
    from tensorflow.keras import layers
    from keras.callbacks import EarlyStopping
    from tensorflow.keras.models import Sequential, Model, clone_model
    from tensorflow.keras.optimizers import Adam

    from dvclive.keras import DVCLiveCallback
    from dvclive import Live

    sys.path.insert(0, "../../")
    from config import DATASETS_PATH

    sys.path.insert(0, "../../packages/python")
    from data import datasets as data_loading

    # Load Dataset

    CROPPED_PATH = os.path.join(DATASETS_PATH, "cropped", "ina", "images")

    image_paths = list()
    for part in os.listdir(CROPPED_PATH):
        for file in os.listdir(os.path.join(CROPPED_PATH, part)):
            image_paths.append(os.path.join(CROPPED_PATH, part, file))

    # Train / val split
    val_size = int(VALIDITAION_SPLIT * len(image_paths))
    train_paths = image_paths[:-val_size]
    val_paths = image_paths[-val_size:]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_paths)

    # Train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
    train_dataset = train_dataset.map(
        lambda x: data_loading.load_image(x, INPUT_SHAPE),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.repeat()

    # Val dataset
    validation_dataset = val_dataset.map(
        lambda x: data_loading.load_image(x, INPUT_SHAPE),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.repeat()

    # Weigthed Loss: SSIM + MAE
    class CustomLoss(tf.keras.losses.Loss):
        def __init__(self, y, z, name="custom_loss"):
            super().__init__(name=name)
            # Weights
            self.y = y
            self.z = z

            # MAE
            self.mae_loss_fn = tf.keras.losses.MeanAbsoluteError()

        # SSIM loss
        def ssim_loss(self, y_true, y_pred):
            ssim = (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))) / 2
            return ssim

        def call(self, y_true, y_pred):
            mae_loss = self.mae_loss_fn(y_true, y_pred)
            ssim = self.ssim_loss(y_true, y_pred)

            return self.y * mae_loss + self.z * ssim

    # Create AutoEncoder

    # Encoder
    encoder_input = layers.Input(shape=INPUT_SHAPE)
    x = encoder_input
    x = layers.Rescaling(1.0 / 255.0)(x)
    x = layers.Conv2D(FILTER_SIZE, KERNEL_SIZE, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2), padding="same")(x)
    x = layers.Conv2D(FILTER_SIZE // 2, KERNEL_SIZE, activation="relu", padding="same")(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2), padding="same")(x)
    x = layers.Conv2D(FILTER_SIZE // 4, KERNEL_SIZE, activation="relu", padding="same")(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2), padding="same")(x)
    x = layers.Conv2D(FILTER_SIZE // 8, KERNEL_SIZE, activation="relu", padding="same")(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2), padding="same")(x)
    x = layers.Conv2D(
        FILTER_SIZE // 16, KERNEL_SIZE, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2), padding="same")(x)
    x = layers.Flatten()(x)
    encoder_output = x
    encoder = Model(encoder_input, encoder_output)

    # Decoder
    decoder_input = layers.Input(shape=[encoder.output_shape[-1]])
    x = decoder_input
    x = layers.Reshape(
        (2 * FILTER_SIZE // 16, 2 * FILTER_SIZE // 16, FILTER_SIZE // 16)
    )(x)
    x = layers.Conv2D(
        FILTER_SIZE // 16, KERNEL_SIZE, activation="relu", padding="same"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(FILTER_SIZE // 8, KERNEL_SIZE, activation="relu", padding="same")(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(FILTER_SIZE // 4, KERNEL_SIZE, activation="relu", padding="same")(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(FILTER_SIZE // 2, KERNEL_SIZE, activation="relu", padding="same")(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(FILTER_SIZE, KERNEL_SIZE, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(1, KERNEL_SIZE, activation="sigmoid", padding="same")(x)

    # x = layers.Rescaling(255.0)(x)
    decoder_output = x
    decoder = Model(decoder_input, decoder_output)

    # Build the autoencoder model
    # autoencoder = Model(encoder_input, decoder_output)
    autoencoder = Sequential([encoder, decoder])

    encoder.summary()
    decoder.summary()
    autoencoder.summary()

    # Compile & fit - Grid search

    # Weight range from 0.2 to 1
    y_values = np.arange(0.2, 1.2, 0.2)
    z_values = np.arange(0.2, 1.2, 0.2)

    # Combinations of weights
    param_combinations = list(itertools.product(y_values, z_values))

    random.seed(34)
    random.shuffle(param_combinations)

    # Normalize weights ---> y + z = 1
    def normalize_weights(weights):
        total_sum = sum(weights)
        return tuple(weight / total_sum for weight in weights)

    param_combinations = [normalize_weights(weights) for weights in param_combinations]
    param_combinations = set(param_combinations)  # delete duplicate
    print("Total combinations: ", len(param_combinations))
    for i, combination in enumerate(param_combinations):
        print(f"Combination {i+1}: ", tuple(map(float, combination)))

    # Train model

    def evaluate_model(
        autoencoder, train_gen, val_gen, params, steps_per_epoch, val_steps
    ):
        y, z = [np.round(arr, 2) for arr in params]

        loss_fn = CustomLoss(y=y, z=z)
        model = clone_model(autoencoder)
        model.compile(loss=loss_fn, optimizer=Adam(learning_rate=1e-3))

        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
        )

        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=20,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=[early_stop, DVCLiveCallback()],
            verbose=1,
        )

        val_loss = min(history.history["val_loss"])
        return val_loss

    best_params = None
    best_loss = float("inf")

    steps_per_epoch = int(len(train_paths) // BATCH_SIZE)
    steps_per_epoch_val = int(len(val_paths) // BATCH_SIZE)

    # Evaluate Autoencoder for all the combinations of weights
    for params in param_combinations:
        try:
            val_loss = evaluate_model(
                autoencoder,
                train_dataset,
                validation_dataset,
                params,
                steps_per_epoch,
                steps_per_epoch_val,
            )
            print(f"Params {params} -> Val Loss: {val_loss}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params
        except Exception as e:
            print(f"{e}")

    # Resultados finales
    print(f"Best Params: {best_params} -> Best Val Loss: {best_loss}")

    # Compile y fit: Best Params
    loss_fn = CustomLoss(y=best_params, z=best_loss)

    steps_per_epoch = int(len(train_paths) // BATCH_SIZE)
    steps_per_epoch_val = int(len(val_paths) // BATCH_SIZE)

    autoencoder.compile(loss=loss_fn, optimizer=Adam(learning_rate=1e-3))
    early_stop = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=0
    )

    autoencoder.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_dataset,
        validation_steps=steps_per_epoch_val,
        callbacks=[early_stop, DVCLiveCallback()],
        verbose=1,
    )

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    encoder.save(os.path.join(OUTPUT_PATH, "encoder.keras"))

    # Track example image
    with Live() as live:
        validation_images = validation_dataset.take(1)

        for images, _ in validation_images:
            shuffled_images = tf.random.shuffle(images)
            first_three_images = shuffled_images[:3]
            break

        processed_images = []
        for img in first_three_images:
            img_with_batch = tf.expand_dims(img, axis=0)
            processed_image = autoencoder(img_with_batch).numpy()
            processed_images.append(processed_image.squeeze())

        # Plot
        n = 0
        fig = plt.figure(dpi=150)

        plt.subplot(3, 2, 1)
        plt.imshow(first_three_images[n], cmap="gray")  # Mostrar la imagen original
        plt.axis(False)
        plt.title(f"Original Image {n+1}")

        plt.subplot(3, 2, 2)
        plt.imshow(processed_images[n], cmap="gray")  # Mostrar la imagen procesada
        plt.axis(False)
        plt.title(f"Processed Image {n+1}")

        n = 1
        plt.subplot(3, 2, 3)
        plt.imshow(first_three_images[n], cmap="gray")
        plt.axis(False)
        plt.title(f"Original Image {n+1}")

        plt.subplot(3, 2, 4)
        plt.imshow(processed_images[n], cmap="gray")
        plt.axis(False)
        plt.title(f"Processed Image {n+1}")

        n = 2
        plt.subplot(3, 2, 5)
        plt.imshow(first_three_images[n], cmap="gray")
        plt.axis(False)
        plt.title(f"Original Image {n+1}")

        plt.subplot(3, 2, 6)
        plt.imshow(processed_images[n], cmap="gray")
        plt.axis(False)
        plt.title(f"Processed Image {n+1}")

        plt.tight_layout()
        # Save and log the image
        filename = os.path.join(OUTPUT_PATH, "encoder_example.png")
        # plt.savefig(filename)
        plt.draw()
        live.log_image(filename, fig)
        plt.close()

    return


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
