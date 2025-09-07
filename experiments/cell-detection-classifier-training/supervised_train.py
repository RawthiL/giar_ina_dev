import sys
import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024 "


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="This script trains a cell detectioon supervised model."
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
        "--epochs",
        "-e",
        type=int,
        required=True,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--color_mode",
        "-cm",
        type=str,
        required=True,
        help="Color mode for keras img loader.",
    )
    parser.add_argument(
        "--train_encoder",
        "-te",
        type=bool,
        required=True,
        help="Wheter to train or not the encoder.",
    )
    parser.add_argument(
        "--dropout",
        "-do",
        type=float,
        required=True,
        help="Weight dropout.",
    )
    parser.add_argument(
        "--l2_reg",
        "-l2",
        type=bool,
        required=True,
        help="L2 regularization.",
    )

    args = parser.parse_args()

    SEED = int(args.seed)
    OUTPUT_PATH = args.output
    INPUT_SHAPE = [int(a) for a in args.input_shape.split(",")]
    COLOR_MODE = args.color_mode
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    TRAIN_ENCODER = args.train_encoder
    DROP_OUT = args.dropout
    L2_REG = args.l2_reg

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
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import keras
    from keras import layers
    from sklearn.metrics import (
        accuracy_score,
        recall_score,
        confusion_matrix,
        precision_score,
    )
    from tensorflow.keras import regularizers
    from dvclive.keras import DVCLiveCallback
    from dvclive import Live
    from datasets import load_dataset

    sys.path.insert(0, "../../")
    from config import DATASETS_PATH, MODELS_PATH

    sys.path.insert(0, "../../packages/python")
    from data import datasets as data_loading

    ENCODER_PATH = os.path.join(MODELS_PATH, "cell_clustering", "encoder.keras")

    VALIDATION_PATH = os.path.join(DATASETS_PATH, "cell_detection", "ina", "validation")
    TEST_PATH = os.path.join(DATASETS_PATH, "cell_detection", "ina", "test")
    # This is the augmented data path
    TRAIN_PATH = "./outputs/train_dataset"

    def plot_confusion_matrix(cm, classes, title="Confusion matrix", cmap=plt.cm.Blues):
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title, fontsize=25)
        # plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
        plt.yticks(tick_marks, classes, fontsize=15)

        fmt = ".2f"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

        plt.ylabel("True label", fontsize=20)
        plt.xlabel("Predicted label", fontsize=20)

    def plot_conf_matrix(model, generator, titulo="Confusion matrix"):
        ### RETRIVE TEST LABEL FROM GENERATOR ###
        # test_num = sum(1 for _ in generator)
        label_test = []
        pred_test = []

        for i, batch in enumerate(generator):
            X, y = batch
            label_test.append(y.numpy())
            predictions = model.predict(X)
            predictions = tf.nn.softmax(predictions, axis=-1)
            pred_test.append(predictions)

        label_test = np.argmax(np.vstack(label_test), axis=1)

        ### COMPUTE PREDICTIONS ON TEST DATA ###
        pred_test = np.argmax(np.vstack(pred_test), axis=1)
        accuracy = accuracy_score(label_test, pred_test)
        recall = recall_score(label_test, pred_test, pos_label=0)
        specificity = recall_score(label_test, pred_test, pos_label=1)
        precision = precision_score(label_test, pred_test, pos_label=0)
        # pred_test = pred_test[:label_test.shape[0],]
        ### ACCURACY ON TEST DATA ###
        print("-" * 40)
        print("ACCURACY:", accuracy)
        print("RECALL:", recall)
        print("PRECISION:", precision)
        print("SPECIFICITY:", specificity)
        print("-" * 40)
        print("\n")
        ### CONFUSION MATRIX ON TEST DATA ###
        cnf_matrix = confusion_matrix(label_test, pred_test)
        # results.append({'Model': titulo, 'Accuracy': accuracy , 'Recall':recall,'Precision':precision,'Specificity':specificity})

        fig = plt.figure(figsize=(7, 7))
        plot_confusion_matrix(cnf_matrix, classes=["cell", "not"], title=titulo)
        plt.title(titulo)
        # plt.show()
        return fig

    def get_model(pretrain_encoder, TRAIN_ENCODER, input_shape, dropout_rate, l2_reg):
        encoder = pretrain_encoder

        inp = keras.Input(shape=input_shape)
        x = inp

        encoder.trainable = TRAIN_ENCODER

        x = encoder(x)

        x = layers.Dropout(dropout_rate)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(
            16, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(
            8, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(
            4, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        out = layers.Dense(2, activation="linear")(x)

        model = keras.Model(inp, out)

        return model

    # Load datasets

    # data generator function
    def data_gen():
        # Generator function
        def gen(dataset):
            for sample in dataset:
                yield data_loading.decode_labeled_example_to_tensorflow(sample, INPUT_SHAPE)
        output_signature = (
            tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(2), dtype=tf.float32)
        )

        # Trainig
        train_ds = load_dataset("parquet", 
                      data_files=os.path.join(TRAIN_PATH, "**", "*.parquet"))["train"]
        train_ds = train_ds.shuffle() # Shuffle here, otherwise it wont really shuffle after
        train_dataset = tf.data.Dataset.from_generator(lambda: gen(train_ds), output_signature=output_signature)
        # Train dataset
        train_dataset = train_dataset.map(
            lambda x, y: (x / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        train_dataset = train_dataset.shuffle(buffer_size=900)
        train_dataset = train_dataset.batch(BATCH_SIZE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        train_dataset = train_dataset.repeat()

        # Validation
        valid_ds = load_dataset("parquet", 
                      data_files=os.path.join(VALIDATION_PATH, "**", "*.parquet"))["train"]
        validation_dataset = tf.data.Dataset.from_generator(lambda: gen(valid_ds), output_signature=output_signature)
        # Validation dataset
        validation_dataset = validation_dataset.map(
            lambda x, y: (x / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        validation_dataset = validation_dataset.batch(BATCH_SIZE)
        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
        # validation_dataset = validation_dataset.repeat()

        # Test
        test_ds = load_dataset("parquet", 
                      data_files=os.path.join(TEST_PATH, "**", "*.parquet"))["train"]
        test_dataset = tf.data.Dataset.from_generator(lambda: gen(test_ds), output_signature=output_signature)
        # Validation dataset
        test_dataset = test_dataset.map(
            lambda x, y: (x / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        test_dataset = test_dataset.batch(BATCH_SIZE)
        test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
        # test_dataset = test_dataset.repeat()


        return train_dataset, validation_dataset, test_dataset, len(train_ds), len(valid_ds), len(test_ds)
    
    # Train Model
    (train_generator, 
     validation_generator, 
     test_generator,
     num_samples_train,
     num_samples_valid,
     num_samples_test) = data_gen()
    stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=2,
    )

    encoder = keras.saving.load_model(ENCODER_PATH)
    model = get_model(encoder, TRAIN_ENCODER, INPUT_SHAPE, DROP_OUT, L2_REG)
    model.compile(
        keras.optimizers.Adam(
            #  1e-4
        ),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    with Live() as live:
        model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[stopping, DVCLiveCallback(live=live)],
            steps_per_epoch=(num_samples_train//BATCH_SIZE)
        )

        os.makedirs(OUTPUT_PATH, exist_ok=True)
        model.save(os.path.join(OUTPUT_PATH, "cell_detection_model.keras"))

        # Log additional data after training
        test_loss, test_acc = model.evaluate(test_generator)
        live.log_metric("test_loss", test_loss, plot=False)
        live.log_metric("test_acc", test_acc, plot=False)

        fig = plot_conf_matrix(
            model, validation_generator, "Confusion Matrix Validation"
        )
        # Save and log the image
        filename = os.path.join(OUTPUT_PATH, "Confusion Matrix Validation.png")
        # plt.savefig(filename)
        plt.draw()
        live.log_image(filename, fig)
        plt.close()
        fig = plot_conf_matrix(model, test_generator, "Confusion Matrix Test")
        # Save and log the image
        filename = os.path.join(OUTPUT_PATH, "Confusion Matrix Test.png")
        # plt.savefig(filename)
        plt.draw()
        live.log_image(filename, fig)
        plt.close()

    return


# Run the main function if the script is executed directly
if __name__ == "__main__":
    print("----------------------------------------------------------------")
    print("- RUNNING CLASSIFIER TRAINING STEP")
    print("----------------------------------------------------------------")
    main()
