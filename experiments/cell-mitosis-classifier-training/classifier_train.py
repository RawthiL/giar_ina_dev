import sys
import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024 "


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="This script performs the training of a classification model for mitosis/no mitosis."
    )

    # Add arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for the resulting model.",
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
        help="Input shape of the classifier.",
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
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        required=True,
        help="Batch size to use.",
    )
    parser.add_argument(
        "--color_mode",
        "-cm",
        type=int,
        required=True,
        help="Color mode to use, 1 for grayscale, 2 for RGB. Defaults to grayscale for every other input",
    )
    parser.add_argument(
        "--l2",
        "-l2",
        type=float,
        required=True,
        help="L2 regularization factor to use in the model.",
    )
    parser.add_argument(
        "--num_classes",
        "-nc",
        type=int,
        required=True,
        help="Number of classes to classify.",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        required=True,
        help="Learning rate to use in the model.",
    )

    args = parser.parse_args()

    SEED = int(args.seed)
    OUTPUT_PATH = args.output
    INPUT_SHAPE = [int(a) for a in args.input_shape.split(",")]
    BATCH_SIZE = args.batch_size
    COLOR_MODE = "rgb" if args.color_mode == 2 else "grayscale"
    L2 = float(args.l2)
    NUM_CLASSES = args.num_classes
    LEARNING_RATE = float(args.learning_rate)
    KERNEL_SIZE = [int(a) for a in args.kernel_size.split(",")]
    FILTER_SIZE = args.filter_size
    EPOCHS = args.epochs    
    OUTPUT_PATH = args.output

    import os

    # Configure the GPU backend to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_list
    # Import and set random seeds
    import numpy as np
    import random
    import tensorflow as tf

    #Configure TensorFlow to use the specified GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print("GPU Available:", gpus)
    print("cuDNN Enabled:", tf.test.is_built_with_cuda())

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Do the rest of imports
    import os
    import sys
    import numpy as np
    import tensorflow as tf
    import keras
    from keras import layers, Model
    from keras.applications import VGG16 # O ResNet50, MobileNetV2, etc.
    from dvclive.keras import DVCLiveCallback
    from dvclive import Live

    sys.path.insert(0, "../../")
    from config import DATASETS_PATH

    # Load Dataset
    TRAIN_PATH = os.path.join(DATASETS_PATH, 'mitosis_classification', 'train')
    VALIDATION_PATH = os.path.join(DATASETS_PATH, 'mitosis_classification', 'valid')
    TEST_PATH = os.path.join(DATASETS_PATH, 'mitosis_classification', 'test')

    def data_gen():
        """
        Genera y preprocesa los datasets de entrenamiento, validación y test
        para un modelo de clasificación de imágenes.
        """

        # Función auxiliar para crear y preprocesar un generador de datos
        def create_and_preprocess_generator(path, shuffle=True):
            generator = tf.keras.utils.image_dataset_from_directory(
                path,
                labels="inferred",
                label_mode="categorical",
                batch_size=BATCH_SIZE,
                image_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                color_mode=COLOR_MODE,
                shuffle=shuffle,
                seed=SEED # Mantener la semilla para reproducibilidad
            )
            generator = generator.map(
                lambda x, y: (x / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            generator = generator.prefetch(buffer_size=tf.data.AUTOTUNE)
            return generator

        # Crear generadores para entrenamiento, validación y test
        train_generator = create_and_preprocess_generator(TRAIN_PATH, shuffle=True)
        validation_generator = create_and_preprocess_generator(VALIDATION_PATH, shuffle=False) # No es necesario shuffear la validación

        # El test set no necesita shuffe y puede cargarse solo una vez si la función se usa para retornar todo
        test_generator = create_and_preprocess_generator(TEST_PATH, shuffle=False)

        return train_generator, validation_generator, test_generator

    train_generator,validation_generator,test_generator=data_gen()


    # --- Construcción del Modelo ---
    # 1. Definir la nueva capa de entrada
    # Se espera (Alto, Ancho, Canales). VGG16 necesita 3 canales.
    input_tensor = layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))

    # 3. Cargar el modelo base pre-entrenado (ej. VGG16)
    # include_top=False: no incluye las capas densas de clasificación originales de VGG16
    # weights='imagenet': usa los pesos pre-entrenados en ImageNet
    x = input_tensor
    base_model = VGG16(input_tensor=x, include_top=False, weights='imagenet')

    # 4. Congelar las capas del modelo base
    # Esto evita que los pesos de VGG16 se modifiquen durante el entrenamiento inicial
    base_model.trainable = False 

    # 5. Agregar tus propias capas de clasificación sobre el modelo base
    x = base_model.output # Salida del modelo base (tensor)
    x = layers.GlobalAveragePooling2D()(x) # O GlobalMaxPooling2D
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.004))(x)
    x = layers.Dropout(0.4)(x)

    # Capa de salida para 2 clases:
    # Para clasificación binaria, una neurona con activación 'sigmoid' es común
    # y se usa con BinaryCrossentropy.
    output_layer = layers.Dense(NUM_CLASSES, activation='linear')(x)

    # Crear el modelo final
    model = Model(inputs=input_tensor, outputs=output_layer)

    # Compilar el modelo
    # Para 2 clases y salida con 'sigmoid', BinaryCrossentropy es la correcta.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 



    # input_layer = layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))

    # x = input_layer 

    # # Bloque Convolucional 1
    # x = layers.Conv2D(FILTER_SIZE, KERNEL_SIZE, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(L2))(input_layer)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPool2D((2, 2), padding='same')(x)
    # x = layers.Dropout(0.25)(x) # Añadido Dropout después del MaxPooling para regularización

    # # Bloque Convolucional 2
    # # Duplicar filtros es una práctica común para capturar características más complejas
    # x = layers.Conv2D(FILTER_SIZE * 2, KERNEL_SIZE, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(L2))(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPool2D((2, 2), padding='same')(x)
    # x = layers.Dropout(0.25)(x)

    # # Bloque Convolucional 3
    # x = layers.Conv2D(FILTER_SIZE * 4, KERNEL_SIZE, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(L2))(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPool2D((2, 2), padding='same')(x)
    # x = layers.Dropout(0.25)(x) # Aumentar ligeramente el Dropout

    # # Bloque Convolucional 4 (Opcional, dado el tamaño de imagen)
    # # Para imágenes de 128x128, 3-4 bloques suelen ser suficientes.
    # x = layers.Conv2D(FILTER_SIZE * 8, KERNEL_SIZE, activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(L2))(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.MaxPool2D((2, 2), padding='same')(x)
    # x = layers.Dropout(0.3)(x)

    # # Capas Densely Connected (Clasificador)
    # x = layers.Flatten()(x)
    # x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.004))(x) # Aumentar neuronas en la capa Dense
    # x = layers.Dropout(0.4)(x) # Aumentar Dropout en la capa Dense

    # # Capa de Salida
    # # Para 2 clases y 'CategoricalCrossentropy(from_logits=True)', 'linear' es correcto.
    # # Si usarías 'CategoricalCrossentropy' (sin from_logits=True), necesitarías 'softmax'.
    # # output_layer = layers.Dense(NUM_CLASSES, activation='linear')(x)

    # model = Model(inputs=input_layer, outputs=output_layer)

    # # --- Compilación del Modelo ---
    # model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), # Especificar learning_rate
    #     loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    #     metrics=['accuracy']
    # )

    model.summary() # Muestra un resumen de la arquitectura del modelo


    # --- Callbacks ---
    stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=8, # Aumentar paciencia para dar más oportunidades al modelo
        verbose=1, # Cambiar a 1 para ver mensajes cuando se detiene
        mode="min", # Asegurarse que es 'min' para val_loss
        restore_best_weights=True,
        start_from_epoch=2,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )

    # --- Entrenamiento del Modelo ---
    with Live() as live:
        model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[stopping, reduce_lr, DVCLiveCallback(live=live)], # Activar el callback EarlyStopping
            verbose=1,
        )

        os.makedirs(OUTPUT_PATH, exist_ok=True)
        model.save(os.path.join(OUTPUT_PATH, "mitosis_classifier.keras"))

    return


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
