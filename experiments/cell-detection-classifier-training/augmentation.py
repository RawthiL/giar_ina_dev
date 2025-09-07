import sys
import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024 "


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="This script augments a labeled datasets to later perform training on it."
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
        "--seed",
        "-s",
        type=int,
        required=False,
        default=0,
        help="Random seed to be used in the run.",
    )
    parser.add_argument(
        "--noise_mean",
        "-nm",
        type=float,
        required=True,
        help="Mean of the added noise.",
    )
    parser.add_argument(
        "--noise_std",
        "-ns",
        type=float,
        required=True,
        help="Std of the added noise.",
    )
    parser.add_argument(
        "--brightness_low",
        "-bl",
        type=float,
        required=True,
        help="Lower limit of the added brightness.",
    )
    parser.add_argument(
        "--brightness_high",
        "-bh",
        type=float,
        required=True,
        help="Higher limit of the added brightness.",
    )
    parser.add_argument(
        "--contrast_low",
        "-cl",
        type=float,
        required=True,
        help="Lower limit of the added contrast.",
    )
    parser.add_argument(
        "--contrast_high",
        "-ch",
        type=float,
        required=True,
        help="Higher limit of the added contrast.",
    )
    parser.add_argument(
        "--saturation_low",
        "-sl",
        type=float,
        required=True,
        help="Lower limit of the added saturation.",
    )
    parser.add_argument(
        "--saturation_high",
        "-sh",
        type=float,
        required=True,
        help="Higher limit of the added saturation.",
    )
    parser.add_argument(
        "--alpha_deform",
        "-ad",
        type=float,
        required=True,
        help="Aplpha. Elastic deformation of images as described in [Simard2003]_ (with modifications).",
    )
    parser.add_argument(
        "--sigma_deform",
        "-sd",
        type=float,
        required=True,
        help="Sigma. Elastic deformation of images as described in [Simard2003]_ (with modifications).",
    )
    parser.add_argument(
        "--alpha_affine_deform",
        "-aad",
        type=float,
        required=True,
        help="Aplpha affine. Elastic deformation of images as described in [Simard2003]_ (with modifications).",
    )
    parser.add_argument(
        "--parquet_shard_size_mb",
        "-ps",
        type=int,
        default=100,
        required=False,
        help="Size of the parquet shard in MB.",
    )


    args = parser.parse_args()

    SEED = int(args.seed)
    NOISE_MEAN = args.noise_mean
    NOISE_STD = args.noise_std
    SATURATION_LOW = args.saturation_low
    SATURATION_HIGH = args.saturation_high
    BRIGHTNESS_LOW = args.brightness_low
    # BRIGHTNESS_HIGH = args.brightness_high
    CONTRAST_LOW = args.contrast_low
    CONTRAST_HIGH = args.contrast_high
    ALPHA_DEFORM = args.alpha_deform
    SIGMA_DEFORM = args.sigma_deform
    ALPHA_AFFINE_DEFORM = args.alpha_affine_deform

    OUTPUT_PATH = args.output
    PARQUET_SHARD_SIZE_MB = int(args.parquet_shard_size_mb)

    # Import and set random seeds
    import numpy as np
    import random

    random.seed(SEED)
    np.random.seed(SEED)

    # Do the rest of imports
    import cv2
    import numpy as np
    from tqdm import tqdm
    from PIL import Image, ImageEnhance, ImageOps
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import map_coordinates
    import shutil
    from datasets import load_dataset
    import io
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path.insert(0, "../../")
    from config import DATASETS_PATH

    DATASET_PATH = os.path.join(DATASETS_PATH, "cell_detection", "ina", "train")

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # Función para añadir ruido a la imagen
    def add_noise(image):
        np_image = np.array(image)
        ruido = np.random.normal(NOISE_MEAN, NOISE_STD, np_image.shape).astype(
            np.int32
        )  # Nivel de ruido
        noisy_image = np.clip(np_image + ruido, 0, 255).astype(
            np.uint8
        )  # Asegurar valores válidos
        return Image.fromarray(noisy_image)

    # Función para deformación elástica
    def elastic_transform(image):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications)."""
        image = np.array(image)
        alpha = image.shape[1] * ALPHA_DEFORM
        sigma = image.shape[1] * SIGMA_DEFORM
        alpha_affine = image.shape[1] * ALPHA_AFFINE_DEFORM

        shape = image.shape
        shape_size = shape[:2]
        random_state = np.random.RandomState(None)

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [
                center_square + square_size,
                [center_square[0] + square_size, center_square[1] - square_size],
                center_square - square_size,
            ]
        )
        pts2 = pts1 + random_state.uniform(
            -alpha_affine, alpha_affine, size=pts1.shape
        ).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(
            image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
        )

        # Elastic deformation (Gaussian noise)
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        # dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(
            np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2])
        )
        indices = (
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1)),
            np.reshape(z, (-1, 1)),
        )

        # Apply elastic deformation
        deformed_image = map_coordinates(
            image, indices, order=1, mode="reflect"
        ).reshape(shape)

        # Convert back to PIL Image
        deformed_image_pil = Image.fromarray(deformed_image.astype(np.uint8))

        return deformed_image_pil

    # Aplicar todas las transformaciones juntas
    def augment_image(image):
        augmented_image = image

        # 1. Volteo horizontal
        augmented_image = ImageOps.mirror(augmented_image)

        # 2. Volteo vertical
        augmented_image = ImageOps.flip(augmented_image)

        # 3. Ajuste de brillo (ajuste más sutil)
        enhancer = ImageEnhance.Brightness(augmented_image)
        # brightness_factor = random.uniform(
        #     BRIGHTNESS_LOW, BRIGHTNESS_HIGH
        # )  # Antes: (0.6, 1.8)
        augmented_image = enhancer.enhance(BRIGHTNESS_LOW)

        # 4. Ajuste de contraste (ajuste más sutil)
        enhancer = ImageEnhance.Contrast(augmented_image)
        contrast_factor = random.uniform(
            CONTRAST_LOW, CONTRAST_HIGH
        )  # Antes: (0.6, 1.8)
        augmented_image = enhancer.enhance(contrast_factor)

        # 5. Cambio de saturación (ajuste más sutil)
        enhancer = ImageEnhance.Color(augmented_image)
        saturation_factor = random.uniform(
            SATURATION_LOW, SATURATION_HIGH
        )  # Ajuste sutil
        augmented_image = enhancer.enhance(saturation_factor)

        # 6. Añadir ruido (menos probabilidad de aplicarlo)
        # if random.random() < 0.5:  # 50% de probabilidad
        augmented_image = add_noise(augmented_image)

        # 7. Aplicar deformación elástica (probabilidad de aplicarla)
        # if random.random() < 0.:  # 30% de probabilidad
        augmented_image = elastic_transform(augmented_image)

        return augmented_image

    def generar_nombre(imagen_original):
        nombre, ext = os.path.splitext(imagen_original)
        return f"{nombre}_aug{ext}"
    
    # Directorio del dataset
    for split in ["cells", "not"]:
        this_path = os.path.join(DATASET_PATH, split)
        save_path = os.path.join(OUTPUT_PATH, split)

        if not os.path.exists(this_path):
            raise FileNotFoundError(f"El directorio {this_path} no existe.")

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # Load Dataset
        imagenes_originales_ds = load_dataset("parquet", 
                        data_files=os.path.join(this_path, "*.parquet"))

        # Determinar la cantidad máxima de imágenes aumentadas
        cantidad_aumentadas = int(len(imagenes_originales_ds['train']) * 1)  # Hasta el 70%
        contador = 0

        # Variables for parquet format saving
        shard_size_bytes = PARQUET_SHARD_SIZE_MB * 1024 * 1024
        shard_index = 0
        records = []
        total_written = 0

        # Selección aleatoria
        imagenes_originales_ds = imagenes_originales_ds.shuffle(seed=42)
        for example in tqdm(imagenes_originales_ds['train']):  
            if contador >= cantidad_aumentadas:
                break

            # Load image            
            img = Image.open(io.BytesIO(example['image']))

            # Aplicar augmentations
            augmented_image = augment_image(img)

            # Guardar la imagen
            nuevo_nombre = generar_nombre(example['filename'])

            # Add to parquet
            img_byte_arr = io.BytesIO()
            augmented_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            records.append({"image": img_byte_arr, "label": split, "filename": f"{nuevo_nombre}"})
            total_written += len(img_byte_arr)
            
            contador += 1

            # When current shard size exceeds limit, flush to disk
            if total_written >= shard_size_bytes:
                parquet_df = pd.DataFrame(records)
                parquet_table = pa.Table.from_pandas(parquet_df)
                out_path = os.path.join(save_path, f"augmented_data_shard-{shard_index:05d}.parquet")
                pq.write_table(parquet_table, out_path)

                # Reset for next shard
                shard_index += 1
                records = []
                total_written = 0
        # Write leftover records
        if records:
            parquet_df = pd.DataFrame(records)
            table = pa.Table.from_pandas(parquet_df)
            out_path = os.path.join(save_path, f"augmented_data_shard-{shard_index:05d}.parquet")
            pq.write_table(table, out_path)

        # Copy original parquet too
        for orig_data in os.listdir(this_path):
            if '.parquet' in orig_data:
                # Copy all dataset to new location with augmenttions
                file_path = os.path.join(this_path, orig_data)
                shutil.copy(file_path, os.path.join(save_path, orig_data))

        print(f"Se han generado {contador} imagenes.")

    return


# Run the main function if the script is executed directly
if __name__ == "__main__":
    print("----------------------------------------------------------------")
    print("- RUNNING AUGMENTATION STEP")
    print("----------------------------------------------------------------")
    main()
