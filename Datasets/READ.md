### Datasets
En este directorio se tratá la preparación de los datasets.
En /Augmentation se encuentra el script image_cutting.py el cual buscará en un directorio dado todas las imagenes y las recortará creando un metadata.csv con información sobre las clases que tenga cada imagen cortada

Uso:

image_cutting.py [-h] [--width WIDTH] [--height HEIGHT] [--type TYPE] [--metadata METADATA] [--overlapX OVERLAPX] [--overlapY OVERLAPY] path

Arugmentos obligatorios:
  path                 Parent directory where the images are

Opciones:
  -h, --help           show this help message and exit
  --width WIDTH        Width to use when cutting images, default is 512
  --height HEIGHT      Height to use when cutting images, default is 512
  --type TYPE          File extention to look for, default is .png
  --metadata METADATA  Create csv metadata of image classes, default is true
  --overlapX OVERLAPX  Overlap to use in X when cutting, default is half of width
  --overlapY OVERLAPY  Overlap to use in Y when cutting, default is half of width

Ejemplo:
$python3 ./Augmentation/image_cutting.py ./raw_dataset

Salida:
Replicará el árbol de directorios de ./raw_dataset pero cortando todas la imagenes y creando un archivo de metadata
```
.
├── raw_dataset (Carpeta original)
    ├── input     
        ├── 331.png
        .
        .
        .
    └── target   
        ├── 331.png
        .
        .
        .
├── raw_dataset_512x512 (Carpeta nueva que indica el recorte de imagenes)
    ├── input     
        ├── 331.png
        .
        .
        .
    └── target   
        ├── 331.png
        .
        .
        .
    └── metadata.csv
```


