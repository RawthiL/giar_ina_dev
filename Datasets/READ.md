## Datasets
En este directorio se tratá la preparación de los datasets.
En /Augmentation se encuentran scripts relacionados al análisis del dataset y realizar data augmentation

### image_cutting
Este script buscará en un directorio dado todas las imagenes y las recortará creando un metadata.csv con información sobre las clases que tenga cada imagen cortada

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
├── raw_dataset_512x512(ov256x256) (Carpeta con recortes, indica ancho, alto y el overlap usado)
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

### complete_cell_detection
Este script recibirá el path donde se encontrán las imágenes a analizar, tomará todas las imagenes y por cada una realizará una erosión para eliminar posibles solapamientos de células, luego detectará cuántas células completas hay por imágen, creando un csv con la cantidad de células enteras por cada imagen. Además si se desea se puede adicionar un análisis del tamaño de cada célula entera. Para esto, luego de la erosión y detección, se toma cada contorno, se lo separa en una imagen nueva, se lo dilata con el mismo kernel que se erosionó para recuperar el tamaño original y se obtiene la información del área, ancho y alto de cada célula que este entera. Este procedimiento produce un error menor al 1% del tamaño de cada célula.

Uso:

image_cutting.py [-h] [--width WIDTH] [--height HEIGHT] [--type TYPE] [--metadata METADATA] [--overlapX OVERLAPX] [--overlapY OVERLAPY] path

complete_cell_detection.py [-h] [--outputDir OUTPUTDIR] [--type TYPE] [--kernel KERNEL] [--eroded ERODED] path

Arugmentos obligatorios:
  path                  Directory where the images are

options:
  -h, --help                    Show this help message and exit
  --outputDir OUTPUTDIR         Directory where to store the csv output, default is parent of PATH
  --type TYPE                   File extention to look for, default is .png
  --kernel KERNEL               Define the NxN kernel to use, where N is KERNEL
  --cellAnalysis CELLANALYSIS   Decides whether to include the analysis of complete cells or not

Ejemplo:
$python3 ./Augmentation/complete_cell_detection.py ./raw_dataset/target --cellAnalysis True

Salida:
Generará dos archivos csv en ./raw_dataset:
    complete_cells_per_image_in_target.csv: Contiene la cantidad de celulas completas por imagen
    complete_cells_data_in_target.csv: Contiene el área, x, y, ancho, alto, area de la bounding box y la imagen de origen de cáda célula completa 
