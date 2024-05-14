# Segmentación y Clasificación de Celulas - Protocolo Allium

Este repositorio contiene el codigo de desarollo de un sistema de segmentación y clasificación de celulas vegetales (especificamente las usadas para el protocolo Allium).

El repositorio se va a dividir en 3 partes principales, las cuales responden a:

### Preparación de Datasets
Esto incluye:
  - Transformar las segmentaciones manuales provistas en objetos binarios y/o colecciones de imágenes que puedan ser consumidas por los procesos de entrenamiento de los modelos a utilizar.
  - Crear procesos de aumentación de datos, ya sean recortes, transformaciones, adicionado de ruido y/o generacion de muestras artificiales.
  - Analisis de composición de los datasets.

### Segmentación
Todo lo que es referente a la segmentacion de las celulas por distintos metodos, como por ejemplo:
  - U-Net
  - SAM
  - etc
Todos los metodos deben acptar los mismos datasets de entrada y producir salidas del mismo tipo para luego porder ser comparados y analizados


### Clasificación
Aqui encontraremos todo lo que sea clasificación de celulas por distintos métodos (a explorar). La entrada a estos métodos seran las salidas de los metodos de segmentación y/o datasets generados especificamente.
Si bien esta tarea puede realizarse en tandem con la segmentación, es interesante explorar los clasificadores como una entidad separada.

