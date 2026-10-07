- organismo_y_tejido : ver si se aplica o no a lo nuestro, ya q solo vemos nucleos perteneciente a la punta de la raiz (sea como sea q se llame)
- estado_mitosis : quitar intermedias, quedarse con las 5 q tenemos
- tipo_tincion : luego analizar frecuencia de tinciones observadas, creo q no tenemos tantas
- tipo_fov : parece  hablar de celulas enteras (nucleo + citoplasma) y no es lo que tenemos en el dataset q solo vemos nucleos
- calidad_enfoque : OK
- morfologia_celular : quitar todos los json anaidados, ya q al final vamos a aplanar la lista en un string
- forma_celular : idem tipo_fov
- integridad_pared_celular : no aplica, vemos nucleos
- morfologia_cromatina_nucleo : para este (y todos) armar un glosario de que es lo que significa cada uno, describiendo como es cada tag
- presencia_vacuola_central : Descripcion, ver luego el analisis de frecuencia (creo q no se ve en el nucleo)
- caption_concatenado_sd : quitar

**EDA** Exploratory Data Analysis
Tomar un subsample de 1000 uniformes entre estados de mitosis y fuen te de dataset (ina, roboflow, etc) contra set de test (tagueado) y ver que tan bien anda y analizar frecuencia de tags.
