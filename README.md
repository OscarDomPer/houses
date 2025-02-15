<div align="center">

# "HOUSE PRICES: ADVANCED REGRESSION TECHNIQUES"  
## Un Enfoque Híbrido con Aprendizaje Supervisado y No Supervisado  
</div>

<br>
<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/0.png">
  
</div>

<br>


## Tecnologías usadas

**Lenguajes:**
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)

**Librerías:**  
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white) 
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white) 
![Seaborn](https://img.shields.io/badge/-Seaborn-0095A7?style=flat&logo=plotly&logoColor=white) 
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat&logo=plotly&logoColor=white) 
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) 
![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)  

## Introducción

Este proyecto utiliza el dataset **"House Prices: Advanced Regression Techniques"** para predecir el valor de las viviendas mediante técnicas avanzadas de **machine learning**.  

### El proceso incluye:  

- **Limpieza y transformación de los datos** para mejorar su calidad.  
- **Evaluación de múltiples modelos predictivos** para seleccionar el más preciso.  
- **Aprendizaje no supervisado** para agrupar las casas en distintos clusters según sus características.  
- **Reentrenamiento supervisado** dentro de cada cluster, mejorando el rendimiento del modelo y obteniendo información valiosa sobre la importancia de las características en cada grupo.  

Este **enfoque híbrido** permite no solo **mejorar la precisión** de las predicciones, sino también **extraer insights clave** sobre los factores que influyen en el precio de las viviendas en diferentes segmentos del mercado.  

<br>

## Análisis exploratorio de los datos

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/02.png">
  
</div>

El dataset está compuesto por **80 variables** (incluida la variable objetivo) de naturaleza tanto **categórica como numérica**.  

Se observa un alto porcentaje de datos faltantes en algunas columnas. Dado que el dataset tiene 1460 filas, ciertas variables presentan hasta un **90 % de valores NaN**.  

Sin embargo, no se descartarán estas variables de inmediato, ya que aún podrían ofrecer información útil. Por ejemplo, los valores nulos en la columna **`Fence`** podrían interpretarse como la **ausencia de cercas** y ser sustituidos por un **0**.  

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/03.png">
  
</div>

Lo primero que llama la atención es que algunas columnas **numéricas** podrían ser, en realidad, **categóricas**. Tras revisar la descripción de los datos proporcionada, podemos concluir que **MSSubClass** es, de hecho, una columna categórica. También se observa que algunas variables presentan un **sesgo a la derecha** en su distribución.  

Algunas variables categóricas tienen una **única categoría**, lo que las hace completamente inútiles, por lo que deben ser **eliminadas**. Además, hay otras variables en las que una **categoría domina significativamente** sobre las demás, lo que las hace susceptibles de algún tipo de transformación para reducir la dimensionalidad. Por último, un tercer grupo de variables presenta **demasiadas categorías**, que en el futuro muy posiblemente deberán ser reducidas para controlar la dimensionalidad del modelo.  

Parece que muchas variables tienen **valores atípicos**. Sin embargo, dada la naturaleza del dataset, no se puede saber a priori si son valores **legítimamente altos** o si se deben a **errores en la recolección de datos**.  

Este hecho, junto con el uso de **modelos robustos** que manejan bien los outliers para la predicción, descarta cualquier tipo de **eliminación generalizada** de estos valores. En su lugar, se tratará **cada caso de forma individual**.  

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/04.png">
  
</div>

Parece que muchas variables tienen **valores atípicos**. Sin embargo, dada la naturaleza del dataset, no se puede saber a priori si son valores **legítimamente altos** o si se deben a **errores en la recolección de datos**.  

Este hecho, junto con el uso de **modelos robustos** que manejan bien los outliers para la predicción, descarta cualquier tipo de **eliminación generalizada** de estos valores. En su lugar, se tratará **cada caso de forma individual**.  

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/05.png">
  
</div>

Se observan **correlaciones** tanto con **SalePrice** como entre otras variables. Por ejemplo, las correlaciones de **YearBuilt** con otras variables sugieren distintas preferencias en el diseño de las casas a lo largo de los años.  

Analizando concretamente las **correlaciones de las variables numéricas** con la variable objetivo, vuelven a aparecer **fuertes correlaciones**, mientras que en otras variables no tanto. Sin embargo, estas últimas pueden ser **importantes para el modelo** al establecer **relaciones no lineales**.  

<br>

## Tratamiento de los datos
A continuación se detallará el tratamiento de los datos de algunas de las variables a modo de ejemplo. Se puede consultar el tratamiento para cada variable en el notebook: [Tratamiento de Variables - Notebook](https://github.com/OscarDomPer/houses/blob/main/02_NTT_trat_train.ipynb)

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/06.png">
  
</div>











  


