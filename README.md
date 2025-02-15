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

### LotFrontage: 

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/06.png">
  
</div>

Esta variable representa la medida del frente del lote que da a la calle. Es un buen ejemplo porque ilustra la política seguida tanto para los **outliers** como para los **NaN**.

### Outliers:  
El máximo de **LotFrontage** son dos propiedades de **313 pies** (aproximadamente 90 metros), lo que parece inverosímil para una propiedad residencial. Además, de acuerdo con sus correspondientes valores de **LotArea**, la propiedad tendría una forma extremadamente alargada, por lo que se asume que es un error de recolección de datos y se elimina. Este es uno de los pocos casos en los que se elimina el **outlier**. Otros casos, como una piscina de un área considerablemente más grande que una olímpica, se mantienen, ya que, si bien son atípicos, entran dentro de lo plausible. Se confía en la capacidad de los modelos más robustos para integrarlos en la predicción.

### Valores Nulos:  
Hay **259 valores nulos**. Teniendo en cuenta que en el dataset no hay valores menores de **20 pies**, no sería descabellado pensar que los **NaN** se correspondan en realidad a valores de **0** o cercanos a 0. Apoya esta teoría la mayor proporción de terrenos de forma irregular entre las filas que tienen valores de **LotFrontage NaN**. Por lo tanto, se sustituyen los **NaN** por **0**. En la práctica, se le ha aplicado este tratamiento a los valores nulos del dataset, porque su presencia parecía responder a la ausencia de la variable más que a la ausencia de su registro.

<br>

### Neighborhood: 

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/07.png">
  
</div>

Es una variable a priori muy interesante, sin embargo, tiene demasiadas **categorías**. Al ser imposible codificarlas de forma ordinal, un **onehot encoding** aumentaría demasiado la dimensionalidad. La solución es **reagrupar los barrios** en tres categorías según su **precio medio**.

Las nuevas categorías son susceptibles de **codificarse de forma ordinal**, lo que contribuye al objetivo de **reducir la dimensionalidad**.

<br>

### BsmtFinType1 y BsmtFinSF1:

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/08.png">
  
</div>

**BsmtFinType1** describe el tipo de acabado principal en el sótano (**GLQ, ALQ, BLQ, Rec, LwQ, Unf, NA**). **BsmtFinSF1** representa los **pies cuadrados acabados** correspondientes al tipo principal de acabado del sótano descrito por **BsmtFinType1**. Es decir, **BsmtFinType1** indica la **calidad del acabado** en una parte del sótano, y **BsmtFinSF1** cuantifica cuántos pies cuadrados corresponden a ese acabado.

2. **BsmtFinType2 y BsmtFinSF2**:  
**BsmtFinType2** describe el **segundo tipo de acabado** en el sótano (si hay más de un tipo de acabado). **BsmtFinSF2** indica los **pies cuadrados acabados** correspondientes a ese segundo tipo de acabado. Si un sótano tiene áreas con diferentes tipos de acabado, **BsmtFinType2** y **BsmtFinSF2** complementan a **BsmtFinType1** y **BsmtFinSF1**. Si no hay dos tipos de acabado, ambos pueden ser **NA** o **0**.

3. **BsmtUnfSF**:  
Representa los **pies cuadrados no acabados** del sótano. Es decir, esta variable indica qué parte del sótano no tiene ningún tipo de acabado.

4. **TotalBsmtSF**:  
Esta variable representa el **área total** del sótano, y es la **suma de todas las áreas**, tanto las acabadas (**BsmtFinSF1 + BsmtFinSF2**) como las no acabadas (**BsmtUnfSF**).

### Resumen de tratamiento:  
Estas seis variables deberían poder resumirse en una sola. En primer lugar, como hasta ahora se le dará un valor de **0** a los **NA**, luego se asignarán **valores ordinales** a **BsmtFinType1** y **BsmtFinType2**. Posteriormente, se ponderará el valor de las áreas correspondientes, con una fórmula que será el **valor normalizado** de la correspondiente área con el valor de **BsmtFinType** para luego **sumar todas las áreas**.











  


