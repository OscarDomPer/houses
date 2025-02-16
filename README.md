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

---

<br>

## Análisis exploratorio de los datos

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/02.png">
  
</div>

El dataset está compuesto por **80 variables** (incluida la variable objetivo) de naturaleza tanto **categórica como numérica**.  

Se observa un alto porcentaje de datos faltantes en algunas columnas. Dado que el dataset tiene 1460 filas, ciertas variables presentan hasta un **90 % de valores NaN**.  

Sin embargo, no se descartarán estas variables de inmediato, ya que aún podrían ofrecer información útil. Por ejemplo, los valores nulos en la columna **`Fence`** podrían interpretarse como la **ausencia de cercas** y ser sustituidos por un **0**.  

---

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/03.png">
  
</div>

Lo primero que llama la atención es que algunas columnas **numéricas** podrían ser, en realidad, **categóricas**. Tras revisar la descripción de los datos proporcionada, podemos concluir que **MSSubClass** es, de hecho, una columna categórica. También se observa que algunas variables presentan un **sesgo a la derecha** en su distribución.  

Algunas variables categóricas tienen una **única categoría**, lo que las hace completamente inútiles, por lo que deben ser **eliminadas**. Además, hay otras variables en las que una **categoría domina significativamente** sobre las demás, lo que las hace susceptibles de algún tipo de transformación para reducir la dimensionalidad. Por último, un tercer grupo de variables presenta **demasiadas categorías**, que en el futuro muy posiblemente deberán ser reducidas para controlar la dimensionalidad del modelo.  

Parece que muchas variables tienen **valores atípicos**. Sin embargo, dada la naturaleza del dataset, no se puede saber a priori si son valores **legítimamente altos** o si se deben a **errores en la recolección de datos**.  

Este hecho, junto con el uso de **modelos robustos** que manejan bien los outliers para la predicción, descarta cualquier tipo de **eliminación generalizada** de estos valores. En su lugar, se tratará **cada caso de forma individual**. 

---

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/04.png">
  
</div>

Parece que muchas variables tienen **valores atípicos**. Sin embargo, dada la naturaleza del dataset, no se puede saber a priori si son valores **legítimamente altos** o si se deben a **errores en la recolección de datos**.  

Este hecho, junto con el uso de **modelos robustos** que manejan bien los outliers para la predicción, descarta cualquier tipo de **eliminación generalizada** de estos valores. En su lugar, se tratará **cada caso de forma individual**.  

---

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/05.png">
  
</div>

Se observan **correlaciones** tanto con **SalePrice** como entre otras variables. Por ejemplo, las correlaciones de **YearBuilt** con otras variables sugieren distintas preferencias en el diseño de las casas a lo largo de los años.  

Analizando concretamente las **correlaciones de las variables numéricas** con la variable objetivo, vuelven a aparecer **fuertes correlaciones**, mientras que en otras variables no tanto. Sin embargo, estas últimas pueden ser **importantes para el modelo** al establecer **relaciones no lineales**.

---
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

---
<br>

### Neighborhood: 

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/07.png">
  
</div>

Es una variable a priori muy interesante, sin embargo, tiene demasiadas **categorías**. Al ser imposible codificarlas de forma ordinal, un **onehot encoding** aumentaría demasiado la dimensionalidad. La solución es **reagrupar los barrios** en tres categorías según su **precio medio**.

Las nuevas categorías son susceptibles de **codificarse de forma ordinal**, lo que contribuye al objetivo de **reducir la dimensionalidad**.

---
<br>

### BsmtFinType1 y BsmtFinSF1:

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/08.png">
  
</div>

1. **BsmtFinType1 y BsmtFinSF1**: BsmtFinType1 describe el tipo de acabado principal en el sótano (**GLQ, ALQ, BLQ, Rec, LwQ, Unf, NA**). BsmtFinSF1 representa los pies cuadrados acabados correspondientes al tipo principal de acabado del sótano descrito por BsmtFinType1. Es decir, BsmtFinType1 indica la calidad del acabado en una parte del sótano, y BsmtFinSF1 cuantifica cuántos pies cuadrados corresponden a ese acabado.

2. **BsmtFinType2 y BsmtFinSF2**: BsmtFinType2 describe el segundo tipo de acabado en el sótano (si hay más de un tipo de acabado). BsmtFinSF2 indica los pies cuadrados acabados correspondientes a ese segundo tipo de acabado. Si un sótano tiene áreas con diferentes tipos de acabado, BsmtFinType2 y BsmtFinSF2 complementan a BsmtFinType1 y BsmtFinSF1. Si no hay dos tipos de acabado, ambos pueden ser NA o 0.

3. **BsmtUnfSF**: Representa los pies cuadrados no acabados del sótano. Es decir, esta variable indica qué parte del sótano no tiene ningún tipo de acabado.

4. **TotalBsmtSF**: Esta variable representa el área total del sótano, y es la suma de todas las áreas, tanto las acabadas (**BsmtFinSF1 + BsmtFinSF2**) como las no acabadas (**BsmtUnfSF**).

**Estas seis variables deberían poder resumirse en una sola**; en primer lugar, como hasta ahora le daremos un valor de 0 a los NA, luego le asignaremos valores ordinales a BsmtFintype 1 y 2, luego se pondera el valor de las áreas correspondientes, con una fórmula que será el valor normalizado de la correspondiente área con el valor de BsmtFintype para luego sumar todas las áreas.

---

<br>

## Selección del modelo

En primera instancia se prueban los **modelos más comunes**, y también un **Stacking** y una **Red Neuronal**. Se usan dos métricas: **RMSLE**, que es la que se usará como referencia en el reto, y **𝑅²**, que se medirán tanto en **train** como en **test**, con el objetivo de valorar el **sobreajuste**.  

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/10.png">
  
</div>

En los **modelos más comunes** se observan métricas prometedoras en los modelos más **robustos** (**Gradient Boosting** y **Random Forest**). Los modelos que no funcionan bien con relaciones **no lineales** tienen un desempeño muy pobre. Todo ello sugiere que las variables se relacionan entre sí de **forma compleja**.

En la familia de los **ensambladores**, solo **CatBoost** mejora a **Gradient**, pero dado que el **sobreajuste** es mayor y que el dataset es distinto a los demás, se descarta. A pesar de que las métricas de **XGBoost** son peores, se mantendrá este modelo, debido a que suele funcionar bien en este tipo de pruebas.

El **Stacking** no ofrece mejores resultados.

La **Red Neuronal** muestra resultados interesantes, ya que el **sobreajuste** es muy bajo, insinuando que es capaz de **generalizar**.

---

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/12.png">
  
</div>

Los resultados son en general **razonables**. Parece que los dos modelos que **sobreajustaban en exceso** (**GBT** y **XGB**) tienen un rendimiento algo superior a los que **generalizaron mejor** (**Stacking** y **Red Neuronal**).  

De esto se puede concluir que los **datos del conjunto de prueba** son **muy similares** a los del de entrenamiento. Seguramente, la **Red Neuronal** mejoraría con **datos menos homogéneos**.

---

<br>

## Aprendizaje no supervisado.


<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/13.png">
  
</div>

---

<br>

La idea aquí es, usando **aprendizaje no supervisado**, dividir el **conjunto de entrenamiento** en **clusters** para **reentrenar** en cada uno de ellos los modelos que dieron mejores resultados.  

Posteriormente, utilizando **labeling**, se divide el **conjunto de prueba** en los mismos **clusters** y se realizan las predicciones con su correspondiente modelo.  

El objetivo es doble: por un lado, **mejorar las predicciones** encontrando patrones específicos en cada cluster y, por otro, **extraer insights clave** sobre los factores que influyen en el **precio de las viviendas** en diferentes segmentos del mercado.  

---

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/14.png">
  
</div>

Dada la **gran dimensionalidad** del modelo, se valora la **inercia** de cada número de **clusters** en el **conjunto completo** de los datos y luego en **subconjuntos** con las **50, 30 y 10 variables más importantes** para el modelo **XGBoost**. En todos los casos, el número de **K** parece ser **2**, aunque no es una decisión clara.  

Se vuelven a usar los **4 conjuntos** para obtener los **clusters** usando **KMeans=2**. Como era de esperar, al usar **todas las variables**, la **clusterización** no es clara, pero con las **50 mejores** ya se observan **dos clusters diferenciados**.  

---

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/15.png">
  
</div>

Se selecciona **K=2** en las **50 mejores características**, pues con esta combinación obtenemos **dos clusters plenamente diferenciados** y no perdemos demasiadas variables.  

Como se observa en estas gráficas, los **clusters** se agrupan claramente en función del **valor de las viviendas**. Esto se refleja tanto en la variable **SalePrice** (que no fue utilizada para la agrupación) como en otras variables relacionadas con **calidad y superficie**, que presentan valores más altos en el **clúster 1**.  

Por otro lado, la variable **Age**, que indica la **antigüedad de la vivienda**, es, como era de esperar, mayor en el **clúster 0**.  

---

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/16.png">
  
</div>

Los **clusters** tienen el mismo comportamiento respecto a las **variables clave** en el **conjunto de prueba**.  

Los datos de **Gradient Boosting** mejoran a los del modelo entrenado con el **conjunto completo**, lo que sugiere que cada grupo tiene **patrones distintos**, es decir, la relación entre las **características** y el **precio** es diferente entre estos subgrupos. Esta teoría se confirma al observar las **gráficas de importancia de características** de cada modelo.  

En conclusión, se consigue en un solo modelo capturar todas las **interacciones posibles**.  

En el caso de la **red neuronal**, los resultados **no mejoran**, seguramente debido al **tamaño limitado** del conjunto de datos original. Al partir de un dataset de **menos de 1500 muestras** y dividirlo en **dos clusters**, el número de muestras disponible para cada **submodelo de red neuronal** no fue suficiente para un entrenamiento efectivo.  

Las **redes neuronales** requieren un **mayor volumen de datos** para aprender **patrones complejos** de manera generalizable, mientras que los modelos como el **GBT** suelen ser más robustos en situaciones de **datos pequeños o medianos**.  

La menor tendencia a **sobreactuarse** de la red frente a **GBT** y la mayor capacidad de **generalización** de las redes neuronales en general podrían hacer que, con un **número mayor de muestras de entrenamiento**, funcionen mejor en un **caso real**.  

---

  <br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/19.png">
  
</div>

Para cada **cluster** varía el **orden de las importancias**, indicando **patrones distintos** a la hora de predecir los **precios**.  

Las diferencias en el **orden de las importancias** se explican en gran parte en relación con el **valor de la vivienda**, estando el **cluster 0** compuesto por **viviendas de valor medio-bajo** y el **cluster 1** por **viviendas de alto valor**.  

Ciertas características, como **GrLivArea** (área total habitable a nivel del suelo), **BsmtQualityScore** (una medida que combina la superficie y la calidad de los acabados del sótano) y **1stFlrSF** (superficie del primer piso), tienen gran **importancia** en ambos **clústeres**.  

De manera general, factores como una **mayor superficie habitable** o un **buen sótano** incrementan el **valor de la propiedad**.  

---

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/20.png">
  
</div>

Características que en **viviendas de gran valor** se dan por sentadas, pueden representar un **factor diferencial** en **viviendas más modestas**.  

Por ejemplo, una **buena chimenea** (**FirePlaceQual**) puede ser determinante en una **vivienda media**. Otro buen ejemplo es el **aire acondicionado** (**CentralAir**) o una **buena calefacción** (**HeatingQC**).  

---

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/21.png">
  
</div>

Las viviendas más caras, por su parte, presentan una serie de **variables exclusivas asociadas con el lujo**: *PoolArea*, terraza de madera (*WoodDeckSF*) o darle importancia a la calidad de los materiales exteriores (*ExterQual*).  

Un caso curioso es **GarageArea**, que es la tercera más importante en el *cluster* 1. Una posible explicación sería que las personas que pueden permitirse esas casas tienen varios vehículos y necesitan garajes más amplios.

---

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/22.png">
  
</div>

Al observar la posición de **OverallQual** (Calidad general de la construcción) y **OverallCond** (Condición-estado general de la construcción) en cada **clúster**, se observa que sus posiciones están **invertidas**. Y no sólo eso, este patrón también se repite en los otros dos pares de **Calidad-Condición** (**ExterQual**, **ExterCond**, **BsmtQual** y **BsmtCond**).  

La explicación podría ser que en **viviendas de alto valor**, la **calidad de los materiales** es más importante que el **estado de conservación** de los mismos. Por ejemplo, unas **escaleras de mármol de Carrara** incrementarán el valor de la propiedad aunque estén algo gastadas. Mientras que en las **casas más baratas**, hechas con materiales más modestos, lo que prima es el **buen estado de los mismos**, ya que esto garantiza su **funcionalidad**.  

La **funcionalidad**, por cierto, también es una variable (**Functional**) que solo aparece en el **top 30** en **Low Price Houses**.  

---

<br>

<div align="center">

  <img src="https://github.com/OscarDomPer/houses/blob/main/imaxes/23.png">
  
</div>

La variable **Neighborhood**, que se refiere al barrio en el que están ubicadas las propiedades, presenta gran **importancia** en ambos **clústeres**. Saber en qué **barrios** ha ido incrementando el **precio de las viviendas** y en cuáles está bajando podría ser una valiosa información para predecir no solo el **valor actual** de una propiedad, sino también su **evolución en el futuro próximo**.  

De acuerdo con los datos, barrios como **Crawford**, **Mitchell** o **Briardale** serían excelentes opciones para **invertir**, mientras que sería recomendable mantenerse alejado de **Meadow Village** o **Bloomington Heights**.  

