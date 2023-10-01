---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.12
  nbformat: 4
  nbformat_minor: 5
---

::: {#d2336a8d .cell .markdown}
## Tabla de contenidos

1.  [**Problema al que nos enfrentamos**](#1)

2.  [**Comprensión de los datos**](#2)

    -   [Importación de los datos](#21)
    -   [Descripción del conjunto de datos](#22)
    -   [Análisis exploratorio de los datos](#23)
        -   [Limpieza inicial de los datos](#231)
        -   [Edad](#232)
        -   [Sexo](#233)
        -   [Tipo de dolor pectoral](#234)
        -   [Análisis de sangre](#235)
        -   [Resultados del Electrocardiograma (ECG)](#236)
        -   [Mapa de correlaciones](#237)
        -   [Conclusiones finales del EDA](#238)

3.  [**Preparación de los datos**](#3)

    -   [Discretización de los datos](#31)
    -   [Conversión de datos discretos](#32)
    -   [Normalización y escalado de datos](#33)
    -   [PCA](#34)

4.  [**Modelado** (Práctica 2)](#4)

    -   [Modelos no supervisados](#41)
    -   [DBSCAN + OPTICS](#42)
    -   [Árboles de decisión](#43)
    -   [KNN](#44)
    -   [Limitaciones y riesgos](#45)

5.  [**Conclusiones**](#5)

6.  [**Bibliografía**](#7)
:::

::: {#da4d9bdd .cell .markdown}
# **Problema al que nos enfrentamos:** `<a id="1">`{=html}`</a>`{=html} {#problema-al-que-nos-enfrentamos-}

Las enfermedades cardiovasculares (ECV) o cardiovasculopatías, más
comúnmente conocidas como enfermedades del corazón, son la primera causa
de muerte a nivel global, causantes de 17,9 millones de muertes sólo
durante el año 2015, lo que supuso un 32,1% del total de muertes de ese
año, con lo cual, es un tema que puede resultar muy interesante para
cualquiera, dado que todos tenemos riesgo de sufrir una ECV, y
sobretodo, a nivel estadístico, cualquiera de nosotros puede fallecer a
causa de una ECV. Las cariovasculopatías más comunes son la cardiopatía
coronaria, la insuficiencia cardíaca, las arrítmias, la hipertensión,
etc.

Por suerte, conocemos muchos factores de riesgo de éste tipo de
enfermedades, con lo cual, si obtenemos información sobre éstos factores
de riesgo, podemos utilizar la minería de datos para realizar una
predicción acerca de si un paciente tiene una elevada probabilidad de
sufrir una ECV, de modo que se puedan realizar tareas de prevención para
reducir su probabilidad de enfermedad o, en última instancia, de
fallecimiento.

[Fuente
1](https://es.wikipedia.org/wiki/Enfermedades_cardiovasculares#),
[Fuente
2](https://medlineplus.gov/spanish/ency/patientinstructions/000759.htm)
:::

::: {#a47afea5 .cell .markdown}
# **Comprensión de los datos:**`<a id="2">`{=html}`</a>`{=html}

Para poder realizar tareas de predicción, lo primero de todo es poder
conseguir datos de pacientes que nos indiquen distintos marcadores que
sabemos que pueden ser factores de riesgo para desarrollar una ECV. Para
conseguirlo, hemos buscado un dataset que cumpliese con nuestras
expectativas y que, a su vez, nos permitiese realizar un correcto
análisis y modelado de los datos.

El dataset utilizado es [Heart Failure Prediction
Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
que podemos hallar en el repositorio de datasets de ***kaggle***. A
continuación realizaremos una descripción y exploración de los datos en
profundidad.
:::

::: {#f3e242cb .cell .markdown}
## Importación de los datos`<a id="21">`{=html}`</a>`{=html}
:::

::: {#eddf250b .cell .code execution_count="1"}
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
```
:::

::: {#492e7e31 .cell .code execution_count="2"}
``` python
data = pd.read_csv("D:/UNI/UOC/Semestre 4/Mineria de Dades/PRA 1/data/heart.csv")
data
```

::: {.output .execute_result execution_count="2"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>ChestPainType</th>
      <th>RestingBP</th>
      <th>Cholesterol</th>
      <th>FastingBS</th>
      <th>RestingECG</th>
      <th>MaxHR</th>
      <th>ExerciseAngina</th>
      <th>Oldpeak</th>
      <th>ST_Slope</th>
      <th>HeartDisease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40</td>
      <td>M</td>
      <td>ATA</td>
      <td>140</td>
      <td>289</td>
      <td>0</td>
      <td>Normal</td>
      <td>172</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>F</td>
      <td>NAP</td>
      <td>160</td>
      <td>180</td>
      <td>0</td>
      <td>Normal</td>
      <td>156</td>
      <td>N</td>
      <td>1.0</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>M</td>
      <td>ATA</td>
      <td>130</td>
      <td>283</td>
      <td>0</td>
      <td>ST</td>
      <td>98</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>F</td>
      <td>ASY</td>
      <td>138</td>
      <td>214</td>
      <td>0</td>
      <td>Normal</td>
      <td>108</td>
      <td>Y</td>
      <td>1.5</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>M</td>
      <td>NAP</td>
      <td>150</td>
      <td>195</td>
      <td>0</td>
      <td>Normal</td>
      <td>122</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>913</th>
      <td>45</td>
      <td>M</td>
      <td>TA</td>
      <td>110</td>
      <td>264</td>
      <td>0</td>
      <td>Normal</td>
      <td>132</td>
      <td>N</td>
      <td>1.2</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>914</th>
      <td>68</td>
      <td>M</td>
      <td>ASY</td>
      <td>144</td>
      <td>193</td>
      <td>1</td>
      <td>Normal</td>
      <td>141</td>
      <td>N</td>
      <td>3.4</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>915</th>
      <td>57</td>
      <td>M</td>
      <td>ASY</td>
      <td>130</td>
      <td>131</td>
      <td>0</td>
      <td>Normal</td>
      <td>115</td>
      <td>Y</td>
      <td>1.2</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>916</th>
      <td>57</td>
      <td>F</td>
      <td>ATA</td>
      <td>130</td>
      <td>236</td>
      <td>0</td>
      <td>LVH</td>
      <td>174</td>
      <td>N</td>
      <td>0.0</td>
      <td>Flat</td>
      <td>1</td>
    </tr>
    <tr>
      <th>917</th>
      <td>38</td>
      <td>M</td>
      <td>NAP</td>
      <td>138</td>
      <td>175</td>
      <td>0</td>
      <td>Normal</td>
      <td>173</td>
      <td>N</td>
      <td>0.0</td>
      <td>Up</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>918 rows × 12 columns</p>
</div>
```
:::
:::

::: {#fb63f870 .cell .markdown}
***Figura 1***`<a id='Figura1'>`{=html}`</a>`{=html}
:::

::: {#d402d425 .cell .markdown}
## Descripción del conjunto de datos`<a id="22">`{=html}`</a>`{=html}
:::

::: {#25a4a56a .cell .markdown}
**1. Age:** Edad del paciente (años)

**2. Sex:** Género del paciente (M: Hombre, F: Mujer)

**3. ChestPainType:** Tipo de dolor de pecho (TA: Dolor típico de
angina, ATA: Dolor atípico de angina, NAP: Sin dolor de angina, ASY:
Asintomático)

**4. RestingBP:** Presión sangínea en reposo (en milímetros de mercurio
\[mm Hg\])

**5. Cholesterol:** Nivel de colesterol en sangre (en milímetros por
decilitro \[mm/dl\])

**6. FastingBS:** Nivel de azucar en sangre en ayunas (1: Nivel de
azucar superior a 120 mg/dl, 0: Nivel de azucar inferior a 120 mg/dl)

**7. RestingECG:** Resultados de electrocardiograma en reposo (Normal:
Normal, ST: Onda ST-T anormal \[inversiones en la onda T y/o elevación o
depresión del segmento ST superiores a 0.05 mV\], LVH: Hipertrofia
ventricular ya sea de forma clara o probavle \[siguiendo el *criterio de
Estes*\]

**8. MaxHR:** Máximo pulso cardíaco conseguido (valor numérico entre 60
y 202)

**9. ExerciseAngina:** Angina inducida mediante ejercicio (Y: Sí, N: No)

**10. Oldpeak:** Segmento T (Valor numérico del segmento T)

**11. ST_Slope:** Elevación del segmento T en el pico del ejercicio (UP:
Elevación positiva, Flat: Elevación neutra, Down: Elevación negativa).

**12. HeartDisease:** Indica si el paciente tiene alguna enfermedad del
corazón (1: Tiene enfermedad del corazón, 2: No tiene enfermedad del
corazón)
:::

::: {#bc8bcb43 .cell .markdown}
***OTROS FACTORES A TENER EN CUENTA***

Los datos estudiados son de pacientes que fueron al médico por problemas
que podían ser causados por una enfermedad del corazón, de modo que no
es una población representativa para todo el mundo, sino que debe
tenerse en cuenta el factor de que los pacientes estudiados ya tenían
una alta probabilidad de tener enfermedades del corazón.
:::

::: {#28a27911 .cell .code execution_count="3" scrolled="true"}
``` python
data.describe()
```

::: {.output .execute_result execution_count="3"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>RestingBP</th>
      <th>Cholesterol</th>
      <th>FastingBS</th>
      <th>MaxHR</th>
      <th>Oldpeak</th>
      <th>HeartDisease</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
      <td>918.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>53.510893</td>
      <td>132.396514</td>
      <td>198.799564</td>
      <td>0.233115</td>
      <td>136.809368</td>
      <td>0.887364</td>
      <td>0.553377</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.432617</td>
      <td>18.514154</td>
      <td>109.384145</td>
      <td>0.423046</td>
      <td>25.460334</td>
      <td>1.066570</td>
      <td>0.497414</td>
    </tr>
    <tr>
      <th>min</th>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>60.000000</td>
      <td>-2.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>47.000000</td>
      <td>120.000000</td>
      <td>173.250000</td>
      <td>0.000000</td>
      <td>120.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>54.000000</td>
      <td>130.000000</td>
      <td>223.000000</td>
      <td>0.000000</td>
      <td>138.000000</td>
      <td>0.600000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>60.000000</td>
      <td>140.000000</td>
      <td>267.000000</td>
      <td>0.000000</td>
      <td>156.000000</td>
      <td>1.500000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>77.000000</td>
      <td>200.000000</td>
      <td>603.000000</td>
      <td>1.000000</td>
      <td>202.000000</td>
      <td>6.200000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#1a383db1 .cell .code execution_count="4" scrolled="false"}
``` python
sns.histplot(data["HeartDisease"])
plt.title("Distribución de las observaciones separadas\n"+
          "entre pacientes sin ECV y pacientes con ECV")
plt.ylabel("Recuento")
plt.xlabel("0 = No tiene ECV | 1 = Tiene ECV")
```

::: {.output .execute_result execution_count="4"}
    Text(0.5, 0, '0 = No tiene ECV | 1 = Tiene ECV')
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/bbd8a5f4469718d1225c60af097ae9fbd6a47f1d.png)
:::
:::

::: {#43ef82ed .cell .markdown}
## Análisis exploratorio de los datos`<a id="23">`{=html}`</a>`{=html}
:::

::: {#6943f0a5 .cell .markdown}
### Limpieza inicial de los datos`<a id="231">`{=html}`</a>`{=html}
:::

::: {#b76e5415 .cell .markdown}
Buscamos valores nulos:
:::

::: {#4d24264a .cell .code execution_count="5"}
``` python
data.isnull().sum()
```

::: {.output .execute_result execution_count="5"}
    Age               0
    Sex               0
    ChestPainType     0
    RestingBP         0
    Cholesterol       0
    FastingBS         0
    RestingECG        0
    MaxHR             0
    ExerciseAngina    0
    Oldpeak           0
    ST_Slope          0
    HeartDisease      0
    dtype: int64
:::
:::

::: {#8cd639c1 .cell .markdown}
En este dataset no tenemos valores nulos.

Aún así, vamos a buscar valores anormales. Para hacerlo, mostraremos qué
columnas tienen valores igual a 0, y analizaremos si tiene sentido éste
valor.
:::

::: {#940ce6d1 .cell .code execution_count="6"}
``` python
for n in data.columns:
    print(n,"\n" ,data.loc[data[n]==0][n],"\n\n\n")
```

::: {.output .stream .stdout}
    Age 
     Series([], Name: Age, dtype: int64) 



    Sex 
     Series([], Name: Sex, dtype: object) 



    ChestPainType 
     Series([], Name: ChestPainType, dtype: object) 



    RestingBP 
     449    0
    Name: RestingBP, dtype: int64 



    Cholesterol 
     293    0
    294    0
    295    0
    296    0
    297    0
          ..
    514    0
    515    0
    518    0
    535    0
    536    0
    Name: Cholesterol, Length: 172, dtype: int64 



    FastingBS 
     0      0
    1      0
    2      0
    3      0
    4      0
          ..
    912    0
    913    0
    915    0
    916    0
    917    0
    Name: FastingBS, Length: 704, dtype: int64 



    RestingECG 
     Series([], Name: RestingECG, dtype: object) 



    MaxHR 
     Series([], Name: MaxHR, dtype: int64) 



    ExerciseAngina 
     Series([], Name: ExerciseAngina, dtype: object) 



    Oldpeak 
     0      0.0
    2      0.0
    4      0.0
    5      0.0
    6      0.0
          ... 
    904    0.0
    909    0.0
    910    0.0
    916    0.0
    917    0.0
    Name: Oldpeak, Length: 368, dtype: float64 



    ST_Slope 
     Series([], Name: ST_Slope, dtype: object) 



    HeartDisease 
     0      0
    2      0
    4      0
    5      0
    6      0
          ..
    903    0
    904    0
    906    0
    910    0
    917    0
    Name: HeartDisease, Length: 410, dtype: int64 
:::
:::

::: {#80384e2e .cell .markdown}
Podemos ver que hay varias columnas con valores igual a 0.

Para las columnas FastingBS y HeartDisease, es normal tener valores
igual a 0, dado que son variables booleanas.

En la columna Oldpeak también tenemos valores igual a 0, pero ello es
debido a que la distrubución va desde rangos negativos hasta rangos
postitivos, con lo que es normal tener valores igual a 0.

Finalmente, tenemos la columna del colesterol, en la cual el valor 0 no
es posible, dado que no es posible tener un colesterol igual a 0, y el
hecho de que aparezcan estos datos puede ser debido a que haya un fallo
en la recolección de los datos. También tenemos una fila con un 0 en la
columna RestingBP, lo cual to tiene sentido, dado que las pulsaciones no
pueden ser igual a 0 para una persona viva.

Así pues tenemos que decidir qué hacemos con estas filas. Para la
columna Cholesterol, dado que hay 172 filas con colesterol igual a 0,
éstas representan un 19% del total de filas, con lo cual es un valor lo
suficientemente bajo como para que no tengamos que eliminar la columna.
Tal y cómo hemos visto en la [Figura 1](#Figura1), los datos no están
ordenados, de modo que para realizar una sustitución de forma aleatoria,
vamos a sustituir cada 0 por el anterior valor distinto a 0 en la
columna. En el caso de la columna RestingBP, como sólo tenemos una fila
con valor igual a 0, simplemente eliminaremos esa observación.
:::

::: {#981dcabc .cell .code execution_count="7"}
``` python
data["Cholesterol"].replace(to_replace=0, method="ffill", inplace=True)
print(data.loc[data["Cholesterol"]==0]["Cholesterol"])

data.drop(data[data["RestingBP"]==0].index, inplace=True)
print(data.loc[data["RestingBP"]==0]["RestingBP"])
```

::: {.output .stream .stdout}
    Series([], Name: Cholesterol, dtype: int64)
    Series([], Name: RestingBP, dtype: int64)
:::
:::

::: {#7b2ceb08 .cell .markdown}
## Análisis del impacto de la edad`<a id="232">`{=html}`</a>`{=html}
:::

::: {#6411bd50 .cell .code execution_count="8"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(data=data, x="Age", ax=ax[0])
ax[0].set_title("Distribución de la edad de los pacientes")
ax[0].set_xlabel("Edad")
ax[0].set_ylabel("Recuento")
sns.histplot(data=data, x="Age", hue="HeartDisease", multiple="stack", ax=ax[1], palette="tab20_r")
ax[1].set_title("Distribución de la edad de los pacientes \nseparados entre enfermos y no enfermos")
ax[1].set_xlabel("Edad")
ax[1].set_ylabel("Recuento")
ax[1].legend(title='ECV', labels=['Sí', 'No'])
```

::: {.output .execute_result execution_count="8"}
    <matplotlib.legend.Legend at 0x2086c419ca0>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/133540946631074d2a8af9a2aba8c1995918b685.png)
:::
:::

::: {#05702d9b .cell .markdown}
En estos gráficos mostramos, en el primero, la distribución de la edad
de la población sobre la que se ha hecho el estudio, y en el segundo, la
misma distribución, pero separando entre personas con alguna ECV y
personas sin ninguna ECV.

Como podemos ver, por debajo de los 30 años, aunque hay pocos datos, se
ve claramente como no hay ningún caso de ECV. A medida que aumentamos el
rango de edad, vemos que entre los 30 y los 45, aunque ya se observan
más casos, el total de observaciones de pacientes con ECV se encuentra
por debajo del 50%. Si nos fijamos en el rango de 45 a 55, el recuento
está sobre el 50%, y a partir de los 55 años, vemos cómo el porcentaje
de pacientes con ECV aumenta considerablemente, sobretodo a partir de
los 70 años, donde ya más de tres cuartas partes de los pacientes
observados tienen alguna ECV.
:::

::: {#39a1a368 .cell .code execution_count="9" scrolled="false"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="Sex", y="Age", ax=ax[0], palette="tab10")
ax[0].set_title("Distribución de la edad de los pacientes\nseparados por sexos")
ax[0].set_xlabel("Sexo")
ax[0].set_ylabel("Edad")
sns.violinplot(data=data, x="Sex", y="Age", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Distribución de la edad de los pacientes\nseparados por sexo y entre enfermos y no enfermos")
ax[1].set_xlabel("Sexo")
ax[1].set_ylabel("Edad")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', loc="lower right")
```

::: {.output .execute_result execution_count="9"}
    <matplotlib.legend.Legend at 0x2086c28ed90>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/cb7122071a7f7467e05199944867bed470476654.png)
:::
:::

::: {#efde475e .cell .markdown}
Vemos que la distribución de los pacientes según su sexo es bastante
similar, con la media de edad de las mujeres siendo un par de años
inferior a la de los hombres. Sin embargo, cuando discriminamos los
datos según si tienen enfermedades del corazón o no, observamos que ésto
se invierte, siendo la media de edad de las mujeres con enfermedades del
corazón ligeramente superior a la de los hombres, lo que puede
significar que los hombres tienen más riesgo de desarrollar enfermedades
del corazón siendo más jovenes que las mujeres.
:::

::: {#19c825b0 .cell .code execution_count="10"}
``` python
sns.scatterplot(data=data, x="RestingBP", y="Age")
plt.title("Relación entre edad y pulso cardíaco en reposo")
```

::: {.output .execute_result execution_count="10"}
    Text(0.5, 1.0, 'Relación entre edad y pulso cardíaco en reposo')
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/940fc96a22c5fb871d0e3bd01efc1198eb0ee79e.png)
:::
:::

::: {#8b67f645 .cell .code execution_count="11"}
``` python
sns.lmplot(data=data, x="RestingBP", y="Age", hue="HeartDisease", palette="tab20_r", scatter_kws={"s": 20}, legend=False)
plt.title("Relación entre edad y pulso cardíaco en reposo\nseparado entre pacientes con y sin ECV")
plt.legend(title='No ECV: 0\n\nECV: 1\n', loc="lower right")
plt.xlabel("Pulsaciones en reposo")
plt.ylabel("Edad")
```

::: {.output .execute_result execution_count="11"}
    Text(24.049999999999997, 0.5, 'Edad')
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/8db7749a14f39d8945aaa41eac9025ad9278edb7.png)
:::
:::

::: {#bfbca29f .cell .markdown}
Estos gráficos nos indican que, a mayor edad, mayor es el pulso cardíaco
en reposo, y a mayor ritmo cardíaco en reposo, mayor es la probabilidad
de desarrollar una ECV. Aún así, es posible que ésto no tenga ningún
impacto signigicativoen el desarrollo de ECV, dado que las líneas de
tendencia son paralelas, de modo que lo que impacta es la edad, así que,
para el estudio de posibles ECV, es posible que estas dos variables
esten autocorrelacionadas.
:::

::: {#5571f3bb .cell .code execution_count="12"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.scatterplot(data=data, x="Cholesterol", y="Age", ax=ax[0], palette="tab10")
ax[0].set_title("Relación entre edad y colesterol")
ax[0].set_ylabel("Edad")
ax[0].set_xlabel("Nivel de colesterol")

sns.scatterplot(data=data, x="Cholesterol", y="Age", hue="HeartDisease", ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre edad y colesterol\nseparado entre pacientes con y sin ECV")
ax[1].set_ylabel("Edad")
ax[1].set_xlabel("Nivel de colesterol")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')
```

::: {.output .execute_result execution_count="12"}
    <matplotlib.legend.Legend at 0x2086c7fbdf0>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/62882d2380874152ef72d1545d10103a694debd1.png)
:::
:::

::: {#00e32e5d .cell .code execution_count="13"}
``` python
sns.lmplot(data=data, x="Cholesterol", y="Age", hue="HeartDisease", palette="tab20_r", scatter_kws={"s": 15}, legend=False)
plt.title("Relación entre edad y colesterol\nseparado entre pacientes con y sin ECV")
plt.ylabel("Edad")
plt.xlabel("Nivel de colesterol")
plt.legend(title='No ECV: 0\n\nECV: 1\n')
```

::: {.output .execute_result execution_count="13"}
    <matplotlib.legend.Legend at 0x2086c874280>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/d27af59d80c7cd9702279a9dc62183adcdd6968f.png)
:::
:::

::: {#bbe7df96 .cell .markdown}
A medida que aumenta el colesterol, se disminuye la edad de los
pacientes con enfermedades del corazón, y se puede ver claramente que
aquellos pacientes con un nivel de colesterol extremadamente alto
(superior a 400), tienen muchas probabilidades de sufrir enfermedades
del corazón, independientemente de la edad que tengan.
:::

::: {#11dab18e .cell .code execution_count="14"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="FastingBS", y="Age", ax=ax[0], palette="tab10")
ax[0].set_title("Relación entre edad y nivel de azúcar")
sns.violinplot(data=data, x="FastingBS", y="Age", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre edad y nivel de azúcar\nseparado entre pacientes con y sin ECV")

for n in [0, 1]:
    ax[n].set_xlabel("Nivel de azúcar (0: valores normales | 1: valores altos)")
    ax[n].set_ylabel("Edad")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', loc="lower right")
```

::: {.output .execute_result execution_count="14"}
    <matplotlib.legend.Legend at 0x2086da10100>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/7550024a110f79ced27bd745248b3df5e2e8dc8f.png)
:::
:::

::: {#e5a7486e .cell .markdown}
A mayor edad, mayor probabilidad de tener los niveles de azúcar en
sangre elevados. En cuanto al impacto del nivel de azucar en sangre, en
relación a la edad y la probabilidad de tener una enfermedad del
corazon, vemos que no tiene una afectación observable.
:::

::: {#a6b48bae .cell .code execution_count="15"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="RestingECG", y="Age", ax=ax[0], palette="tab10")
ax[0].set_title("Relación entre edad y anomalías en el ECG")

sns.violinplot(data=data, x="RestingECG", y="Age", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre edad y anomalías en el ECG\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Anomalías en el ECG")
    ax[n].set_ylabel("Edad")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/8999baf206817969d9cf39d1f9d8a5f14bce07f0.png)
:::
:::

::: {#dd56e9f8 .cell .markdown}
Si el paciente tiene una anomalía ST o LVH, tiene altas probabilidades
de desarrollar una enfermedad en el corazón, sobretodo a partir de los
50 años. En caso de tener un ECG normal, las probabilidades de
desarrollar una ECV no sólo se reducen, sino que también se distribuyen
de forma más uniforme a lo largo de todo el rango de edad.
:::

::: {#2ba9f793 .cell .code execution_count="16"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.scatterplot(data=data, x="MaxHR", y="Age", ax=ax[0], palette="tab10")
ax[0].set_title("Relación entre edad y pulso máximo")

sns.scatterplot(data=data, x="MaxHR", y="Age", hue="HeartDisease", ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre edad y pulso máximo\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Pulso máximo")
    ax[n].set_ylabel("Edad")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/140c9bdf9dcb40f7f424d7d477239472bff9966e.png)
:::
:::

::: {#c9b303de .cell .code execution_count="17"}
``` python

sns.lmplot(data=data, x="MaxHR", y="Age", hue="HeartDisease", palette="tab20_r", scatter_kws={"s": 15}, legend=False)
plt.title("Relación entre edad y pulso máximo\nseparado entre pacientes con y sin ECV")
plt.legend(title='No ECV: 0\n\nECV: 1\n')
plt.xlabel("Pulsaciones máximas")
plt.ylabel("Edad")
```

::: {.output .execute_result execution_count="17"}
    Text(24.049999999999997, 0.5, 'Edad')
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/f972d5491d4bf0c8ca04a1eb0ce36882f1f7406e.png)
:::
:::

::: {#09d0d2f0 .cell .markdown}
A mayor edad, menor es el pulso máximo del corazón, y un pulso máximo
del corazón bajo hace que la probabilidad de tener una enfermedad del
corazón aumente, con lo que el pulso máximo puede ser una causa de
enfermedad del corazón.
:::

::: {#8ff20cce .cell .code execution_count="18"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="ExerciseAngina", y="Age", ax=ax[0], palette="tab10")
ax[0].set_title("Relación entre edad y angina inducida por ejercicio")

sns.violinplot(data=data, x="ExerciseAngina", y="Age", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre edad y angina inducida por ejercicio\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Angina inducida por ejercicio\n(N: No | Y: Sí)")
    ax[n].set_ylabel("Edad")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/9fa34864355a525fb77a199a2deccfb25ec5dee2.png)
:::
:::

::: {#3f8a8072 .cell .markdown}
En ésta comparativa podemos ver que, a mayor edad, más probable es que
se induzca una angina mediante el ejercicio físico. Aún así, el hecho de
que aparezca una angina mediante el ejericio, no parece tener impacto en
la probabilidad de desarrollar una ECV.
:::

::: {#0db68b58 .cell .code execution_count="19"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.scatterplot(data=data, x="Oldpeak", y="Age", ax=ax[0], palette="tab10")
ax[0].set_title("Relación entre edad y segmento T del ECG")

sns.scatterplot(data=data, x="Oldpeak", y="Age", hue="HeartDisease", ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre edad y segmento T del ECG\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Segmento T del ECG")
    ax[n].set_ylabel("Edad")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/99f02deaaae3057e3f686dd34442817093f646f5.png)
:::
:::

::: {#67a5d50b .cell .markdown}
Cuando comparamos el segmento T del ECG con la edad, observamos que a
partir de los 50 años, aumentan considerablemente las observaciones en
las que se ve un segmento T con valores superiores a 0, mientras que en
gente más joven, la distribución tiende más a estar en 0. Al analizar el
impacto de este segmento T, observamos que, a medida que se aleja del 0
(ya sea en positivo o en negativo), se incrementa muy notablemente la
probabilidad de sufrir una ECV.-
:::

::: {#a141512b .cell .markdown}
### ***CONCLUSIONES DIMENSIÓN EDAD***

Hasta ahora hemos visto el impacto de la edad en distintos factores de
riesgo, y hemos podido observar claramente que con la edad, aumentan
muchos de los indicadores de riesgo, de modo que, a mayor edad, mayor
riesgo de tener alguna ECV.
:::

::: {#71a4728b .cell .markdown}
## Análisis del impacto del sexo`<a id="233">`{=html}`</a>`{=html}
:::

::: {#44522a8e .cell .code execution_count="20"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12,6))
sns.histplot(data=data, x="Sex", ax=ax[0])
ax[0].set_title("Distribución del género de los pacientes")

sns.histplot(data=data, x="Sex", hue="HeartDisease", multiple="stack", ax=ax[1], palette="tab20_r")
ax[1].set_title("Distribución del género de los paciente\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Género de los pacientes")
    ax[n].set_ylabel("Recuento")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/fa0084fe538806cd3b516c98741ce545d61f1a29.png)
:::
:::

::: {#d5263079 .cell .markdown}
En esta distribución, podemos observar que la mayoría de la población
estudiada son hombres. Además, cuando discretizamos entre personas con
ECV o sin ellas, podemos ver que en el caso de los hombres la proporción
de pacientes con alguna ECV es muy elevada, mientras que en el caso de
las mujeres es notablemente baja.
:::

::: {#b0636bf0 .cell .code execution_count="21" scrolled="false"}
``` python
g = sns.catplot(data=data, x="ChestPainType", kind="count", hue="Sex", col="HeartDisease", legend=False)
axes = g.axes.flatten()

g.set_ylabels("Recuento")
g.set_xlabels("Tipo de dolor de pecho\n\n"+
             "TA: Dolor típico de angina | ATA: Dolor atípico de angina\n" + 
              "NAP: Sin dolor de angina | ASY: Asintomático")
g.fig.subplots_adjust(top=0.85)
plt.suptitle("Distribución de los tipos de dolor de pecho separados por género y entre personas con y sin ECV\n\n")
axes[0].set_title("Sin ECV")
axes[1].set_title("Con ECV")
g.add_legend(title="Sexo", **{"labels":["Hombre", "Mujer"]})
```

::: {.output .execute_result execution_count="21"}
    <seaborn.axisgrid.FacetGrid at 0x2086c538f10>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/6f15f2c2355cda605efd2d683bbeab0b9c00bbcb.png)
:::
:::

::: {#ef358f20 .cell .markdown}
Al observar los tipos de dolor de pecho, vemos que la distribución es
bastante similar entre hombres y mujeres cuando no tienen ninguna
enfermedad cardiovascular, en cambio, para los pacientes con ECV, vemos
que, aunque para ambos géneros la mayor parte no tiene ningun síntoma de
éste tipo, en el caso de los hombres vemos que hay más casos con dolor
de pecho que en las mujeres, en las que prácticamente no se observa
ningún tipo de dolor de pecho.
:::

::: {#0fb5a19e .cell .code execution_count="22" scrolled="false"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="Sex", y="RestingBP", ax=ax[0], palette="tab10")
ax[0].set_title("Relación entre el sexo y las pulsaciones en reposo")

sns.violinplot(data=data, x="Sex", y="RestingBP", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre el sexo y las pulsaciones en reposo\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Sexo")
    ax[n].set_ylabel("Pulsaciones en reposo")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/6e99a68ae6b03a8ffa4a0afe13b7547a16f41bfc.png)
:::
:::

::: {#9acdc438 .cell .markdown}
Mientras que la distribución es bastante similar entre hombres y
mujeres, las mujeres necesitan tener un pulso en reposo mayor que los
hombres para tener las mismas probabilidades para desarrollar una
enfermedad del corazón, mientras que en el caso de los hombres, tienen
una elevada probabilidad de tener una enfermedad del corazón aún
teniendo un pulso en reposo en unos valores normales.
:::

::: {#14a0d8d3 .cell .code execution_count="23" scrolled="false"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="Sex", y="Cholesterol", ax=ax[0], palette="tab10")
ax[0].set_title("Relación entre el sexo y el nivel de colesterol")

sns.violinplot(data=data, x="Sex", y="Cholesterol", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre el sexo y el nivel de colesterol\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Sexo")
    ax[n].set_ylabel("Colesterol")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/5e50ef2842f31637027ad8c0d58b35c87aa14e78.png)
:::
:::

::: {#a6c66e73 .cell .markdown}
Lo primero que observamos al ver la distribución del colesterol separado
por sexos es que, mientras en los hombres se concentra principalmente
alrededor de 200, en las mujeres se ven unos valores más distribuidos y
ligeramente superiores, encontrándose la mayoría entre 200 y 300. Al
analizar los casos de ECV, vemos que, mientras que en los hombre, la
mayotía de casos se encuentra con unos valores cerca de 200, y los casos
sin ECV estan por encima de 200, en las mujeres, los valores se
distribuyen de forma prácticamente uniforme desde 200 hasta 300, de
forma que parece ser que, mientras que en los hombres, el colesterol no
tiene mucho impacto en las ECV, en las mujeres sí podemos ver como a
mayor colesterol, mayor es la probabilidad de sufrir una ECV.
:::

::: {#d6b14a34 .cell .code execution_count="24"}
``` python
g = sns.catplot(data=data, x="FastingBS", kind="count", hue="Sex", col="HeartDisease", legend=False)
axes = g.axes.flatten()
g.set_ylabels("Recuento")
g.set_xlabels("Nivel de azúcar en sangre\n\n0: Normal | 1: Alto")
g.fig.subplots_adjust(top=0.85)
plt.suptitle("Distribución del nivel de azúcar en sangre separado por género y entre personas con y sin ECV\n\n")
axes[0].set_title("Sin ECV")
axes[1].set_title("Con ECV")
g.add_legend(title="Sexo", **{"labels":["Hombre", "Mujer"]})
```

::: {.output .execute_result execution_count="24"}
    <seaborn.axisgrid.FacetGrid at 0x2086df7a340>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/2543132c66e89186f50e198f65e3b1ef0b5e2acb.png)
:::
:::

::: {#81948019 .cell .markdown}
En el caso del nivel de azúcar en sangre, no vemos un gran impacto, y lo
único que puede llamarnos la atención es que en el caso de los hombres,
cuando el nivel de azucar es alto, aumentan ligeramente las
probabilidades de tener una enfermedad del corazón, pero tampoco es algo
que se vea de forma clara.
:::

::: {#5137f659 .cell .code execution_count="25"}
``` python
sns.histplot(data=data, x="RestingECG", hue="Sex", multiple="stack")
```

::: {.output .execute_result execution_count="25"}
    <AxesSubplot:xlabel='RestingECG', ylabel='Count'>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/f5fb184a780819c0e2169817400fcd0580af2307.png)
:::
:::

::: {#fe2b5aa6 .cell .code execution_count="26"}
``` python
g = sns.catplot(data=data, x="RestingECG", kind="count", hue="Sex", col="HeartDisease", legend=False)
axes = g.axes.flatten()
g.set_ylabels("Recuento")
g.set_xlabels("Resultado del ECG\n\n"+
             "Normal: Normal\nST: Onda ST-T anormal\nLVH: Hipertrofia ventricular")
g.fig.subplots_adjust(top=0.85)
plt.suptitle("Distribución de los tipos resultado del ECG separados por género y entre personas con y sin ECV\n\n")
axes[0].set_title("Sin ECV")
axes[1].set_title("Con ECV")
g.add_legend(title="Sexo", **{"labels":["Hombre", "Mujer"]})
```

::: {.output .execute_result execution_count="26"}
    <seaborn.axisgrid.FacetGrid at 0x2086e2010a0>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/ed70c336cd3efce89f19a134624cddb50d76de53.png)
:::
:::

::: {#6ec00279 .cell .markdown}
Para los resultados del electrocardiograma en reposo, no se pueden
observar grandes diferencias entre sexos más allá de que las mujeres
tienen una ligera tendencia superior a los hombres a mostrar una
hipertrofia ventricular, pero sin demasiado impacto. Lo que sí que es
curioso, es que en el caso de los hombres con alguna ECV, la proporción
de observaciones con un ECG normal es muy elevada, con lo cual puede ser
más dificil para éste sexo el hacer un diagnóstico claro con sólo un
EGC, y aunque esta prueba puede aportar una información realmente
valiosa, hay más variables a tener en cuenta.
:::

::: {#d15f604e .cell .code execution_count="27"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="Sex", y="MaxHR", ax=ax[0], palette="tab10")
ax[0].set_title("Relación entre el sexo y las pulsaciones máximas")

sns.violinplot(data=data, x="Sex", y="MaxHR", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre el sexo y las pulsaciones máximas\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Sexo")
    ax[n].set_ylabel("Pulsaciones máximas")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/8bcb9486727777ce0a037e7cd15c1fa222dab668.png)
:::
:::

::: {#c318860f .cell .markdown}
Ya habíamos visto que cuanto menor fuese el pulso máximo, más
probabilidades había de tener una enfermedad del corazón.

Lo primero que podemos observar aquí es que, en general, los hombres
tienen el pulso máximo por debajo de el de las mujeres. Aún así, podemos
observar que las mujeres no necesitan tener un pulso máximo tan bajo
como los hombres para tener las mismas probabilidades de desarrollar una
enfermedad del corazón, dado que con las mujeres, el pico de la
distribución de las pacientes con enfermedades del corazón está en
aproximadamente 150 pulsos por minuto, mientras que en los hombres está
en 125 pulsos por minuto.
:::

::: {#5e6b979f .cell .code execution_count="28"}
``` python
g = sns.catplot(data=data, x="ExerciseAngina", kind="count", hue="Sex", col="HeartDisease", legend=False)
axes = g.axes.flatten()
g.set_ylabels("Recuento")
g.set_xlabels("Aparición de angina inducida\n\n"+
             "N: No | Y: Sí")
g.fig.subplots_adjust(top=0.85)
plt.suptitle("Distribución de los paceintes según si han mostrado una angina inducida separados por"+
             "género y entre personas con y sin ECV\n\n")
axes[0].set_title("Sin ECV")
axes[1].set_title("Con ECV")
g.add_legend(title="Sexo", **{"labels":["Hombre", "Mujer"]})
```

::: {.output .execute_result execution_count="28"}
    <seaborn.axisgrid.FacetGrid at 0x2086e01aa30>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/35550869ae39601a84855ef8e6f4c43bae728734.png)
:::
:::

::: {#8679cf1a .cell .markdown}
A diferencia de lo que hemos visto en el caso anterior, aquí podemos ver
claramente como el hecho de que aparezca una angina inducida, es un
claro indicador de que puede haber una ECV, aunque el hecho de que no
aparezca tampoco sirve para descartar una enfermedad de tal tipo. En
cuanto a la separación entre géneros, éste indicador es mucho más claro
para los hombres, mientras que para las mujeres, el hecho de que no
aparezca sí que tiene más peso en deducir que las probabilidades de que
no exista una ECV son elevadas.
:::

::: {#a16a04e6 .cell .code execution_count="29"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="Sex", y="Oldpeak", ax=ax[0], palette="tab10")
ax[0].set_title("Relación entre el sexo y el valor del Segmento T")

sns.violinplot(data=data, x="Sex", y="Oldpeak", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre el sexo y el valor del Segmento T\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Sexo")
    ax[n].set_ylabel("Segmento T")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/496110f06ed26f2616f7e2e1896914989cb34deb.png)
:::
:::

::: {#83400060 .cell .markdown}
0 o ligeramente positivo: sin enfermedad. el resto: con enfermedad.
Hombres tienden a Oldpeak superior a las mujeres. Al comparar el
segmento T entre hombres y mujeres, vemos que las mujeres tienen una
elevada concentración de casos en los que el segmento T es 0, y los
casos que se salen de este valor, no se distancian mucho. En cambio, en
los hombres, vemos como los valores se distribuyen más a lo largo del
rango entre 0 y 3.

Este indicador podemos ver que tiene un alto impacto en las ECV, dado
que mientras que los casos con el segmento T tienden a ser negativos en
ECV, los casos con ECV tienen unos valores del segmento T mucho más
distribuidos, de forma que, con esta variable en concreto, los hombres
presentan un riesgo superior a sufrir una ECV por el hechode tener mayor
tendencia a distanciarse del 0.
:::

::: {#1e58fe03 .cell .code execution_count="30" scrolled="false"}
``` python
g = sns.catplot(data=data, x="ST_Slope", kind="count", hue="Sex", col="HeartDisease")
axes = g.axes.flatten()
g.set_ylabels("Recuento")
g.set_xlabels("Elevación del segmento T\n\n"+
             "Up: Hacia arriba | Flat: Plano | Down: Hacia abajo")

plt.suptitle("Distribución entre los pacientes según la elevación del segmento T en el pico del ejercicio\n"+
                "separado por el género y entre personas con y sin ECV\n\n")
g.fig.subplots_adjust(top=0.8)
axes[0].set_title("Sin ECV")
axes[1].set_title("Con ECV")
g.legend.remove()
g.add_legend(title="Sexo", **{"labels":["Hombre", "Mujer"]})
```

::: {.output .execute_result execution_count="30"}
    <seaborn.axisgrid.FacetGrid at 0x2086e01a7f0>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/4d09882c7f68141a82c28226e46edf7f55bfbec1.png)
:::
:::

::: {#a30cc5a5 .cell .markdown}
Finalmente, observamos la elevación del segmento T, en la que vemos que
hay una distribución muy similar entre sexos, tanto para los casos sin
ECV, como con los casos con ECV, con lo que, aunque la variable en sí
nos aporta información de valor, no es preciso estudiarla ahora y lo
haremos más adelante, sin necesidad de discriminar entre sexos.
:::

::: {#2abbfad7 .cell .markdown}
### ***CONCLUSIONES SEXO***

En este sector hemos observado, principalmente que los hombres tienen
mayor riesgo a sufrir una ECV que las mujeres, sobretodo si nos
centramos en las variables del segmento T, el axucar en sangre, las
pulsaciones máximas o la angina inducida.

Sin embargo, para las mujeres, en cuanto aumenta el pulso mínimo en
reposo o el colesterol, su riesgo a sufrir una ECV aumenta de forma muy
considerable.
:::

::: {#5d4e92cd .cell .markdown}
## Análisis del impacto del tipo de dolor pectoral`<a id="234">`{=html}`</a>`{=html}
:::

::: {#edde0aca .cell .code execution_count="31"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data=data, x="ChestPainType", ax=ax[0], palette="tab10")
ax[0].set_title("Distribución del tipo de dolor pectoral")
sns.histplot(data=data, x="ChestPainType", hue="HeartDisease", multiple="stack", ax=ax[1], palette="tab20_r")
ax[1].set_title("Distribución del tipo de dolor pectoral\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Tipo de dolor pectoral\n\n"+
             "TA: Dolor típico de angina | ATA: Dolor atípico de angina\n" + 
              "NAP: Sin dolor de angina | ASY: Asintomático")
    ax[n].set_ylabel("Recuento")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/99807dd205d4d99b355bbf2a91b46e3f29445887.png)
:::
:::

::: {#63a52736 .cell .code execution_count="32"}
``` python
sns.histplot(data=data, x="HeartDisease", hue="ChestPainType", multiple="stack")
plt.title("Distribución del tipo de dolor pectoral\nseparado entre pacientes con y sin ECV\n(No ECV: 0 | ECV: 1)")
plt.xlabel("ECV")
plt.ylabel("Recuento")
```

::: {.output .execute_result execution_count="32"}
    Text(0, 0.5, 'Recuento')
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/e0592ea66d69c3e7684af88fc25b3bdbe539f507.png)
:::
:::

::: {#98a95eb3 .cell .markdown}
En estos gráficos en los que vemos la distribución de los tipos de dolor
de pecho de cada paciente, observamos que, en la mayoría de los casos,
no existe ningún tipo de dolor en el pecho, y lo que es aún más curioso
y puede plantear un reto es que si nos fijamos en los casos en los que
existe una ECV, la proporción de casos sin ningún tipo de dolor pectoral
se dispara, de forma que el tipo de dolor en el pecho puede ser un
factor que puede llevar a confusión tanto a los médicos como a los
propios pacientes, y por ello será interesante observar cómo se
relaciona esta variable con las otras para poder ver si hay alguna causa
subyacente a éste dolor, tanto si está relacionada con una ECV como si
no lo está.
:::

::: {#383a816a .cell .code execution_count="33"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="ChestPainType", y="Age", ax=ax[0])
ax[0].set_title("Relación entre el tipo de dolor pectoral y la edad")

sns.violinplot(data=data, x="ChestPainType", y="Age", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre el tipo de dolor pectoral y la edad\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Tipo de dolor pectoral\n\n"+
             "TA: Dolor típico de angina | ATA: Dolor atípico de angina\n" + 
              "NAP: Sin dolor de angina | ASY: Asintomático")
    ax[n].set_ylabel("Edad")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/1e91cb1a5197159a4928007aaabeb3ba1461fea7.png)
:::
:::

::: {#34cd0f38 .cell .markdown}
Si separamos el tipo de dolor pectoral a lo largo del rango de edad,
vemos que hay una distribución bastante clara en la que cada rango de
edad tiene un tipo de dolor. Se observa que tanto el dolor atípico de
angina como los casos sin dolor de angina se dan mayoritariamente en
personas con una edad por debajo de los 55 años. En cambio, los
pacientes asintomáticos tienen la mayor parte de la distribución sobre
los 60 años, y finalmente, el dolor de angina típico se distribuye a lo
largo de todo el rango de edad, pero con su pico en la personas mayores
de 60 años.

El único impacto que se observa al tener en cuenta el dolor pectoral
junto con la edad, es que para los pacientes con un dolor atípico de
angina, a partir de los 55 años se dispara su probabilidad de sufrur una
ECV. En los otros casos, la distribución de edad es bastante similar a
cuando no discriminabamos entre pacientes con y sin ECV.
:::

::: {#538f588d .cell .code execution_count="34"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="ChestPainType", y="RestingBP", ax=ax[0])
ax[0].set_title("Relación entre el tipo de dolor pectoral y\nlas pulsaciones en reposo")

sns.violinplot(data=data, x="ChestPainType", y="RestingBP", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre el tipo de dolor pectoral y\nlas pulsaciones en reposo\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Tipo de dolor pectoral\n\n"+
             "TA: Dolor típico de angina | ATA: Dolor atípico de angina\n" + 
              "NAP: Sin dolor de angina | ASY: Asintomático")
    ax[n].set_ylabel("Pulsaciones en reposo")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/25b20b83b4b5fd2900a85754f6ed72eeccf1dfb4.png)
:::
:::

::: {#bb27328f .cell .markdown}
Si comparamos los tipos de dolor pectoral junto con las pulsaciones en
reposo, observamos que en todos los casos de dolor pectoral excepto con
el dolor típico de angina, las pulsaciones tienen una distribución
bastante similar. En el caso del lolor típico de angina, las pulsaciones
en reposo aumentan considerablemente.

También observamos que cuando hay un dolor típico de angina, el hecho de
tener las pulsaciones altas supone un alto riesgo para tener una ECV,
mientras que en los otros casos, no parece que haya un gran impacto de
éstas a la hora de tener una ECV.
:::

::: {#8ec959d5 .cell .code execution_count="35"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="ChestPainType", y="MaxHR", ax=ax[0])
ax[0].set_title("Relación entre el tipo de dolor pectoral y\nlas pulsaciones máximas")

sns.violinplot(data=data, x="ChestPainType", y="MaxHR", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre el tipo de dolor pectoral y\nlas pulsaciones máximas\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Tipo de dolor pectoral\n\n"+
             "TA: Dolor típico de angina | ATA: Dolor atípico de angina\n" + 
              "NAP: Sin dolor de angina | ASY: Asintomático")
    ax[n].set_ylabel("Pulsaciones máximas")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/4c4d89d1f4392f7d02f3076644b091d4c51b9bbf.png)
:::
:::

::: {#b05271de .cell .markdown}
En estos gráficos vemos la comparativa entre los tipos de dolor pectoral
y las pulsaciones máximas, y podemos ver ciertas diferencias en algunos
de los campos. En el dolor atípico de angina, las pulsaciones máximas
son muy elevadas, lo cual tiene sentido, dado que antes hemos visto que
éste tipo de dolor se daba en pacientes más jóvenes, los cuales tienen
mayor capacidad para tener altas pulsaciones. En el resto de casos, las
pulsaciones máximas se distribuyen de forma bastante similar, aunque
para los casos asintomáticos, las pulsaciones máximas se reducen
ligeramente.

La única información que nos aporta la separación entre pacientes con y
sin ECV, es que los pacientes con mayores pulsaciones máximas tienen
menor riesgo a sufrir una ECV, pero no se ve ningún impacto en la
separación entre tipos de dolor pectoral.
:::

::: {#a8c52d2e .cell .code execution_count="36"}
``` python
g = sns.catplot(data=data, x="ChestPainType", kind="count", hue="RestingECG", col="HeartDisease")
axes = g.axes.flatten()

g.set_ylabels("Recuento")
g.set_xlabels("Tipo de dolor pectoral\n\n"+
             "TA: Dolor típico de angina | ATA: Dolor atípico de angina\n" + 
              "NAP: Sin dolor de angina | ASY: Asintomático")

plt.suptitle("Distribución entre los pacientes según el tipo de dolor pectoral\n"+
                "separado por el resultado del ECG con y sin ECV\n\n")
g.fig.subplots_adjust(top=0.8)
axes[0].set_title("Sin ECV")
axes[1].set_title("Con ECV")
g.legend.remove()
g.add_legend(title="Resultado del ECG", **{"labels":["Normal", "Anormal", "Hipertrofia ventricular"]})
```

::: {.output .execute_result execution_count="36"}
    <seaborn.axisgrid.FacetGrid at 0x2086f2266a0>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/7bc782a99d28f641bf7760d791d9ce6a58c5aafc.png)
:::
:::

::: {#f2bf1f5e .cell .markdown}
La comparativa entre el resultado del ECG y los tipos de dolor pectoral
no nos aporta ninguna información de valor.
:::

::: {#14d84d0c .cell .code execution_count="37"}
``` python
g = sns.catplot(data=data, x="ChestPainType", kind="count", hue="ExerciseAngina", col="HeartDisease")
axes = g.axes.flatten()

g.set_ylabels("Recuento")
g.set_xlabels("Tipo de dolor pectoral\n\n"+
             "TA: Dolor típico de angina | ATA: Dolor atípico de angina\n" + 
              "NAP: Sin dolor de angina | ASY: Asintomático")

plt.suptitle("Distribución entre los pacientes según el tipo de dolor pectoral\n"+
                "separado por si el paciente mostró una angina inducida\n\n")
g.fig.subplots_adjust(top=0.8)
axes[0].set_title("Sin ECV")
axes[1].set_title("Con ECV")
g.legend.remove()
g.add_legend(title="Angina inducida", **{"labels":["No", "Sí"]})
```

::: {.output .execute_result execution_count="37"}
    <seaborn.axisgrid.FacetGrid at 0x208704728e0>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/a724e1fed628e546c2329c934a44e7f774d3175f.png)
:::
:::

::: {#474e49f9 .cell .markdown}
En el caso de la comparativa entre los tipos de dolor pectoral y la
presencia de angina inducida, no vemos que tengan gran impacto una
variable con la otra.

Lo que sí que podemos observar, es que las personas con angina inducida
tienen alto riesgo de ECV, aunque no tengan dolor pectoral.
:::

::: {#bc5a444a .cell .code execution_count="38"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="ChestPainType", y="Oldpeak", ax=ax[0])
sns.violinplot(data=data, x="ChestPainType", y="Oldpeak", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre el tipo de dolor pectoral y\nlos valores del segmento T\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Tipo de dolor pectoral\n\n"+
             "TA: Dolor típico de angina | ATA: Dolor atípico de angina\n" + 
              "NAP: Sin dolor de angina | ASY: Asintomático")
    ax[n].set_ylabel("Segmento T")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/02bf7d3f0d5edee89775eb51b1fe1ac50583744e.png)
:::
:::

::: {#f55e0734 .cell .markdown}
Para la comparativa entre el segmento T y el tipo de dolor pectoral,
vamos que las personas con dolor atípico de angina tienen el segmento T
muy concentrado alrededor de 0, y para el resto de casos, está
distribuido de forma más uniforme entre 0 y 2.

Vemos también que para las personas con dolor típico de angina, el
segmento T no tiene impacto en el riesgo. Para el resto de casos, si el
segmento T se distancia de 0, existe un alto riesgo de ECV.
:::

::: {#25d02862 .cell .markdown}
### ***CONCLUSIONES TIPO DE DOLOR PECTORAL***

Para este sector, hemos visto que, para muchas variables, el tipo de
dolor pectoral no tenía influencia en el aumento de riesgo de ECV,
aunque también es debido a que tenemos una gran cantidad de
observaciones de pacientes con ECV que son asintomáticos en cuanto a el
dolor pectoral.

Donde sí hemos visto un impacto ha sido en la relación entre el segmento
T y el dolor pectoral, dado que mientras que cuando no hay un dolor
típico de angina, el tener el segmento T cercano a 0 reduce
drásticamente la probabilidad de tener una ECV, cuando existe un dolor
típico de angina, ésta varianble deja de ser relevante y se puede sufrir
o no una ECV independientemente del valor del segmento T.
:::

::: {#5d1015a9 .cell .markdown}
## Análisis del impacto de los resultados de los análisis de sangre`<a id="235">`{=html}`</a>`{=html}
:::

::: {#fa06031d .cell .code execution_count="39"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data=data, x="Cholesterol", bins=30, ax=ax[0])
ax[0].set_ylim(0, 200)
ax[0].set_title("Distribución del nivel de colesterol en sangre")

sns.histplot(data=data, x="Cholesterol", hue="HeartDisease", bins=30, ax=ax[1], multiple="stack", palette="tab20_r")
ax[1].set_ylim(0, 200)
ax[1].set_title("Distribución del nivel de colesterol en sangre\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Colesterol")
    ax[n].set_ylabel("Recuento")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/7b034a2baf3dc09ad492d74cf957b1df10d32b94.png)
:::
:::

::: {#f1ede39e .cell .markdown}
Aquí vemos la distribución del nivel de colesterol en sangre, y vemos
que el pico de la distribución está sobre 200.

Podemos ver también claramente, que cuando los valores del nivel de
colesterol aumentan, sobretodo a partir de 300, aumenta el riesgo de
sufrir una ECV.
:::

::: {#c72d015f .cell .code execution_count="40" scrolled="false"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data=data, x="FastingBS", ax=ax[0])
ax[0].set_title("Distribución del nivel de azúcar en sangre")

sns.histplot(data=data, x="FastingBS", hue="HeartDisease", ax=ax[1], multiple="stack", palette="tab20_r")
ax[1].set_title("Distribución del nivel de azúcar en sangre\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Azúcar en sangre\n\n(0: Valores normales | 1: Valores elevados)")
    ax[n].set_ylabel("Recuento")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/c4c683948f039c94f1d99771fc1d9f9deaef3f8b.png)
:::
:::

::: {#1bc4e17d .cell .markdown}
Para la variable de el nivel de azúcar en sangre, vemos que, mientras
que hay casi 700 observaciones con valores normales de azúcar en sangre
y 200 para valores elevados, cuando nos fijamos en los casos con ECV,
hay algo más de 300 casos con niveles normales de azúcar en sangre, y
casi 200 casos con niveles elevados, de forma que podemos ver que el
hecho de tener un nivel elevado de azúcar en sangre, aumenta
consideramblemente el riesgo de sufrir una ECV.
:::

::: {#0fd4f0b0 .cell .code execution_count="41" scrolled="false"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.violinplot(data=data, x="FastingBS", y="Cholesterol", ax=ax[0])
ax[0].set_title("Distribución del colesterol\nsegún el nivel de azúcar en sangre")

sns.violinplot(data=data, x="FastingBS", y="Cholesterol", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Distribución del colesterol\nsegún el nivel de azúcar en sangre\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Azúcar en sangre\n\n(0: Valores normales | 1: Valores elevados)")
    ax[n].set_ylabel("Colesterol")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/0aa06c95d3accd1adbe5bdcc92351d784cc4a664.png)
:::
:::

::: {#94f26626 .cell .markdown}
No podemos ver una relación entre estas dos variables.
:::

::: {#495b9765 .cell .markdown}
### ***CONCLUSIONES ANÁLISIS DE SANGRE***

Ambas variables tienen un elevado impacto en el riesgo de sufrir una
ECV, aunque no se observa una gran correlación ellas.
:::

::: {#a468e705 .cell .markdown}
## Análisis del impacto de los resultados del electrocardiograma`<a id="236">`{=html}`</a>`{=html}
:::

::: {#8a73c536 .cell .code execution_count="42"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data=data, x="RestingBP", bins=30, ax=ax[0])
ax[0].set_ylim(0, 160)
ax[0].set_title("Distribución de las pulsaciones en reposo")

sns.histplot(data=data, x="RestingBP", hue="HeartDisease", bins=30, ax=ax[1], multiple="stack", palette="tab20_r")
ax[1].set_ylim(0, 160)
ax[1].set_title("Distribución de las pulsaciones en reposo\nseparadas entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Pulsaciones en reposo")
    ax[n].set_ylabel("Recuento")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/de159e73b411e9ab725c2bf5af7d75eb6d029cdb.png)
:::
:::

::: {#f5ae240c .cell .markdown}
La distribución de las pulsaciones en reposo es una distribución normal,
con si pico alrededor de 130.

Se puede observar que a medida que aumentan éstas pulsaciones en reposo,
mayor es el riesgo de sufrir una ECV.
:::

::: {#6c25c85a .cell .code execution_count="43"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data=data, x="MaxHR", bins=30, ax=ax[0])
ax[0].set_ylim(0, 100)
ax[0].set_title("Distribución de las pulsaciones máximas")

sns.histplot(data=data, x="MaxHR", hue="HeartDisease", bins=30, ax=ax[1], multiple="stack", palette="tab20_r")
ax[1].set_ylim(0, 100)
ax[1].set_title("Distribución de las pulsaciones máximas\nseparadas entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Pulsaciones máximas")
    ax[n].set_ylabel("Recuento")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/10cddf5ef063685e135ec392d276ed757debe9ec.png)
:::
:::

::: {#3fd15cc3 .cell .markdown}
La distribución de las pulsaciones máximas, tiene su pico alrerdedor de
160, aunque la mayoría de valores se encuentran entre 120 y 170.

Se ve claramente que a mayores pulsaciones máximas, menor es el riesgo
de sufrir una ECV.
:::

::: {#bb4130f1 .cell .code execution_count="44"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data=data, x="Oldpeak", bins=30, ax=ax[0])
ax[0].set_title("Distribución del segmento T")

sns.histplot(data=data, x="Oldpeak", hue="HeartDisease", bins=30, ax=ax[1], multiple="stack", palette="tab20_r")
ax[1].set_title("Distribución del segmento T\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Segmento T")
    ax[n].set_ylabel("Recuento")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/3615e834c4e3c0e6adb05c8e1a0cb1f898644ff9.png)
:::
:::

::: {#893679c3 .cell .markdown}
La distribución del segmento T, se encuentra mayoritáriamente
concentrada en el 0, que es su valor normal, y podemos ver que a medida
que se aleja de éste valor, aumentan drásticamente los casos de ECV.
:::

::: {#640e1743 .cell .code execution_count="45"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data=data, x="RestingECG", bins=30, ax=ax[0])
ax[0].set_title("Distribución del resultado del ECG")

sns.histplot(data=data, x="RestingECG", hue="HeartDisease", bins=30, ax=ax[1], multiple="stack", palette="tab20_r")
ax[1].set_title("Distribución del resultado del ECG\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Resultado del ECG\n\n"+
                    "Normal: Normal\nST: Onda ST-T anormal\nLVH: Hipertrofia ventricular")
    ax[n].set_ylabel("Recuento")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/45c7658033f25b5bf0810903881154471cb4fc86.png)
:::
:::

::: {#156417cd .cell .markdown}
La mayor parte de los pacientes observados mostraron un ECG normal, sin
embargo, vemos que cuando el ECG sale de los valores normales, aumenta
ligeramente el riesgo de ECV.
:::

::: {#53c36d8f .cell .code execution_count="46"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data=data, x="ExerciseAngina", bins=30, ax=ax[0])
ax[1].set_title("Distribución del resultado de la angina inducida")

sns.histplot(data=data, x="ExerciseAngina", hue="HeartDisease", bins=30, ax=ax[1], multiple="stack", palette="tab20_r")
ax[1].set_title("Distribución del resultado de la angina inducida\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Angina inducida por ejercicio\n\n"+
                    "N: No | Y: Sí")
    ax[n].set_ylabel("Recuento")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/17ca389e216632c6a38c92e2c724f86861673a8b.png)
:::
:::

::: {#66e8a60a .cell .markdown}
Aunque observamos que hay más casos en los que no hay una angina
inducida, hay más casos de ECV que han presentado una angina inducida
que casos que no lo han hecho.
:::

::: {#d3d0d125 .cell .code execution_count="47"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(data=data, x="ST_Slope", bins=30, ax=ax[0])
ax[0].set_title("Distribución de la elevación del segmento T")

sns.histplot(data=data, x="ST_Slope", hue="HeartDisease", bins=30, ax=ax[1], multiple="stack", palette="tab20_r")
ax[1].set_title("Distribución de la elevación del segmento T\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Elevación del segmento T")
    ax[n].set_ylabel("Recuento")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/c639c345012fcd7cb0bed81c1005f9c6a96b96f7.png)
:::
:::

::: {#f30f87b8 .cell .markdown}
La mayor parte de las observaciones muestran una elevación del segmento
T hacia arriba o plana, y los casos de más riesgo de ECV son tanto la
plana como hacia abajo.
:::

::: {#c5aa7cff .cell .code execution_count="48"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.scatterplot(data=data, x="RestingBP", y="MaxHR", ax=ax[0])
ax[0].set_title("Relación entre las pulsaciones máximas y las pulsaciones en reposo")

sns.scatterplot(data=data, x="RestingBP", y="MaxHR", hue="HeartDisease", ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre las pulsaciones máximas y las pulsaciones en reposo\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n')

for n in [0, 1]:
    ax[n].set_xlabel("Pulsaciones en reposo")
    ax[n].set_ylabel("Pulsaciones máximas")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/7b2079a8d23a72932e276f05d6459ef63289bad4.png)
:::
:::

::: {#710a9998 .cell .code execution_count="49"}
``` python
sns.lmplot(data=data, x="RestingBP", y="MaxHR", hue="HeartDisease", palette="tab20_r", scatter_kws={"s": 15}, legend=False)
plt.title("Relación pulso en reposo y pulso máximo\nseparado entre pacientes con y sin ECV")
plt.legend(title='No ECV: 0\n\nECV: 1\n')
plt.xlabel("Pulsaciones en reposo")
plt.ylabel("Pulsaciones máximas")
```

::: {.output .execute_result execution_count="49"}
    Text(17.67500000000001, 0.5, 'Pulsaciones máximas')
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/622c4a1bc73b75915d4ce4ae053272a44d954eaa.png)
:::
:::

::: {#80b76f44 .cell .markdown}
Aquí vemos cómo se relacionan las variables pulso máximo y pulso en
reposo, y no parece que tengan un gran impacto la una con la otra. Al
fijarmos en la separación entre casos con y sin ECV, vemos que a mayores
pulsaciones en reposo y a menores pulsaciones máximas, mayor es el
riesgo de tener alguna ECV.
:::

::: {#cf5f2118 .cell .code execution_count="50"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.scatterplot(data=data, x="RestingBP", y="Oldpeak", ax=ax[0])
ax[0].set_title("Relación entre el valor del segmento T y\nlas pulsaciones en reposo")

sns.scatterplot(data=data, x="RestingBP", y="Oldpeak", hue="HeartDisease", ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre el valor del segmento T y\nlas pulsaciones en reposo\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Pulsaciones en reposo")
    ax[n].set_ylabel("Segmento T")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/b32250707c65326239be60a947ce1c7836230581.png)
:::
:::

::: {#2be3a952 .cell .markdown}
No se aprecia una relación entre estas dos variables.
:::

::: {#01ced54a .cell .code execution_count="51"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.scatterplot(data=data, x="MaxHR", y="Oldpeak", ax=ax[0])
ax[0].set_title("Relación entre el valor del segmento T y\nlas pulsaciones máximas")

sns.scatterplot(data=data, x="MaxHR", y="Oldpeak", hue="HeartDisease", ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre el valor del segmento T y\nlas pulsaciones máximas\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Pulsaciones máximas")
    ax[n].set_ylabel("Segmento T")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/94669363f2d6715123cdb107252ca4a3e8d536f9.png)
:::
:::

::: {#35721aed .cell .markdown}
Aquí podemos ver que a medida que aumentan las pulsaciones máximas, más
se acerca el segmento T a 0, con lo cual se reduce el riesgo de tener
alguna ECV.
:::

::: {#93fa8792 .cell .code execution_count="52"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="RestingECG", y="RestingBP", ax=ax[0])
ax[0].set_title("Relación entre el resultado del ECGy\nlas pulsaciones en reposo")

sns.violinplot(data=data, x="RestingECG", y="RestingBP", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre el resultado del ECGy\nlas pulsaciones en reposo\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Resultado del ECG\n\n"+
                    "Normal: Normal\nST: Onda ST-T anormal\nLVH: Hipertrofia ventricular")
    ax[n].set_ylabel("Pulsaciones en reposo")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/7767337b26b3c5c3cf4ce94fc1b458ca026839d8.png)
:::
:::

::: {#01a3d26b .cell .markdown}
En cuanto a la comparativa entre el resultado del ECG y las pulsaciones
en reposo, no se puede ver ninguna información interesante más alla de
que para un ECG normal, las pulsaciones en reposo son ligeramente más
bajas.
:::

::: {#2dea7daf .cell .code execution_count="53"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.violinplot(data=data, x="RestingECG", y="MaxHR", ax=ax[0])
ax[0].set_title("Relación entre el resultado del ECGy\nlas pulsaciones máximas")

sns.violinplot(data=data, x="RestingECG", y="MaxHR", hue="HeartDisease", split=True, ax=ax[1], palette="tab20_r")
ax[1].set_title("Relación entre el resultado del ECGy\nlas pulsaciones máximas\nseparado entre pacientes con y sin ECV")
ax[1].legend(title='No ECV: 0\n\nECV: 1\n', labels=[0, 1])

for n in [0, 1]:
    ax[n].set_xlabel("Resultado del ECG\n\n"+
                    "Normal: Normal\nST: Onda ST-T anormal\nLVH: Hipertrofia ventricular")
    ax[n].set_ylabel("Pulsaciones máximas")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/a7a298688975e9af0e6f026ea95765ed6a0b7d86.png)
:::
:::

::: {#6e9e88cc .cell .markdown}
En esta comparativa, podemos observar que para un electrocardiograma
normal, las pulsaciones máximas son ligeramente superiores a cuando
existe una onda ST-T anormal, pero aún son más elevadas cuando hay una
hipertrofia ventricular. Sin embargo, cuando existe una hipertrofia
ventricular, aun teniendo unas pulsaciones elevadas, sigue existiendo
riesgo a tener una ECV.
:::

::: {#f1c1966f .cell .code execution_count="54"}
``` python
g = sns.catplot(data=data, x="RestingECG", kind="count", hue="ST_Slope")
g.set_ylabels("Recuento")
g.set_xlabels("Resultado del ECG\n\n"+
                    "Normal: Normal\nST: Onda ST-T anormal\nLVH: Hipertrofia ventricular")
plt.suptitle("Distribución entre los pacientes según el resultado del EXG"+
                "separado por la elevación del segmento T\n\n")
g.fig.subplots_adjust(top=0.8)
g.legend.remove()
g.add_legend(title="Elevación del segmento T", **{"labels":["Hacia arriba", "Plano", "Hacia abajo"]})
```

::: {.output .execute_result execution_count="54"}
    <seaborn.axisgrid.FacetGrid at 0x2086e316b50>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/1a0ed8010d717568d0b3af72cc00441112a8d8d2.png)
:::
:::

::: {#be37beee .cell .code execution_count="55" scrolled="false"}
``` python
g = sns.catplot(data=data, x="RestingECG", kind="count", hue="ST_Slope", col="HeartDisease")
axes = g.axes.flatten()
g.set_ylabels("Recuento")
g.set_xlabels("Resultado del ECG\n\n"+
                    "Normal: Normal\nST: Onda ST-T anormal\nLVH: Hipertrofia ventricular")

plt.suptitle("Distribución entre los pacientes según el resultado del EXG"+
                "separado por la elevación del segmento T con y sin ECV\n\n")
g.fig.subplots_adjust(top=0.8)
axes[0].set_title("Sin ECV")
axes[1].set_title("Con ECV")
g.legend.remove()
g.add_legend(title="Elevación del segmento T", **{"labels":["Hacia arriba", "Plano", "Hacia abajo"]})
```

::: {.output .execute_result execution_count="55"}
    <seaborn.axisgrid.FacetGrid at 0x2086e89cf70>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/8089b9e4e73a686df971071f1ce9fe985ac1bfef.png)
:::
:::

::: {#d6f6131a .cell .markdown}
En este caso no vemos que haya relación entre el resultado del ECG y la
elevación del segmento T. Sin embargo, sí que vemos que para una
elevación del segmento T plana, hay muchas probabilidades de tener una
ECV, aunque el resultado del ECG sea normal.
:::

::: {#8d0c5cf0 .cell .code execution_count="56"}
``` python
g = sns.catplot(data=data, x="RestingECG", kind="count", hue="ExerciseAngina")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/f6a5c40d7ac147055a8ce0556fa501fc7ade3770.png)
:::
:::

::: {#b505bc23 .cell .code execution_count="57"}
``` python
g = sns.catplot(data=data, x="RestingECG", kind="count", hue="ExerciseAngina", col="HeartDisease")
axes = g.axes.flatten()
g.set_ylabels("Recuento")
g.set_xlabels("Resultado del ECG\n\n"+
                    "Normal: Normal\nST: Onda ST-T anormal\nLVH: Hipertrofia ventricular")

plt.suptitle("Distribución entre los pacientes según el resultado del ECG "+
                "separado si mostraron una angina inducida con y sin ECV\n\n")
g.fig.subplots_adjust(top=0.8)
axes[0].set_title("Sin ECV")
axes[1].set_title("Con ECV")
g.legend.remove()
g.add_legend(title="Angina inducida", **{"labels":["Sí", "No"]})
```

::: {.output .execute_result execution_count="57"}
    <seaborn.axisgrid.FacetGrid at 0x2086e804520>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/aefe8cceeb67f3fe385ce6b29d6552d5c5845ae9.png)
:::
:::

::: {#cb8bd027 .cell .markdown}
Aquí podemos ver que para un ECG con una onda ST-T anormal, hay muchas
probabilidades de tener una angina inducida, y esta anggina inducida es
un fuerte indicador de las ECV. La relación entre un ECG normal y con
hipertrofia ventricular y la angina inducida, no parece ser clara.
:::

::: {#bd5a26ac .cell .code execution_count="58"}
``` python
g = sns.catplot(data=data, x="ST_Slope", kind="count", hue="ExerciseAngina")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/240e423d8cfb672b48615ea71b762e37a914e0be.png)
:::
:::

::: {#530d1ded .cell .code execution_count="59"}
``` python
g = sns.catplot(data=data, x="ST_Slope", kind="count", hue="ExerciseAngina", col="HeartDisease")
axes = g.axes.flatten()
g.set_ylabels("Recuento")
g.set_xlabels("Elevación del segmento T\n"+
              "Up: Hacia arriba | Flat: Plano | Down: Hacia abajo")

plt.suptitle("Distribución entre los pacientes según la elevación del segmento T"+
                "separado si mostraron una angina inducida con y sin ECV\n\n")
g.fig.subplots_adjust(top=0.8)
axes[0].set_title("Sin ECV")
axes[1].set_title("Con ECV")
g.legend.remove()
g.add_legend(title="Angina inducida", **{"labels":["Sí", "No"]})
```

::: {.output .execute_result execution_count="59"}
    <seaborn.axisgrid.FacetGrid at 0x208705a06a0>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/f343cef2f148ca2712e1f155bb83484c95b52b1f.png)
:::
:::

::: {#597da873 .cell .markdown}
Para la elevación del segmento T en comparación con la angina inducida,
vemos que cuando la elevación es plana o hacia abajo, hay muchísimas más
probabilidades de que se muestre una angina inducida, y ya hemos
comentado anteriormente que cuando hay angina inducida, es muy probable
que el paciente tenga una ECV.
:::

::: {#9b09a4ed .cell .markdown}
### ***CONCLUSIONES RESULTADOS DEL ECG***

En este sector, hemos analizado y comparado distintas variables
derivadas del resultado del ECG, y hemos visto varias cosas que nos
pueden aportar información.

Podemos ver que en cuanto a las pulsaciones, cuando las pulsaciones en
reposo son elevadas o cuando las pulsaciones máximas son bajas, aumenta
la probabilidad de tener una ECV. Aún así, no observamos una clara
relación entre estas dos variables. Para el segmento T, cuando éste se
aleja de 0, o la elevación es plana o hacia abajo, aumenta la
probabilidad de mostrar una ECV. Además, vemos que cuando las
pulsaciones máximas son altas, el segmento T tiende a estar cerca de 0.
También hemos visto que, cuando el resultado del ECG muestra una onda
ST-T anormal, hay más porcentaje de observaciones con ECV que con el
resto de resultados, y la pulsación máxima se reduce. Cuando el ECG
muestra una hipertrofia, la pulsación máxima aumenta. Además, vemos que
cuando hay este tipo de resultados, es más probable mostrar una angina
inducida. Continuando con la angina inducida, cuando ésta se produce,
las probabilidades de observar una ECV aumenta drásticamente, y ésta se
muestra también cuando la elevación del segmento T es plana o hacia
abajo.
:::

::: {#3d8ec2ac .cell .markdown}
## Mapa de correlación`<a id="237">`{=html}`</a>`{=html}
:::

::: {#8c609eed .cell .code execution_count="60"}
``` python
corrmap = data.corr().abs()
fig, ax = plt.subplots(figsize=(15,12))
labels = ["Edad", "Pulsaciones_en_reposo", "Colesterol", "Nivel_de_azúcar", "Pulsaciones_máximas", "Segmento_T", "ECV"]
sns.heatmap(corrmap, annot=True, vmax=0.5, ax=ax, xticklabels=labels, yticklabels=labels)
plt.title("Mapa de correlación entre las variables numéricas")
plt.xticks(rotation=90)
```

::: {.output .execute_result execution_count="60"}
    (array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
     [Text(0.5, 0, 'Edad'),
      Text(1.5, 0, 'Pulsaciones_en_reposo'),
      Text(2.5, 0, 'Colesterol'),
      Text(3.5, 0, 'Nivel_de_azúcar'),
      Text(4.5, 0, 'Pulsaciones_máximas'),
      Text(5.5, 0, 'Segmento_T'),
      Text(6.5, 0, 'ECV')])
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/9629c18e7f765e3d65f7643fe1908b0b0488d2d4.png)
:::
:::

::: {#9534202f .cell .markdown}
Al centrarnos en el mapa de correlaciones, vemos que hay relaciones
entre variables que hemos comentado anteriormente.

En cuanto a las principales causas de ECV, vemos que son el valor del
segmento T, las pulsaciones máximas y, en menor medida, la edad y el
nivel de azúcar en sangre.

También vemos como están fuertemente correlacionadas la edad con las
pulsacines en reposo, con las pulsaciones máximas y con el valor del
Segmento-T.
:::

::: {#f2d90d63 .cell .markdown}
### ***CONCLUSIONES FINALES DEL EDA***`<a id="238">`{=html}`</a>`{=html}

Al estudiar el impacto de la edad tanto en el efecto que tiene en las
ECV como al compararla con otras variables, hemos visto que claramente,
a mayor edad, mayor riesgo de ECV y mayor riesgo de que aumenten los
indicadores de riesgo para las ECV.

En cuanto a las comparativas entre sexos, hemos visto que los hombres
tienen mayor riesgo a sufrir una ECV que las mujeres, sobretodo si nos
centramos en las variables del segmento T, el axucar en sangre, las
pulsaciones máximas o la angina inducida. Sin embargo, para las mujeres,
en cuanto aumenta el pulso mínimo en reposo o el colesterol, su riesgo a
sufrir una ECV aumenta de forma muy considerable.

Si nos fijamos en el tipo de dolor pectoral, podemos ver que no tiene
muchai influencia en el riesgo de sufrir una ECV,, aunque no podemos
sacar conclusiones seguras, debido a que la mayoría de las observaciones
no presentan dolor pectoral.

Donde sí hemos visto un impacto ha sido en la relación entre el segmento
T y el dolor pectoral, dado que mientras que cuando no hay un dolor
típico de angina, el tener el segmento T cercano a 0 reduce
drásticamente la probabilidad de tener una ECV, cuando existe un dolor
típico de angina, ésta varianble deja de ser relevante y se puede sufrir
o no una ECV independientemente del valor del segmento T.

Viendo los resultados de las analíticas de sangre, hemos podido ver que
tanto el colesterol como el nivel de azúcar en sangre tienen un elevado
impacto en el riesgo de sufrir una ECV, aunque no se observa una gran
correlación ellas.

En el sector de los resultados del electrocardiograma, hemos analizado y
comparado distintas variables derivadas del resultado del ECG, y hemos
visto varias cosas que nos pueden aportar información.

Hemos observado que, en cuanto a las pulsaciones, cuando las pulsaciones
en reposo son elevadas o cuando las pulsaciones máximas son bajas,
aumenta la probabilidad de tener una ECV. Aún así, no observamos una
clara relación entre estas dos variables. Para el segmento T, cuando
éste se aleja de 0, o la elevación es plana o hacia abajo, aumenta la
probabilidad de mostrar una ECV. Además, vemos que cuando las
pulsaciones máximas son altas, el segmento T tiende a estar cerca de 0.
También hemos visto que, cuando el resultado del ECG muestra una onda
ST-T anormal, hay más porcentaje de observaciones con ECV que con el
resto de resultados, y la pulsación máxima se reduce. Cuando el ECG
muestra una hipertrofia, la pulsación máxima aumenta. Además, vemos que
cuando hay este tipo de resultados, es más probable mostrar una angina
inducida. Continuando con la angina inducida, cuando ésta se produce,
las probabilidades de observar una ECV aumenta drásticamente, y ésta se
muestra también cuando la elevación del segmento T es plana o hacia
abajo.
:::

::: {#54b0dc7c .cell .markdown}
# **Preparación de los datos:**`<a id="3">`{=html}`</a>`{=html}

En esta fase, vamos a realizar una limpieza de los datos. Además, los
acomodaremos para que sean una buena entrada para el modelado de forma
que podamos obtener unos buenos resultados. Para ello, realizaremos
tareas de normalización, discretización, *feature engineering* como PCA
o SVD, y todo aquello que sea necesario para poder tener la máxima
precisión en el modelado.
:::

::: {#bf7558c5 .cell .markdown}
## Discretización de los datos`<a id="31">`{=html}`</a>`{=html}
:::

::: {#cdcc9e1c .cell .markdown}
AgeDisc

Vamos a discretizar la variable edad siguiendo el siguiente criterio: 0:
\<45 \| 1: \>=45 & \<55 \| 2: \>=55 & \<70 \| 3: \>70
:::

::: {#20b195a8 .cell .code execution_count="61"}
``` python
data["AgeDisc"] = data["Age"]
data.loc[data["Age"]<45, "AgeDisc"] = 0
data.loc[(data["Age"]>=45) & (data["Age"]<55), "AgeDisc"] = 1
data.loc[(data["Age"]>=55) & (data["Age"]<70), "AgeDisc"] = 2
data.loc[data["Age"]>=70, "AgeDisc"] = 3
```
:::

::: {#9346133b .cell .code execution_count="62"}
``` python
sns.histplot(data=data, x="AgeDisc", hue="HeartDisease", multiple="stack")
```

::: {.output .execute_result execution_count="62"}
    <AxesSubplot:xlabel='AgeDisc', ylabel='Count'>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/76af8978d59f5d2f878a0a0bb8863241b2192380.png)
:::
:::

::: {#5b0edca5 .cell .markdown}
Discretizamos el segmento T siguiendo el criterio en el que
consideraremos que si está encima de -1 y por debajo de 0.5, lo
consideraremos normal (0), y para el resto de valores, lo consideraremos
anormal (1).
:::

::: {#37f84a23 .cell .code execution_count="63"}
``` python
data["OldpeakDisc"] = data["Oldpeak"]
data.loc[(data["Oldpeak"]>-1) & (data["Oldpeak"]<0.5), "OldpeakDisc"] = 0
data.loc[data["Oldpeak"]<=-1, "OldpeakDisc"] = 1
data.loc[data["Oldpeak"]>=0.5, "OldpeakDisc"] = 1
```
:::

::: {#48209f04 .cell .code execution_count="64"}
``` python
sns.histplot(data=data, x="OldpeakDisc", hue="HeartDisease", multiple="stack")
```

::: {.output .execute_result execution_count="64"}
    <AxesSubplot:xlabel='OldpeakDisc', ylabel='Count'>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/0e5285ead11d7095abbd7a98f728848eb07f85d8.png)
:::
:::

::: {#f8497d23 .cell .markdown}
Discretizamos las pulsaciones máximas, y asignamos un 0 si están por
debajo o son iguales a 130, y un 1 si están por encima.
:::

::: {#704dd5db .cell .code execution_count="65"}
``` python
data["MaxHRDisc"] = data["MaxHR"]
data.loc[data["MaxHR"]<=130, "MaxHRDisc"] = 0
data.loc[data["MaxHR"]>130, "MaxHRDisc"] = 1
```
:::

::: {#78f5683c .cell .code execution_count="66" scrolled="true"}
``` python
sns.histplot(data=data, x="MaxHRDisc", hue="HeartDisease", multiple="stack")
```

::: {.output .execute_result execution_count="66"}
    <AxesSubplot:xlabel='MaxHRDisc', ylabel='Count'>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/c9ca5d8eafd38c7060acc7ccdae0a9f9302e84f4.png)
:::
:::

::: {#2a7e17f4 .cell .markdown}
## Conversión de datos discretos`<a id="32">`{=html}`</a>`{=html}
:::

::: {#6251f6eb .cell .markdown}
En este apartado, asignaremos valores numéricos a todas las variables
categóricas que tenían valores de tipo texto.
:::

::: {#1c8df9f3 .cell .code execution_count="67"}
``` python
data["ChestPainTypeNum"] = data["ChestPainType"]
data.loc[data["ChestPainType"]=="ATA", "ChestPainTypeNum"] = 3
data.loc[data["ChestPainType"]=="NAP", "ChestPainTypeNum"] = 1
data.loc[data["ChestPainType"]=="ASY", "ChestPainTypeNum"] = 0
data.loc[data["ChestPainType"]=="TA", "ChestPainTypeNum"] = 2
data["ChestPainTypeNum"] = data["ChestPainTypeNum"].astype('int')
```
:::

::: {#c3469af5 .cell .code execution_count="68"}
``` python
data["SexNum"] = data["Sex"]
data.loc[data["Sex"]=="M", "SexNum"] = 0
data.loc[data["Sex"]=="F", "SexNum"] = 1
data["SexNum"] = data["SexNum"].astype('int')
```
:::

::: {#d3ee5033 .cell .code execution_count="69"}
``` python
data["RestingECGNum"] = data["RestingECG"]
data.loc[data["RestingECG"]=="Normal", "RestingECGNum"] = 0
data.loc[data["RestingECG"]=="ST", "RestingECGNum"] = 1
data.loc[data["RestingECG"]=="LVH", "RestingECGNum"] = 2

data["RestingECGNum"] = data["RestingECGNum"].astype('int')
```
:::

::: {#5ae117b7 .cell .code execution_count="70"}
``` python
data["ExerciseAnginaNum"] = data["ExerciseAngina"]
data.loc[data["ExerciseAngina"]=="N", "ExerciseAnginaNum"] = 0
data.loc[data["ExerciseAngina"]=="Y", "ExerciseAnginaNum"] = 1
data["ExerciseAnginaNum"] = data["ExerciseAnginaNum"].astype('int')
```
:::

::: {#4aba76c9 .cell .code execution_count="71"}
``` python
data["ST_SlopeNum"] = data["ST_Slope"]
data.loc[data["ST_Slope"]=="Up", "ST_SlopeNum"] = 0
data.loc[data["ST_Slope"]=="Flat", "ST_SlopeNum"] = 1
data.loc[data["ST_Slope"]=="Down", "ST_SlopeNum"] = 2

data["ST_SlopeNum"] = data["ST_SlopeNum"].astype('int')
```
:::

::: {#3f8f6e40 .cell .markdown}
Volvemos a analizar cómo se correlacionan las variables ahora que
tenemos variables discretizadas que las variables categóricas se pueden
calcular.
:::

::: {#5ded3c11 .cell .code execution_count="72" scrolled="false"}
``` python
plt.subplots(figsize=(16,12))
cormap = data.corr().abs()
labels = ["Edad", "Pulsaciones_en_reposo", "Colesterol", "Nivel_de_azúcar", "Pulsaciones_máximas", "Segmento_T", 
          "ECV", "Edad (discretizada)", "Segmento_T (discretizado)", "Pulsaciones_máximas (discretizadas)", 
          "Tipo_dolor_pecho", "Sexo", "Resultados_ECG", "Angina_inducida", "Elevacion_Segmento_T"]

sns.heatmap(cormap, annot=True, xticklabels=labels, yticklabels=labels)
```

::: {.output .execute_result execution_count="72"}
    <AxesSubplot:>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/27c519d3ee560d3edfacf3838226f0256183d5c0.png)
:::
:::

::: {#630848d6 .cell .markdown}
Vemos que al discretizar la edad y el segmento T, aumenta su correlación
con la variable ECV, aunque no es así con las pulsaciones máximas.

Podemos cer también que la mayoría de las variables categóricas tienen
una elevada correlación con las ECV, de modo que será interesante
tenerlas de forma numérica para poder introducirlas en los modelos.
:::

::: {#70fbb866 .cell .markdown}
## Normalización y escalado de los datos`<a id="33">`{=html}`</a>`{=html}
:::

::: {#3b32074e .cell .markdown}
Ahora normalizamos y escalamos los datos para que sean una entrada de
mayor calidad y de este modo le podamos facilitar los cálculos al
modelo.
:::

::: {#d215c4cb .cell .markdown}
Colesterol:
:::

::: {#4052894c .cell .code execution_count="73"}
``` python
data["CholesterolNorm"] = data["Cholesterol"]

data["CholesterolNorm"] = stats.boxcox(data["CholesterolNorm"])[0]

data["CholesterolNorm"] = StandardScaler().fit_transform(data["CholesterolNorm"].array.reshape(-1, 1))


fig, ax=plt.subplots(1, 2, figsize=(15, 3))


sns.histplot(data["Cholesterol"], ax=ax[0])
sns.histplot(data["CholesterolNorm"], ax=ax[1])
```

::: {.output .execute_result execution_count="73"}
    <AxesSubplot:xlabel='CholesterolNorm', ylabel='Count'>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/f3771e6ef16fd0a444f7919bb226491bd351f251.png)
:::
:::

::: {#8be94bce .cell .markdown}
Segmento T:
:::

::: {#ba43e836 .cell .code execution_count="74"}
``` python
data["OldpeakNorm"] = data["Oldpeak"]

data["OldpeakNorm"].array.reshape(-1, 1)

data["OldpeakNorm"] = normalize(data["OldpeakNorm"].array.reshape(-1, 1))

fig, ax=plt.subplots(1, 2, figsize=(15, 3))


sns.histplot(data["Oldpeak"], ax=ax[0])
sns.histplot(data["OldpeakNorm"], ax=ax[1])
```

::: {.output .execute_result execution_count="74"}
    <AxesSubplot:xlabel='OldpeakNorm', ylabel='Count'>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/8538724c439f74c15de6d41d23412ba6f5c6108e.png)
:::
:::

::: {#8959a3a8 .cell .markdown}
Pulsaciones máximas:
:::

::: {#97d42b59 .cell .code execution_count="75"}
``` python
data["MaxHRNorm"] = data["MaxHR"]


#data["MaxHRNorm"] = normalize(data["MaxHRNorm"].array.reshape(-1, 1), norm='l2')
data["MaxHRNorm"] = stats.boxcox(data["MaxHRNorm"])[0]

data["MaxHRNorm"] = StandardScaler().fit_transform(data["MaxHRNorm"].array.reshape(-1, 1))

fig, ax=plt.subplots(1, 2, figsize=(15, 3))


sns.histplot(data["MaxHR"], ax=ax[0])
sns.histplot(data["MaxHRNorm"], ax=ax[1])
```

::: {.output .execute_result execution_count="75"}
    <AxesSubplot:xlabel='MaxHRNorm', ylabel='Count'>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/846bd6b51912fd0ef4de7602160a90283d327eda.png)
:::
:::

::: {#096c1cfb .cell .markdown}
Edad:
:::

::: {#5d2f2c74 .cell .code execution_count="76"}
``` python
data["AgeNorm"] = data["Age"]


#data["MaxHRNorm"] = normalize(data["MaxHRNorm"].array.reshape(-1, 1), norm='l2')
data["AgeNorm"] = stats.boxcox(data["AgeNorm"])[0]

data["AgeNorm"] = StandardScaler().fit_transform(data["AgeNorm"].array.reshape(-1, 1))

fig, ax=plt.subplots(1, 2, figsize=(15, 3))


sns.histplot(data["Age"], ax=ax[0])
sns.histplot(data["AgeNorm"], ax=ax[1])
```

::: {.output .execute_result execution_count="76"}
    <AxesSubplot:xlabel='AgeNorm', ylabel='Count'>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/bbe7f8a72971d704af30cfb770ff74a3f4c96104.png)
:::
:::

::: {#d5b3e1ec .cell .markdown}
Pulsaciones en reposo:
:::

::: {#0011634a .cell .code execution_count="77"}
``` python
data["RestingBPNorm"] = data["RestingBP"]


#data["MaxHRNorm"] = normalize(data["MaxHRNorm"].array.reshape(-1, 1), norm='l2')
data["RestingBPNorm"] = stats.boxcox(data["RestingBPNorm"])[0]

data["RestingBPNorm"] = StandardScaler().fit_transform(data["RestingBPNorm"].array.reshape(-1, 1))

fig, ax=plt.subplots(1, 2, figsize=(15, 3))


sns.histplot(data["RestingBP"], ax=ax[0])
sns.histplot(data["RestingBPNorm"], ax=ax[1])
```

::: {.output .execute_result execution_count="77"}
    <AxesSubplot:xlabel='RestingBPNorm', ylabel='Count'>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/7d3a35c460fff2223b7481a7e6e9bdaa3afcd55b.png)
:::
:::

::: {#4cbdcd32 .cell .markdown}
Volvemos a analizar las correlaciones para ver si hemos mejorado tras la
normalización y el escalado:
:::

::: {#c0edcade .cell .code execution_count="78"}
``` python
plt.subplots(figsize=(16,12))
cormap = data.corr().abs()
labels = ["Edad", "Pulsaciones_en_reposo", "Colesterol", "Nivel_de_azúcar", "Pulsaciones_máximas", "Segmento_T", 
          "ECV", "Edad (discretizada)", "Segmento_T (discretizado)", "Pulsaciones_máximas (discretizadas)", 
          "Tipo_dolor_pecho", "Sexo", "Resultados_ECG", "Angina_inducida", "Elevacion_Segmento_T", "Colesterol (Normalizado)",
          "Segmento_T (Normalizado)", "Pulsaciones_máximas (Normalizadas)", "Edad (Normalizada)", "Pulsaciones_en_reposo (Normalizadas)"]

sns.heatmap(cormap, annot=True, xticklabels=labels, yticklabels=labels)
```

::: {.output .execute_result execution_count="78"}
    <AxesSubplot:>
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/a23dd32dca379783e69cb0e9c1a6f6e429f0e895.png)
:::
:::

::: {#d8fe20b5 .cell .markdown}
Aunque no se observa una gran mejora, sí que mejoramos la entrada del
algoritmo, de modo que nos será útil mantener estas variables.
:::

::: {#a6a10c8f .cell .markdown}
## PCA `<a id="34">`{=html}`</a>`{=html} {#pca-}
:::

::: {#57f7cabe .cell .markdown}
Creamos una función que nos sirva para hacer el PCA cambiando las
variables que se tienen en cuenta.
:::

::: {#ff5fe666 .cell .code execution_count="79"}
``` python
def analisisPCA(feats, data):
    '''Función para realizar un análisis PCA eligiendo las variables a estudiar'''
    feats=feats
    X = data[feats]
    pca_pipe = make_pipeline(StandardScaler(), PCA())
    pca_pipe.fit(X)
    # Se extrae el modelo entrenado del pipeline
    modelo_pca = pca_pipe.named_steps['pca']

    comps = []
    for n in range(len(modelo_pca.components_)):
        comps.append("PC"+str(n+1))

    pca_ = pd.DataFrame(data = modelo_pca.components_, columns = feats, index=comps)

    print(pca_.round(2))
    sns.heatmap(pca_.T, annot=True, cmap='viridis')
    plt.title("Mapa de calor del impacto de cada componente")
    plt.show()
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.bar(x = np.arange(modelo_pca.n_components_) + 1, height = modelo_pca.explained_variance_ratio_)

    for x, y in zip(np.arange(len(feats)) + 1, modelo_pca.explained_variance_ratio_):
        label = round(y, 2)
        ax.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center')

    ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
    ax.set_ylim(0, 1.1)
    ax.set_title('Porcentaje de varianza explicada por cada componente')
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('% varianza explicada')
    plt.show()
    prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
   
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.plot(np.arange(len(feats)) + 1, prop_varianza_acum, marker = 'o')

    for x, y in zip(np.arange(len(feats)) + 1, prop_varianza_acum):
        label = round(y, 2)
        ax.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center')

    ax.set_ylim(0, 1.1)
    ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
    ax.set_title('Porcentaje de varianza explicada acumulada')
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('% varianza acumulada')
```
:::

::: {#475c5403 .cell .markdown}
Variables iniciales numéricas sin tratar
:::

::: {#190fa255 .cell .code execution_count="80"}
``` python
feats=["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
analisisPCA(feats, data)
```

::: {.output .stream .stdout}
          Age  RestingBP  Cholesterol  FastingBS  MaxHR  Oldpeak
    PC1  0.59       0.40         0.03       0.29  -0.48     0.41
    PC2 -0.04       0.36         0.73      -0.42   0.27     0.28
    PC3 -0.06       0.28         0.26       0.78   0.40    -0.28
    PC4  0.02       0.74        -0.40      -0.32   0.04    -0.43
    PC5 -0.20       0.16        -0.48       0.09   0.48     0.68
    PC6  0.78      -0.24        -0.08      -0.15   0.54    -0.13
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/79ee7ba68ea3dd509270bf7990be7025868baa5a.png)
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/5d5b05abc5c2d65a3a4bd6c74f7447c74ed0a18a.png)
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/59f89b409f90e972f2fb0ebd3262d23a911c7e46.png)
:::
:::

::: {#7d294c28 .cell .markdown}
feats = \[\'Age\', \'RestingBP\', \'Cholesterol\', \'FastingBS\',
\'MaxHR\', \'Oldpeak\', \'HeartDisease\', \'AgeDisc\', \'OldpeakDisc\',
\'MaxHRDisc\', \'ChestPainTypeNum\', \'SexNum\', \'RestingECGNum\',
\'ExerciseAnginaNum\', \'ST_SlopeNum\', \'CholesterolNorm\',
\'OldpeakNorm\', \'MaxHRNorm\', \'AgeNorm\', \'RestingBPNorm\'\]
analisisPCA(feats, data)
:::

::: {#4f4d986c .cell .markdown}
Hacemos el PCA ahora teniendo en cuenta las variables discretizadas.
:::

::: {#6851e3e4 .cell .code execution_count="81"}
``` python
feats=["RestingBP", "Cholesterol", "FastingBS", 'AgeDisc', 'OldpeakDisc', 'MaxHRDisc']
analisisPCA(feats, data)
```

::: {.output .stream .stdout}
         RestingBP  Cholesterol  FastingBS  AgeDisc  OldpeakDisc  MaxHRDisc
    PC1       0.38         0.03       0.33     0.56         0.47      -0.46
    PC2       0.42         0.79      -0.40     0.01         0.06       0.18
    PC3       0.50        -0.07       0.56     0.09        -0.41       0.50
    PC4      -0.47         0.44       0.56    -0.17         0.42       0.27
    PC5      -0.22         0.41       0.25     0.06        -0.64      -0.55
    PC6       0.41        -0.03       0.19    -0.80         0.15      -0.36
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/3f7709d6a490bdceee96d61fdafe7f59b9ca0f25.png)
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/209786e4616e3680021a87d643c4d434fbd82617.png)
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/6ae2f2ad5c806d20aea4c6f5477c9b48b61f5df7.png)
:::
:::

::: {#da74d196 .cell .markdown}
Ahora con las variables normalizadas.
:::

::: {#9d29640e .cell .code execution_count="82"}
``` python
feats=["AgeNorm", "MaxHRNorm", "OldpeakDisc", "CholesterolNorm"]
analisisPCA(feats, data)
```

::: {.output .stream .stdout}
         AgeNorm  MaxHRNorm  OldpeakDisc  CholesterolNorm
    PC1    -0.62       0.60        -0.50             0.02
    PC2    -0.03      -0.22        -0.28            -0.93
    PC3    -0.36       0.32         0.82            -0.31
    PC4    -0.70      -0.69         0.03             0.18
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/07e161541a669bc0563aceed54fae223cc2ba788.png)
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/9960bc98d9248244fbade674c46dd92d07d9fe4b.png)
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/322e575804cf97bfbe7b346d64fb8f03e7fe85fa.png)
:::
:::

::: {#fcb53222 .cell .markdown}
También con las variables categóricas.
:::

::: {#5fcd0d00 .cell .code execution_count="83"}
``` python
feats=["AgeDisc", "SexNum", "MaxHRNorm", "OldpeakDisc", "ChestPainTypeNum", "ExerciseAnginaNum", "ST_SlopeNum"]
analisisPCA(feats, data)
```

::: {.output .stream .stdout}
         AgeDisc  SexNum  MaxHRNorm  OldpeakDisc  ChestPainTypeNum  \
    PC1    -0.31    0.20       0.39        -0.41              0.39   
    PC2     0.26    0.89       0.10         0.29              0.15   
    PC3    -0.70    0.00       0.50         0.38             -0.15   
    PC4     0.33   -0.41       0.32         0.38              0.64   
    PC5    -0.39    0.08      -0.52        -0.07              0.62   
    PC6    -0.24    0.01      -0.26        -0.12             -0.03   
    PC7    -0.18   -0.02      -0.38         0.66             -0.04   

         ExerciseAnginaNum  ST_SlopeNum  
    PC1              -0.44        -0.44  
    PC2              -0.00         0.17  
    PC3               0.23         0.22  
    PC4              -0.13         0.22  
    PC5               0.41         0.06  
    PC6              -0.64         0.67  
    PC7              -0.40        -0.48  
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/415f5057604c0a7c385aa2641e367efdd7306177.png)
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/04ad45c2ad709f66e451fe9c87cc4aa29902a582.png)
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/bb0b296f8619aa38536579b252e205c50bac7f51.png)
:::
:::

::: {#ab287709 .cell .markdown}
Vemos como los mejores resultados los hemos obtenido cuando hemos usado
las variables Edad (normalizada), Pulsaciones máximas (normalizadas),
Segmento T (discretizado) y Colesterol (discretizado), dado que sólo con
el primer componente principal, ya explicamos el 40% de la varianza.

También hemos obtenido buenos resultados con las variables Edad
(discretizada), Sexo, Pulsaciones máximas (normalizadas), Segmento T
(discretizado), Tipo de dolor pectoral, Angina inducida y elevación del
Segmento T, con resultados similares a cuando hemos usado las variables
anteriores.
:::

::: {#d225cf44 .cell .markdown}
# **Modelado**`<a id="4">`{=html}`</a>`{=html}
:::

::: {#d6caad3f .cell .markdown}
En esta segunda práctica, vamos a realizar la parte del modelado usando
los datos de la práctica anterior.

Para realizar el modelado, usaremos las variables que hemos visto que
mejores resultados nos daban en la práctica anterior. Éstos son:

-   Edad (discretizada y normalizada)
-   Pulsaciones máximas (normalizadas)
-   Edad (discretizada)
-   Sexo
-   Tipo de dolor pectoral,
-   Angina inducida
-   Elevación del Segmento T

Empezamos importando todas las librerías necesarias:
:::

::: {#50c2e93c .cell .code execution_count="84"}
``` python
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from matplotlib.pylab import rcParams
from dtreeviz import trees
from dtreeviz.models.xgb_decision_tree import ShadowXGBDTree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
```
:::

::: {#0ff4d3bc .cell .markdown}
## Modelos no supervisados`<a id="41">`{=html}`</a>`{=html}
:::

::: {#8d59dd8e .cell .markdown}
Ahora vamos a realizar un modelo no supervisado de tipo K-Means.
:::

::: {#6a74d7d9 .cell .code execution_count="85"}
``` python
# Edad (normalizada), Pulsaciones máximas (normalizadas), Segmento T (discretizado),
# Colesterol (discretizado), Edad (discretizada), Sexo, Tipo de dolor pectoral,
# Angina inducida y elevación del Segmento T

feats=["AgeDisc", "SexNum", "MaxHRNorm", "OldpeakDisc", "ChestPainTypeNum", "ExerciseAnginaNum",
       "ST_SlopeNum"]
data_cl = data
data_cl = data_cl[feats]
data_cl = data_cl.join(data["HeartDisease"])


resultados = []
indices = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for n in indices:
    kmeans = KMeans(n_clusters=n, random_state=0)
    data_cd = kmeans.fit_transform(data_cl[feats])
    resultados.append(data_cd.mean())
sns.lineplot(indices, resultados)
plt.xlabel("Número de clústers")
plt.ylabel("Media de la distancia entre los puntos y el centroide")
plt.title("Selección del mejor número de clústers")
```

::: {.output .execute_result execution_count="85"}
    Text(0.5, 1.0, 'Selección del mejor número de clústers')
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/f8c79d7d8727b2fc5ec55b8b281f4bacd233a0f5.png)
:::
:::

::: {#e69c9eec .cell .markdown}
Podemos observar que según el cálculo de la media de distancia entre los
puntos y el ccentroide, lo más optimo es realizar un modelo con 2
clústers, lo cual tiene mucho sentido, dado que nuestra variable target
nos indica si el paciente tiene alguna ECV o no, de modo que es una
variable booleana (sólo puede ser 0 o 1).
:::

::: {#9f73b1ad .cell .markdown}
Realizamos el modelado y mostramos los datos por pantalla para poder ver
cómo ha realizado la clasificación el modelo.
:::

::: {#348c3857 .cell .code execution_count="86"}
``` python
kmeans = KMeans(n_clusters=2, random_state=0)
data["Cluster"] = kmeans.fit_predict(data_cl[feats])
#data_cl["Cluster"] = kmeans.fit_predict(data_cl[feats])


data["Cluster"].replace([0, 1], [1, 0], inplace=True)
```
:::

::: {#be6904e4 .cell .code execution_count="87"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.scatterplot(data=data, x="Age", y="MaxHR", hue="Cluster", ax=ax[0])
ax[0].set_title("Clasificación kmeans")
ax[0].set_xlabel("Edad")
ax[0].set_ylabel("Pulsaciones máximas")

sns.scatterplot(data=data, x="Age", y="MaxHR", hue="HeartDisease", ax=ax[1])
ax[1].set_title("Clasificación real")
ax[1].set_xlabel("Edad")
ax[1].set_ylabel("Pulsaciones máximas")

plt.show()
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/5a3d06d7cf8aa804ecd7909c7cd9c5bb163dcd68.png)
:::
:::

::: {#ddca8021 .cell .markdown}
A primera vista vemos buenos resultados, pero recordemos que es un
algoritmo no supervisado, de modo que vamos a usar métricas de éste tipo
para ver la calidad del modelo.
:::

::: {#cc0a71fe .cell .code execution_count="88"}
``` python
estimator = make_pipeline(StandardScaler(), kmeans).fit(data_cl[feats])
print("Silhouette: " + 
      str(round((silhouette_score(data_cl, estimator[-1].labels_, metric="euclidean")), 3)))
```

::: {.output .stream .stdout}
    Silhouette: 0.272
:::
:::

::: {#fb4cc41a .cell .markdown}
Como vemos, este modelo con 2 clústers nos proporciona una *Silhouette*
de 0.278 que, aunque está muy lejos de ser un resultado perfecto, para
ser datos tan entremezclados como hemos visto que son estos, podemos
decir que es un buen resultado.
:::

::: {#9e6a8c5a .cell .markdown}
Hacemos ahora la prueba con 4 clústers.
:::

::: {#d7f40119 .cell .code execution_count="89"}
``` python
kmeans = KMeans(n_clusters=4, random_state=0)
data["Cluster"] = kmeans.fit_predict(data_cl[feats])
```
:::

::: {#b5796f82 .cell .code execution_count="90"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.scatterplot(data=data, x="Age", y="MaxHR", hue="Cluster", ax=ax[0])
ax[0].set_title("Clasificación kmeans")
ax[0].set_xlabel("Edad")
ax[0].set_ylabel("Pulsaciones máximas")

sns.scatterplot(data=data, x="Age", y="MaxHR", hue="HeartDisease", ax=ax[1])
ax[1].set_title("Clasificación real")
ax[1].set_xlabel("Edad")
ax[1].set_ylabel("Pulsaciones máximas")

plt.show()
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/ef419e3aaffac8ac433bc6b0f6c175f08d97d382.png)
:::
:::

::: {#f271a672 .cell .markdown}
Aquí vemos que los clústers están muy entremezclados y no se aprecia
ningún patron visual que pueda indicarnos que las separaciones son
correctas.
:::

::: {#9654bb61 .cell .code execution_count="91"}
``` python
estimator = make_pipeline(StandardScaler(), kmeans).fit(data_cl)
print("Silhouette: " + 
      str(round((silhouette_score(data_cl, estimator[-1].labels_, metric="euclidean")), 3)))
```

::: {.output .stream .stdout}
    Silhouette: 0.117
:::
:::

::: {#114ed6bb .cell .markdown}
Vemos pues, que con 4 clústers obtenemos una puntuación de *Silhouette*
de 0.141, la cual es notablemente peor que con 2 clústers.
:::

::: {#b1aba644 .cell .markdown}
Finalmente, realizamos la prueba con 3 clústers.
:::

::: {#1fa9c614 .cell .code execution_count="92"}
``` python
kmeans = KMeans(n_clusters=3, random_state=0)
data["Cluster"] = kmeans.fit_predict(data_cl[feats])
data_cl["Cluster"] = kmeans.fit_predict(data_cl[feats])
```
:::

::: {#97179b8e .cell .code execution_count="93"}
``` python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.scatterplot(data=data, x="Age", y="MaxHR", hue="Cluster", ax=ax[0])
ax[0].set_title("Clasificación kmeans")
ax[0].set_xlabel("Edad")
ax[0].set_ylabel("Pulsaciones máximas")

sns.scatterplot(data=data, x="Age", y="MaxHR", hue="HeartDisease", ax=ax[1])
ax[1].set_title("Clasificación real")
ax[1].set_xlabel("Edad")
ax[1].set_ylabel("Pulsaciones máximas")

plt.show()
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/22e5895cfbf4d9781cbb533086664ebdc45be813.png)
:::
:::

::: {#5cd32034 .cell .code execution_count="94"}
``` python
estimator = make_pipeline(StandardScaler(), kmeans).fit(data_cl)
print("Silhouette: " + 
      str(round((silhouette_score(data_cl, estimator[-1].labels_, metric="euclidean")), 3)))
```

::: {.output .stream .stdout}
    Silhouette: 0.268
:::
:::

::: {#d654987b .cell .markdown}
Al calcular la métrica *Silhouette* vemos que, aunque se acerca a los
niveles obtenidos con 2 clústers, el modelo no llega a ser tan bueno.
:::

::: {#7ff9bb0f .cell .markdown}
### Otra métrica (Davies Bouldin)
:::

::: {#661d1f6f .cell .code execution_count="95"}
``` python
def dbScore(clusters):
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    data["Cluster"] = kmeans.fit_predict(data_cl[feats])
    data_cl["Cluster"] = kmeans.fit_predict(data_cl[feats])
    result = davies_bouldin_score(data_cl, estimator[-1].labels_)
    print("Métrica Davies Bouldin para " + str(clusters) + " clústers: " + str(result) + "\n")
```
:::

::: {#67e5334a .cell .code execution_count="96"}
``` python
dbScore(2)
dbScore(3)
dbScore(4)
```

::: {.output .stream .stdout}
    Métrica Davies Bouldin para 2 clústers: 1.9909097671907057

    Métrica Davies Bouldin para 3 clústers: 1.83014523689354

    Métrica Davies Bouldin para 4 clústers: 2.3141334498871675
:::
:::

::: {#b61cbb1f .cell .markdown}
Con ésta métrica podemos ver que el modelo con 3 clústers tiene mejor
resultado de proximidad, lo cual puede ser algo interesante a investigar
en otro momento.
:::

::: {#5d05994f .cell .markdown}
## DBSCAN + OPTICS`<a id="42">`{=html}`</a>`{=html} {#dbscan--optics}
:::

::: {#a4236131 .cell .markdown}
Vamos a hacer ahora un modelo DBSCAN y un modelo OPTICS.
:::

::: {#e536ffec .cell .markdown}
### DBSCAN
:::

::: {#73a36994 .cell .code execution_count="97"}
``` python
data_db = data
```
:::

::: {#93d7e3cf .cell .code execution_count="98"}
``` python
data_db["Cluster"] = DBSCAN(eps=0.5, min_samples=25).fit_predict(data_cl[feats])

estimator = make_pipeline(StandardScaler(), DBSCAN(eps=0.5, min_samples=30)).fit(data_db[feats])


fig, ax = plt.subplots(1, 2, figsize=(12, 6))


sns.scatterplot(data=data_db, x="Age", y="MaxHR", hue="Cluster", ax=ax[0])
ax[0].set_title("Clasificación DBSCAN")
ax[0].set_xlabel("Edad")
ax[0].set_ylabel("Pulsaciones máximas")
sns.scatterplot(data=data_db, x="Age", y="MaxHR", hue="HeartDisease", ax=ax[1])
ax[1].set_title("Clasificación real")
ax[1].set_xlabel("Edad")
ax[1].set_ylabel("Pulsaciones máximas")

plt.show()
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/1895267985269f36a0b7e6cabadeadba792fcbb1.png)
:::
:::

::: {#0cc95f07 .cell .code execution_count="99"}
``` python
print("Silhouette: " + 
      str(round(metrics.silhouette_score(data_db[feats], 
                                         estimator[-1].labels_, 
                                         metric="euclidean"), 3)))
```

::: {.output .stream .stdout}
    Silhouette: -0.074
:::
:::

::: {#74f75559 .cell .markdown}
Si hacemos el modelo con una $\epsilon$ de 0.5 y un mínimo de 25
muestras, aunque vemos que nos genera los clústers que necesitamos, la
nube de puntos y de observaciones que el modelo considera outliers es
muy dispersa, y no aporta información relevante. Además, la métrica
*Silhouette* está muy cercana a 0, lo cual indica que los clústers se
superponen unos encima de otros. Así pues, probaremos con otros
parámetros.
:::

::: {#46c85dfd .cell .code execution_count="100"}
``` python
data_db["Cluster"] = DBSCAN(eps=1.0, min_samples=25).fit_predict(data_cl[feats])

estimator = make_pipeline(StandardScaler(), DBSCAN(eps=1.0, min_samples=30)).fit(data_db[feats])


fig, ax = plt.subplots(1, 2, figsize=(12, 6))


sns.scatterplot(data=data_db, x="Age", y="MaxHR", hue="Cluster", ax=ax[0])
ax[0].set_title("Clasificación DBSCAN")
ax[0].set_xlabel("Edad")
ax[0].set_ylabel("Pulsaciones máximas")
sns.scatterplot(data=data_db, x="Age", y="MaxHR", hue="HeartDisease", ax=ax[1])
ax[1].set_title("Clasificación real")
ax[1].set_xlabel("Edad")
ax[1].set_ylabel("Pulsaciones máximas")

plt.show()


fig, ax = plt.subplots(1, 2, figsize=(12, 6))


sns.scatterplot(data=data_db, x="Oldpeak", y="RestingBP", hue="Cluster", ax=ax[0])
ax[0].set_title("Clasificación DBSCAN")
ax[0].set_xlabel("Segmento T")
ax[0].set_ylabel("Pulsaciones en reposo")
sns.scatterplot(data=data_db, x="Oldpeak", y="RestingBP", hue="HeartDisease", ax=ax[1])
ax[1].set_title("Clasificación real")
ax[1].set_xlabel("Segmento T")
ax[1].set_ylabel("Pulsaciones en reposo")

plt.show()
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/72977d6e2f3e56b5490596bd825758d432339d8c.png)
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/c240e13e0174b74e87d57cf95016ad6d50dc1093.png)
:::
:::

::: {#83323377 .cell .code execution_count="101"}
``` python
print("Silhouette: " + 
      str(round(metrics.silhouette_score(data_db[feats], 
                                         estimator[-1].labels_, 
                                         metric="euclidean"), 3)))
```

::: {.output .stream .stdout}
    Silhouette: -0.045
:::
:::

::: {#07bce8ec .cell .markdown}
Manteniendo el mínimo de observaciones pero cambiando la $\epsilon$ a 1,
vemos que mejora bastabte el modelo y ya se observa una separación más
clara a simple vista incluso al comparar distintas variables. Sin
embargo, al observar la métrica *Silhouette*, vemos algo que tambiéne se
puede ver en las gráficas, que es la superposición de los clústers,
aunque no es tan exagerada como en el caso anterior.
:::

::: {#73846924 .cell .markdown}
### OPTICS
:::

::: {#6aa47608 .cell .markdown}
Vamos a ver ahora la alcanzabilidad de este modelo mediante OPTICS.
:::

::: {#78a1b11b .cell .code execution_count="102"}
``` python
clust = OPTICS(min_samples=250, max_eps=25).fit(data_cl[feats])
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]


reachability.round(1)

plt.plot(reachability)
plt.title("Reachability plot con $\epsilon$ = 2")
plt.show()
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/b9988b26b51000e0fe86f5babaca500055ca0ab0.png)
:::
:::

::: {#4ba85fd4 .cell .markdown}
Con unos parámetros demasiado altos, casi ninguna observaciónn entra en
algun clúster.
:::

::: {#b201ec2e .cell .code execution_count="103"}
``` python
clust = OPTICS(min_samples=10, max_eps=0.8).fit(data_cl[feats])
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]


reachability.round(1)

plt.plot(reachability)
plt.title("Reachability plot con $\epsilon$ = 0.5")
plt.show()
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/49eed12de5ff720c0f5873a1ced9bb340f1a2ecf.png)
:::
:::

::: {#242bd012 .cell .markdown}
Con unos parámetros demasiado pequeños, vemos que hay muchas
observaciones que no se encuentran en nuestra función y se crean
demasiados clústers.
:::

::: {#ce43e75b .cell .code execution_count="104" scrolled="false"}
``` python
clust = OPTICS(min_samples=25, eps=1).fit(data_cl[feats])
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]


reachability.round(1)

plt.plot(reachability)
plt.title("Reachability plot con $\epsilon$ = 1")
plt.show()
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/a9e475add68859f832af8766504dedddc76e17e9.png)
:::
:::

::: {#7d61c536 .cell .markdown}
En un punto intermedio, vemos que se sigue una forma en la que se ven
claramente 2 clústers que se pueden formar.
:::

::: {#7b3fbea8 .cell .code execution_count="105"}
``` python
X = data_cl[feats].to_numpy()

m_s = 25

eps = 1

clust = OPTICS(min_samples=m_s, max_eps=eps)
clust.fit(X)


space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 8))
G = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(G[0, :])
#ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 0])
ax4 = plt.subplot(G[1, 1])

# Reachability plot
colors = ["g.", "r.", "b.", "y.", "c."]
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color)
ax1.plot(space[labels == -1], reachability[labels == -1], "k.")
ax1.set_ylabel("Reachability (epsilon distance)")
ax1.set_title("Reachability Plot ($\epsilon$ = 1)")

'''# OPTICS
colors = ["g.", "r.", "b.", "y.", "c."]
for klass, color in zip(range(0, 5), colors):
    Xk = X[clust.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.5)
ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], "k+", alpha=0.5)
ax2.set_title("OPTICS\ncon epsilon = 25")
'''
# DBSCAN
data_db["Cluster"] = DBSCAN(min_samples=m_s, eps=eps).fit_predict(data_cl[feats])

sns.scatterplot(data=data_db, x="Age", y="MaxHR", hue="Cluster", ax=ax3, legend=False, alpha=0.5, size=0.5
                , palette=['tab:red', 'tab:green', 'tab:blue'])
ax3.set_title("DBSCAN\ncon epsilon = 1")


# Datos originales
sns.scatterplot(data=data_db, x="Age", y="MaxHR", hue="HeartDisease", ax=ax4, legend=False, alpha=0.5, size=0.5, palette=['tab:red', 'tab:blue'])
ax4.set_title("Datos originales")

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/fbd580dd62cfa01ea8698d08c632aaf9cad4ba70.png)
:::
:::

::: {#804cf0b0 .cell .markdown}
Con los valores $\epsilon$ = 1 y como mínimo 25 observaciones en cada
clúster, se observan unos buneos niveles de alcanzabilidad y se ven
clústers bastante definidos y acorde con la realidad.
:::

::: {#0264059a .cell .markdown}
## Árboles de decisión`<a id="43">`{=html}`</a>`{=html}
:::

::: {#645f9f86 .cell .markdown}
Vamos a modelar ahora un árbol de decisión que nos genere unas reglas
para, poniéndonos en la piel de un médico, que con unas pocas pruebas o
preguntas pueda hacer ya una estimación de si el paciente tiene una ECV
o no.
:::

::: {#02fc6da5 .cell .code execution_count="106"}
``` python
X = data_cl[feats]
y = data["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model_XGB = XGBClassifier(max_depth=4, random_state=0)

model_XGB.fit(X_train, y_train)

y_test_pred = model_XGB.predict(X_test)


model_XGB.best_iteration
best_tree = model_XGB.best_iteration

viz = trees.model(model_XGB, X_test, y_test, target_name='ECV', tree_index=best_tree,
                  feature_names=list(X.columns.values), class_names=[0, 1])

viz.view()
```

::: {.output .execute_result execution_count="106"}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/92da6ff12ac8779ab18b99235543bfa8ae74fc8d.svg)
:::
:::

::: {#55526116 .cell .markdown}
### Reglas del modelo
:::

::: {#6c327275 .cell .markdown}
Podemos observar que la primera gran separación está en la variable de
la elevación del segmento T, donde vemos que en los casos en los que
ésta esté haca arriba (la primera columna), las probabilidades de que el
paciente tenga una ECV se reducen muchísimo, y sólo se ven incrementadas
cuando su Segmento T se distancia de 0 (en la variable \'OldpeakDisc\').

Sin embargo, cuando la elevación del Segmento T sea plana o hacia abajo,
las probabilidades de desarrollar una ECV se disparan, y sólo se ven
reducidas según el tipo de dolor pectoral que experimente el paciente.
:::

::: {#4d82ddcc .cell .code execution_count="107"}
``` python
x = data_cl[feats].iloc[11]

viz.view(x=x)
```

::: {.output .execute_result execution_count="107"}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/18ad0a0ad2d79574b5df2747c419846c0590f1ce.svg)
:::
:::

::: {#3f0bf4e9 .cell .markdown}
En éste ejemplo vemos un caso bastante claro de un paciente con ECV, que
no sólo tiene la elevación del Segmento T plana sinó que además sus
pulsaciones máximas son bastante bajas (algo que normalmente está
relacionado con la edad, tal y como vimos en la práctica anterior).
:::

::: {#13875387 .cell .code execution_count="108"}
``` python
print("Score del modelo:")
print(str(cross_val_score(model_XGB, X_test, y_test).mean().round(2)*100) + "%")
```

::: {.output .stream .stdout}
    Score del modelo:
    83.0%
:::
:::

::: {#9a668a6a .cell .code execution_count="109"}
``` python
cm = confusion_matrix(y_test, y_test_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.show()

print("Porcentaje de falsos negativos: " + str(round((cm[1][0]/(cm[1][1]+cm[1][0]))*100, 2)) + "%")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/a06d15565a0645d4657bd8076e65df7b7277e36f.png)
:::

::: {.output .stream .stdout}
    Porcentaje de falsos negativos: 12.98%
:::
:::

::: {#6b4ea296 .cell .markdown}
Podemos observar que éste modelo tiene una *Score* bastante alta, de
83%.

Por otro lado, en este caso creemos que lo más significativo para tener
en cuenta son los falsos negativos, dado que en caso de que se diese un
falso negativo, un paciente podría no ser intervenido o explorado en
mayor profundidad y se podría agravar la ECV, mientras que en el caso de
los falsos positivos, sólo habría el inconveniente que se deberían
realizar más pruebas posteriormente para descartar la existencia de éste
tipo de enfermedad.

Así pues, con éste árbol, hemos conseguido un porcentaje de falsos
negativos del 13% aproximadamente, lo cual nos indica que, aunque es un
buen modelo, no debería usarse como único método, sino quizas, como
primera aproximación.
:::

::: {#a93043e6 .cell .markdown}
## Otro enfoque algorítmico (KNN)`<a id="44">`{=html}`</a>`{=html}
:::

::: {#402e42c3 .cell .markdown}
Ahora vamos a realizar otro enfoque en el que usaremos el algoritmo KNN,
y compararemos los resultados al utilizar distintos parámetros.
:::

::: {#38702996 .cell .markdown}
Empezamos con **250** *Neighbours*:
:::

::: {#c2fdcb1a .cell .code execution_count="110"}
``` python
data_knn = data

X = data_knn[feats]
y = data_knn["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

neigh = KNeighborsClassifier(n_neighbors=250)
neigh.fit(X_train, y_train)
y_test_pred = neigh.predict(X_test)
```
:::

::: {#9dd52ed3 .cell .code execution_count="111"}
``` python
data_knn["prediction"] = neigh.predict(X)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))


sns.scatterplot(data=data_knn, x="Age", y="MaxHR", hue="prediction", ax=ax[0])
ax[0].set_title("Clasificación KNN")
ax[0].set_xlabel("Edad")
ax[0].set_ylabel("Pulsaciones máximas")
sns.scatterplot(data=data_knn, x="Age", y="MaxHR", hue="HeartDisease", ax=ax[1])
ax[1].set_title("Clasificación real")
ax[1].set_xlabel("Edad")
ax[1].set_ylabel("Pulsaciones máximas")

plt.show()

cm = confusion_matrix(y_test, y_test_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.show()

print("Porcentaje de falsos negativos: " + str(round((cm[1][0]/(cm[1][1]+cm[1][0]))*100, 2)) + "%")
print("Score del modelo: " + str(round(neigh.score(X, y)*100, 2)) + "%")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/05f453cfa6f009eac2e2c1de767b17c7e0ecaa0d.png)
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/9129261c5515b3980f87819a315af73f1900d4a2.png)
:::

::: {.output .stream .stdout}
    Porcentaje de falsos negativos: 8.4%
    Score del modelo: 78.74%
:::
:::

::: {#3d7332a5 .cell .markdown}
Podemos ver que con 250 Neighbours, la *Score* del modelo es de 78.74%,
mientras que tenemos un 8.4% de falsos negativos, lo cual es una mejora
respecto al árbol anterior. Sin embargo, podemos ver como han aumentado
considerablemente los falsos positivos (41.4%)
:::

::: {#010d0dfd .cell .markdown}
Modelamos ahora con **150** *Neighbours*:
:::

::: {#50572308 .cell .code execution_count="112"}
``` python
data_knn = data

X = data_knn[feats]
y = data_knn["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

neigh = KNeighborsClassifier(n_neighbors=150)
neigh.fit(X_train, y_train)
y_test_pred = neigh.predict(X_test)
```
:::

::: {#56b3b5da .cell .code execution_count="113"}
``` python
data_knn["prediction"] = neigh.predict(X)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))


sns.scatterplot(data=data_knn, x="Age", y="MaxHR", hue="prediction", ax=ax[0])
ax[0].set_title("Clasificación KNN")
ax[0].set_xlabel("Edad")
ax[0].set_ylabel("Pulsaciones máximas")
sns.scatterplot(data=data_knn, x="Age", y="MaxHR", hue="HeartDisease", ax=ax[1])
ax[1].set_title("Clasificación real")
ax[1].set_xlabel("Edad")
ax[1].set_ylabel("Pulsaciones máximas")

plt.show()

cm = confusion_matrix(y_test, y_test_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.show()

print("Porcentaje de falsos negativos: " + str(round((cm[1][0]/(cm[1][1]+cm[1][0]))*100, 2)) + "%")
print("Score del modelo: " + str(round(neigh.score(X, y)*100, 2)) + "%")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/97b70241e14c72827d322c14c04adcf8bcbe7d40.png)
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/6ce00deb20b2ab46a0e5535b690c513fdfabcf30.png)
:::

::: {.output .stream .stdout}
    Porcentaje de falsos negativos: 12.21%
    Score del modelo: 80.81%
:::
:::

::: {#2c061c4c .cell .markdown}
En este caso nos pasa al revés, dado que aumenta la *Score* del modelo,
pero también aumentan los falsos negativos, aunque se han reducido
notablemente los falsos positivos hasta aproximadamente el 25%.
:::

::: {#c525e46e .cell .markdown}
Finalmente, hacemos la prueba con **350** *Neighbours*:
:::

::: {#0d863382 .cell .code execution_count="114"}
``` python
data_knn = data

X = data_knn[feats]
y = data_knn["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

neigh = KNeighborsClassifier(n_neighbors=350)
neigh.fit(X_train, y_train)
y_test_pred = neigh.predict(X_test)
```
:::

::: {#40350cfb .cell .code execution_count="115"}
``` python
data_knn["prediction"] = neigh.predict(X)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))


sns.scatterplot(data=data_knn, x="Age", y="MaxHR", hue="prediction", ax=ax[0])
ax[0].set_title("Clasificación KNN")
ax[0].set_xlabel("Edad")
ax[0].set_ylabel("Pulsaciones máximas")
sns.scatterplot(data=data_knn, x="Age", y="MaxHR", hue="HeartDisease", ax=ax[1])
ax[1].set_title("Clasificación real")
ax[1].set_xlabel("Edad")
ax[1].set_ylabel("Pulsaciones máximas")

plt.show()

cm = confusion_matrix(y_test, y_test_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm)
cm_display.plot()
plt.show()

print("Porcentaje de falsos negativos: " + str(round((cm[1][0]/(cm[1][1]+cm[1][0]))*100, 2)) + "%")
print("Score del modelo: " + str(round(neigh.score(X, y)*100, 2)) + "%")
```

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/9191beabcbe8d2eba4cb4495cbed73698f9117ce.png)
:::

::: {.output .display_data}
![](vertopal_912d3094a48a4d9b8dbc491767b4fab5/6037d818d4d4426bcb273ff2c680ccd31bfd813c.png)
:::

::: {.output .stream .stdout}
    Porcentaje de falsos negativos: 5.34%
    Score del modelo: 77.75%
:::
:::

::: {#920db4b5 .cell .markdown}
Finalmente, con 350 neighbours, se vuelven a reducir tanto la *Score*
como los falsos negativos, pero se mantienen aproximadamente los falsos
positivos.
:::

::: {#68248bda .cell .markdown}
En resumen, podemos ver que a medida que aumentamos los neighbours,
reducimos los falsos negativos a cambio de aumentar los falsos positivos
y reducir la *Score* del modelo, de modo que dependiendo de lo que se
quiera conseguir, habrá que utilizar unos u otros parámetros.
:::

::: {#cc9d40a0 .cell .markdown}
## Limitaciones y riesgos`<a id="45">`{=html}`</a>`{=html}
:::

::: {#048dcc84 .cell .markdown}
Hemos visto que con los datos que tenemos, lo primero de todo y más
importante es hacer una adaptación para que se acomoden lo mejor posible
al modelo, y aún así, no obtenemos unos resultados ni mucho menos
perfectos, aunque sí que mejoran mucho que si tan sólo utilizásemos las
observaciones iniciales o un *EDA*.

Siguiendo con los datos, hemos podido ver que usando las variables que
tienen los médicos para poder diagnosticar una ECV, es muy difícil
conseguir un correcto diagnóstico, aún adaptando los datos y usando
modelos y algoritmos avanzados, lo cual pone en valor la labor que éstos
realizan al hacer diagnósticos con unas variables y observaciones que no
crean claramente dos grupos separados entre pacientes con ECV y
pacientes sin éstas, sino que vemos muchas observaciones entremezcladas
y pacientes que a priori podría parecer que tienen una ECV, resulta que
no la tienen y otros que aparentemente están sanos, tienen una ECV
contra todo pronóstico.

Otra consideración a tener en cuenta, es que éste conjunto de datos
tiene un cierto sesgo, dado que la distribución no es la habitual entre
la población tanto a nivel de sexo, de edad, etc., pero sobretodo al ver
la cantidad de observaciones de pacientes con ECV, dado que vemos que
aproximadamente la mitad de las observaciones tienen una ECV, lo cual en
la población real esto no ocurre, lo que hace pensar que se han tomado
observaciones de pacientes de los cuales ya había una primera sospecha
de que tuviesen algún tipo de patología de éste tipo y se haya realizado
ya un primer filtro que no nos permita hacer el modelo más preciso.

Finalmente, es importante tener en cuenta el principal riesgo que puede
suponer un modelo de éste tipo, y es que puede darse el caso de que
aparezcan falsos negativos, de forma que se podría dar el alta a
pacientes que realmente tuviesen una ECV, posibilitando así el hecho de
que ésta se agravase o que no se puediese combatir a tiempo, de modo
que, aunque éste tipo de modelos avanzan a pasos agigantados, en el
campo de la medicina (así como en muchos otros campos), éstos modelos
deban ser una herramienta que sirva para ayudar al profesional, pero
nunca un sustituto de éste profesional, dado que es quien debe tener la
última palabra.
:::

::: {#81ef01d2 .cell .markdown}
# Conclusiones práctica 1`<a id="5">`{=html}`</a>`{=html}
:::

::: {#63825c32 .cell .markdown}
En esta práctica hemos hecho un planteamiento de lo que podría ser un
problema real, en el que hemos hecho toda la preparación previa de un
juego de datos antes de la aplicación del modelado.

El problema que hemos decidido plantear ha sido el de las enfermedades
cardiovasculares (ECV) y cuáles pueden ser sus principales causas o sus
indicadores de riesgo. Hemos decidido hacer este estudio dado que las
ECV son la principal causa de muerte en todo el mundo, de modo que
consideramos importante saber cuales pueden ser sus raíces ya sea o bien
para prevenirlas, o bien para detectarlas sa tiempo y así poder
tratarlas a tiempo.

En primer lugar hemos importado y descrito los datos, y luego hemos
hecho una limpieza de los datos, aunque el dataset ya venía con los
datos muy bien acomodados.

Luego hemos realizado un análisis exploratorio de los datos viéndolos
desde distintas perspectivas o dimensiones para poder ver cómo se
relacionan los datos tanto entre sí como en relación a las ECV. Las
principales conclusiones que hemos podido sacar han sido que, a primera
vista, los factores más decisivos son la edad, el sexo, el nivel de
azucar, las pulsaciones máximas, la angina inducida por ejercicio y el
segmento T. Un dato curioso que hemos observado ha sido que el
colesterol no tiene un impacto muy decisivo.

Luego hemos hecho una acomodación de los datos para facilitar el
modelado, y hemos hecho tareas de discretización, conversión de
variables categóricas en variables numéricas, normalización y escalado.

Finalmente, hemos realizado un PCA, en el que hemos usado distintos
grupos de variables, y hemos observado que las que mejor resultado nos
han dado han sido las siguientes: Edad (normalizada), Pulsaciones
máximas (normalizadas), Segmento T (discretizado) Colesterol
(discretizado), Edad (discretizada), Sexo, Tipo de dolor pectoral,
Angina inducida y elevación del Segmento T, de modo que serán variables
que deberemos tener en cuenta al realizar el modelado, y ver como
interactúan con el riesgo a sufrir una ECV.

En la próxima práctica veremos el modelado de estos datos y a ver cómo
de precisas serán nuestras predicciones tras esta preparación de los
datos.
:::

::: {#04e0b3da .cell .markdown}
# Bibliografía`<a id="7">`{=html}`</a>`{=html}
:::

::: {#109658bf .cell .markdown}
[Información Enfermedades Cardiovasculares
(Wikipedia)](https://es.wikipedia.org/wiki/Enfermedades_cardiovasculares#)

[Información Enfermedades Cardiovasculares
(MedlinePlus)](https://medlineplus.gov/spanish/ency/patientinstructions/000759.htm)

[Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

[PCA en Python
(cienciadedatos.net)](https://www.cienciadedatos.net/documentos/py19-pca-python.html)

[Documentación PCA
scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html#sklearn.preprocessing.normalize)

[Silhouette en
Python](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

[Davies Bouldin en
Python](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score)

[Uso de DBSCAN en
python](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

[OPTICS en
python](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS)

[Matriz de confusión en
Python](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)

[KNN en python](https://scikit-learn.org/stable/modules/neighbors.html)
:::
