
# SNIES Analytics Platform

### *De datos dispersos a decisiones estratégicas en educación superior colombiana*

![Python](https://img.shields.io/badge/Python-3.11.9-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.4.1-orange)
![MLlib](https://img.shields.io/badge/MLlib-KMeans%20%C2%B7%20RF%20%C2%B7%20GBT%20%C2%B7%20LR-green)
![Parquet](https://img.shields.io/badge/DataLake-Parquet-lightgrey)
![Período](https://img.shields.io/badge/Per%C3%ADodo-2013--2024-yellow)

---

## Tabla de Contenidos

1. [Contexto del proyecto](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#1-contexto-del-proyecto)
2. [Qué resuelve este sistema](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#2-qu%C3%A9-resuelve-este-sistema)
3. [Arquitectura completa — 5 capas](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#3-arquitectura-completa--5-capas)
4. [Justificación técnica del Data Lake Parquet](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#4-justificaci%C3%B3n-t%C3%A9cnica-del-data-lake-parquet)
5. [Estructura del repositorio](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#5-estructura-del-repositorio)
6. [Requisitos e instalación](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#6-requisitos-e-instalaci%C3%B3n)
7. [Datos fuente](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#7-datos-fuente)
8. [Orden de ejecución completo](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#8-orden-de-ejecuci%C3%B3n-completo)
9. [Descripción de cada componente](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#9-descripci%C3%B3n-de-cada-componente)
10. [Desafíos técnicos resueltos](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#10-desaf%C3%ADos-t%C3%A9cnicos-resueltos)
11. [Salidas esperadas](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#11-salidas-esperadas)
12. [Preguntas de negocio respondidas](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#12-preguntas-de-negocio-respondidas)
13. [Agregar un año nuevo (2025+)](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#13-agregar-un-a%C3%B1o-nuevo-2025)
14. [Autores](https://claude.ai/chat/0a92c600-3cbb-4bff-9ac9-e38e48eed1d5#14-autores)

---

## 1. Contexto del proyecto

El **Sistema Nacional de Información de la Educación Superior (SNIES)** es la fuente oficial del Ministerio de Educación Nacional de Colombia para el seguimiento de instituciones y programas académicos en todo el país. Publica anualmente datos sobre seis categorías: Inscritos, Admitidos, Matriculados, Graduados, Docentes y Matriculados Primer Curso.

El problema estructural es que cada año se publica como archivos Excel independientes por categoría — sin integración histórica oficial. Cruzar 12 años de historia para responder una sola pregunta estratégica puede tomar días de trabajo manual. La tasa de graduación nacional fluctúa entre el 10% y el 15% histórico, una cifra invisible para quienes toman decisiones porque los datos que la revelan están dispersos en más de 68 archivos desconectados.

**SNIES Analytics Platform** construye la infraestructura que el SNIES no tiene: un pipeline ETL de 5 capas completamente automático, un Data Lake en formato Parquet particionado por año, y tres capas de analítica —descriptiva, de segmentación y predictiva— que convierten esa fragmentación en respuestas accionables.

| Dimensión                      | Valor                                               |
| ------------------------------- | --------------------------------------------------- |
| Período cubierto               | 2013 – 2024 (12 años)                             |
| Categorías integradas          | 6                                                   |
| Archivos Excel procesados       | ~68                                                 |
| Instituciones únicas           | ~350+ IES                                           |
| Observaciones institución-año | ~3,500                                              |
| Tecnología central             | Python · PySpark 3.4.1 · PySpark MLlib · Parquet |

---

## 2. Qué resuelve este sistema

| Problema                                                      | Solución implementada                                            |
| ------------------------------------------------------------- | ----------------------------------------------------------------- |
| 68+ archivos Excel dispersos sin integración                 | Pipeline ETL de 5 capas que consolida todo automáticamente       |
| 60+ variantes históricas de nombres de columnas              | Mapa canónico centralizado + renombrado por índice posicional   |
| Encoding roto Windows-1252 en columnas (`CËDIGO`,`AÐO`) | Lectura `latin-1`+ renombrado por posición, inmune al encoding |
| Schema evolutivo: columnas CINE aparecen solo desde 2019      | `unionByName(allowMissingColumns=True)`en PySpark               |
| Sin análisis histórico longitudinal integrado               | Data Lake Parquet particionado con lectura columnar selectiva     |
| Sin segmentación institucional objetiva                      | K-Means con evaluación Silhouette para k=3..7 + PCA 2D           |
| Sin alerta temprana de riesgo académico                      | 3 clasificadores comparados: RF · GBT · Regresión Logística   |
| Datos 2025+ requieren reprocesamiento manual                  | Pipeline append-only: nueva partición sin tocar el histórico    |

---

## 3. Arquitectura completa — 5 capas

```
┌─────────────────────────────────────────────────────────────────┐
│  CAPA 1 — INGESTA  (pandas + openpyxl)                          │
│  data/raw/Analitica_educativa/<año>/                            │
│  • Detección automática del header real (scoring palabras clave)│
│  • Eliminación de filas de metadata institucional               │
│  • Parche multisheet para archivos 2023-2024                    │
│  • Caso especial: graduados 2001-2017 en archivo único          │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓  data/prepared/<categoria>/year=<año>/
┌──────────────────────────┴──────────────────────────────────────┐
│  CAPA 2 — PREPROCESAMIENTO  (PySpark)                           │
│  • Renombrado canónico: 60+ variantes → nombre estándar         │
│  • Normalización de valores categóricos                         │
│  • Manejo de nulos y casteo de tipos                            │
│  • Validación de schema post-renombrado                         │
│  • Sanitización de caracteres especiales en nombres de columnas │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────┴──────────────────────────────────────┐
│  CAPA 3 — INTEGRACIÓN HISTÓRICA  (PySpark)                      │
│  • unionByName(allowMissingColumns=True) por categoría          │
│  • Tolerancia a schema evolutivo (CINE 2019+)                   │
│  • Filtro de período relevante (graduados históricos)           │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────┴──────────────────────────────────────┐
│  CAPA 4 — DATA LAKE PARQUET                                     │
│  data/lake/<categoria>/year=<año>/part-*.parquet                │
│  • Compresión snappy · Particionado por año                     │
│  • Partition pruning para queries analíticas                    │
│  • Append-only: años nuevos no tocan el histórico              │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌──────────────────────────┴──────────────────────────────────────┐
│  CAPA 5 — LIMPIEZA ANALÍTICA  (PySpark → CSV)                   │
│  • Imputación final para columnas de ML                         │
│  • Exportación a data/processed/<categoria>.csv (UTF-8)         │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
         ┌─────────────────┴──────────────────┐
         ↓                                    ↓
┌────────────────────┐              ┌──────────────────────────┐
│  feature_builder   │              │     Notebooks             │
│  12 features       │              │  01_descriptive           │
│  derivadas IES-año │              │  02_clustering            │
└────────┬───────────┘              │  03_risk_prediction       │
         ↓                         │  04_shark_tank_answers    │
  ┌──────┴────────────────────┐    └──────────────────────────┘
  │  clustering.py            │
  │  K-Means · Silhouette k=3..7 │
  │  PCA 2D · 5 perfiles IES  │
  └──────────────────────────-┘
  ┌───────────────────────────┐
  │  risk_predictor.py        │
  │  RF · GBT · Log. Reg.     │
  │  AUC-ROC · Accuracy · F1  │
  │  Split temporal 2013-2021 │
  │  / test 2022-2024         │
  └───────────────────────────┘
```

---

## 4. Justificación técnica del Data Lake Parquet

La elección de Parquet como formato de almacenamiento del Data Lake fue evaluada frente a las alternativas convencionales:

| Criterio                              | Data Lake Parquet                                                          | Data Warehouse SQL (MySQL)                    |
| ------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------- |
| Velocidad de lectura para ML          | **~10x más rápido**— lectura columnar selectiva                   | Lento — lectura por filas completas          |
| Integración con PySpark MLlib        | **Nativa**— sin conversión                                         | Requiere JDBC driver + conversión en memoria |
| Agregar año nuevo                    | **Sin tocar el histórico**— nueva partición `year=2025/`        | INSERT masivo o ALTER TABLE                   |
| Evolución del schema                 | **Tolerante**— columnas nuevas conviven con particiones antiguas    | Requiere ALTER TABLE + migración             |
| Compresión de almacenamiento         | **~70% menos**que CSV con codec snappy                               | Sin compresión nativa eficiente              |
| Consultas analíticas (GROUP BY, AGG) | **Predicate pushdown**— solo lee columnas necesarias                | Full table scan en tablas grandes             |
| Portabilidad                          | **Estándar de industria**— Spark, Databricks, AWS Athena, BigQuery | Vendor lock-in por motor SQL                  |

### Partition pruning en acción

El particionado `year=<año>/` permite que PySpark aplique **partition pruning** automáticamente. Al consultar solo el test set 2022-2024, Spark no lee las particiones de los 9 años anteriores, reduciendo el I/O proporcionalmente al rango consultado.

### Por qué no spark-excel

`spark-excel` generó errores de dependencias Maven en el entorno del proyecto. La combinación `pandas + openpyxl` para lectura inicial + PySpark para todo el procesamiento posterior es el patrón estándar en AWS Glue y Databricks, lo que además documenta una decisión técnica de madurez industrial.

---

## 5. Estructura del repositorio

```
Proyecto_Final_BigDataV/
│
├── README.md
│
├── data/
│   ├── raw/                               ← Excel originales sin modificar
│   │   └── Analitica_educativa/
│   │       ├── 2013/
│   │       │   ├── inscritos_2013.xlsx
│   │       │   └── ...
│   │       └── ... 2024/
│   │
│   ├── prepared/                          ← CSV intermedios post-ingesta
│   │   └── <categoria>/
│   │       └── year=<año>/
│   │
│   ├── lake/                              ← Data Lake Parquet particionado
│   │   ├── inscritos/
│   │   │   ├── year=2013/part-0000.parquet
│   │   │   └── ... year=2024/
│   │   ├── admitidos/
│   │   ├── matriculados/
│   │   ├── graduados/
│   │   ├── docentes/
│   │   └── matriculados_primer_curso/
│   │
│   └── processed/
│       ├── inscritos.csv                  ← CSV limpios para analítica
│       ├── admitidos.csv
│       ├── matriculados.csv
│       ├── graduados.csv
│       ├── docentes.csv
│       ├── matriculados_primer_curso.csv
│       │
│       ├── aggregations/
│       │   ├── features_institucionales.csv   ← tabla central ML (~3,500 filas)
│       │   ├── ranking_instituciones_riesgo.csv
│       │   ├── tabla_prioridad_intervencion.csv
│       │   └── viz_*.png                      ← 19 visualizaciones exportadas
│       │
│       └── ml/
│           ├── institutional_clusters.csv
│           ├── cluster_profiles.csv
│           ├── pca_coords.csv
│           ├── silhouette_scores.csv
│           ├── risk_predictions.csv
│           ├── model_comparison.csv
│           ├── feature_importance.csv
│           └── roc_data.csv
│
├── src/
│   ├── pipeline/
│   │   ├── run_pipeline.py                ← Orquestador principal
│   │   ├── ingestion.py                   ← Capa 1: lectura Excel
│   │   ├── preprocessing.py               ← Capa 2: normalización
│   │   ├── integration.py                 ← Capa 3: unión histórica
│   │   └── canonical_schemas.py           ← Mapa canónico de columnas (fuente de verdad)
│   │
│   └── ml/
│       ├── feature_builder.py             ← Features derivadas con PySpark
│       ├── clustering.py                  ← K-Means + Silhouette + PCA
│       └── risk_predictor.py              ← RF + GBT + Logística comparados
│
├── notebooks/
│   ├── clean_for_analytics.ipynb          ← Limpieza final → CSV procesados
│   ├── 01_descriptive_analysis.ipynb      ← Análisis histórico + 6 visualizaciones
│   ├── 02_institutional_clustering.ipynb  ← Clustering + 4 visualizaciones
│   ├── 03_risk_prediction.ipynb           ← Predicción + 4 visualizaciones
│   └── 04_shark_tank_answers.ipynb        ← 5 preguntas de negocio respondidas
│
└── requirements.txt
```

---

## 6. Requisitos e instalación

### Prerrequisitos del sistema

* **Java 8 o 11** instalado y `JAVA_HOME` configurado (requerido por PySpark)
* **Python 3.9+** (probado en 3.11.9)

```bash
# Verificar Java
java -version

# Verificar Python
python --version
```

### Entorno virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar — Windows PowerShell
.\venv\Scripts\Activate.ps1

# Activar — macOS / Linux
source venv/bin/activate
```

### Dependencias

```bash
pip install pyspark==3.4.1 pandas numpy matplotlib seaborn openpyxl jupyter
```

O desde el archivo de dependencias:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```
pyspark==3.4.1
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl==3.1.5
jupyter>=1.0.0
```

### Verificar instalación

```bash
python -c "from pyspark.sql import SparkSession; print('PySpark OK')"
python -c "from pyspark.ml.clustering import KMeans; print('MLlib OK')"
```

---

## 7. Datos fuente

Los archivos Excel originales del SNIES deben depositarse en `data/raw/Analitica_educativa/<año>/` antes de ejecutar el pipeline. Los CSV limpios en `data/processed/` son el insumo para los scripts de ML y los notebooks.

### Esquema de columnas (post-limpieza, por índice posicional)

El pipeline renombra columnas por  **posición** , no por nombre, para ser inmune al encoding roto Windows-1252 de los archivos originales (`CËDIGO`, `AÐO`, `CAR┴CTER`):

| Índice | Contenido                                            | Alias canónico       |
| ------- | ---------------------------------------------------- | --------------------- |
| 0       | Código de la institución                           | `COD_IES`           |
| 2       | Nombre de la IES                                     | `NOMBRE_IES`        |
| 5       | Sector (Oficial / Privado)                           | `SECTOR_IES`        |
| 7       | Carácter IES                                        | `CARACTER_IES`      |
| 9       | Código departamento IES                             | `COD_DEPTO_IES`     |
| 10      | Departamento de domicilio                            | `DEPTO_IES`         |
| 22      | Área de conocimiento*(categorías con programa)*    | `AREA_CONOCIMIENTO` |
| -2      | Métrica principal (MATRICULADOS / INSCRITOS / etc.) | Según categoría     |
| -1      | Año                                                 | `AÑO`              |

### Particularidades conocidas del dato

| Situación                                                        | Solución implementada                                         |
| ----------------------------------------------------------------- | -------------------------------------------------------------- |
| 5-8 filas de metadata antes del header real                       | Algoritmo de scoring por palabras clave en primeras 20 filas   |
| Archivos 2023-2024 con hoja índice extra                         | Módulo de detección de hoja correcta por scoring por hoja    |
| Graduados 2001-2017 en un único archivo                          | Filtro de período post-lectura: solo 2013-2024                |
| Archivos 2013 con formato pivoteado (sem/sexo como columnas)      | Excluidos con justificación técnica; limitación documentada |
| Columnas CINE solo desde 2019                                     | `unionByName(allowMissingColumns=True)`en integración       |
| Punto en `NO. DE DOCENTES`interpretado por Spark como separador | Sanitización automática de nombres post-renombrado           |

---

## 8. Orden de ejecución completo

> **Regla de oro:** respetar el orden estrictamente. Cada componente depende de la salida del anterior.

---

### FASE I — Pipeline ETL

#### Paso 0 — Ejecutar el pipeline completo de ingesta y Data Lake

```bash
python src/pipeline/run_pipeline.py --mode full --categories todas
```

Esto ejecuta en secuencia las capas 1 a 5: ingesta de Excel, preprocesamiento, integración histórica, escritura del Data Lake Parquet y exportación de CSV limpios a `data/processed/`.

**Para agregar solo un año nuevo sin reprocesar el histórico:**

```bash
python src/pipeline/run_pipeline.py --mode append --categories todas --year 2025
```

**Tiempo estimado:** 15–40 minutos según hardware y número de archivos.

**Salida esperada:**

```
[INGESTA]      ✅ 68 archivos procesados sin errores
[PREPROCESSING] ✅ Schema canónico aplicado — 60+ variantes normalizadas
[INTEGRACIÓN]  ✅ Union histórica completada por categoría
[DATA LAKE]    ✅ Parquet escrito en data/lake/ — particionado por año
[LIMPIEZA]     ✅ CSV limpios exportados a data/processed/
```

#### Paso 0b — Limpieza analítica final (si es necesario ejecutar por separado)

```bash
jupyter nbconvert --to notebook --execute notebooks/clean_for_analytics.ipynb
```

---

### FASE II — Feature Engineering y Machine Learning

#### Paso 1 — Construir features institucionales

```bash
python src/ml/feature_builder.py
```

**Qué hace:** Lee los 6 CSV limpios, realiza el JOIN central por `(COD_IES, AÑO)`, calcula 8 variables derivadas más la variable objetivo `riesgo_academico` y guarda `features_institucionales.csv` en `data/processed/aggregations/`.

**Tiempo estimado:** 3–8 minutos.

**Salida esperada:**

```
[1/6] Leyendo CSV limpios...
[2/6] Agregando métricas por institución-año...
[3/6] Construyendo tabla central IES-AÑO...
[4/6] Calculando features derivadas...
[5/6] Construyendo variable objetivo riesgo_academico...
   → Percentil 33 tasa_graduacion:      0.XXXX
   → Percentil 33 tasa_permanencia:     0.XXXX
   → Percentil 75 ratio_doc_estudiante: XX.XX
[6/6] Guardando features_institucionales.csv...
✅ Features guardadas | Filas: X,XXX | Instituciones: XXX | En riesgo: XXX (XX%)
```

---

#### Paso 2 — Clustering institucional

```bash
python src/ml/clustering.py
```

**Qué hace:** Evalúa K-Means para k=3..7 con Silhouette Score, entrena el modelo óptimo, aplica PCA 2D para visualización y exporta los 4 archivos de resultados.

**Tiempo estimado:** 5–12 minutos.

**Salida esperada:**

```
[1/5] Cargando features institucionales...
   → Instituciones únicas: XXX
[2/5] Evaluando silhouette para k = 3..7...
   k=3  silhouette=0.XXXX
   k=4  silhouette=0.XXXX
   k=5  silhouette=0.XXXX
   k=6  silhouette=0.XXXX
   k=7  silhouette=0.XXXX
   ✅ k óptimo por silhouette: X
[3/5] Entrenando modelo final con k=X...
[4/5] Calculando coordenadas PCA 2D...
[5/5] Guardando resultados...
✅ Clustering completado — k=X
```

---

#### Paso 3 — Predictor de riesgo académico

```bash
python src/ml/risk_predictor.py
```

**Qué hace:** Entrena y compara Random Forest, GBT y Regresión Logística usando split temporal estricto (train 2013-2021 / test 2022-2024). Exporta métricas comparativas, curvas ROC, importancia de features y predicciones del modelo ganador.

**Tiempo estimado:** 8–20 minutos (GBT es el más lento).

**Salida esperada:**

```
[1/5] Cargando datos...
   Total filas: X,XXX | En riesgo (=1): XXX (XX%) | Sin riesgo (=0): XXX
[2/5] Split temporal: train=2013-2021  test=2022-2024
   Train: X,XXX filas | Test: XXX filas
[3/5] Entrenando y evaluando modelos...
   ▶ Entrenando RandomForest...
   [RandomForest]       AUC-ROC=0.XXXX  Accuracy=0.XXXX  F1=0.XXXX
   ▶ Entrenando GBT...
   [GBT]                AUC-ROC=0.XXXX  Accuracy=0.XXXX  F1=0.XXXX
   ▶ Entrenando LogisticRegression...
   [LogisticRegression] AUC-ROC=0.XXXX  Accuracy=0.XXXX  F1=0.XXXX
[5/5] Guardando resultados...
   🏆 Mejor modelo: XXXXX (AUC-ROC=0.XXXX)
```

---

#### Paso 4 — Notebooks de análisis y visualización

```bash
jupyter notebook
```

Ejecutar en orden con  **Kernel → Restart & Run All** :

| Notebook                              | Contenido                                                                                                                                 | Prerequisito | Visualizaciones |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------ | --------------- |
| `01_descriptive_analysis.ipynb`     | Evolución histórica nacional, top 10 IES, heatmap departamental, scatter docente-eficiencia, sector oficial vs privado, embudo nacional | Paso 1       | 6               |
| `02_institutional_clustering.ipynb` | Silhouette k=3..7, PCA 2D con clusters, radar de perfiles, distribución por sector                                                       | Paso 2       | 4               |
| `03_risk_prediction.ipynb`          | Comparación de modelos, curvas ROC, feature importance, mapa de riesgo por departamento                                                  | Paso 3       | 4               |
| `04_shark_tank_answers.ipynb`       | Respuestas directas a las 5 preguntas de negocio + resumen ejecutivo                                                                      | Pasos 1-3    | 5               |

**Para ejecución no interactiva (CI/CD o reproducibilidad):**

```bash
jupyter nbconvert --to notebook --execute notebooks/01_descriptive_analysis.ipynb --output notebooks/01_descriptive_analysis.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_institutional_clustering.ipynb --output notebooks/02_institutional_clustering.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_risk_prediction.ipynb --output notebooks/03_risk_prediction.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_shark_tank_answers.ipynb --output notebooks/04_shark_tank_answers.ipynb
```

---

## 9. Descripción de cada componente

### `src/pipeline/canonical_schemas.py`

Fuente de verdad única del sistema. Contiene el `CANONICAL_COLUMN_MAP` con las 60+ variantes históricas de nombres de columnas del SNIES y su alias canónico. Agregar soporte para el año 2025 requiere únicamente actualizar este archivo si aparecen variantes nuevas.

---

### `src/pipeline/run_pipeline.py`

Orquestador principal. Acepta argumentos `--mode` (full / append), `--categories` y `--year` para controlar qué se procesa. En modo `append`, solo procesa el año indicado y escribe la nueva partición Parquet sin tocar el histórico.

---

### `src/ml/feature_builder.py`

Núcleo de la capa analítica. Realiza el JOIN central de las 6 categorías y computa las features derivadas:

| Variable                       | Fórmula                         | Interpretación                |
| ------------------------------ | -------------------------------- | ------------------------------ |
| `tasa_graduacion`            | Graduados / Matriculados         | Eficiencia de egreso           |
| `tasa_permanencia`           | Matriculados / Inscritos         | Conversión inscrito → activo |
| `tasa_admision`              | Admitidos / Inscritos            | Selectividad institucional     |
| `ratio_docente_estudiante`   | Matriculados / Docentes          | Carga docente                  |
| `brecha_ingreso_egreso`      | Inscritos − Graduados           | Pérdida neta de talento       |
| `tasa_crecimiento_matricula` | (Mat_t − Mat_{t-1}) / Mat_{t-1} | Crecimiento YoY                |
| `tasa_grad_anho_anterior`    | Lag(tasa_graduacion, 1 año)     | Graduación año anterior      |
| `riesgo_academico`           | OR de 3 criterios                | Variable objetivo ML (0/1)     |

**Definición de `riesgo_academico = 1`** — la institución cumple al menos uno de:

* `tasa_graduacion` < percentil 33 histórico
* `tasa_permanencia` < percentil 33 histórico
* `ratio_docente_estudiante` > percentil 75 histórico

> **Nota:** La condición OR fue adoptada tras verificar que la condición AND original (`tasa_graduacion < P25 AND tasa_crecimiento > P75`) producía cero casos positivos — el lag del primer año de cada IES genera nulos en `tasa_crecimiento_matricula`, y el join implícito entre ambas columnas al calcular percentiles dejaba el subconjunto vacío.

---

### `src/ml/clustering.py`

Segmentación no supervisada con K-Means (PySpark MLlib).

**Pipeline interno:**

```
VectorAssembler → StandardScaler (μ=0, σ=1) → KMeans
```

**Evaluación del k óptimo** via Silhouette Score:

```
S(i) = [b(i) − a(i)] / max{a(i), b(i)}
```

donde `a(i)` = distancia media intra-cluster y `b(i)` = distancia media al cluster vecino más próximo.

**Visualización:** PCA con 2 componentes (también en MLlib) para proyección 2D del espacio de features.

**5 perfiles resultantes:**

| Cluster | Perfil                          | Señal principal                                 |
| ------- | ------------------------------- | ------------------------------------------------ |
| 0       | Alta eficiencia                 | Alta graduación + bajo ratio docente            |
| 1       | Crecimiento acelerado           | Alto crecimiento matrícula + eficiencia media   |
| 2       | Riesgo académico               | Baja graduación + baja permanencia + alta carga |
| 3       | Pequeña estable                | Bajo volumen + métricas estables                |
| 4       | Alta docencia / baja eficiencia | Ratio alto + graduación media-baja              |

> Las etiquetas se asignan post-análisis de centroides. Validar con `cluster_profiles.csv` tras cada ejecución.

---

### `src/ml/risk_predictor.py`

Clasificación supervisada con split temporal estricto para eliminar data leakage.

**Split:** entrenamiento 2013-2021 / test 2022-2024 — replica la condición real de predicción prospectiva.

**Modelos comparados:**

| Modelo                | Hiperparámetros                                 | Importancia de features |
| --------------------- | ------------------------------------------------ | ----------------------- |
| Random Forest         | 100 árboles · profundidad máx. 6 · seed=42   | Gini impurity           |
| GBT                   | 50 iteraciones · profundidad máx. 5 · seed=42 | Information gain        |
| Regresión Logística | regParam=0.1 · elasticNet=0.0 · maxIter=100    | Coeficientes            |

**Métricas computadas:**

```
AUC-ROC  = ∫ TPR d(FPR)
Accuracy = (TP + TN) / (P + N)
F1-Score = 2 · (Precision · Recall) / (Precision + Recall)
```

**El modelo con mayor AUC-ROC se selecciona automáticamente como predictor de producción.**

---

## 10. Desafíos técnicos resueltos

| # | Desafío                                                                  | Solución implementada                                                                                     |
| - | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1 | 5-8 filas de metadata antes del header real en los Excel                  | Algoritmo de scoring por palabras clave en primeras 20 filas; tie-breaking por densidad de celdas no nulas |
| 2 | 60+ variantes históricas de nombres de columnas                          | `CANONICAL_COLUMN_MAP`centralizado en `canonical_schemas.py`como única fuente de verdad               |
| 3 | Columnas CINE aparecen solo desde 2019                                    | `unionByName(allowMissingColumns=True)`en PySpark                                                        |
| 4 | Graduados 2013-2017 en un único archivo con datos desde 2001             | Filtro de período post-lectura                                                                            |
| 5 | Archivos 2013 con semestres/sexos como columnas (formato pivoteado)       | Excluidos con documentación técnica explícita                                                           |
| 6 | Hoja índice en archivos 2023-2024                                        | Módulo de parche con scoring de hojas por palabras clave                                                  |
| 7 | Punto en `NO. DE DOCENTES`interpretado por Spark como `tabla.columna` | Sanitización de nombres post-renombrado antes de entrar al plan de ejecución Spark                       |
| 8 | Encoding roto Windows-1252 en columnas CSV (`CËDIGO`,`AÐO`)         | Lectura `encoding='latin-1'`+ renombrado por índice posicional                                          |
| 9 | Variable objetivo con cero casos positivos en condición AND              | Redefinición con lógica OR sobre 3 criterios percentílicos independientes                               |

---

## 11. Salidas esperadas

Tras ejecutar el pipeline completo, los scripts de ML y los 4 notebooks:

```
data/processed/aggregations/
├── features_institucionales.csv        ← tabla central (~3,500 filas, 15 columnas)
├── ranking_instituciones_riesgo.csv    ← top 20 IES por años en riesgo
├── tabla_prioridad_intervencion.csv    ← lista accionable para el Ministerio
├── viz_01_evolucion_historica.png      ← Notebook 01
├── viz_02_top10_instituciones.png      ← Notebook 01
├── viz_03_heatmap_departamental.png    ← Notebook 01
├── viz_04_scatter_docente_eficiencia.png ← Notebook 01
├── viz_05_sector_oficial_privado.png   ← Notebook 01
├── viz_06_embudo_nacional.png          ← Notebook 01
├── viz_cl1_silhouette.png              ← Notebook 02
├── viz_cl2_pca_clusters.png            ← Notebook 02
├── viz_cl3_radar_perfiles.png          ← Notebook 02
├── viz_cl4_clusters_sector.png         ← Notebook 02
├── viz_ml1_comparacion_modelos.png     ← Notebook 03
├── viz_ml2_curvas_roc.png              ← Notebook 03
├── viz_ml3_feature_importance.png      ← Notebook 03
├── viz_ml4_riesgo_departamentos.png    ← Notebook 03
├── viz_q1_sostenibilidad.png           ← Notebook 04
├── viz_q2_riesgo_regional.png          ← Notebook 04
├── viz_q3_docente_eficiencia.png       ← Notebook 04
├── viz_q4_perdida_area.png             ← Notebook 04
└── viz_q5_prioridad_intervencion.png   ← Notebook 04

data/processed/ml/
├── institutional_clusters.csv          ← cluster asignado por IES
├── cluster_profiles.csv                ← perfil promedio por cluster
├── pca_coords.csv                      ← coordenadas PC1/PC2
├── silhouette_scores.csv               ← evaluación k=3..7
├── risk_predictions.csv                ← predicciones del mejor modelo
├── model_comparison.csv                ← tabla AUC-ROC / Accuracy / F1
├── feature_importance.csv              ← importancia RF y GBT
└── roc_data.csv                        ← puntos FPR/TPR para los 3 modelos
```

---

## 12. Preguntas de negocio respondidas

| #  | Pregunta                                                                 | Stakeholder              | Notebook — Celda | Tipo de análisis                        |
| -- | ------------------------------------------------------------------------ | ------------------------ | ----------------- | ---------------------------------------- |
| Q1 | ¿Qué instituciones tienen crecimiento no sostenible?                   | Ministerio de Educación | 04 — P1          | feature_builder + scatter sostenibilidad |
| Q2 | ¿Qué regiones muestran riesgo académico creciente?                    | Rectores / Planeación   | 04 — P2          | Descriptivo + risk_predictor             |
| Q3 | ¿Existe relación entre capacidad docente y eficiencia de graduación?  | Investigadores / MEN     | 04 — P3          | Correlación Pearson + hexbin density    |
| Q4 | ¿Qué áreas de conocimiento tienen alta demanda pero baja permanencia? | Ministerio / Decanatos   | 04 — P4          | Embudo por área de conocimiento         |
| Q5 | ¿Qué instituciones priorizar para intervención o inversión?          | Gobierno / Fondos        | 04 — P5          | Clustering + ML combinados               |

---

## 13. Agregar un año nuevo (2025+)

El sistema está diseñado para absorber datos futuros con intervención manual mínima.

**1. Depositar los Excel del SNIES 2025:**

```
data/raw/Analitica_educativa/2025/
├── inscritos_2025.xlsx
├── admitidos_2025.xlsx
├── matriculados_2025.xlsx
├── graduados_2025.xlsx
├── docentes_2025.xlsx
└── matriculados_primer_curso_2025.xlsx
```

**2. Ejecutar el pipeline en modo append:**

```bash
python src/pipeline/run_pipeline.py --mode append --categories todas --year 2025
```

Esto escribe únicamente la partición `year=2025/` en el Data Lake sin modificar ninguna partición histórica.

**3. Reconstruir features y modelos:**

```bash
python src/ml/feature_builder.py
python src/ml/clustering.py
python src/ml/risk_predictor.py
```

**4. Re-ejecutar notebooks:**

```bash
jupyter nbconvert --to notebook --execute notebooks/01_descriptive_analysis.ipynb --output notebooks/01_descriptive_analysis.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_institutional_clustering.ipynb --output notebooks/02_institutional_clustering.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_risk_prediction.ipynb --output notebooks/03_risk_prediction.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_shark_tank_answers.ipynb --output notebooks/04_shark_tank_answers.ipynb
```

El sistema detecta automáticamente el año nuevo, recalcula los percentiles globales para `riesgo_academico` con el histórico ampliado y reentrena los modelos. **Intervención manual en código: ninguna.**

> Si el SNIES 2025 introduce nuevas variantes de nombres de columnas, agregar el alias correspondiente en `canonical_schemas.py` es el único cambio necesario.

---

*SNIES Analytics Platform —  2025*
