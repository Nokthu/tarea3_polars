# Documentación - Tarea 3 - Procesamiento de Datos con Polars

En la carpeta adjunta se encontrará un pipeline completo de análisis de datos y aprendizaje automático construido con Polars, 
y comparado contra Pandas en rendimiento computacional, para la tercera tarea del curso de Computación Paralela.

## Requisitos

- python 3.x
- polars
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- kagglehub
- psutil

Se instalan las dependencias en caso de no tenerlas. El dataset se descarga
automáticamente vía kagglehub, por lo que no es necesario tenerlo en el repositorio.

# Estructura en orden de prioridad

- tarea3_polars.ipynb # donde se orquesta todo el pipeline: carga, preprocesamiento, ingeniería de características, machine learning, benchmark, experimentos y visualizaciones
- distribucion_clases.png # gráfica donde se muestra el balance entre vuelos a tiempo y retrasados
- benchmark_tiempos.png # comparativa de tiempos de Polars y Pandas por operación del pipeline
- speedup_operaciones.png # speedup de Polars sobre Pandas por operación, con línea de referencia en 1x
- escalabilidad.png # tiempos y speedup vs tamaño del dataset al 25%, 50%, 75% y 100%
- resultados_modelos.png # comparación de Accuracy, F1 y AUC para los tres modelos entrenados

# Para ejecutar

1) Abrir el notebook tarea3_polars.ipynb en Jupyter (Kabré) o Google Colab.
2) Ejecutar la celda de instalación de dependencias al inicio y reiniciar el kernel.
3) Ejecutar las celdas en orden de arriba a abajo.
4) El dataset se descarga automáticamente desde Kaggle con kagglehub.
5) Las gráficas se generan y guardan automáticamente al final del notebook.

# Dataset

Flight Delay Prediction — Tunisair (2016-2018)
Fuente: https://www.kaggle.com/datasets/abderrahimalakouche/flight-delay-prediction

El dataset contiene registros de vuelos operados por Tunisair y aerolíneas
asociadas entre 2016 y 2018. Cada fila representa un vuelo con información
de origen, destino, horarios programados, estado del vuelo y minutos de retraso.

Columnas originales:

- ID: identificador único del vuelo
- DATOP: fecha de operación (formato YYYY-MM-DD)
- FLTID: código de vuelo (ej: TU 0712, donde TU es el código IATA de Tunisair)
- DEPSTN: aeropuerto de salida (código IATA, ej: TUN para Túnez)
- ARRSTN: aeropuerto de llegada (código IATA)
- STD: fecha y hora programada de salida
- STA: fecha y hora programada de llegada
- STATUS: estado del vuelo (ATA=aterrizó, SCH=programado, DEP=despegó, RTR=retornó, DEL=cancelado)
- AC: código de aeronave (ej: TU 320IMU, donde 320 corresponde a un Airbus A320)
- target: minutos de retraso en la llegada (variable continua)

# Funcionalidades por sección del notebook

## Sección 0: Instalación e imports

Se instalan todas las dependencias usando sys.executable para garantizar que
el pip instala en el mismo entorno del kernel de Jupyter. Esto es necesario
en entornos de HPC como el Kabré donde el kernel y el sistema pueden tener
entornos Python distintos. Luego se importan todas las bibliotecas.

## Sección 1: Carga del dataset

Se descarga el dataset con kagglehub.dataset_download() y se carga con
pl.read_csv(). Se realiza una exploración inicial que incluye dimensiones,
tipos de columnas, estadísticas descriptivas completas y conteo de valores
nulos por columna. El dataset tiene 107,833 filas, 10 columnas y ningún
valor nulo en ninguna columna.

## Sección 2: Análisis Exploratorio (Polars)

Se analiza la estructura del dataset usando exclusivamente Polars:

- Se identifican dos formatos de hora mezclados en las columnas STD y STA:
  algunos registros usan puntos (HH.MM.SS) y otros usan dos puntos (HH:MM:SS).
  Esto requiere una función de parseo personalizada con pl.coalesce() que
  intenta ambos formatos y toma el primero que funcione.
- Se analiza la distribución de STATUS, identificando que solo los vuelos ATA
  (93,679 de 107,833) tienen datos reales de retraso medido.
- Se analiza la variable target mostrando que va de 0 a 3,451 minutos, con
  una mediana de 19 minutos y valores extremos que corresponden a cancelaciones
  reprogramadas.
- Se exploran los aeropuertos y aerolíneas más frecuentes, identificando que
  el 98% de los vuelos son de Tunisair (código TU).

## Sección 3: Preprocesamiento

Se limpian y transforman los datos crudos antes de construir las features:

- **Parseo de fechas**: DATOP se convierte a tipo Date con str.to_date().
  STD y STA se convierten a Datetime con una función auxiliar parsear_datetime_mixto()
  que usa pl.coalesce() para manejar los dos formatos detectados. Se extraen
  mes, día de la semana, hora de salida y hora de llegada programada.
- **Filtrado por STATUS**: se conservan solo los vuelos ATA porque son los únicos
  con datos reales de aterrizaje. Los vuelos SCH (programados a futuro), DEP
  (en vuelo), RTR (retornados) y DEL (cancelados) no tienen retraso medido real.
  Esto reduce el dataset a 93,679 filas.
- **Variable objetivo binaria**: se crea la columna retrasado como
  (target > 15).cast(Int8), usando el umbral de 15 minutos que es el
  estándar oficial de la industria aérea (IATA). El balance de clases resultante
  es casi perfecto: 48.4% a tiempo vs 51.6% retrasados, lo que elimina la
  necesidad de técnicas de balanceo como SMOTE.

## Sección 4: Ingeniería de Características (Polars)

Se construyen las variables predictoras usando expresiones vectorizadas de Polars:

- **codigo_aerolinea**: se extrae con str.slice(0, 2) de FLTID. Permite
  al modelo distinguir entre las 7 aerolíneas presentes en el dataset.
- **modelo_aeronave**: se extrae de AC con la expresión regular
  `([A-Z]{0,2}\d{2,3}[A-Z]?)`, que captura tanto modelos con prefijo numérico
  (ej: 320I para A320) como con prefijo alfabético (ej: AT7 para ATR 72).
  Se usa fill_null("desconocido") para los casos que no coincidan con el patrón.
- **mes y dia_semana**: extraídos de fecha_operacion con .dt.month()
  y .dt.weekday().
- **hora_salida y hora_llegada_programada**: extraídas de datetime_salida
  y datetime_llegada con .dt.hour().
- **retraso_promedio_aeropuerto**: calculado con group_by("DEPSTN").agg(pl.col("target").mean())
  y unido al dataset con un join izquierdo sobre DEPSTN. Representa el historial
  de retrasos de cada aeropuerto de salida y es una de las features más informativas
  para el modelo.
- **retraso_promedio_ruta**: calculado con group_by(["DEPSTN", "ARRSTN"])
  y unido con un segundo join sobre el par origen-destino. Captura el
  comportamiento específico de cada ruta.

Las variables categóricas se convierten a enteros con cast(pl.Categorical).to_physical()
directamente en Polars, sin necesidad de LabelEncoder de sklearn.

## Sección 5: Machine Learning

Se entrena y evalúa el dataset con tres modelos de clasificación binaria.
El dataset se convierte a numpy con .to_numpy() para compatibilidad con sklearn
y se divide en 80% entrenamiento (72,239 filas) y 20% prueba (18,060 filas)
con stratify=y para preservar el balance de clases.

Se define una función auxiliar evaluar_modelo() que encapsula el entrenamiento,
predicción y cálculo de métricas, permitiendo evaluar los tres modelos de forma
consistente y comparable:

- **Regresión Logística**: modelo base lineal con max_iter=1000 para garantizar
  convergencia. Sirve como punto de comparación mínimo esperado.
- **Random Forest**: 100 árboles con n_jobs=-1 para usar todos los núcleos
  disponibles. Captura relaciones no lineales entre features.
- **XGBoost**: 100 árboles en secuencia con n_jobs=-1. Cada árbol corrige
  los errores del anterior, lo que lo hace el más poderoso de los tres para
  datos tabulares.

Se reportan Accuracy, F1 Score, AUC y tiempo de entrenamiento para cada modelo,
junto con la matriz de confusión mostrando VP, VN, FP y FN.

## Sección 6: Benchmark Polars vs Pandas

Se reimplementa exactamente el mismo pipeline usando Pandas y se mide el tiempo
de cada operación con time.time() antes y después de cada paso. Las operaciones
medidas son: lectura del CSV, filtrado, agregación con group_by, join y feature
engineering. Se calcula el speedup como tiempo_pandas / tiempo_polars para
cada operación. Se reporta además la información del sistema: número de núcleos,
RAM total y tamaño del dataset.

## Sección 7: Experimento de Escalabilidad

Se construyen subconjuntos del dataset al 25%, 50%, 75% y 100% de las filas
originales usando .head(n_filas) tanto en Polars como en Pandas. Se corre
el pipeline completo (filtrado + group_by + join + feature engineering) en
cada subconjunto y se mide el tiempo total y el speedup. El objetivo es
observar cómo escala el beneficio de Polars conforme crece el volumen de datos,
dado que su paralelismo interno aprovecha mejor los múltiples núcleos con más trabajo.

## Sección 8: Experimento de Lazy Execution

Se comparan dos modos de procesamiento en Polars sobre el mismo pipeline
(filtrado + group_by):

- **Eager** (pl.read_csv): lee todo el CSV inmediatamente a memoria y
  ejecuta cada operación al momento de ser llamada.
- **Lazy** (pl.scan_csv(...).collect()): construye un plan de ejecución
  sin ejecutar nada. Polars optimiza el plan completo antes de materializarlo
  con .collect(), pudiendo aplicar el filtro durante la lectura y eliminar
  columnas innecesarias antes de cargarlas.

El uso de memoria se mide con psutil.Process().memory_info().rss antes y
después de cada modo, ya que tracemalloc no captura la memoria gestionada
por Rust (el lenguaje interno de Polars).

# Ejecución en el Kabré (CeNAT)

El notebook está pensado para ejecutarse en el clúster Kabré del CeNAT de
Costa Rica usando Jupyter Notebook interactivo. La configuración recomendada
para reproducir los resultados es:

- Cola: nukwa
- Nodos: 1
- Cores: 8
- Memoria: 16G

Con esta configuración Polars puede paralelizar sus operaciones sobre múltiples
núcleos, lo que hace que el speedup sea visible y medible. Los experimentos
reportados en este README fueron ejecutados en un nodo con 24 núcleos y 30.8 GB
de RAM disponibles.

**Nota importante**: el Kabré no tiene acceso a internet general, pero sí
tiene acceso a la API de Kaggle, por lo que kagglehub.dataset_download()
funciona correctamente desde los nodos de cómputo.

Mejor modelo: **XGBoost** con AUC de 0.7681, siendo además el más rápido
en entrenar (0.17s). 
