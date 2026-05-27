"""
clean_columns.py
================
Primera etapa de preprocessing con PySpark.

Responsabilidades:
- Leer archivos preparados (data/prepared/) con PySpark.
- Renombrar columnas a sus nombres canónicos usando canonical_schemas.
- Eliminar columnas completamente irrelevantes (pivoteadas, totales históricos).
- Agregar columnas de metadata (category, source_year) si no existen.
- Devolver DataFrame limpio listo para normalize_values.py

Nota: pandas solo para leer Excel → convertir a Spark DF.
      Todo el procesamiento desde aquí es PySpark.
"""

import pandas as pd
import logging
import sys
import os
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.ingestion.canonical_schemas import (
    resolve_canonical,
    rename_columns_to_canonical,
    is_pivoted_file,
    get_canonical_columns,
    NULLABLE_COLUMNS,
)

logger = logging.getLogger(__name__)

PREPARED_PATH = Path("data/prepared")

# Patrones de columnas basura detectadas en schema_report
# (columnas pivoteadas tipo "Hombre 2009-2", "Total 2013-1*")
JUNK_COLUMN_PATTERNS = [
    r"^Hombre \d{4}",
    r"^Mujer \d{4}",
    r"^Total \d{4}",
    r"^Admisiones \d{4}",
]


def get_spark(app_name: str = "SNIES_Preprocessing") -> SparkSession:
    """Inicializa o reutiliza SparkSession en modo local."""

    #os.environ["HADOOP_HOME"] = "C:\\hadoop"
    #os.environ["PATH"] = os.environ["PATH"] + ";C:\\hadoop\\bin"

    os.environ["HADOOP_HOME"] = r"C:\\hadoop"
    os.environ["hadoop.home.dir"] = r"C:\\hadoop"
    
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def _is_junk_column(col: str) -> bool:
    """Detecta columnas pivoteadas/basura por patrón."""
    import re
    for pattern in JUNK_COLUMN_PATTERNS:
        if re.match(pattern, col.strip()):
            return True
    return False


def read_prepared_to_spark(
    spark: SparkSession,
    file_path: Path,
    category: str,
    year: str,
) -> DataFrame:
    """
    Lee un archivo Excel preparado con pandas y lo convierte a Spark DataFrame.

    Args:
        spark: SparkSession activa.
        file_path: Ruta al archivo .xlsx en data/prepared/.
        category: Categoría del archivo.
        year: Año del archivo.

    Returns:
        Spark DataFrame con datos crudos del archivo.
    """
    logger.info(f"  Leyendo: {file_path.name}")

    # pandas solo para leer Excel — luego inmediatamente a Spark
    pdf = pd.read_excel(file_path, engine="openpyxl", dtype=str)

    # Convertir a Spark DF
    sdf = spark.createDataFrame(pdf)

    # Agregar metadata si no vienen en el archivo
    if "AÑO" not in [resolve_canonical(c) for c in sdf.columns]:
        sdf = sdf.withColumn("AÑO", F.lit(str(year)))

    sdf = sdf.withColumn("_category", F.lit(category))
    sdf = sdf.withColumn("_source_year", F.lit(str(year)))

    return sdf


def drop_junk_columns(sdf: DataFrame) -> DataFrame:
    """
    Elimina columnas pivoteadas/basura detectadas por patrón.
    Loguea cada columna eliminada.
    """
    cols_to_drop = [col for col in sdf.columns if _is_junk_column(col)]

    if cols_to_drop:
        logger.warning(f"  Eliminando {len(cols_to_drop)} columnas basura: {cols_to_drop[:5]}...")
        sdf = sdf.drop(*cols_to_drop)

    return sdf


def apply_canonical_renaming(sdf: DataFrame) -> DataFrame:
    """
    Renombra todas las columnas del DataFrame a sus nombres canónicos
    usando COLUMN_ALIAS_MAP de canonical_schemas.

    Columnas sin mapeo se conservan con advertencia.
    """
    rename_map = rename_columns_to_canonical(sdf.columns)

    for original, canonical in rename_map.items():
        logger.debug(f"  Renombrando: '{original}' → '{canonical}'")
        sdf = sdf.withColumnRenamed(original, canonical)

    if rename_map:
        logger.info(f"  Columnas renombradas: {len(rename_map)}")

    return sdf

def sanitize_column_names(sdf: DataFrame) -> DataFrame:
    """Reemplaza puntos en nombres de columnas — Spark los interpreta como separadores."""
    for col in sdf.columns:
        if "." in col:
            new_name = col.replace(".", "")
            sdf = sdf.withColumnRenamed(col, new_name)
    return sdf

def add_missing_nullable_columns(sdf: DataFrame, category: str) -> DataFrame:
    """
    Agrega columnas nullable que no existen en el archivo
    (ej. columnas CINE ausentes en años anteriores a 2019).
    Las rellena con null para mantener schema consistente.
    """
    canonical_cols = get_canonical_columns(category)
    existing = set(sdf.columns)

    for col in canonical_cols:
        if col not in existing and col in NULLABLE_COLUMNS:
            sdf = sdf.withColumn(col, F.lit(None).cast("string"))
            logger.debug(f"  Columna nullable agregada como null: '{col}'")

    return sdf


def clean_columns(
    spark: SparkSession,
    file_path: Path,
    category: str,
    year: str,
) -> DataFrame:
    """
    Pipeline completo de limpieza de columnas para un archivo.

    Pasos:
    1. Leer Excel preparado → Spark DF
    2. Eliminar columnas basura/pivoteadas
    3. Renombrar a canónico
    4. Agregar columnas nullable faltantes como null
    5. Retornar DF limpio

    Args:
        spark: SparkSession activa.
        file_path: Ruta al archivo preparado.
        category: Categoría SNIES.
        year: Año del archivo.

    Returns:
        Spark DataFrame con columnas canónicas limpias.
    """
    if is_pivoted_file(category, str(year)):
        logger.warning(
            f"  [SKIP] Archivo pivoteado detectado: {category} {year}. "
            f"Requiere tratamiento especial — omitido en este pipeline."
        )
        return None

    sdf = read_prepared_to_spark(spark, file_path, category, year)
    sdf = drop_junk_columns(sdf)
    sdf = apply_canonical_renaming(sdf)
    sdf = sanitize_column_names(sdf)
    sdf = add_missing_nullable_columns(sdf, category)

    logger.info(f"  Columnas finales: {len(sdf.columns)} | Filas: {sdf.count()}")
    return sdf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    spark = get_spark()

    # Test rápido sobre un archivo
    test_file = next(PREPARED_PATH.rglob("*.xlsx"), None)
    if test_file:
        parts = test_file.parts
        year = parts[-2].replace("year=", "")
        category = parts[-3]
        sdf = clean_columns(spark, test_file, category, year)
        if sdf:
            sdf.printSchema()
            sdf.show(5, truncate=True)