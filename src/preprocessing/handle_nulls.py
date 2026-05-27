"""
handle_nulls.py
===============
Tercera etapa de preprocessing con PySpark.

Responsabilidades:
- Definir estrategia de manejo de nulos por tipo de columna.
- Columnas de ID/código: conservar null (no imputar).
- Columnas de texto dimensional: imputar "DESCONOCIDO" donde aplique.
- Columnas métricas: imputar 0 si null (ausencia = sin registro).
- Columnas CINE (nullable históricamente): conservar null.
- Generar reporte de nulos antes y después.
"""

import logging
import sys
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.ingestion.canonical_schemas import (
    get_metric_column,
    NULLABLE_COLUMNS,
    get_canonical_columns,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# ESTRATEGIAS POR GRUPO DE COLUMNAS
# ──────────────────────────────────────────────────────────────

# Columnas que NUNCA se imputan — null es información válida
NEVER_IMPUTE = NULLABLE_COLUMNS | {
    "CÓDIGO DE LA INSTITUCIÓN",
    "CÓDIGO SNIES DEL PROGRAMA",
    "CÓDIGO DEL DEPARTAMENTO (IES)",
    "CÓDIGO DEL MUNICIPIO IES",
    "CÓDIGO DEL DEPARTAMENTO (PROGRAMA)",
    "CÓDIGO DEL MUNICIPIO (PROGRAMA)",
    "ID SECTOR IES",
    "ID CARÁCTER IES",
    "ID NIVEL ACADÉMICO",
    "ID NIVEL DE FORMACIÓN",
    "ID MODALIDAD",
    "ID ÁREA",
    "ID NÚCLEO",
    "ID SEXO",
    "AÑO",
    "SEMESTRE",
}

# Columnas de texto dimensional que se imputan con "DESCONOCIDO"
IMPUTE_UNKNOWN = {
    "INSTITUCIÓN DE EDUCACIÓN SUPERIOR (IES)",
    "SECTOR IES",
    "CARÁCTER IES",
    "DEPARTAMENTO DE DOMICILIO DE LA IES",
    "MUNICIPIO DE DOMICILIO DE LA IES",
    "DEPARTAMENTO DE OFERTA DEL PROGRAMA",
    "MUNICIPIO DE OFERTA DEL PROGRAMA",
    "PROGRAMA ACADÉMICO",
    "NIVEL ACADÉMICO",
    "NIVEL DE FORMACIÓN",
    "MODALIDAD",
    "ÁREA DE CONOCIMIENTO",
    "NÚCLEO BÁSICO DEL CONOCIMIENTO (NBC)",
    "SEXO",
    "TIPO IES",
    "CARÁCTER IES",
    "PRINCIPAL O SECCIONAL",
}


# ──────────────────────────────────────────────────────────────
# REPORTE DE NULOS
# ──────────────────────────────────────────────────────────────

def null_report(sdf: DataFrame, label: str = "") -> dict:
    """
    Genera un reporte de nulos por columna.

    Returns:
        Dict {col: null_count} ordenado de mayor a menor.
    """
    total = sdf.count()
    null_counts = {}

    # Calcular nulos en una sola pasada (eficiente en Spark)
    agg_exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in sdf.columns]
    result = sdf.agg(*agg_exprs).collect()[0].asDict()

    for col, count in result.items():
        if count and count > 0:
            null_counts[col] = {
                "null_count": count,
                "null_pct": round(count / total * 100, 2) if total > 0 else 0,
            }

    null_counts = dict(sorted(null_counts.items(), key=lambda x: -x[1]["null_count"]))

    if label:
        logger.info(f"\n  Nulos [{label}] — Total filas: {total}")
        for col, stats in null_counts.items():
            logger.info(f"    {col}: {stats['null_count']} ({stats['null_pct']}%)")

    return null_counts


# ──────────────────────────────────────────────────────────────
# ESTRATEGIAS DE IMPUTACIÓN
# ──────────────────────────────────────────────────────────────

def impute_unknown_text(sdf: DataFrame) -> DataFrame:
    """Imputa 'DESCONOCIDO' en columnas de texto dimensional con nulos."""
    for col in IMPUTE_UNKNOWN:
        if col in sdf.columns:
            sdf = sdf.withColumn(
                col,
                F.when(F.col(col).isNull(), F.lit("DESCONOCIDO"))
                 .otherwise(F.col(col))
            )
    return sdf


def impute_metric_zeros(sdf: DataFrame, category: str) -> DataFrame:
    """
    Imputa 0 en la columna métrica si es null.
    Null en métrica = sin registro = 0 válido para agregaciones.
    """
    metric_col = get_metric_column(category)
    if metric_col in sdf.columns:
        sdf = sdf.withColumn(
            metric_col,
            F.when(F.col(metric_col).isNull(), F.lit(0).cast(LongType()))
             .otherwise(F.col(metric_col))
        )
    return sdf


def drop_rows_without_key_columns(sdf: DataFrame) -> DataFrame:
    """
    Elimina filas donde las columnas clave son null.
    Sin institución y sin programa no hay registro útil.
    """
    key_cols = ["CÓDIGO DE LA INSTITUCIÓN", "CÓDIGO SNIES DEL PROGRAMA"]
    existing_keys = [c for c in key_cols if c in sdf.columns]

    if not existing_keys:
        return sdf

    before = sdf.count()
    condition = F.lit(True)
    for col in existing_keys:
        condition = condition & F.col(col).isNotNull()

    sdf = sdf.filter(condition)
    after = sdf.count()

    if before != after:
        logger.warning(f"  Filas eliminadas por claves nulas: {before - after}")

    return sdf


# ──────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ──────────────────────────────────────────────────────────────

def handle_nulls(sdf: DataFrame, category: str, verbose: bool = False) -> DataFrame:
    """
    Pipeline completo de manejo de nulos.

    Estrategia:
    - Eliminar filas sin clave de institución/programa.
    - Imputar 'DESCONOCIDO' en texto dimensional.
    - Imputar 0 en métrica.
    - Conservar null en IDs, códigos y columnas CINE históricas.

    Args:
        sdf: Spark DataFrame normalizado (salida de normalize_values).
        category: Categoría SNIES.
        verbose: Si True, imprime reporte de nulos antes y después.

    Returns:
        Spark DataFrame con nulos manejados.
    """
    logger.info(f"  Manejando nulos: {category}")

    if verbose:
        null_report(sdf, label="ANTES")

    sdf = drop_rows_without_key_columns(sdf)
    sdf = impute_unknown_text(sdf)
    sdf = impute_metric_zeros(sdf, category)

    if verbose:
        null_report(sdf, label="DESPUÉS")

    logger.info(f"  Manejo de nulos completado.")
    return sdf