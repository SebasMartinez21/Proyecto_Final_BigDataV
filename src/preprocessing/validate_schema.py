"""
validate_schema.py
==================
Cuarta etapa de preprocessing con PySpark.

Responsabilidades:
- Validar que el DataFrame tenga las columnas canónicas esperadas.
- Reportar columnas faltantes requeridas vs nullable.
- Reportar columnas inesperadas.
- Reordenar columnas en el orden canónico definido.
- Generar reporte de validación por archivo.
- NO bloquea el pipeline — advierte y continúa.
"""

import logging
import sys
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.ingestion.canonical_schemas import (
    validate_schema as _validate_schema_meta,
    get_canonical_columns,
    is_nullable,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# VALIDACIÓN
# ──────────────────────────────────────────────────────────────

def validate_and_report(sdf: DataFrame, category: str, year: str) -> dict:
    """
    Valida el schema del DataFrame contra el canónico.

    Returns:
        Dict con resultados de validación:
        - valid: bool (True si no hay columnas requeridas faltantes)
        - missing_required: columnas canónicas requeridas ausentes
        - missing_nullable: canónicas nullable ausentes (informativo)
        - unexpected: columnas presentes fuera del schema canónico
    """
    result = _validate_schema_meta(list(sdf.columns), category)
    result["category"] = category
    result["year"] = year
    result["valid"] = len(result["missing_required"]) == 0

    if result["missing_required"]:
        logger.error(
            f"  [SCHEMA ERROR] {category} {year} — "
            f"Columnas requeridas faltantes: {result['missing_required']}"
        )
    if result["missing_nullable"]:
        logger.info(
            f"  [INFO] {category} {year} — "
            f"Columnas nullable ausentes (normal en años anteriores): "
            f"{result['missing_nullable']}"
        )
    if result["unexpected"]:
        logger.warning(
            f"  [WARN] {category} {year} — "
            f"Columnas fuera del schema canónico: {result['unexpected']}"
        )

    if result["valid"]:
        logger.info(f"  [OK] Schema válido: {category} {year}")

    return result


# ──────────────────────────────────────────────────────────────
# REORDENAMIENTO Y ALINEACIÓN
# ──────────────────────────────────────────────────────────────

def align_to_canonical_schema(sdf: DataFrame, category: str) -> DataFrame:
    """
    Reordena y alinea el DataFrame al orden canónico definido.

    - Columnas canónicas presentes → en orden canónico.
    - Columnas canónicas nullable ausentes → se agregan como null.
    - Columnas fuera del schema → se colocan al final (no se descartan).
    - Columnas de metadata (_category, _source_year) → al final siempre.

    Returns:
        Spark DataFrame con columnas en orden canónico.
    """
    canonical_cols = get_canonical_columns(category)
    existing = set(sdf.columns)

    # Columnas de metadata internas
    meta_cols = [c for c in sdf.columns if c.startswith("_")]

    # Columnas canónicas presentes
    ordered = []
    for col in canonical_cols:
        if col in existing:
            ordered.append(F.col(col))
        elif is_nullable(col):
            # Agregar como null si es nullable y no existe
            ordered.append(F.lit(None).cast(StringType()).alias(col))
        # Si es requerida y no existe → ya fue reportada en validate_and_report

    # Columnas inesperadas (fuera del schema) → al final
    unexpected = [
        c for c in sdf.columns
        if c not in canonical_cols and not c.startswith("_")
    ]
    for col in unexpected:
        ordered.append(F.col(col))

    # Metadata al final
    for col in meta_cols:
        ordered.append(F.col(col))

    return sdf.select(ordered)


# ──────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ──────────────────────────────────────────────────────────────

def validate_schema(sdf: DataFrame, category: str, year: str) -> tuple:
    """
    Pipeline completo de validación y alineación de schema.

    Pasos:
    1. Validar columnas contra schema canónico.
    2. Reportar diferencias.
    3. Alinear y reordenar columnas al orden canónico.

    Args:
        sdf: Spark DataFrame (salida de handle_nulls).
        category: Categoría SNIES.
        year: Año del archivo.

    Returns:
        Tuple (DataFrame alineado, dict reporte de validación)
        El DataFrame se retorna siempre, incluso si hay errores de schema.
    """
    logger.info(f"  Validando schema: {category} {year}")

    report = validate_and_report(sdf, category, year)
    sdf_aligned = align_to_canonical_schema(sdf, category)

    return sdf_aligned, report