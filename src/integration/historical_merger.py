"""
historical_merger.py
====================
Módulo de integración histórica SNIES con PySpark.

Responsabilidades:
- Orquestar el pipeline de preprocessing por cada archivo preparado.
- Unir todos los años de una categoría en un único DataFrame histórico.
- Manejar caso especial: graduados 2001-2017 (filtrar 2013-2017, explotar por año).
- Garantizar consistencia de schema entre años antes de unir.
- Devolver DataFrame histórico completo por categoría.

Flujo por archivo:
    prepared .xlsx
        → clean_columns
        → normalize_values
        → handle_nulls
        → validate_schema
        → union histórico

Caso especial graduados:
    estudiantes_graduados_2001-2017.xlsx contiene múltiples años en filas.
    Solo se filtran 2013-2017 y se integran al histórico normal.
"""

import logging
import sys
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.preprocessing.clean_columns import clean_columns, get_spark, PREPARED_PATH
from src.preprocessing.normalize_values import normalize_values
from src.preprocessing.handle_nulls import handle_nulls
from src.preprocessing.validate_schema import validate_schema
from src.ingestion.canonical_schemas import (
    CATEGORY_SCHEMAS,
    is_pivoted_file,
)

logger = logging.getLogger(__name__)

# Años válidos del proyecto
VALID_YEARS = set(range(2013, 2025))

# Año especial graduados histórico
GRADUADOS_SPECIAL_YEAR = "2001_2017"
GRADUADOS_CATEGORY = "graduados"


# ──────────────────────────────────────────────────────────────
# DESCUBRIMIENTO DE ARCHIVOS PREPARADOS
# ──────────────────────────────────────────────────────────────

def discover_prepared_by_category(category: str) -> list:
    """
    Devuelve lista de {path, year} para una categoría en data/prepared/.
    Ordena por año ascendente.
    """
    category_path = PREPARED_PATH / category
    if not category_path.exists():
        logger.warning(f"Categoría no encontrada en prepared: {category}")
        return []

    files = []
    for year_dir in sorted(category_path.iterdir()):
        if not year_dir.is_dir():
            continue
        year = year_dir.name.replace("year=", "")
        for file_path in year_dir.glob("*.xlsx"):
            files.append({"path": file_path, "year": year})

    return files


# ──────────────────────────────────────────────────────────────
# CASO ESPECIAL: GRADUADOS 2001-2017
# ──────────────────────────────────────────────────────────────

def process_graduados_historical(
    spark: SparkSession,
    file_meta: dict,
) -> DataFrame:
    """
    Procesa el archivo histórico de graduados 2001-2017.

    Pasos:
    1. Aplicar pipeline normal de preprocessing.
    2. Filtrar solo años 2013-2017.
    3. Verificar que la columna AÑO tenga valores válidos.

    Returns:
        DataFrame con graduados 2013-2017 en formato estándar.
    """
    logger.info("  [ESPECIAL] Procesando graduados histórico 2001-2017")

    sdf = clean_columns(spark, file_meta["path"], GRADUADOS_CATEGORY, "2001_2017")
    if sdf is None:
        return None

    sdf = normalize_values(sdf, GRADUADOS_CATEGORY)
    sdf = handle_nulls(sdf, GRADUADOS_CATEGORY)
    sdf, report = validate_schema(sdf, GRADUADOS_CATEGORY, "2001_2017")

    # Filtrar solo 2013-2017
    if "AÑO" in sdf.columns:
        #sdf = sdf.filter(F.col("AÑO").between(2013, 2017))
        sdf = sdf.filter(F.col("AÑO").cast("int").between(2013, 2017))
        count = sdf.count()
        logger.info(f"  Graduados 2013-2017 después de filtro: {count} filas")
    else:
        logger.error("  Columna AÑO no encontrada en graduados histórico — no se puede filtrar")
        return None

    return sdf


# ──────────────────────────────────────────────────────────────
# PIPELINE POR ARCHIVO
# ──────────────────────────────────────────────────────────────

def process_single_file(
    spark: SparkSession,
    file_meta: dict,
    category: str,
) -> tuple:
    """
    Aplica el pipeline completo de preprocessing a un archivo.

    Returns:
        Tuple (DataFrame procesado, reporte de validación)
        DataFrame es None si el archivo fue saltado.
    """
    path = file_meta["path"]
    year = file_meta["year"]

    logger.info(f"\n  Procesando: [{category}] año={year} | {path.name}")

    # Caso especial graduados histórico
    if category == GRADUADOS_CATEGORY and year == GRADUADOS_SPECIAL_YEAR:
        sdf = process_graduados_historical(spark, file_meta)
        return sdf, {"category": category, "year": year, "special": True}

    # Saltar archivos pivoteados
    if is_pivoted_file(category, year):
        logger.warning(f"  [SKIP] Archivo pivoteado: {category} {year}")
        return None, {"category": category, "year": year, "skipped": "pivoted"}

    try:
        sdf = clean_columns(spark, path, category, year)
        if sdf is None:
            return None, {"category": category, "year": year, "skipped": "clean_failed"}

        sdf = normalize_values(sdf, category)
        sdf = handle_nulls(sdf, category)
        sdf, report = validate_schema(sdf, category, year)

        return sdf, report

    except Exception as e:
        logger.error(f"  [ERROR] {category} {year}: {e}")
        return None, {"category": category, "year": year, "error": str(e)}


# ──────────────────────────────────────────────────────────────
# INTEGRACIÓN HISTÓRICA POR CATEGORÍA
# ──────────────────────────────────────────────────────────────

def merge_category(
    spark: SparkSession,
    category: str,
) -> tuple:
    """
    Integra todos los años de una categoría en un único DataFrame histórico.

    Pasos:
    1. Descubrir archivos preparados de la categoría.
    2. Procesar cada archivo con el pipeline completo.
    3. Hacer union progresiva de DataFrames.
    4. Agregar columna _category al resultado final.

    Args:
        spark: SparkSession activa.
        category: Categoría SNIES a integrar.

    Returns:
        Tuple (DataFrame histórico completo, lista de reportes de validación)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"INTEGRANDO CATEGORÍA: {category.upper()}")
    logger.info(f"{'='*60}")

    files = discover_prepared_by_category(category)

    if not files:
        logger.warning(f"Sin archivos para categoría: {category}")
        return None, []

    logger.info(f"Archivos encontrados: {len(files)}")

    historical_df = None
    reports = []
    processed = 0
    skipped = 0
    errors = 0

    for file_meta in files:
        sdf, report = process_single_file(spark, file_meta, category)
        reports.append(report)

        if sdf is None:
            if report.get("skipped"):
                skipped += 1
            else:
                errors += 1
            continue

        # Union progresiva — allowMissingColumns para tolerancia entre años
        if historical_df is None:
            historical_df = sdf
        else:
            historical_df = historical_df.unionByName(sdf, allowMissingColumns=True)

        processed += 1

    logger.info(f"\nResumen {category}:")
    logger.info(f"  ✓ Procesados: {processed}")
    logger.info(f"  → Saltados:   {skipped}")
    logger.info(f"  ✗ Errores:    {errors}")

    if historical_df is not None:
        total_rows = historical_df.count()
        logger.info(f"  Total filas históricas: {total_rows:,}")
        logger.info(f"  Columnas: {len(historical_df.columns)}")

    return historical_df, reports


# ──────────────────────────────────────────────────────────────
# INTEGRACIÓN COMPLETA — TODAS LAS CATEGORÍAS
# ──────────────────────────────────────────────────────────────

def merge_all_categories(
    spark: SparkSession,
    categories: list = None,
) -> dict:
    """
    Integra históricamente todas las categorías SNIES.

    Args:
        spark: SparkSession activa.
        categories: Lista de categorías a procesar.
                    Default: todas las definidas en CATEGORY_SCHEMAS.

    Returns:
        Dict {category: DataFrame histórico}
    """
    if categories is None:
        categories = list(CATEGORY_SCHEMAS.keys())

    logger.info(f"\nCategorías a integrar: {categories}")

    result = {}
    all_reports = {}

    for category in categories:
        df, reports = merge_category(spark, category)
        if df is not None:
            result[category] = df
            all_reports[category] = reports

    logger.info(f"\n{'='*60}")
    logger.info("INTEGRACIÓN HISTÓRICA COMPLETADA")
    logger.info(f"{'='*60}")
    for cat, df in result.items():
        logger.info(f"  {cat}: {df.count():,} filas")

    return result


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Integración histórica SNIES por categoría."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Categorías a procesar. Default: todas.",
    )
    args = parser.parse_args()

    spark = get_spark("SNIES_HistoricalMerger")
    dfs = merge_all_categories(spark, categories=args.categories)

    logger.info(f"\nDataFrames listos para storage: {list(dfs.keys())}")