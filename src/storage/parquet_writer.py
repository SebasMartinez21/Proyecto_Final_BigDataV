"""
parquet_writer.py
=================
Módulo de escritura del Data Lake en formato Parquet.

Responsabilidades:
- Escribir DataFrames históricos por categoría en data/lake/.
- Particionar por categoría y año (para consultas eficientes).
- Soportar modo append (nuevos años) y overwrite (reprocesamiento).
- Verificar integridad de archivos escritos.
- Ser totalmente automático para años futuros 2025+.

Estructura del Data Lake:
    data/lake/
        admitidos/year=2013/part-*.parquet
        admitidos/year=2014/part-*.parquet
        ...
        graduados/year=2013/part-*.parquet
"""

import logging
import sys
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.preprocessing.clean_columns import get_spark

logger = logging.getLogger(__name__)

LAKE_PATH = Path("data/lake")


# ──────────────────────────────────────────────────────────────
# ESCRITURA PARQUET
# ──────────────────────────────────────────────────────────────

def write_category_to_lake(
    sdf: DataFrame,
    category: str,
    mode: str = "overwrite",
) -> dict:
    """
    Escribe el DataFrame histórico de una categoría al Data Lake en Parquet,
    particionado por AÑO.

    Args:
        sdf: DataFrame histórico completo de la categoría.
        category: Nombre de la categoría (ej. 'admitidos').
        mode: 'overwrite' para reprocesar, 'append' para años nuevos.

    Returns:
        Dict con resultado: {category, path, n_rows, n_partitions, status}
    """
    output_path = LAKE_PATH / category
    result = {
        "category": category,
        "path": str(output_path),
        "status": None,
        "n_rows": None,
        "n_partitions": None,
        "error": None,
    }

    try:
        # Eliminar columnas internas de metadata antes de escribir
        meta_cols = [c for c in sdf.columns if c.startswith("_")]
        if meta_cols:
            sdf = sdf.drop(*meta_cols)

        # Asegurar que AÑO existe para particionar
        if "AÑO" not in sdf.columns:
            logger.error(f"  [ERROR] Columna AÑO no encontrada en {category} — no se puede particionar")
            result["status"] = "error"
            result["error"] = "Columna AÑO ausente"
            return result

        n_rows = sdf.count()

        # Contar particiones (años únicos)
        #years = [r["AÑO"] for r in sdf.select("AÑO").distinct().collect()]
        years = [r["AÑO"] for r in sdf.select("AÑO").distinct().collect() if r["AÑO"] is not None]
        n_partitions = len(years)

        logger.info(f"\n  Escribiendo {category} → {output_path}")
        logger.info(f"  Filas: {n_rows:,} | Años: {sorted(years)} | Modo: {mode}")

        (
            sdf
            .write
            .mode(mode)
            .partitionBy("AÑO")
            .parquet(str(output_path))
        )

        result["status"] = "success"
        result["n_rows"] = n_rows
        result["n_partitions"] = n_partitions

        logger.info(f"  [OK] {category} escrito en Parquet — {n_partitions} particiones")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"  [ERROR] {category}: {e}")

    return result


def write_all_to_lake(
    category_dfs: dict,
    mode: str = "overwrite",
) -> list:
    """
    Escribe todos los DataFrames históricos al Data Lake.

    Args:
        category_dfs: Dict {category: DataFrame} (salida de historical_merger).
        mode: 'overwrite' o 'append'.

    Returns:
        Lista de resultados por categoría.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"ESCRIBIENDO DATA LAKE — Modo: {mode.upper()}")
    logger.info(f"{'='*60}")

    LAKE_PATH.mkdir(parents=True, exist_ok=True)
    results = []

    for category, sdf in category_dfs.items():
        result = write_category_to_lake(sdf, category, mode=mode)
        results.append(result)

    # Resumen
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    total_rows = sum(r["n_rows"] or 0 for r in results)

    logger.info(f"\n{'='*60}")
    logger.info(f"DATA LAKE — RESUMEN")
    logger.info(f"  ✓ Categorías escritas: {success}")
    logger.info(f"  ✗ Errores:             {errors}")
    logger.info(f"  Total filas escritas:  {total_rows:,}")
    logger.info(f"  Ubicación:             {LAKE_PATH.resolve()}")
    logger.info(f"{'='*60}")

    return results


# ──────────────────────────────────────────────────────────────
# LECTURA DESDE LAKE (para módulos posteriores)
# ──────────────────────────────────────────────────────────────

def read_category_from_lake(
    spark: SparkSession,
    category: str,
    years: list = None,
) -> DataFrame:
    """
    Lee una categoría del Data Lake con filtro opcional por años.

    Args:
        spark: SparkSession activa.
        category: Categoría a leer.
        years: Lista de años a filtrar (ej. [2020, 2021]).
               None = leer todos los años disponibles.

    Returns:
        Spark DataFrame con datos del lake.
    """
    lake_path = LAKE_PATH / category

    if not lake_path.exists():
        raise FileNotFoundError(
            f"Categoría '{category}' no encontrada en Data Lake: {lake_path}"
        )

    sdf = spark.read.parquet(str(lake_path))

    if years:
        sdf = sdf.filter(F.col("AÑO").isin(years))
        logger.info(f"  Leyendo {category} años {years}: {sdf.count():,} filas")
    else:
        logger.info(f"  Leyendo {category} completo: {sdf.count():,} filas")

    return sdf


def read_all_from_lake(
    spark: SparkSession,
    categories: list = None,
    years: list = None,
) -> dict:
    """
    Lee múltiples categorías del Data Lake.

    Returns:
        Dict {category: DataFrame}
    """
    if categories is None:
        # Descubrir categorías disponibles en el lake
        categories = [d.name for d in LAKE_PATH.iterdir() if d.is_dir()]

    result = {}
    for category in categories:
        try:
            result[category] = read_category_from_lake(spark, category, years=years)
        except FileNotFoundError as e:
            logger.warning(str(e))

    return result


# ──────────────────────────────────────────────────────────────
# VERIFICACIÓN DE INTEGRIDAD
# ──────────────────────────────────────────────────────────────

def verify_lake_integrity(spark: SparkSession) -> dict:
    """
    Verifica que el Data Lake esté completo y legible.

    Returns:
        Dict {category: {years, total_rows, status}}
    """
    logger.info("\nVerificando integridad del Data Lake...")
    report = {}

    for category_dir in sorted(LAKE_PATH.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        try:
            sdf = spark.read.parquet(str(category_dir))
            years = sorted([r["AÑO"] for r in sdf.select("AÑO").distinct().collect() if r["AÑO"] is not None])
            #years = sorted([r["AÑO"] for r in sdf.select("AÑO").distinct().collect()])
            #years = [r["AÑO"] for r in sdf.select("AÑO").distinct().collect() if r["AÑO"] is not None]
            total = sdf.count()
            report[category] = {
                "status": "ok",
                "years": years,
                "total_rows": total,
                "n_years": len(years),
            }
            logger.info(f"  ✓ {category}: {len(years)} años | {total:,} filas")
        except Exception as e:
            report[category] = {"status": "error", "error": str(e)}
            logger.error(f"  ✗ {category}: {e}")

    return report


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

    parser = argparse.ArgumentParser(description="Verificar integridad del Data Lake.")
    parser.add_argument("--verify", action="store_true", help="Solo verificar, no escribir.")
    args = parser.parse_args()

    spark = get_spark("SNIES_LakeVerifier")

    if args.verify:
        verify_lake_integrity(spark)