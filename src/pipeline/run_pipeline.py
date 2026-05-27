"""
run_pipeline.py
===============
Orquestador principal del pipeline SNIES.

Ejecuta en orden:
    1. [Opcional] Ingesta y prelimpieza (prepare_raw_files)
    2. Integración histórica por categoría (historical_merger)
    3. Escritura Data Lake Parquet (parquet_writer)
    4. Verificación de integridad del lake

Diseñado para:
- Ejecutarse con un solo comando.
- Soportar años futuros (2025+) sin modificar código.
- Permitir ejecución parcial por etapa o categoría.
- Loguear todo en consola y archivo de log.

Uso:
    # Pipeline completo
    python src/pipeline/run_pipeline.py

    # Solo integración y storage (ingesta ya hecha)
    python src/pipeline/run_pipeline.py --skip-ingestion

    # Solo una categoría
    python src/pipeline/run_pipeline.py --categories admitidos graduados

    # Años nuevos (append al lake existente)
    python src/pipeline/run_pipeline.py --mode append --skip-ingestion
"""

import logging
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ingestion.prepare_raw_files import prepare_all_files
from src.ingestion.canonical_schemas import CATEGORY_SCHEMAS
from src.integration.historical_merger import merge_all_categories
from src.storage.parquet_writer import write_all_to_lake, verify_lake_integrity
from src.preprocessing.clean_columns import get_spark

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────

LOGS_PATH = Path("logs")

def setup_logging():
    LOGS_PATH.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_PATH / f"pipeline_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return log_file


# ──────────────────────────────────────────────────────────────
# ETAPAS
# ──────────────────────────────────────────────────────────────

def run_ingestion(force: bool = False):
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("ETAPA 1 — INGESTA Y PRELIMPIEZA")
    logger.info("="*60)
    t0 = time.time()
    results = prepare_all_files(force=force)
    success = sum(1 for r in results if r.get("status") == "success")
    logger.info(f"Ingesta completada en {time.time()-t0:.1f}s — {success} archivos preparados")
    return results


def run_integration(spark, categories, mode):
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("ETAPA 2 — INTEGRACIÓN HISTÓRICA")
    logger.info("="*60)
    t0 = time.time()
    category_dfs = merge_all_categories(spark, categories=categories)
    logger.info(f"Integración completada en {time.time()-t0:.1f}s")
    return category_dfs


def run_storage(category_dfs, mode):
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("ETAPA 3 — ESCRITURA DATA LAKE (PARQUET)")
    logger.info("="*60)
    t0 = time.time()
    results = write_all_to_lake(category_dfs, mode=mode)
    logger.info(f"Storage completado en {time.time()-t0:.1f}s")
    return results


def run_verification(spark):
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*60)
    logger.info("ETAPA 4 — VERIFICACIÓN DATA LAKE")
    logger.info("="*60)
    return verify_lake_integrity(spark)


# ──────────────────────────────────────────────────────────────
# ORQUESTADOR
# ──────────────────────────────────────────────────────────────

def run_pipeline(
    skip_ingestion: bool = False,
    categories: list = None,
    mode: str = "overwrite",
    force_ingestion: bool = False,
):
    logger = logging.getLogger(__name__)
    log_file = setup_logging()

    logger.info("="*60)
    logger.info("PIPELINE SNIES — INICIO")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Categorías: {categories or 'todas'}")
    logger.info(f"Modo storage: {mode}")
    logger.info(f"Log: {log_file}")
    logger.info("="*60)

    t_total = time.time()

    # ── Etapa 1: Ingesta
    if not skip_ingestion:
        run_ingestion(force=force_ingestion)
    else:
        logger.info("\n[SKIP] Ingesta omitida — usando data/prepared/ existente")

    # ── Spark Session
    logger.info("\nInicializando SparkSession...")
    spark = get_spark("SNIES_Pipeline")
    logger.info("SparkSession lista.")

    # ── Etapa 2: Integración
    category_dfs = run_integration(spark, categories, mode)

    if not category_dfs:
        logger.error("No se generaron DataFrames. Revisa data/prepared/")
        return

    # ── Etapa 3: Storage
    run_storage(category_dfs, mode)

    # ── Etapa 4: Verificación
    run_verification(spark)

    # ── Resumen final
    elapsed = time.time() - t_total
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETADO")
    logger.info(f"Tiempo total: {elapsed/60:.1f} minutos")
    logger.info(f"Categorías procesadas: {list(category_dfs.keys())}")
    logger.info("="*60)

    spark.stop()


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline principal SNIES.")

    parser.add_argument(
        "--skip-ingestion", action="store_true",
        help="Saltar etapa de ingesta (usar data/prepared/ existente).",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Categorías a procesar. Default: todas.",
    )
    parser.add_argument(
        "--mode", choices=["overwrite", "append"], default="overwrite",
        help="Modo de escritura Parquet. Default: overwrite.",
    )
    parser.add_argument(
        "--force-ingestion", action="store_true",
        help="Forzar reprocesamiento de ingesta aunque ya existan archivos.",
    )

    args = parser.parse_args()

    run_pipeline(
        skip_ingestion=args.skip_ingestion,
        categories=args.categories,
        mode=args.mode,
        force_ingestion=args.force_ingestion,
    )