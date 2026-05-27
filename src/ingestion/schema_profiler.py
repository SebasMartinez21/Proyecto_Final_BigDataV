"""
schema_profiler.py
==================
Profiling de schemas sobre archivos ya preparados (data/prepared/).

Responsabilidades:
- Leer archivos limpios desde data/prepared/ (headers correctos garantizados).
- Extraer columnas reales por categoría y año.
- Normalizar nombres de columnas para comparación (tildes, mayúsculas, espacios).
- Detectar inconsistencias históricas de columnas entre años.
- Identificar columnas comunes (canónicas) por categoría.
- Generar reporte de schemas en JSON y log legible.
- Servir como base para construir schemas canónicos del ETL.

Nota: Lee Excel con pandas (openpyxl), el procesamiento analítico
posterior usará PySpark sobre los archivos preparados.
"""

import pandas as pd
import json
import re
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

# ──────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────

PREPARED_PATH = Path("data/prepared")
REPORTS_PATH = Path("data/processed/schema_reports")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Normalización de columnas
# ──────────────────────────────────────────────────────────────

# Mapeo de variantes históricas conocidas al nombre canónico.
# Se expande a medida que se detectan nuevas inconsistencias.
CANONICAL_COLUMN_MAP = {
    # Año
    "año": "AÑO",
    "ano": "AÑO",
    "año*": "AÑO",
    # Carácter IES
    "id caracter ies": "ID CARÁCTER IES",
    "id carácter ies": "ID CARÁCTER IES",
    "id caracter": "ID CARÁCTER IES",
    "id carácter": "ID CARÁCTER IES",
    # Código institución
    "código de la institución": "CÓDIGO DE LA INSTITUCIÓN",
    "codigo de la institucion": "CÓDIGO DE LA INSTITUCIÓN",
    # IES
    "institución de educación superior (ies)": "INSTITUCIÓN DE EDUCACIÓN SUPERIOR (IES)",
    "institucion de educacion superior (ies)": "INSTITUCIÓN DE EDUCACIÓN SUPERIOR (IES)",
    # Sector
    "sector ies": "SECTOR IES",
    # Código SNIES
    "código snies del programa": "CÓDIGO SNIES DEL PROGRAMA",
    "codigo snies del programa": "CÓDIGO SNIES DEL PROGRAMA",
    # Programa
    "programa académico": "PROGRAMA ACADÉMICO",
    "programa academico": "PROGRAMA ACADÉMICO",
    # Área
    "área de conocimiento": "ÁREA DE CONOCIMIENTO",
    "area de conocimiento": "ÁREA DE CONOCIMIENTO",
    # Núcleo
    "núcleo básico del conocimiento": "NÚCLEO BÁSICO DEL CONOCIMIENTO (NBC)",
    "nucleo basico del conocimiento": "NÚCLEO BÁSICO DEL CONOCIMIENTO (NBC)",
    "núcleo básico del conocimiento (nbc)": "NÚCLEO BÁSICO DEL CONOCIMIENTO (NBC)",
}


def normalize_column_name(col: str) -> str:
    """
    Normaliza un nombre de columna:
    1. Strip y colapsa espacios múltiples.
    2. Pasa a minúsculas para comparación.
    3. Elimina tildes.
    4. Busca en el mapa canónico.
    5. Si no hay match canónico, devuelve la versión con strip original.

    Returns:
        Nombre canónico si existe, original limpio si no.
    """
    if not col or pd.isna(col):
        return ""

    cleaned = str(col).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)  # Colapsar espacios múltiples

    # Versión normalizada para lookup
    normalized = cleaned.lower()
    replacements = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
    for acc, plain in replacements.items():
        normalized = normalized.replace(acc, plain)

    canonical = CANONICAL_COLUMN_MAP.get(normalized)
    if canonical:
        return canonical

    return cleaned


# ──────────────────────────────────────────────────────────────
# Descubrimiento de archivos preparados
# ──────────────────────────────────────────────────────────────

def discover_prepared_files() -> list:
    """
    Descubre todos los archivos en data/prepared/ siguiendo la estructura:
    data/prepared/<category>/year=<year>/<filename>.xlsx

    Returns:
        Lista de dicts {category, year, path, filename}
    """
    discovered = []

    if not PREPARED_PATH.exists():
        logger.error(f"Carpeta prepared no existe: {PREPARED_PATH}")
        return discovered

    for category_dir in PREPARED_PATH.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for year_dir in category_dir.iterdir():
            if not year_dir.is_dir():
                continue
            # Extraer año del nombre "year=XXXX"
            year_match = re.match(r"year=(.+)", year_dir.name)
            year = year_match.group(1) if year_match else year_dir.name

            for file_path in year_dir.glob("*.xlsx"):
                discovered.append({
                    "category": category,
                    "year": year,
                    "path": str(file_path),
                    "filename": file_path.name,
                })

    return discovered


# ──────────────────────────────────────────────────────────────
# Extracción de schema por archivo
# ──────────────────────────────────────────────────────────────

def extract_file_schema(file_meta: dict) -> Optional[dict]:
    """
    Lee un archivo preparado y extrae su schema:
    - Columnas originales (tal como vienen)
    - Columnas normalizadas (canónicas)
    - N filas, N columnas
    - Sample de valores por columna (para detectar tipos)

    Returns:
        Dict con schema, o None si el archivo falla.
    """
    path = Path(file_meta["path"])
    try:
        # Solo leer header + primeras filas para profiling (eficiente)
        df = pd.read_excel(path, engine="openpyxl", nrows=5, dtype=str)

        # Contar filas totales sin cargar todo en memoria
        df_full = pd.read_excel(path, engine="openpyxl", dtype=str, usecols=[0])
        n_rows = len(df_full)

        original_cols = list(df.columns)
        normalized_cols = [normalize_column_name(c) for c in original_cols]

        # Detectar columnas que no tienen nombre canónico (posibles inconsistencias)
        non_canonical = [
            orig for orig, norm in zip(original_cols, normalized_cols)
            if orig.strip() == norm and orig.lower() not in CANONICAL_COLUMN_MAP
        ]

        return {
            "category": file_meta["category"],
            "year": file_meta["year"],
            "filename": file_meta["filename"],
            "n_rows": n_rows,
            "n_cols": len(original_cols),
            "original_columns": original_cols,
            "normalized_columns": normalized_cols,
            "non_canonical_columns": non_canonical,
            "status": "ok",
        }

    except Exception as e:
        logger.error(f"  [ERROR] {file_meta['filename']}: {e}")
        return {
            "category": file_meta["category"],
            "year": file_meta["year"],
            "filename": file_meta["filename"],
            "status": "error",
            "error": str(e),
        }


# ──────────────────────────────────────────────────────────────
# Análisis de inconsistencias por categoría
# ──────────────────────────────────────────────────────────────

def analyze_category_schemas(schemas: list) -> dict:
    """
    Agrupa schemas por categoría y analiza:
    - Columnas comunes a todos los años (candidatas canónicas).
    - Columnas que aparecen en algunos años (variantes/inconsistencias).
    - Columnas únicas por año.

    Args:
        schemas: Lista de schemas extraídos por extract_file_schema.

    Returns:
        Dict {category: análisis}
    """
    by_category = defaultdict(list)
    for s in schemas:
        if s.get("status") == "ok":
            by_category[s["category"]].append(s)

    analysis = {}

    for category, cat_schemas in by_category.items():
        # Columnas normalizadas por año (set por año)
        cols_by_year = {}
        for s in cat_schemas:
            year = s["year"]
            cols_by_year[year] = set(s["normalized_columns"])

        all_col_sets = list(cols_by_year.values())

        if not all_col_sets:
            continue

        # Columnas presentes en TODOS los años → canónicas seguras
        common_columns = set.intersection(*all_col_sets) if all_col_sets else set()

        # Columnas presentes en ALGÚN año → variantes
        all_columns = set.union(*all_col_sets) if all_col_sets else set()
        variant_columns = all_columns - common_columns

        # Qué años tienen cada columna variante
        variant_presence = {}
        for col in variant_columns:
            years_present = [y for y, cols in cols_by_year.items() if col in cols]
            variant_presence[col] = sorted(years_present)

        analysis[category] = {
            "total_files": len(cat_schemas),
            "years_covered": sorted(cols_by_year.keys()),
            "common_columns": sorted(common_columns),
            "variant_columns": variant_presence,
            "total_unique_columns": len(all_columns),
        }

        logger.info(f"\n[{category.upper()}]")
        logger.info(f"  Archivos: {len(cat_schemas)} | Años: {sorted(cols_by_year.keys())}")
        logger.info(f"  Columnas comunes: {len(common_columns)}")
        logger.info(f"  Columnas variantes: {len(variant_columns)}")

        if variant_columns:
            logger.info("  Variantes detectadas:")
            for col, years in sorted(variant_presence.items()):
                logger.info(f"    - '{col}' → presente en: {years}")

    return analysis


# ──────────────────────────────────────────────────────────────
# Generación de reporte
# ──────────────────────────────────────────────────────────────

def generate_report(schemas: list, analysis: dict) -> dict:
    """
    Construye el reporte completo del profiling.

    Returns:
        Dict con reporte completo (serializable a JSON).
    """
    report = {
        "summary": {
            "total_files_profiled": len(schemas),
            "files_ok": sum(1 for s in schemas if s.get("status") == "ok"),
            "files_error": sum(1 for s in schemas if s.get("status") == "error"),
        },
        "by_category": analysis,
        "file_details": schemas,
    }
    return report


def save_report(report: dict, output_path: Optional[Path] = None):
    """Guarda el reporte en JSON."""
    if output_path is None:
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_PATH / "schema_report.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"\nReporte guardado en: {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────────
# Orquestador principal
# ──────────────────────────────────────────────────────────────

def run_schema_profiling(save: bool = True) -> dict:
    """
    Ejecuta el profiling completo sobre data/prepared/.

    Args:
        save: Si True, guarda reporte JSON en data/processed/schema_reports/.

    Returns:
        Dict con reporte completo.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO SCHEMA PROFILING SNIES")
    logger.info("=" * 60)

    # ── Descubrir archivos preparados
    prepared_files = discover_prepared_files()
    logger.info(f"Archivos preparados encontrados: {len(prepared_files)}")

    if not prepared_files:
        logger.warning("No hay archivos en data/prepared/. Ejecuta prepare_raw_files.py primero.")
        return {}

    # ── Extraer schema por archivo
    logger.info("\nExtrayendo schemas por archivo...")
    schemas = []
    for file_meta in prepared_files:
        schema = extract_file_schema(file_meta)
        if schema:
            schemas.append(schema)
            if schema["status"] == "ok":
                logger.info(
                    f"  [OK] [{schema['category']}] {schema['year']} → "
                    f"{schema['n_cols']} cols | {schema['n_rows']} filas"
                )

    # ── Analizar inconsistencias por categoría
    logger.info("\nAnalizando consistencia histórica por categoría...")
    analysis = analyze_category_schemas(schemas)

    # ── Generar y guardar reporte
    report = generate_report(schemas, analysis)

    if save:
        save_report(report)

    logger.info("\n" + "=" * 60)
    logger.info("PROFILING COMPLETADO")
    logger.info(f"  Archivos procesados: {report['summary']['files_ok']}")
    logger.info(f"  Errores:             {report['summary']['files_error']}")
    logger.info("=" * 60)

    return report


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Schema profiling de archivos preparados SNIES."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar reporte JSON (solo mostrar en consola).",
    )
    args = parser.parse_args()

    report = run_schema_profiling(save=not args.no_save)