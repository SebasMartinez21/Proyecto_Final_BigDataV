"""
prepare_raw_files.py
====================
Etapa de prelimpieza automática de archivos Excel SNIES.

Responsabilidades:
- Detectar automáticamente la fila real donde empieza la tabla (header real).
- Eliminar títulos, subtítulos y metadata institucional superiores.
- Limpiar columnas basura (Unnamed, columnas vacías).
- Exportar archivos limpios a data/prepared/ manteniendo estructura de carpetas.
- Ser totalmente automático y mantenible para años futuros (2025+).

Estrategia de detección de header:
- Buscar la primera fila donde la mayoría de celdas son strings no vacíos
  y que contengan palabras clave típicas de columnas SNIES.
- Fallback: buscar la fila con mayor densidad de valores no nulos.
"""

import pandas as pd
from pathlib import Path
import logging
import re
import sys
import os

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.ingestion.discover_files import discover_files

# ──────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────

PREPARED_PATH = Path("data/prepared")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Palabras clave que suelen aparecer en headers reales SNIES
# Se usan para validar que una fila candidata es el header real
HEADER_KEYWORDS = [
    "código", "codigo", "institución", "institucion", "ies",
    "departamento", "municipio", "programa", "académico", "academico",
    "nivel", "metodología", "metodologia", "área", "area",
    "núcleo", "nucleo", "sexo", "año", "semestre",
    "admitidos", "inscritos", "matriculados", "graduados", "docentes",
    "sector", "carácter", "caracter", "snies", "padre",
]

# Máximo de filas a escanear buscando el header (los títulos no suelen
# ocupar más de 15 filas)
MAX_ROWS_TO_SCAN = 20

# Mínimo de columnas reales que debe tener el header encontrado
MIN_COLUMNS = 5


# ──────────────────────────────────────────────────────────────
# Lógica de detección de header
# ──────────────────────────────────────────────────────────────

def _normalize_text(value) -> str:
    """Normaliza un valor de celda a string minúscula sin tildes para comparación."""
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    replacements = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
    for accented, plain in replacements.items():
        text = text.replace(accented, plain)
    return text


def _row_keyword_score(row: pd.Series) -> int:
    """Cuenta cuántas palabras clave SNIES aparecen en una fila."""
    score = 0
    for value in row:
        normalized = _normalize_text(value)
        if any(kw in normalized for kw in HEADER_KEYWORDS):
            score += 1
    return score


def _row_non_null_ratio(row: pd.Series) -> float:
    """Fracción de celdas no vacías en una fila."""
    total = len(row)
    if total == 0:
        return 0.0
    non_null = sum(1 for v in row if not pd.isna(v) and str(v).strip() != "")
    return non_null / total


def detect_header_row(df_raw: pd.DataFrame) -> int:
    """
    Detecta el índice de la fila que contiene el header real.

    Estrategia:
    1. Entre las primeras MAX_ROWS_TO_SCAN filas, busca la que tenga
       mayor score de palabras clave SNIES.
    2. Si empate o score 0, usa la fila con mayor densidad de valores
       no nulos como fallback.

    Returns:
        Índice (0-based) de la fila del header real.
    """
    scan_limit = min(MAX_ROWS_TO_SCAN, len(df_raw))
    candidates = df_raw.iloc[:scan_limit]

    best_row = 0
    best_keyword_score = -1
    best_density = -1.0

    for idx in range(len(candidates)):
        row = candidates.iloc[idx]
        kw_score = _row_keyword_score(row)
        density = _row_non_null_ratio(row)

        # Primero priorizar keyword score; en empate, densidad
        if (kw_score > best_keyword_score) or (
            kw_score == best_keyword_score and density > best_density
        ):
            best_keyword_score = kw_score
            best_density = density
            best_row = idx

    return best_row


# ──────────────────────────────────────────────────────────────
# Limpieza de columnas
# ──────────────────────────────────────────────────────────────

def clean_column_names(columns) -> list:
    """
    Limpia y normaliza nombres de columnas:
    - Elimina columnas Unnamed
    - Strip de espacios
    - No modifica el contenido semántico (eso es trabajo del schema_profiler)
    """
    cleaned = []
    for col in columns:
        col_str = str(col).strip()
        # Columnas Unnamed generadas por pandas cuando hay celdas vacías en header
        if re.match(r"^Unnamed", col_str, re.IGNORECASE):
            cleaned.append(None)
        else:
            cleaned.append(col_str)
    return cleaned


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina columnas cuyos nombres son None (Unnamed/vacías)."""
    cols_to_keep = [col for col in df.columns if col is not None]
    return df[cols_to_keep]


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas completamente vacías."""
    return df.dropna(how="all").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Procesamiento por archivo
# ──────────────────────────────────────────────────────────────

def prepare_single_file(file_meta: dict) -> dict:
    """
    Procesa un archivo Excel SNIES individual:
    1. Lee el archivo sin asumir header (header=None).
    2. Detecta automáticamente la fila del header real.
    3. Reconstruye el DataFrame con ese header.
    4. Limpia columnas y filas vacías.
    5. Exporta a data/prepared/ con la misma estructura de carpetas.

    Args:
        file_meta: dict con keys {year, category, path, filename}

    Returns:
        dict con resultado: {status, output_path, header_row, n_rows, n_cols, error}
    """
    source_path = Path(file_meta["path"])
    category = file_meta["category"]
    year = file_meta["year"]
    filename = file_meta["filename"]

    result = {
        "filename": filename,
        "category": category,
        "year": year,
        "status": None,
        "output_path": None,
        "header_row_detected": None,
        "n_rows": None,
        "n_cols": None,
        "error": None,
    }

    try:
        # ── Paso 1: Leer sin asumir header para poder escanear todas las filas
        df_raw = pd.read_excel(
            source_path,
            header=None,
            dtype=str,          # Todo como string en esta etapa para no perder nada
            engine="openpyxl",
        )

        if df_raw.empty:
            result["status"] = "skipped_empty"
            logger.warning(f"  [SKIP] Archivo vacío: {filename}")
            return result

        # ── Paso 2: Detectar fila del header real
        header_row_idx = detect_header_row(df_raw)
        result["header_row_detected"] = header_row_idx

        # ── Paso 3: Reconstruir DataFrame con header correcto
        header = df_raw.iloc[header_row_idx].tolist()
        data_rows = df_raw.iloc[header_row_idx + 1 :].reset_index(drop=True)
        data_rows.columns = clean_column_names(header)

        # ── Paso 4: Limpiar columnas Unnamed y filas vacías
        df_clean = drop_unnamed_columns(data_rows)
        df_clean = drop_empty_rows(df_clean)

        if df_clean.empty or len(df_clean.columns) < MIN_COLUMNS:
            result["status"] = "skipped_insufficient_data"
            logger.warning(
                f"  [SKIP] Datos insuficientes tras limpieza: {filename} "
                f"({len(df_clean.columns)} columnas)"
            )
            return result

        # ── Paso 5: Determinar ruta de salida
        # Estructura: data/prepared/<category>/year=<year>/<filename>.xlsx
        year_folder = f"year={year}"
        output_dir = PREPARED_PATH / category / year_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # ── Paso 6: Exportar a Excel limpio
        df_clean.to_excel(output_path, index=False, engine="openpyxl")

        result["status"] = "success"
        result["output_path"] = str(output_path)
        result["n_rows"] = len(df_clean)
        result["n_cols"] = len(df_clean.columns)

        logger.info(
            f"  [OK] {filename} → header en fila {header_row_idx} | "
            f"{result['n_rows']} filas | {result['n_cols']} cols"
        )

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"  [ERROR] {filename}: {e}")

    return result


# ──────────────────────────────────────────────────────────────
# Orquestador principal
# ──────────────────────────────────────────────────────────────

def prepare_all_files(force: bool = False) -> list:
    """
    Descubre todos los archivos raw y los prepara automáticamente.

    Args:
        force: Si True, reprocesa archivos aunque ya existan en data/prepared/.
               Si False (default), salta archivos ya preparados.

    Returns:
        Lista de dicts con resultados por archivo.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO PRELIMPIEZA AUTOMÁTICA SNIES")
    logger.info("=" * 60)

    files = discover_files()
    logger.info(f"Archivos descubiertos: {len(files)}")

    results = []
    counts = {"success": 0, "skipped_existing": 0, "skipped_empty": 0,
              "skipped_insufficient_data": 0, "error": 0}

    for file_meta in files:
        filename = file_meta["filename"]
        category = file_meta["category"]
        year = file_meta["year"]

        logger.info(f"Procesando: [{category}] {filename}")

        # Verificar si ya fue preparado (para no reprocesar innecesariamente)
        if not force:
            expected_output = (
                PREPARED_PATH / category / f"year={year}" / filename
            )
            if expected_output.exists():
                logger.info(f"  [SKIP] Ya preparado: {filename}")
                results.append({
                    "filename": filename,
                    "category": category,
                    "year": year,
                    "status": "skipped_existing",
                    "output_path": str(expected_output),
                })
                counts["skipped_existing"] += 1
                continue

        result = prepare_single_file(file_meta)
        results.append(result)
        counts[result["status"]] = counts.get(result["status"], 0) + 1

    # ── Resumen final
    logger.info("=" * 60)
    logger.info("RESUMEN PRELIMPIEZA")
    logger.info(f"  ✓ Exitosos:          {counts['success']}")
    logger.info(f"  → Ya existían:       {counts['skipped_existing']}")
    logger.info(f"  ! Vacíos:            {counts['skipped_empty']}")
    logger.info(f"  ! Datos insuf.:      {counts['skipped_insufficient_data']}")
    logger.info(f"  ✗ Errores:           {counts['error']}")
    logger.info("=" * 60)

    return results


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prelimpieza automática de archivos Excel SNIES."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocesar archivos aunque ya existan en data/prepared/.",
    )
    args = parser.parse_args()

    results = prepare_all_files(force=args.force)