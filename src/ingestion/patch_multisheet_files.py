"""
patch_multisheet_files.py
=========================
Parche quirúrgico para archivos SNIES con múltiples hojas (sheets).

Problema detectado:
- Archivos de años 2023 y 2024 tienen dos sheets:
    - Sheet 0: "ÍNDICE" (metadata, no datos)
    - Sheet 1: "1." (datos reales)
- prepare_raw_files.py leyó solo sheet 0 por defecto.

Solución:
- Detectar automáticamente qué archivos en data/prepared/ tienen datos
  incorrectos (pocas columnas, o columnas que parecen índice).
- Releer el archivo raw correspondiente usando la sheet correcta.
- Sobreescribir el archivo preparado con los datos correctos.
- NO tocar archivos que ya están bien.

Uso:
    python src/ingestion/patch_multisheet_files.py
    python src/ingestion/patch_multisheet_files.py --years 2023 2024
    python src/ingestion/patch_multisheet_files.py --dry-run
"""

import pandas as pd
from pathlib import Path
import logging
import re
import sys
import argparse

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.ingestion.prepare_raw_files import (
    detect_header_row,
    clean_column_names,
    drop_unnamed_columns,
    drop_empty_rows,
    MIN_COLUMNS,
    HEADER_KEYWORDS,
)

# ──────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────

RAW_DATA_PATH = Path("data/raw/Analitica_educativa")
PREPARED_PATH = Path("data/prepared")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Años conocidos con problema multisheet
DEFAULT_AFFECTED_YEARS = ["2023", "2024"]


# ──────────────────────────────────────────────────────────────
# Detección de sheet correcta
# ──────────────────────────────────────────────────────────────

def _normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip()
    for acc, plain in {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ñ":"n"}.items():
        text = text.replace(acc, plain)
    return text


def _sheet_keyword_score(df: pd.DataFrame) -> int:
    """Cuenta palabras clave SNIES en las primeras filas de un sheet."""
    score = 0
    for _, row in df.head(20).iterrows():
        for val in row:
            normalized = _normalize_text(val)
            if any(kw in normalized for kw in HEADER_KEYWORDS):
                score += 1
    return score


def find_data_sheet(raw_path: Path) -> int:
    """
    Dado un archivo Excel con múltiples sheets, detecta automáticamente
    cuál sheet contiene los datos reales SNIES.

    Estrategia:
    - Lee cada sheet sin header.
    - Calcula keyword score para cada una.
    - Devuelve el índice de la sheet con mayor score.

    Returns:
        Índice (0-based) de la sheet con datos reales.
    """
    xl = pd.ExcelFile(raw_path, engine="openpyxl")
    sheet_names = xl.sheet_names

    if len(sheet_names) == 1:
        return 0  # Solo una sheet, no hay ambigüedad

    best_sheet_idx = 0
    best_score = -1

    for idx, sheet_name in enumerate(sheet_names):
        try:
            df = xl.parse(sheet_name, header=None, dtype=str, nrows=25)
            score = _sheet_keyword_score(df)
            logger.info(f"    Sheet [{idx}] '{sheet_name}' → keyword score: {score}")
            if score > best_score:
                best_score = score
                best_sheet_idx = idx
        except Exception as e:
            logger.warning(f"    Sheet [{idx}] '{sheet_name}' → error al leer: {e}")

    return best_sheet_idx


# ──────────────────────────────────────────────────────────────
# Lógica de parche por archivo
# ──────────────────────────────────────────────────────────────

def patch_file(prepared_path: Path, raw_path: Path, dry_run: bool = False) -> dict:
    """
    Reprocesa un archivo preparado usando la sheet correcta del raw.

    Returns:
        Dict con resultado del parche.
    """
    result = {
        "filename": prepared_path.name,
        "status": None,
        "sheet_used": None,
        "n_rows": None,
        "n_cols": None,
        "error": None,
    }

    try:
        logger.info(f"  Analizando sheets de: {raw_path.name}")
        correct_sheet_idx = find_data_sheet(raw_path)
        result["sheet_used"] = correct_sheet_idx

        if dry_run:
            logger.info(f"  [DRY-RUN] Usaría sheet índice {correct_sheet_idx}")
            result["status"] = "dry_run"
            return result

        # Leer la sheet correcta sin asumir header
        df_raw = pd.read_excel(
            raw_path,
            sheet_name=correct_sheet_idx,
            header=None,
            dtype=str,
            engine="openpyxl",
        )

        if df_raw.empty:
            result["status"] = "skipped_empty"
            return result

        # Detectar header real dentro de la sheet correcta
        header_row_idx = detect_header_row(df_raw)
        header = df_raw.iloc[header_row_idx].tolist()
        data_rows = df_raw.iloc[header_row_idx + 1:].reset_index(drop=True)
        data_rows.columns = clean_column_names(header)

        df_clean = drop_unnamed_columns(data_rows)
        df_clean = drop_empty_rows(df_clean)

        if df_clean.empty or len(df_clean.columns) < MIN_COLUMNS:
            result["status"] = "skipped_insufficient_data"
            logger.warning(f"  [SKIP] Datos insuficientes: {len(df_clean.columns)} columnas")
            return result

        # Sobreescribir el archivo preparado
        df_clean.to_excel(prepared_path, index=False, engine="openpyxl")

        result["status"] = "patched"
        result["n_rows"] = len(df_clean)
        result["n_cols"] = len(df_clean.columns)

        logger.info(
            f"  [PATCHED] Sheet {correct_sheet_idx} → "
            f"{result['n_rows']} filas | {result['n_cols']} cols"
        )

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"  [ERROR] {e}")

    return result


# ──────────────────────────────────────────────────────────────
# Orquestador
# ──────────────────────────────────────────────────────────────

def run_patch(years: list = None, dry_run: bool = False) -> list:
    """
    Recorre data/prepared/ filtrando por años afectados,
    busca el raw correspondiente y aplica el parche.

    Args:
        years: Lista de años a parchear (strings). Default: ["2023", "2024"].
        dry_run: Si True, solo reporta sin modificar nada.
    """
    if years is None:
        years = DEFAULT_AFFECTED_YEARS

    logger.info("=" * 60)
    logger.info(f"PARCHE MULTISHEET — Años: {years}" + (" [DRY-RUN]" if dry_run else ""))
    logger.info("=" * 60)

    results = []
    counts = {"patched": 0, "skipped": 0, "error": 0, "raw_not_found": 0}

    for year in years:
        year_str = str(year)
        # Buscar todos los archivos preparados de ese año
        pattern = PREPARED_PATH / "*" / f"year={year_str}" / "*.xlsx"
        prepared_files = list(PREPARED_PATH.rglob(f"year={year_str}/*.xlsx"))

        if not prepared_files:
            logger.warning(f"No se encontraron archivos preparados para año {year_str}")
            continue

        logger.info(f"\nAño {year_str}: {len(prepared_files)} archivos a parchear")

        for prepared_path in prepared_files:
            filename = prepared_path.name
            # Extraer categoría del path: data/prepared/<category>/year=XXXX/file.xlsx
            category = prepared_path.parts[-3]

            logger.info(f"\n[{category}] {filename}")

            # Buscar el raw correspondiente
            # Estructura raw: data/raw/Analitica_educativa/<year>/<filename>
            raw_path = RAW_DATA_PATH / year_str / filename

            if not raw_path.exists():
                # Intentar buscar recursivamente por si la estructura varía
                matches = list(RAW_DATA_PATH.rglob(filename))
                if matches:
                    raw_path = matches[0]
                else:
                    logger.warning(f"  [NOT FOUND] Raw no encontrado: {raw_path}")
                    results.append({
                        "filename": filename,
                        "year": year_str,
                        "category": category,
                        "status": "raw_not_found",
                    })
                    counts["raw_not_found"] += 1
                    continue

            result = patch_file(prepared_path, raw_path, dry_run=dry_run)
            result["year"] = year_str
            result["category"] = category
            results.append(result)

            if result["status"] == "patched":
                counts["patched"] += 1
            elif result["status"] in ("skipped_empty", "skipped_insufficient_data", "dry_run"):
                counts["skipped"] += 1
            elif result["status"] == "error":
                counts["error"] += 1

    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN PARCHE")
    logger.info(f"  ✓ Parcheados:        {counts['patched']}")
    logger.info(f"  → Saltados:          {counts['skipped']}")
    logger.info(f"  ! Raw no encontrado: {counts['raw_not_found']}")
    logger.info(f"  ✗ Errores:           {counts['error']}")
    logger.info("=" * 60)

    return results


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parche multisheet para archivos SNIES 2023-2024."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=DEFAULT_AFFECTED_YEARS,
        help="Años a parchear. Default: 2023 2024",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo reportar qué haría, sin modificar archivos.",
    )
    args = parser.parse_args()

    run_patch(years=args.years, dry_run=args.dry_run)