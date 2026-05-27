"""
normalize_values.py
===================
Segunda etapa de preprocessing con PySpark.

Responsabilidades:
- Estandarizar valores de columnas categóricas (SEXO, SECTOR, MODALIDAD, etc.)
- Normalizar texto: mayúsculas, strip, tildes donde aplique.
- Castear columnas numéricas (AÑO, SEMESTRE, métricas).
- Estandarizar códigos históricos inconsistentes.
- Devolver DataFrame con tipos y valores limpios.
"""

import logging
import sys
from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.ingestion.canonical_schemas import get_metric_column

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# MAPAS DE ESTANDARIZACIÓN DE VALORES
# ──────────────────────────────────────────────────────────────

# Estandarización de SEXO
SEXO_MAP = {
    "f": "Femenino",
    "m": "Masculino",
    "femenino": "Femenino",
    "masculino": "Masculino",
    "mujer": "Femenino",
    "hombre": "Masculino",
    "female": "Femenino",
    "male": "Masculino",
}

# Estandarización de SECTOR IES
SECTOR_MAP = {
    "oficial": "Oficial",
    "privado": "Privado",
    "public": "Oficial",
    "private": "Privado",
}

# Estandarización de MODALIDAD
MODALIDAD_MAP = {
    "presencial": "Presencial",
    "distancia": "Distancia",
    "distancia (tradicional)": "Distancia",
    "virtual": "Virtual",
    "distancia virtual": "Virtual",
}

# Estandarización de NIVEL DE FORMACIÓN
NIVEL_FORMACION_MAP = {
    "universitaria": "Universitaria",
    "tecnologica": "Tecnológica",
    "tecnológica": "Tecnológica",
    "tecnica profesional": "Técnica Profesional",
    "técnica profesional": "Técnica Profesional",
    "especializacion": "Especialización",
    "especialización": "Especialización",
    "maestria": "Maestría",
    "maestría": "Maestría",
    "doctorado": "Doctorado",
    "especializacion tecnica": "Especialización Técnica",
    "especialización técnica": "Especialización Técnica",
}


# ──────────────────────────────────────────────────────────────
# FUNCIONES DE NORMALIZACIÓN
# ──────────────────────────────────────────────────────────────

def _build_mapping_expr(col_name: str, mapping: dict):
    """
    Construye una expresión PySpark CASE WHEN para mapear valores
    de una columna usando un diccionario de estandarización.
    """
    expr = F.when(F.lit(False), F.lit(None))  # base vacía
    for raw, canonical in mapping.items():
        expr = expr.when(
            F.lower(F.trim(F.col(col_name))) == raw,
            F.lit(canonical)
        )
    # Si no hay match → conservar valor original en uppercase strip
    expr = expr.otherwise(F.upper(F.trim(F.col(col_name))))
    return expr


def normalize_text_column(sdf: DataFrame, col_name: str) -> DataFrame:
    """
    Normaliza una columna de texto:
    - Strip de espacios
    - Eliminar saltos de línea
    - Reemplazar nulos string ("nan", "none", "null") por null real
    """
    if col_name not in sdf.columns:
        return sdf

    return sdf.withColumn(
        col_name,
        F.when(
            F.lower(F.trim(F.col(col_name))).isin("nan", "none", "null", ""),
            F.lit(None)
        ).otherwise(
            F.trim(F.regexp_replace(F.col(col_name), r"[\n\r\t]", " "))
        )
    )


def normalize_categorical_column(sdf: DataFrame, col_name: str, mapping: dict) -> DataFrame:
    """
    Aplica un mapa de estandarización a una columna categórica.
    Valores sin mapeo se convierten a UPPERCASE trimmed.
    """
    if col_name not in sdf.columns:
        return sdf

    # Primero limpiar nulos string
    sdf = normalize_text_column(sdf, col_name)

    # Construir expresión de mapeo
    cases = F.when(F.lit(False), F.lit(None))
    for raw, canonical in mapping.items():
        cases = cases.when(
            F.lower(F.trim(F.col(col_name))) == F.lit(raw),
            F.lit(canonical)
        )
    cases = cases.otherwise(F.trim(F.col(col_name)))

    return sdf.withColumn(col_name, cases)


def cast_numeric_columns(sdf: DataFrame, category: str) -> DataFrame:
    """
    Castea columnas numéricas al tipo correcto:
    - AÑO → Integer
    - SEMESTRE → Integer
    - Métrica de la categoría → Long (puede ser grande)
    """
    metric_col = get_metric_column(category)

    if "AÑO" in sdf.columns:
        sdf = sdf.withColumn(
            "AÑO",
            F.col("AÑO").cast(IntegerType())
        )

    if "SEMESTRE" in sdf.columns:
        sdf = sdf.withColumn(
            "SEMESTRE",
            F.col("SEMESTRE").cast(IntegerType())
        )

    if metric_col in sdf.columns:
        sdf = sdf.withColumn(
            metric_col,
            F.regexp_replace(F.col(metric_col), r"[,\.]", "")
            .cast(LongType())
        )

    return sdf


def normalize_codigo_snies(sdf: DataFrame) -> DataFrame:
    """
    Normaliza CÓDIGO SNIES DEL PROGRAMA:
    - Eliminar decimales si vino como float string ("12345.0" → "12345")
    - Strip
    """
    col = "CÓDIGO SNIES DEL PROGRAMA"
    if col not in sdf.columns:
        return sdf

    return sdf.withColumn(
        col,
        F.regexp_replace(F.trim(F.col(col)), r"\.0$", "")
    )


def normalize_codigo_institucion(sdf: DataFrame) -> DataFrame:
    """
    Normaliza CÓDIGO DE LA INSTITUCIÓN:
    - Eliminar decimales ("1234.0" → "1234")
    - Strip
    """
    col = "CÓDIGO DE LA INSTITUCIÓN"
    if col not in sdf.columns:
        return sdf

    return sdf.withColumn(
        col,
        F.regexp_replace(F.trim(F.col(col)), r"\.0$", "")
    )


def normalize_anho_semestre(sdf: DataFrame) -> DataFrame:
    """
    Asegura que AÑO y SEMESTRE tengan valores válidos:
    - SEMESTRE solo puede ser 1 o 2
    - AÑO entre 2013 y 2030
    """
    if "SEMESTRE" in sdf.columns:
        sdf = sdf.withColumn(
            "SEMESTRE",
            F.when(F.col("SEMESTRE").isin(1, 2), F.col("SEMESTRE"))
             .otherwise(F.lit(None).cast(IntegerType()))
        )

    if "AÑO" in sdf.columns:
        sdf = sdf.withColumn(
            "AÑO",
            F.when(
                (F.col("AÑO") >= 2013) & (F.col("AÑO") <= 2030),
                F.col("AÑO")
            ).otherwise(F.lit(None).cast(IntegerType()))
        )

    return sdf


# ──────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ──────────────────────────────────────────────────────────────

def normalize_values(sdf: DataFrame, category: str) -> DataFrame:
    """
    Pipeline completo de normalización de valores para un DataFrame SNIES.

    Pasos:
    1. Normalizar columnas de texto libre
    2. Estandarizar columnas categóricas (SEXO, SECTOR, MODALIDAD, NIVEL)
    3. Normalizar códigos (SNIES, institución)
    4. Castear numéricos (AÑO, SEMESTRE, métrica)
    5. Validar rangos de AÑO y SEMESTRE

    Args:
        sdf: Spark DataFrame con columnas canónicas (salida de clean_columns).
        category: Categoría SNIES del DataFrame.

    Returns:
        Spark DataFrame con valores normalizados.
    """
    logger.info(f"  Normalizando valores: {category}")

    # ── Texto libre
    text_cols = [
        "INSTITUCIÓN DE EDUCACIÓN SUPERIOR (IES)",
        "PROGRAMA ACADÉMICO",
        "DEPARTAMENTO DE DOMICILIO DE LA IES",
        "MUNICIPIO DE DOMICILIO DE LA IES",
        "DEPARTAMENTO DE OFERTA DEL PROGRAMA",
        "MUNICIPIO DE OFERTA DEL PROGRAMA",
        "ÁREA DE CONOCIMIENTO",
        "NÚCLEO BÁSICO DEL CONOCIMIENTO (NBC)",
        "DESC CINE CAMPO AMPLIO",
        "DESC CINE CAMPO ESPECÍFICO",
        "DESC CINE CAMPO DETALLADO",
        "IES PADRE",
        "TIPO IES",
        "CARÁCTER IES",
        "NIVEL ACADÉMICO",
        "NIVEL DE FORMACIÓN",
    ]
    for col in text_cols:
        sdf = normalize_text_column(sdf, col)

    # ── Categóricas estandarizadas
    sdf = normalize_categorical_column(sdf, "SEXO", SEXO_MAP)
    sdf = normalize_categorical_column(sdf, "SECTOR IES", SECTOR_MAP)
    sdf = normalize_categorical_column(sdf, "MODALIDAD", MODALIDAD_MAP)
    sdf = normalize_categorical_column(sdf, "NIVEL DE FORMACIÓN", NIVEL_FORMACION_MAP)

    # ── Códigos
    sdf = normalize_codigo_snies(sdf)
    sdf = normalize_codigo_institucion(sdf)

    # ── Numéricos y rangos
    sdf = cast_numeric_columns(sdf, category)
    sdf = normalize_anho_semestre(sdf)

    logger.info(f"  Normalización completada.")
    return sdf