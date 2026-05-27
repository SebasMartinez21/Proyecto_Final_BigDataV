"""
feature_builder.py
==================
Construye las features derivadas institucionales por año a partir de los
.csv limpios en data/processed/.

Salida: data/processed/aggregations/features_institucionales.csv

Regla arquitectónica:
  - pandas SOLO para leer CSV (encoding corrupto requiere detección manual)
  - PySpark para TODO el procesamiento
"""

import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
DATA_PROCESSED = "data/processed"
OUTPUT_DIR     = "data/processed/aggregations"
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, "features_institucionales.csv")

# Encoding real de los CSV (Windows-1252 generado en Colombia)
CSV_ENCODING = "latin-1"

# Columnas clave de agrupación institucional
KEY_IES   = "COD_IES"       # alias limpio que asignamos
KEY_YEAR  = "AÑO"
KEY_DEPT  = "DEPTO_IES"
KEY_SECTOR = "SECTOR_IES"
KEY_CARACTER = "CARACTER_IES"
KEY_NOMBRE = "NOMBRE_IES"

# ─────────────────────────────────────────────
# MAPEO DE COLUMNAS (encoding roto → alias limpio)
# ─────────────────────────────────────────────
# Los nombres reales en los CSV tienen caracteres corruptos tipo
# "CËDIGO DE LA INSTITUCIËN", "AÐO", etc.
# Usamos índice posicional para renombrar de forma robusta.

COLS_COMUNES = {
    0:  KEY_IES,        # CÓDIGO DE LA INSTITUCIÓN
    2:  KEY_NOMBRE,     # INSTITUCIÓN DE EDUCACIÓN SUPERIOR
    3:  "TIPO_IES",
    4:  "ID_SECTOR_IES",
    5:  KEY_SECTOR,
    6:  "ID_CARACTER_IES",
    7:  KEY_CARACTER,
    8:  "PRINCIPAL_SECCIONAL",
    9:  "COD_DEPTO_IES",
    10: KEY_DEPT,
    11: "COD_MUNICIPIO_IES",
    12: "MUNICIPIO_IES",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def start_spark(app_name="SNIES_FeatureBuilder"):
    return (SparkSession.builder
            .appName(app_name)
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "4g")
            .getOrCreate())


def read_csv_spark(spark, path, metrica_col_idx, metrica_alias):
    """
    Lee un CSV con pandas (encoding seguro) y lo convierte a Spark DataFrame.
    Renombra columnas comunes por índice posicional + la métrica específica.
    
    metrica_col_idx : índice de la columna numérica (ej. -2 = penúltima)
    metrica_alias   : nombre limpio para esa métrica
    """
    pdf = pd.read_csv(path, encoding=CSV_ENCODING, low_memory=False)
    
    # Renombrar por índice — inmune al encoding roto
    rename_map = {}
    cols = list(pdf.columns)
    
    for idx, alias in COLS_COMUNES.items():
        if idx < len(cols):
            rename_map[cols[idx]] = alias
    
    # Columna AÑO (siempre última)
    rename_map[cols[-1]] = "AÑO"
    # Métrica (penúltima o la indicada)
    rename_map[cols[metrica_col_idx]] = metrica_alias
    
    pdf = pdf.rename(columns=rename_map)
    
    # Castear métrica a numérico
    pdf[metrica_alias] = pd.to_numeric(pdf[metrica_alias], errors="coerce")
    pdf["AÑO"]         = pd.to_numeric(pdf["AÑO"], errors="coerce")
    pdf[KEY_IES]       = pd.to_numeric(pdf[KEY_IES], errors="coerce")
    
    sdf = spark.createDataFrame(pdf)
    return sdf


def agg_por_ies_anho(sdf, metrica_col, group_extra=None):
    """
    Agrega la métrica sumando por IES + AÑO.
    group_extra: columnas adicionales de grupo (ej. KEY_DEPT para región)
    """
    group_cols = [KEY_IES, KEY_NOMBRE, KEY_SECTOR, KEY_CARACTER,
                  KEY_DEPT, "AÑO"]
    if group_extra:
        group_cols += [c for c in group_extra if c not in group_cols]
    
    return (sdf
            .groupBy(*group_cols)
            .agg(F.sum(metrica_col).cast("double").alias(metrica_col)))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def build_features(spark):
    print("=" * 60)
    print("  SNIES — Feature Builder")
    print("=" * 60)

    # ── 1. Leer cada categoría ──────────────────────────────────
    print("\n[1/6] Leyendo CSV limpios...")

    mat = read_csv_spark(
        spark,
        f"{DATA_PROCESSED}/matriculados.csv",
        metrica_col_idx=-2,
        metrica_alias="MATRICULADOS"
    )
    ins = read_csv_spark(
        spark,
        f"{DATA_PROCESSED}/inscritos.csv",
        metrica_col_idx=-2,
        metrica_alias="INSCRITOS"
    )
    adm = read_csv_spark(
        spark,
        f"{DATA_PROCESSED}/admitidos.csv",
        metrica_col_idx=-2,
        metrica_alias="ADMITIDOS"
    )
    gra = read_csv_spark(
        spark,
        f"{DATA_PROCESSED}/graduados.csv",
        metrica_col_idx=-2,
        metrica_alias="GRADUADOS"
    )
    doc = read_csv_spark(
        spark,
        f"{DATA_PROCESSED}/docentes.csv",
        metrica_col_idx=-2,
        metrica_alias="NUM_DOCENTES"
    )
    mpc = read_csv_spark(
        spark,
        f"{DATA_PROCESSED}/matriculados_primer_curso.csv",
        metrica_col_idx=-2,
        metrica_alias="MATRICULADOS_PRIMER_CURSO"
    )

    # ── 2. Agregar por IES + AÑO ────────────────────────────────
    print("[2/6] Agregando métricas por institución-año...")

    mat_agg = agg_por_ies_anho(mat, "MATRICULADOS")
    ins_agg = agg_por_ies_anho(ins, "INSCRITOS")
    adm_agg = agg_por_ies_anho(adm, "ADMITIDOS")
    gra_agg = agg_por_ies_anho(gra, "GRADUADOS")
    mpc_agg = agg_por_ies_anho(mpc, "MATRICULADOS_PRIMER_CURSO")

    # Docentes: no tiene programa, solo IES
    doc_agg = (doc
               .groupBy(KEY_IES, KEY_NOMBRE, KEY_SECTOR, KEY_CARACTER, KEY_DEPT, "AÑO")
               .agg(F.sum("NUM_DOCENTES").cast("double").alias("NUM_DOCENTES")))

    # ── 3. JOIN central ─────────────────────────────────────────
    print("[3/6] Construyendo tabla central IES-AÑO...")

    join_keys = [KEY_IES, KEY_NOMBRE, KEY_SECTOR, KEY_CARACTER, KEY_DEPT, "AÑO"]

    central = (mat_agg
               .join(ins_agg,  join_keys, "left")
               .join(adm_agg,  join_keys, "left")
               .join(gra_agg,  join_keys, "left")
               .join(mpc_agg,  join_keys, "left")
               .join(doc_agg,  join_keys, "left"))

    # Rellenar nulos en métricas con 0
    metricas = ["MATRICULADOS", "INSCRITOS", "ADMITIDOS", "GRADUADOS",
                "MATRICULADOS_PRIMER_CURSO", "NUM_DOCENTES"]
    central = central.fillna(0, subset=metricas)

    # ── 4. Features derivadas ────────────────────────────────────
    print("[4/6] Calculando features derivadas...")

    # Window para cálculos YoY por institución
    w_ies = Window.partitionBy(KEY_IES).orderBy("AÑO")

    central = (central
        # Tasa de graduación: graduados / matriculados
        .withColumn("tasa_graduacion",
            F.when(F.col("MATRICULADOS") > 0,
                   F.col("GRADUADOS") / F.col("MATRICULADOS"))
             .otherwise(None))

        # Tasa de permanencia: matriculados / inscritos (selectividad)
        .withColumn("tasa_permanencia",
            F.when(F.col("INSCRITOS") > 0,
                   F.col("MATRICULADOS") / F.col("INSCRITOS"))
             .otherwise(None))

        # Tasa de admisión: admitidos / inscritos
        .withColumn("tasa_admision",
            F.when(F.col("INSCRITOS") > 0,
                   F.col("ADMITIDOS") / F.col("INSCRITOS"))
             .otherwise(None))

        # Ratio docente-estudiante: matriculados / docentes (carga docente)
        .withColumn("ratio_docente_estudiante",
            F.when(F.col("NUM_DOCENTES") > 0,
                   F.col("MATRICULADOS") / F.col("NUM_DOCENTES"))
             .otherwise(None))

        # Brecha ingreso-egreso: inscritos - graduados
        .withColumn("brecha_ingreso_egreso",
            F.col("INSCRITOS") - F.col("GRADUADOS"))

        # Matriculados año anterior (lag)
        .withColumn("mat_anho_anterior",
            F.lag("MATRICULADOS", 1).over(w_ies))

        # Tasa de crecimiento YoY de matrícula
        .withColumn("tasa_crecimiento_matricula",
            F.when(F.col("mat_anho_anterior").isNotNull() &
                   (F.col("mat_anho_anterior") > 0),
                   (F.col("MATRICULADOS") - F.col("mat_anho_anterior")) /
                   F.col("mat_anho_anterior"))
             .otherwise(None))

        # Tasa graduación año anterior (para predictor)
        .withColumn("tasa_grad_anho_anterior",
            F.lag("tasa_graduacion", 1).over(w_ies))

        # Indicador: institución principal (no seccional)
        # (columna PRINCIPAL_SECCIONAL puede no estar en todos → ignorar si falla)

        .drop("mat_anho_anterior")
    )

    # ── 5. Variable objetivo para ML ────────────────────────────
    print("[5/6] Construyendo variable objetivo riesgo_academico...")

    # Percentiles calculados de forma INDEPENDIENTE por columna
    # (evita intersección vacía cuando una columna tiene nulos)
    stats = central.filter(
        F.col("tasa_graduacion").isNotNull()
    ).select(
        F.percentile_approx("tasa_graduacion", 0.33).alias("p33_grad"),
        F.percentile_approx("tasa_graduacion", 0.25).alias("p25_grad"),
    ).collect()[0]

    stats_perm = central.filter(
        F.col("tasa_permanencia").isNotNull()
    ).select(
        F.percentile_approx("tasa_permanencia", 0.33).alias("p33_perm"),
    ).collect()[0]

    stats_ratio = central.filter(
        F.col("ratio_docente_estudiante").isNotNull()
    ).select(
        F.percentile_approx("ratio_docente_estudiante", 0.75).alias("p75_ratio"),
    ).collect()[0]

    p33_grad  = stats["p33_grad"]
    p25_grad  = stats["p25_grad"]
    p33_perm  = stats_perm["p33_perm"]
    p75_ratio = stats_ratio["p75_ratio"]

    print(f"   → Percentil 33 tasa_graduacion:       {p33_grad:.4f}")
    print(f"   → Percentil 25 tasa_graduacion:       {p25_grad:.4f}")
    print(f"   → Percentil 33 tasa_permanencia:      {p33_perm:.4f}")
    print(f"   → Percentil 75 ratio_doc_estudiante:  {p75_ratio:.2f}")

    # Definición de riesgo: UNO de tres criterios es suficiente (OR)
    # ─────────────────────────────────────────────────────────────
    # C1: Baja graduación (peor tercio histórico)
    # C2: Baja permanencia (muchos inscritos que no se matriculan)
    # C3: Alta carga docente (demasiados estudiantes por docente)
    # ─────────────────────────────────────────────────────────────
    criterio_1 = F.col("tasa_graduacion") < p33_grad
    criterio_2 = F.col("tasa_permanencia") < p33_perm
    criterio_3 = F.col("ratio_docente_estudiante") > p75_ratio

    central = central.withColumn("riesgo_academico",
        F.when(
            criterio_1 | criterio_2 | criterio_3,
            F.lit(1)
        ).otherwise(F.lit(0)).cast("int")
    )

    # ── 6. Guardar ───────────────────────────────────────────────
    print("[6/6] Guardando features_institucionales.csv...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_out = central.toPandas()
    pdf_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    n_rows  = len(pdf_out)
    n_ies   = pdf_out[KEY_IES].nunique()
    n_years = pdf_out["AÑO"].nunique()
    n_riesgo = pdf_out["riesgo_academico"].sum()

    print("\n" + "=" * 60)
    print(f"  ✅ Features guardadas en: {OUTPUT_FILE}")
    print(f"     Filas:          {n_rows:,}")
    print(f"     Instituciones:  {n_ies:,}")
    print(f"     Años:           {n_years}")
    print(f"     En riesgo (=1): {int(n_riesgo):,}  ({100*n_riesgo/n_rows:.1f}%)")
    print("=" * 60)

    return central


if __name__ == "__main__":
    spark = start_spark()
    spark.sparkContext.setLogLevel("ERROR")
    build_features(spark)
    spark.stop()