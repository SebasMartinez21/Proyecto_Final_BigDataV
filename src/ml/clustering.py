"""
clustering.py
=============
Segmentación institucional con K-Means (PySpark MLlib).

Entrada : data/processed/aggregations/features_institucionales.csv
Salidas :
  - data/processed/ml/institutional_clusters.csv
  - data/processed/ml/cluster_profiles.csv       ← perfil promedio por cluster
  - data/processed/ml/silhouette_scores.csv       ← evaluación k=3..7
  - data/processed/ml/pca_coords.csv              ← coordenadas PCA para visualizar

Uso:
  python src/ml/clustering.py
"""

import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA as SparkPCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE  = "data/processed/aggregations/features_institucionales.csv"
OUTPUT_DIR  = "data/processed/ml"

FEATURES = [
    "tasa_graduacion",
    "ratio_docente_estudiante",
    "tasa_crecimiento_matricula",
    "tasa_permanencia",
    "brecha_ingreso_egreso",      # se escala luego
    "MATRICULADOS",               # volumen institucional
]

# Etiquetas descriptivas por cluster (se asignan post-análisis de centroides)
# Se actualizan automáticamente al inspeccionar medias → ver notebook 02
CLUSTER_LABELS = {
    0: "🟢 Alta eficiencia",
    1: "🟡 Crecimiento acelerado",
    2: "🔴 Riesgo académico",
    3: "🔵 Pequeña estable",
    4: "🟠 Alta docencia / baja eficiencia",
}

K_RANGE   = range(3, 8)   # evaluar k=3..7
K_OPTIMAL = 5             # puede sobreescribirse tras ver silhouette

# ─────────────────────────────────────────────
# SPARK
# ─────────────────────────────────────────────

def start_spark():
    return (SparkSession.builder
            .appName("SNIES_Clustering")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "4g")
            .getOrCreate())

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_features(spark):
    """Lee el CSV de features y prepara el DataFrame de Spark."""
    pdf = pd.read_csv(INPUT_FILE, encoding="utf-8", low_memory=False)
    sdf = spark.createDataFrame(pdf)

    # Filtrar filas con features completas
    for col in FEATURES:
        sdf = sdf.filter(F.col(col).isNotNull())

    return sdf


def build_pipeline(k):
    """Construye el pipeline MLlib: VectorAssembler → StandardScaler → KMeans."""
    assembler = VectorAssembler(
        inputCols=FEATURES,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features_scaled",
        withMean=True,
        withStd=True
    )
    kmeans = KMeans(
        featuresCol="features_scaled",
        predictionCol="cluster",
        k=k,
        seed=42,
        maxIter=50
    )
    return Pipeline(stages=[assembler, scaler, kmeans])


def eval_silhouette(model, sdf):
    evaluator = ClusteringEvaluator(
        predictionCol="cluster",
        featuresCol="features_scaled",
        metricName="silhouette"
    )
    predictions = model.transform(sdf)
    return evaluator.evaluate(predictions), predictions


def build_pca_coords(sdf_with_clusters, spark):
    """Reduce features a 2D con PCA para visualización."""
    pca = SparkPCA(
        inputCol="features_scaled",
        outputCol="pca_features",
        k=2
    )
    # Re-ensamblar y escalar (pipeline ya aplicado en sdf)
    pca_model = pca.fit(sdf_with_clusters)
    pca_df = pca_model.transform(sdf_with_clusters)

    # Extraer coordenadas
    extract_pc = F.udf(lambda v: float(v[0]) if v else None,
                       returnType=__import__("pyspark.sql.types", fromlist=["DoubleType"]).DoubleType())
    extract_pc2 = F.udf(lambda v: float(v[1]) if v else None,
                        returnType=__import__("pyspark.sql.types", fromlist=["DoubleType"]).DoubleType())

    pca_df = (pca_df
              .withColumn("PC1", extract_pc(F.col("pca_features")))
              .withColumn("PC2", extract_pc2(F.col("pca_features"))))
    return pca_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_clustering(k_optimal=K_OPTIMAL):
    spark = start_spark()
    spark.sparkContext.setLogLevel("ERROR")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  SNIES — Clustering Institucional (K-Means MLlib)")
    print("=" * 60)

    # ── 1. Cargar features ───────────────────────────────────────
    print("\n[1/5] Cargando features institucionales...")
    sdf = load_features(spark)
    # Usar promedio histórico por institución (un punto por IES)
    sdf_ies = sdf.groupBy("COD_IES", "NOMBRE_IES", "SECTOR_IES",
                           "CARACTER_IES", "DEPTO_IES").agg(
        *[F.avg(c).alias(c) for c in FEATURES]
    )
    print(f"   → Instituciones únicas: {sdf_ies.count():,}")

    # ── 2. Evaluar k óptimo ─────────────────────────────────────
    print("\n[2/5] Evaluando silhouette para k = 3..7...")
    silhouette_records = []
    for k in K_RANGE:
        pipeline = build_pipeline(k)
        model = pipeline.fit(sdf_ies)
        score, _ = eval_silhouette(model, sdf_ies)
        silhouette_records.append({"k": k, "silhouette": round(score, 4)})
        print(f"   k={k}  silhouette={score:.4f}")

    sil_df = pd.DataFrame(silhouette_records)
    sil_df.to_csv(f"{OUTPUT_DIR}/silhouette_scores.csv", index=False)
    k_best = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])
    print(f"\n   ✅ k óptimo por silhouette: {k_best}")

    # Usar k_optimal si fue especificado explícitamente, si no usar el mejor
    k_final = k_optimal if k_optimal else k_best

    # ── 3. Entrenar modelo final ─────────────────────────────────
    print(f"\n[3/5] Entrenando modelo final con k={k_final}...")
    pipeline_final = build_pipeline(k_final)
    model_final = pipeline_final.fit(sdf_ies)
    _, predictions = eval_silhouette(model_final, sdf_ies)

    # ── 4. PCA 2D para visualización ────────────────────────────
    print("\n[4/5] Calculando coordenadas PCA 2D...")
    from pyspark.ml.feature import PCA as SparkPCA2
    from pyspark.sql.types import DoubleType

    pca_spark = SparkPCA2(inputCol="features_scaled", outputCol="pca_features", k=2)
    pca_model = pca_spark.fit(predictions)
    pca_result = pca_model.transform(predictions)

    get_pc1 = F.udf(lambda v: float(v[0]) if v else None, DoubleType())
    get_pc2 = F.udf(lambda v: float(v[1]) if v else None, DoubleType())

    pca_result = (pca_result
                  .withColumn("PC1", get_pc1(F.col("pca_features")))
                  .withColumn("PC2", get_pc2(F.col("pca_features"))))

    # ── 5. Guardar resultados ────────────────────────────────────
    print("\n[5/5] Guardando resultados...")

    # 5a. Dataset completo con cluster asignado
    cols_out = ["COD_IES", "NOMBRE_IES", "SECTOR_IES", "CARACTER_IES",
                "DEPTO_IES", "cluster"] + FEATURES
    pdf_clusters = predictions.select(*cols_out).toPandas()
    pdf_clusters["cluster_label"] = pdf_clusters["cluster"].map(CLUSTER_LABELS)
    pdf_clusters.to_csv(f"{OUTPUT_DIR}/institutional_clusters.csv",
                        index=False, encoding="utf-8")

    # 5b. Perfil promedio por cluster
    profile_cols = ["cluster"] + FEATURES
    pdf_profiles = (predictions.select(*profile_cols)
                    .groupBy("cluster")
                    .agg(*[F.avg(c).alias(f"avg_{c}") for c in FEATURES])
                    .orderBy("cluster")
                    .toPandas())
    pdf_profiles["cluster_label"] = pdf_profiles["cluster"].map(CLUSTER_LABELS)
    pdf_profiles.to_csv(f"{OUTPUT_DIR}/cluster_profiles.csv",
                        index=False, encoding="utf-8")

    # 5c. Coordenadas PCA
    pca_cols = ["COD_IES", "NOMBRE_IES", "SECTOR_IES", "DEPTO_IES",
                "cluster", "PC1", "PC2"] + FEATURES
    pdf_pca = pca_result.select(*pca_cols).toPandas()
    pdf_pca["cluster_label"] = pdf_pca["cluster"].map(CLUSTER_LABELS)
    pdf_pca.to_csv(f"{OUTPUT_DIR}/pca_coords.csv",
                   index=False, encoding="utf-8")

    # ── Resumen ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  ✅ Clustering completado — k={k_final}")
    print(f"     institutional_clusters.csv : {len(pdf_clusters):,} instituciones")
    print(f"     cluster_profiles.csv       : perfil de {k_final} clusters")
    print(f"     pca_coords.csv             : {len(pdf_pca):,} puntos para viz")
    print(f"     silhouette_scores.csv      : evaluación k=3..7")
    print("=" * 60)

    dist = pdf_clusters["cluster_label"].value_counts()
    print("\nDistribución de clusters:")
    print(dist.to_string())

    spark.stop()
    return pdf_clusters, pdf_profiles, pdf_pca, sil_df


if __name__ == "__main__":
    run_clustering(k_optimal=K_OPTIMAL)