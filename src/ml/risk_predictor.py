"""
risk_predictor.py
=================
Predicción de riesgo académico institucional.
Compara tres clasificadores y selecciona el mejor por AUC-ROC.

Clasificadores:
  1. Random Forest
  2. Gradient Boosted Trees (GBT)
  3. Regresión Logística

Métricas: AUC-ROC, Accuracy, F1-Score

Entrada : data/processed/aggregations/features_institucionales.csv
Salidas :
  - data/processed/ml/risk_predictions.csv          ← predicciones del mejor modelo
  - data/processed/ml/model_comparison.csv          ← tabla comparativa de los 3 modelos
  - data/processed/ml/feature_importance.csv        ← importancia de variables (RF y GBT)
  - data/processed/ml/roc_data_<modelo>.csv         ← datos para curva ROC (por modelo)

Uso:
  python src/ml/risk_predictor.py
"""

import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import (
    RandomForestClassifier,
    GBTClassifier,
    LogisticRegression,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE = "data/processed/aggregations/features_institucionales.csv"
OUTPUT_DIR = "data/processed/ml"

# Features numéricas
NUM_FEATURES = [
    "tasa_graduacion",
    "tasa_grad_anho_anterior",
    "ratio_docente_estudiante",
    "tasa_crecimiento_matricula",
    "tasa_permanencia",
    "MATRICULADOS",
    "brecha_ingreso_egreso",
]

# Features categóricas (se one-hot-encodean)
CAT_FEATURES = [
    "SECTOR_IES",
    "CARACTER_IES",
]

LABEL_COL    = "riesgo_academico"
TRAIN_YEARS  = list(range(2013, 2022))   # entrenar: 2013-2021
TEST_YEARS   = list(range(2022, 2025))   # evaluar:  2022-2024

SEED = 42

# ─────────────────────────────────────────────
# SPARK
# ─────────────────────────────────────────────

def start_spark():
    return (SparkSession.builder
            .appName("SNIES_RiskPredictor")
            .master("local[*]")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "4g")
            .getOrCreate())

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_data(spark):
    pdf = pd.read_csv(INPUT_FILE, encoding="utf-8", low_memory=False)
    sdf = spark.createDataFrame(pdf)

    # Filtrar filas con features y label completas
    required = NUM_FEATURES + [LABEL_COL, "AÑO"]
    for col in required:
        sdf = sdf.filter(F.col(col).isNotNull())

    # Castear label a int
    sdf = sdf.withColumn(LABEL_COL, F.col(LABEL_COL).cast("int"))

    return sdf


def build_feature_pipeline(classifier, cat_features=CAT_FEATURES):
    """
    Pipeline: StringIndexer → OHE → VectorAssembler → StandardScaler → Classifier
    """
    stages = []

    # Indexar + OHE para categóricas
    ohe_output_cols = []
    for cat in cat_features:
        idx_col = f"{cat}_idx"
        ohe_col = f"{cat}_ohe"
        stages.append(StringIndexer(
            inputCol=cat, outputCol=idx_col,
            handleInvalid="keep"
        ))
        stages.append(OneHotEncoder(
            inputCols=[idx_col], outputCols=[ohe_col],
            handleInvalid="keep"
        ))
        ohe_output_cols.append(ohe_col)

    all_feature_cols = NUM_FEATURES + ohe_output_cols

    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol="features_raw",
        handleInvalid="skip"
    )
    stages.append(assembler)

    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,
        withStd=True
    )
    stages.append(scaler)
    stages.append(classifier)

    return Pipeline(stages=stages)


def evaluate_model(model, test_df, model_name):
    """Calcula AUC-ROC, Accuracy y F1 del modelo sobre test set."""
    preds = model.transform(test_df)

    auc_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    acc_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
        metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
        metricName="f1"
    )

    auc      = auc_eval.evaluate(preds)
    accuracy = acc_eval.evaluate(preds)
    f1       = f1_eval.evaluate(preds)

    print(f"   [{model_name}]  AUC-ROC={auc:.4f}  Accuracy={accuracy:.4f}  F1={f1:.4f}")
    return {"modelo": model_name, "AUC_ROC": auc, "Accuracy": accuracy, "F1_Score": f1}, preds


def extract_feature_importance(model, feature_names, model_name):
    """Extrae importancia de features para RF y GBT."""
    try:
        # El clasificador es el último stage del pipeline
        clf = model.stages[-1]
        importance = clf.featureImportances.toArray()

        # Las features OHE generan varias columnas → agrupar por nombre base
        # Para simplicidad, mostramos las numéricas directamente
        n_num = len(NUM_FEATURES)
        num_importance = importance[:n_num]

        df_imp = pd.DataFrame({
            "feature":    NUM_FEATURES,
            "importance": num_importance,
            "modelo":     model_name
        }).sort_values("importance", ascending=False)

        return df_imp
    except AttributeError:
        return pd.DataFrame()  # Logística no tiene featureImportances


def get_roc_data(model, test_df, model_name):
    """Extrae puntos de la curva ROC (FPR, TPR)."""
    preds = model.transform(test_df)

    # Extraer probabilidad de clase positiva
    get_prob = F.udf(lambda v: float(v[1]) if v else 0.0,
                     __import__("pyspark.sql.types",
                                fromlist=["DoubleType"]).DoubleType())
    preds = preds.withColumn("prob_pos", get_prob(F.col("probability")))

    pdf = preds.select("prob_pos", LABEL_COL).toPandas()

    # Calcular curva ROC manualmente con numpy
    thresholds = np.linspace(0, 1, 100)
    roc_points = []
    y_true = pdf[LABEL_COL].values
    y_prob = pdf["prob_pos"].values

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        roc_points.append({"threshold": t, "FPR": fpr, "TPR": tpr, "modelo": model_name})

    return pd.DataFrame(roc_points)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_predictor():
    spark = start_spark()
    spark.sparkContext.setLogLevel("ERROR")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  SNIES — Predictor de Riesgo Académico")
    print("  Modelos: Random Forest | GBT | Regresión Logística")
    print("=" * 60)

    # ── 1. Cargar datos ──────────────────────────────────────────
    print("\n[1/5] Cargando datos...")
    sdf = load_data(spark)

    # Balance de clases
    total   = sdf.count()
    riesgo  = sdf.filter(F.col(LABEL_COL) == 1).count()
    no_riesgo = total - riesgo
    print(f"   Total filas:      {total:,}")
    print(f"   En riesgo (=1):   {riesgo:,}  ({100*riesgo/total:.1f}%)")
    print(f"   Sin riesgo (=0):  {no_riesgo:,}  ({100*no_riesgo/total:.1f}%)")

    # ── 2. Split temporal ────────────────────────────────────────
    print(f"\n[2/5] Split temporal: train={TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}  "
          f"test={TEST_YEARS[0]}-{TEST_YEARS[-1]}")

    train_df = sdf.filter(F.col("AÑO").isin(TRAIN_YEARS))
    test_df  = sdf.filter(F.col("AÑO").isin(TEST_YEARS))

    print(f"   Train: {train_df.count():,} filas")
    print(f"   Test:  {test_df.count():,} filas")

    # ── 3. Definir clasificadores ────────────────────────────────
    classifiers = {
        "RandomForest": RandomForestClassifier(
            labelCol=LABEL_COL,
            featuresCol="features",
            numTrees=100,
            maxDepth=6,
            seed=SEED
        ),
        "GBT": GBTClassifier(
            labelCol=LABEL_COL,
            featuresCol="features",
            maxIter=50,
            maxDepth=5,
            seed=SEED,
            # GBT no soporta probabilidades directas → usar rawPrediction
        ),
        "LogisticRegression": LogisticRegression(
            labelCol=LABEL_COL,
            featuresCol="features",
            maxIter=100,
            regParam=0.1,
            elasticNetParam=0.0
        ),
    }

    # ── 4. Entrenar y evaluar ────────────────────────────────────
    print("\n[3/5] Entrenando y evaluando modelos...")

    results     = []
    best_model  = None
    best_auc    = -1
    best_name   = ""
    all_preds   = {}
    all_importance = []
    all_roc     = []

    for name, clf in classifiers.items():
        print(f"\n   ▶ Entrenando {name}...")

        # GBT no soporta probabilidades en rawPrediction para BinaryEval
        # → necesita ajuste
        if name == "GBT":
            # Usar umbral manual con rawPrediction
            clf_adjusted = GBTClassifier(
                labelCol=LABEL_COL,
                featuresCol="features",
                maxIter=50,
                maxDepth=5,
                seed=SEED,
            )
            pipeline = build_feature_pipeline(clf_adjusted)
            model = pipeline.fit(train_df)
            preds = model.transform(test_df)

            # Para GBT usamos rawPrediction[1] como score continuo
            from pyspark.sql.types import DoubleType
            get_raw = F.udf(lambda v: float(v[1]) if v else 0.0, DoubleType())
            preds = preds.withColumn("prob_gbt", get_raw(F.col("rawPrediction")))

            # AUC con rawPrediction
            auc_eval_gbt = BinaryClassificationEvaluator(
                labelCol=LABEL_COL,
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
            auc = auc_eval_gbt.evaluate(preds)

            acc_eval = MulticlassClassificationEvaluator(
                labelCol=LABEL_COL, predictionCol="prediction", metricName="accuracy")
            f1_eval = MulticlassClassificationEvaluator(
                labelCol=LABEL_COL, predictionCol="prediction", metricName="f1")
            accuracy = acc_eval.evaluate(preds)
            f1       = f1_eval.evaluate(preds)

            print(f"   [GBT]  AUC-ROC={auc:.4f}  Accuracy={accuracy:.4f}  F1={f1:.4f}")
            metrics = {"modelo": "GBT", "AUC_ROC": auc,
                       "Accuracy": accuracy, "F1_Score": f1}
            all_preds[name] = preds

            # ROC data para GBT (usar prob_gbt)
            pdf_gbt = preds.select("prob_gbt", LABEL_COL).toPandas()
            thresholds = np.linspace(0, 1, 100)
            y_true = pdf_gbt[LABEL_COL].values
            y_prob = pdf_gbt["prob_gbt"].values
            roc_pts = []
            for t in thresholds:
                y_pred = (y_prob >= t).astype(int)
                tp = int(((y_pred == 1) & (y_true == 1)).sum())
                fp = int(((y_pred == 1) & (y_true == 0)).sum())
                fn = int(((y_pred == 0) & (y_true == 1)).sum())
                tn = int(((y_pred == 0) & (y_true == 0)).sum())
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                roc_pts.append({"threshold": t, "FPR": fpr, "TPR": tpr, "modelo": "GBT"})
            all_roc.append(pd.DataFrame(roc_pts))

        else:
            pipeline = build_feature_pipeline(clf)
            model = pipeline.fit(train_df)
            metrics, preds = evaluate_model(model, test_df, name)
            all_preds[name] = preds
            all_roc.append(get_roc_data(model, test_df, name))

        results.append(metrics)

        # Importancia de features (RF y GBT)
        if name in ("RandomForest", "GBT"):
            imp_df = extract_feature_importance(model, NUM_FEATURES, name)
            if not imp_df.empty:
                all_importance.append(imp_df)

        # Guardar mejor modelo
        if metrics["AUC_ROC"] > best_auc:
            best_auc   = metrics["AUC_ROC"]
            best_model = model
            best_name  = name

    # ── 5. Guardar resultados ────────────────────────────────────
    print(f"\n[5/5] Guardando resultados...")
    print(f"\n   🏆 Mejor modelo: {best_name} (AUC-ROC={best_auc:.4f})")

    # 5a. Comparación de modelos
    df_results = pd.DataFrame(results).sort_values("AUC_ROC", ascending=False)
    df_results["mejor"] = df_results["modelo"] == best_name
    df_results.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

    # 5b. Predicciones del mejor modelo
    best_preds = all_preds[best_name]
    out_cols = ["COD_IES", "NOMBRE_IES", "SECTOR_IES", "CARACTER_IES",
                "DEPTO_IES", "AÑO", LABEL_COL, "prediction"]
    available = [c for c in out_cols if c in best_preds.columns]
    pdf_preds = best_preds.select(*available).toPandas()
    pdf_preds["modelo_ganador"] = best_name
    pdf_preds.to_csv(f"{OUTPUT_DIR}/risk_predictions.csv", index=False, encoding="utf-8")

    # 5c. Feature importance
    if all_importance:
        df_imp = pd.concat(all_importance, ignore_index=True)
        df_imp.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

    # 5d. Datos ROC por modelo
    df_roc = pd.concat(all_roc, ignore_index=True)
    df_roc.to_csv(f"{OUTPUT_DIR}/roc_data.csv", index=False)

    # ── Resumen final ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ RESULTADOS FINALES")
    print("=" * 60)
    print(df_results[["modelo", "AUC_ROC", "Accuracy", "F1_Score"]].to_string(index=False))
    print("\nArchivos guardados en data/processed/ml/")
    print("=" * 60)

    spark.stop()
    return df_results, best_name


if __name__ == "__main__":
    run_predictor()