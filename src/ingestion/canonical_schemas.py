"""
canonical_schemas.py
====================
Diccionario central de schemas canónicos del proyecto SNIES.

Responsabilidades:
- Definir las columnas canónicas (nombre oficial) por categoría.
- Mapear todas las variantes históricas (2013-2024) al nombre canónico.
- Definir tipos de datos esperados por columna.
- Proveer funciones de normalización reutilizables por el ETL PySpark.

Este módulo es importado por:
- preprocessing/normalizer.py
- integration/historical_merger.py
- storage/parquet_writer.py

Regla de oro:
- Si una columna no está en el mapa → se conserva tal cual y se loguea como advertencia.
- Nunca se descarta una columna silenciosamente.
"""

# ──────────────────────────────────────────────────────────────
# MAPA GLOBAL DE VARIANTES → CANÓNICO
# Cubre todas las inconsistencias detectadas en el schema_report
# ──────────────────────────────────────────────────────────────

"""
canonical_schemas.py  — VERSIÓN CORREGIDA
==========================================
Diccionario central de schemas canónicos SNIES.
Aliases extraídos directamente de los nombres reales de columnas
detectados en todos los archivos 2013-2024.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# MAPA GLOBAL DE VARIANTES → CANÓNICO
# Normalización: lower + sin tildes + strip + colapso de espacios
# ──────────────────────────────────────────────────────────────

COLUMN_ALIAS_MAP = {

    # ── Código Institución
    "codigo de \nla institucion":                   "CÓDIGO DE LA INSTITUCIÓN",
    "codigo de la institucion":                     "CÓDIGO DE LA INSTITUCIÓN",
    "codigo de la institucion ":                    "CÓDIGO DE LA INSTITUCIÓN",
    "codigo de \nla institucion ":                  "CÓDIGO DE LA INSTITUCIÓN",

    # ── IES Padre
    "ies padre":                                    "IES PADRE",
    "ies_padre":                                    "IES PADRE",

    # ── Institución IES
    "institucion de educacion superior (ies)":      "INSTITUCIÓN DE EDUCACIÓN SUPERIOR (IES)",

    # ── Principal o Seccional
    "principal\n o\nseccional":                     "PRINCIPAL O SECCIONAL",
    "principal o seccional":                        "PRINCIPAL O SECCIONAL",
    "principal \no\nseccional":                     "PRINCIPAL O SECCIONAL",
    "principal o\nseccional":                       "PRINCIPAL O SECCIONAL",

    # ── Tipo IES (2023+)
    "tipo ies":                                     "TIPO IES",

    # ── Sector IES
    "sector ies":                                   "SECTOR IES",
    "ies sector ies":                               "SECTOR IES",
    "id_sector":                                    "ID SECTOR IES",
    "id sector ies":                                "ID SECTOR IES",
    "id sector":                                    "ID SECTOR IES",

    # ── Carácter IES
    "caracter ies":                                 "CARÁCTER IES",
    "id_caracter":                                  "ID CARÁCTER IES",
    "id caracter ies":                              "ID CARÁCTER IES",
    "id caracter":                                  "ID CARÁCTER IES",
    "id caracter ies ":                             "ID CARÁCTER IES",

    # ── IES Acreditada
    "ies acreditada":                               "IES ACREDITADA",

    # ── Departamento IES
    "codigo del \ndepartamento\n(ies)":             "CÓDIGO DEL DEPARTAMENTO (IES)",
    "codigo del departamento (ies)":                "CÓDIGO DEL DEPARTAMENTO (IES)",
    "codigo del\ndepartamento":                     "CÓDIGO DEL DEPARTAMENTO (IES)",
    "departamento de \ndomicilio de la ies":        "DEPARTAMENTO DE DOMICILIO DE LA IES",
    "departamento de domicilio de la ies":          "DEPARTAMENTO DE DOMICILIO DE LA IES",
    "departamento de\ndomicilio de la ies":         "DEPARTAMENTO DE DOMICILIO DE LA IES",

    # ── Municipio IES
    "codigo del \nmunicipio\n(ies)":                "CÓDIGO DEL MUNICIPIO IES",
    "codigo del municipio (ies)":                   "CÓDIGO DEL MUNICIPIO IES",
    "codigo del\nmunicipio":                        "CÓDIGO DEL MUNICIPIO IES",
    "codigo del municipio":                         "CÓDIGO DEL MUNICIPIO IES",
    "codigo del municipio ies":                     "CÓDIGO DEL MUNICIPIO IES",
    "municipio de\ndomicilio de la ies":            "MUNICIPIO DE DOMICILIO DE LA IES",
    "municipio de domicilio de la ies":             "MUNICIPIO DE DOMICILIO DE LA IES",
    "municipio de domicilio de la ies ":            "MUNICIPIO DE DOMICILIO DE LA IES",

    # ── SNIES Programa
    "codigo \nsnies del\nprograma":                 "CÓDIGO SNIES DEL PROGRAMA",
    "codigo snies del programa":                    "CÓDIGO SNIES DEL PROGRAMA",
    "codigo snies del programa ":                   "CÓDIGO SNIES DEL PROGRAMA",

    # ── Programa Académico
    "programa academico":                           "PROGRAMA ACADÉMICO",
    "programa acreditado":                          "PROGRAMA ACREDITADO",

    # ── Nivel Académico
    "id_nivel":                                     "ID NIVEL ACADÉMICO",
    "id nivel academico":                           "ID NIVEL ACADÉMICO",
    "nivel academico":                              "NIVEL ACADÉMICO",

    # ── Nivel de Formación
    "id_nivel_formacion":                           "ID NIVEL DE FORMACIÓN",
    "id nivel de formacion":                        "ID NIVEL DE FORMACIÓN",
    "nivel de formacion":                           "NIVEL DE FORMACIÓN",

    # ── Metodología / Modalidad
    "id_metodologia":                               "ID MODALIDAD",
    "id metodologia":                               "ID MODALIDAD",
    "id modalidad":                                 "ID MODALIDAD",
    "metodologia \ndel programa":                   "MODALIDAD",
    "metodologia del programa":                     "MODALIDAD",
    "metodologia":                                  "MODALIDAD",
    "modalidad":                                    "MODALIDAD",

    # ── Área
    "id_area":                                      "ID ÁREA",
    "id area":                                      "ID ÁREA",
    "id area de conocimiento":                      "ID ÁREA",
    "area de conocimiento":                         "ÁREA DE CONOCIMIENTO",

    # ── Núcleo
    "id_nucleo":                                    "ID NÚCLEO",
    "id nucleo":                                    "ID NÚCLEO",
    "nucleo basico del conocimiento (nbc)":         "NÚCLEO BÁSICO DEL CONOCIMIENTO (NBC)",
    "nucleo basico del conocimiento":               "NÚCLEO BÁSICO DEL CONOCIMIENTO (NBC)",

    # ── CINE (2019+)
    "id cine campo amplio":                         "ID CINE CAMPO AMPLIO",
    "id cine campo amplio\tdesc":                   "ID CINE CAMPO AMPLIO",
    "desc cine campo amplio":                       "DESC CINE CAMPO AMPLIO",
    "id cine campo especifico":                     "ID CINE CAMPO ESPECÍFICO",
    "desc cine campo especifico":                   "DESC CINE CAMPO ESPECÍFICO",
    "id cine campo detallado":                      "ID CINE CAMPO DETALLADO",
    "desc cine campo detallado":                    "DESC CINE CAMPO DETALLADO",
    # Variante 2019-2022: "CODIGO DETALLADO" en vez de "CAMPO DETALLADO"
    "id cine codigo detallado":                     "ID CINE CAMPO DETALLADO",
    "desc cine codigo detallado":                   "DESC CINE CAMPO DETALLADO",
    # Variante con tab en graduados 2020/2022
    "id cine campo amplio\tdesc":                   "ID CINE CAMPO AMPLIO",
    "cine campo amplio":                            "DESC CINE CAMPO AMPLIO",

    # ── Departamento Programa
    "codigo del \ndepartamento\n(programa)":        "CÓDIGO DEL DEPARTAMENTO (PROGRAMA)",
    "codigo del departamento (programa)":           "CÓDIGO DEL DEPARTAMENTO (PROGRAMA)",
    "codigo del departamento\n(programa)":          "CÓDIGO DEL DEPARTAMENTO (PROGRAMA)",
    "departamento de oferta del programa":          "DEPARTAMENTO DE OFERTA DEL PROGRAMA",
    "departamento de\noferta del programa":         "DEPARTAMENTO DE OFERTA DEL PROGRAMA",

    # ── Municipio Programa
    "codigo del \nmunicipio\n(programa)":           "CÓDIGO DEL MUNICIPIO (PROGRAMA)",
    "codigo del municipio (programa)":              "CÓDIGO DEL MUNICIPIO (PROGRAMA)",
    "codigo del municipio\n(programa)":             "CÓDIGO DEL MUNICIPIO (PROGRAMA)",
    # Typo en graduados 2022
    "cdigo del municipio (programa)":               "CÓDIGO DEL MUNICIPIO (PROGRAMA)",
    "municipio de oferta del programa":             "MUNICIPIO DE OFERTA DEL PROGRAMA",
    "municipio de oferta del programa ":            "MUNICIPIO DE OFERTA DEL PROGRAMA",

    # ── Sexo / Género (variantes históricas)
    "id_genero":                                    "ID SEXO",
    "id genero":                                    "ID SEXO",
    "id sexo":                                      "ID SEXO",
    "genero":                                       "SEXO",
    "sexo":                                         "SEXO",

    # ── Tiempo
    "a\u00f1o":                                     "AÑO",
    "ano":                                          "AÑO",
    "a\u00f1o*":                                    "AÑO",
    "ano*":                                         "AÑO",
    "semestre":                                     "SEMESTRE",
    "a±o*":                                         "AÑO",
    "A±o*":                                         "AÑO",

    # ── Métricas por categoría
    # admitidos
    "admitidos 2014":                               "ADMITIDOS",
    "admitdos 2015":                                "ADMITIDOS",
    "admitdos 2016":                                "ADMITIDOS",
    "admisiones 2017":                              "ADMITIDOS",
    "Admisiones 2017":                              "ADMITIDOS",
    "admisiones 2018":                              "ADMITIDOS",
    "Admisiones 2018":                              "ADMITIDOS",
    "admitidos":                                    "ADMITIDOS",

    # inscritos
    "inscritos 2014":                               "INSCRITOS",
    "primer_curso 2015":                            "INSCRITOS",
    "inscritos 2016":                               "INSCRITOS",
    "inscripciones 2017":                           "INSCRITOS",
    "inscripciones 2018":                           "INSCRITOS",
    "inscritos":                                    "INSCRITOS",

    # matriculados
    "matriculados 2014":                            "MATRICULADOS",
    "matriculados 2015":                            "MATRICULADOS",
    "matriculados 2016":                            "MATRICULADOS",
    "matriculados 2017":                            "MATRICULADOS",
    "matriculados 2018":                            "MATRICULADOS",
    "matriculados":                                 "MATRICULADOS",

    # matriculados primer año
    "primer curso 2014":                            "MATRICULADOS PRIMER CURSO",
    "inscritos 2015":                               "MATRICULADOS PRIMER CURSO",
    "primer curso 2016":                            "MATRICULADOS PRIMER CURSO",
    "primer curso 2017":                            "MATRICULADOS PRIMER CURSO",
    "primer curso 2018":                            "MATRICULADOS PRIMER CURSO",
    "primer curso 2019":                            "MATRICULADOS PRIMER CURSO",
    "primer curso":                                 "MATRICULADOS PRIMER CURSO",
    "matriculados primer curso":                    "MATRICULADOS PRIMER CURSO",

    # graduados
    "graduados":                                    "GRADUADOS",

    # docentes — métricas
    "no. de docentes":                              "NUM DE DOCENTES",
    "docentes":                                     "NUM DE DOCENTES",

    # ── Docentes — columnas específicas
    "genero del\ndocente":                          "SEXO",
    "genero del docente":                           "SEXO",
    "sexo del\ndocente":                            "SEXO",
    "sexo del docente":                             "SEXO",
    "id genero":                                    "ID SEXO",
    "id sexo":                                      "ID SEXO",
    "tipo de documento":                            "TIPO DE DOCUMENTO",
    "maximo nivel \nde formacion\ndel docente":     "MÁXIMO NIVEL DE FORMACIÓN DEL DOCENTE",
    "maximo nivel de formacion del docente":        "MÁXIMO NIVEL DE FORMACIÓN DEL DOCENTE",
    "id maximo_nivel":                              "ID MÁXIMO NIVEL DE FORMACIÓN DEL DOCENTE",
    "id maximo nivel de formacion del docente":     "ID MÁXIMO NIVEL DE FORMACIÓN DEL DOCENTE",
    "tiempo de dedicacion\ndel docente":            "TIEMPO DE DEDICACIÓN DEL DOCENTE",
    "tiempo de dedicacion\ndel docente1":           "TIEMPO DE DEDICACIÓN DEL DOCENTE",
    "tiempo de dedicacion del docente1":            "TIEMPO DE DEDICACIÓN DEL DOCENTE",
    "tiempo de dedicacion del docente 1":           "TIEMPO DE DEDICACIÓN DEL DOCENTE",
    "tiempo de dedicacion del docente":             "TIEMPO DE DEDICACIÓN DEL DOCENTE",
    # Columnas duplicadas con .1 .2 .3 en docentes 2019 (pandas auto-rename)
    "tiempo de dedicacion\ndel docente.1":          "TIEMPO DE DEDICACIÓN DEL DOCENTE 2",
    "tiempo de dedicacion del docente.1":           "TIEMPO DE DEDICACIÓN DEL DOCENTE 1",
    "tiempo de dedicacion\ndel docente.2":          "TIEMPO DE DEDICACIÓN DEL DOCENTE 3",
    "tiempo de dedicacion\ndel docente.3":          "TIEMPO DE DEDICACIÓN DEL DOCENTE 4",
    "id dedicacion":                                "ID TIEMPO DE DEDICACIÓN",
    "id tiempo de dedicacion":                      "ID TIEMPO DE DEDICACIÓN",
    "tipo de contrato\ndel docente":                "TIPO DE CONTRATO DEL DOCENTE",
    "tipo de contrato del docente":                 "TIPO DE CONTRATO DEL DOCENTE",
    "tipo de contrato":                             "TIPO DE CONTRATO DEL DOCENTE",
    "id tipo_contrato":                             "ID TIPO DE CONTRATO",
    "id tipo de contrato":                          "ID TIPO DE CONTRATO",
    "capital o\nmunicipio":                         "CAPITAL O MUNICIPIO",
    "nivel cine":                                   "NIVEL CINE",

    # ── Variantes con saltos de línea en municipio IES docentes
    "codigo del\nmunicipio":                        "CÓDIGO DEL MUNICIPIO IES",
    "municipio de\ndomicilio de la ies":            "MUNICIPIO DE DOMICILIO DE LA IES",
}


# ──────────────────────────────────────────────────────────────
# SCHEMAS CANÓNICOS POR CATEGORÍA
# ──────────────────────────────────────────────────────────────

_DIM_INSTITUCION = [
    "CÓDIGO DE LA INSTITUCIÓN",
    "IES PADRE",
    "INSTITUCIÓN DE EDUCACIÓN SUPERIOR (IES)",
    "TIPO IES",
    "ID SECTOR IES",
    "SECTOR IES",
    "ID CARÁCTER IES",
    "CARÁCTER IES",
    "IES ACREDITADA",
    "PRINCIPAL O SECCIONAL",
    "CÓDIGO DEL DEPARTAMENTO (IES)",
    "DEPARTAMENTO DE DOMICILIO DE LA IES",
    "CÓDIGO DEL MUNICIPIO IES",
    "MUNICIPIO DE DOMICILIO DE LA IES",
]

_DIM_PROGRAMA = [
    "CÓDIGO SNIES DEL PROGRAMA",
    "PROGRAMA ACADÉMICO",
    "PROGRAMA ACREDITADO",
    "ID NIVEL ACADÉMICO",
    "NIVEL ACADÉMICO",
    "ID NIVEL DE FORMACIÓN",
    "NIVEL DE FORMACIÓN",
    "ID MODALIDAD",
    "MODALIDAD",
    "ID ÁREA",
    "ÁREA DE CONOCIMIENTO",
    "ID NÚCLEO",
    "NÚCLEO BÁSICO DEL CONOCIMIENTO (NBC)",
    "ID CINE CAMPO AMPLIO",
    "DESC CINE CAMPO AMPLIO",
    "ID CINE CAMPO ESPECÍFICO",
    "DESC CINE CAMPO ESPECÍFICO",
    "ID CINE CAMPO DETALLADO",
    "DESC CINE CAMPO DETALLADO",
    "CÓDIGO DEL DEPARTAMENTO (PROGRAMA)",
    "DEPARTAMENTO DE OFERTA DEL PROGRAMA",
    "CÓDIGO DEL MUNICIPIO (PROGRAMA)",
    "MUNICIPIO DE OFERTA DEL PROGRAMA",
]

_DIM_TIEMPO = ["AÑO", "SEMESTRE"]
_DIM_SEXO = ["ID SEXO", "SEXO"]

_DIM_DOCENTES_EXTRA = [
    "TIPO DE DOCUMENTO",
    "ID MÁXIMO NIVEL DE FORMACIÓN DEL DOCENTE",
    "MÁXIMO NIVEL DE FORMACIÓN DEL DOCENTE",
    "ID TIEMPO DE DEDICACIÓN",
    "TIEMPO DE DEDICACIÓN DEL DOCENTE",
    "ID TIPO DE CONTRATO",
    "TIPO DE CONTRATO DEL DOCENTE",
]

# Columnas nullable (ausentes en años anteriores)
NULLABLE_COLUMNS = {
    "TIPO IES", "IES ACREDITADA", "PROGRAMA ACREDITADO",
    "ID SECTOR IES", "ID CARÁCTER IES", "ID NIVEL ACADÉMICO",
    "ID NIVEL DE FORMACIÓN", "ID MODALIDAD", "ID ÁREA", "ID NÚCLEO",
    "ID SEXO", "PRINCIPAL O SECCIONAL", "IES PADRE",
    "CÓDIGO DEL DEPARTAMENTO (IES)", "CÓDIGO DEL MUNICIPIO IES",
    "CÓDIGO DEL DEPARTAMENTO (PROGRAMA)", "CÓDIGO DEL MUNICIPIO (PROGRAMA)",
    "ID CINE CAMPO AMPLIO", "DESC CINE CAMPO AMPLIO",
    "ID CINE CAMPO ESPECÍFICO", "DESC CINE CAMPO ESPECÍFICO",
    "ID CINE CAMPO DETALLADO", "DESC CINE CAMPO DETALLADO",
    "ID MÁXIMO NIVEL DE FORMACIÓN DEL DOCENTE",
    "ID TIEMPO DE DEDICACIÓN", "ID TIPO DE CONTRATO",
    "TIPO DE DOCUMENTO", "CAPITAL O MUNICIPIO", "NIVEL CINE",
}

CATEGORY_SCHEMAS = {
    "admitidos": {
        "columns": _DIM_INSTITUCION + _DIM_PROGRAMA + _DIM_SEXO + _DIM_TIEMPO + ["ADMITIDOS"],
        "metric_column": "ADMITIDOS",
    },
    "inscritos": {
        "columns": _DIM_INSTITUCION + _DIM_PROGRAMA + _DIM_SEXO + _DIM_TIEMPO + ["INSCRITOS"],
        "metric_column": "INSCRITOS",
    },
    "matriculados": {
        "columns": _DIM_INSTITUCION + _DIM_PROGRAMA + _DIM_SEXO + _DIM_TIEMPO + ["MATRICULADOS"],
        "metric_column": "MATRICULADOS",
    },
    "matriculados_primer_anho": {
        "columns": _DIM_INSTITUCION + _DIM_PROGRAMA + _DIM_SEXO + _DIM_TIEMPO + ["MATRICULADOS PRIMER CURSO"],
        "metric_column": "MATRICULADOS PRIMER CURSO",
    },
    "graduados": {
        "columns": _DIM_INSTITUCION + _DIM_PROGRAMA + _DIM_SEXO + _DIM_TIEMPO + ["GRADUADOS"],
        "metric_column": "GRADUADOS",
    },
    "docentes": {
        "columns": _DIM_INSTITUCION + _DIM_SEXO + _DIM_DOCENTES_EXTRA + _DIM_TIEMPO + ["NUM DE DOCENTES"],
        "metric_column": "NUM DE DOCENTES",
    },
}

# Archivos pivoteados que requieren tratamiento especial
PIVOTED_FILES = {
    ("admitidos", "2013"),
    ("inscritos", "2013"),
    ("matriculados", "2013"),
    ("matriculados_primer_anho", "2013"),
    ("docentes", "2013"),
}


# ──────────────────────────────────────────────────────────────
# FUNCIONES DE NORMALIZACIÓN
# ──────────────────────────────────────────────────────────────

def normalize_col_key(col: str) -> str:
    if not col:
        return ""
    text = str(col).strip().lower()
    text = re.sub(r"\s+", " ", text)
    for acc, plain in {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ñ":"n"}.items():
        text = text.replace(acc, plain)
    return text


def resolve_canonical(col: str) -> str:
    key = normalize_col_key(col)
    canonical = COLUMN_ALIAS_MAP.get(key)
    if canonical:
        return canonical
    cleaned = str(col).strip()
    logger.debug(f"Sin mapeo canónico: '{cleaned}' (key='{key}')")
    return cleaned


def get_canonical_columns(category: str) -> list:
    if category not in CATEGORY_SCHEMAS:
        raise KeyError(f"Categoría desconocida: '{category}'")
    return CATEGORY_SCHEMAS[category]["columns"]


def get_metric_column(category: str) -> str:
    return CATEGORY_SCHEMAS[category]["metric_column"]


def is_nullable(col: str) -> bool:
    return col in NULLABLE_COLUMNS


def is_pivoted_file(category: str, year: str) -> bool:
    return (category, str(year)) in PIVOTED_FILES


def rename_columns_to_canonical(columns: list) -> dict:
    rename_map = {}
    for col in columns:
        canonical = resolve_canonical(col)
        if canonical != col:
            rename_map[col] = canonical
    return rename_map


def validate_schema(columns: list, category: str) -> dict:
    expected = set(get_canonical_columns(category))
    present = set(columns)
    missing_all = expected - present
    missing_required = {c for c in missing_all if not is_nullable(c)}
    missing_nullable = {c for c in missing_all if is_nullable(c)}
    unexpected = present - expected
    return {
        "missing_required": sorted(missing_required),
        "missing_nullable": sorted(missing_nullable),
        "unexpected": sorted(unexpected),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("SCHEMAS CANÓNICOS SNIES")
    for category, schema in CATEGORY_SCHEMAS.items():
        cols = schema["columns"]
        print(f"\n[{category.upper()}] — {len(cols)} cols | métrica: '{schema['metric_column']}'")
    print(f"\nTotal variantes mapeadas: {len(COLUMN_ALIAS_MAP)}")
    print(f"Archivos pivoteados: {PIVOTED_FILES}")