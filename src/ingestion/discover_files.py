from pathlib import Path
import re


RAW_DATA_PATH = Path("data/raw/Analitica_educativa")


CATEGORY_PATTERNS = {
    "matriculados_primer_anho": r"primer_anho",
    "admitidos": r"admitidos",
    "inscritos": r"inscritos",
    "matriculados": r"matriculados",
    "docentes": r"docentes",
    "graduados": r"graduados",
}


def detect_category(filename: str) -> str:
    """
    Detecta automáticamente la categoría del archivo.
    """

    filename = filename.lower()

    if "primer_anho" in filename:
        return "matriculados_primer_anho"

    for category, pattern in CATEGORY_PATTERNS.items():
        if re.search(pattern, filename):
            return category

    return "unknown"


def extract_year(filename: str):
    """
    Extrae el año desde el nombre del archivo.
    """

    match = re.search(r"(20\d{2})", filename)

    if match:
        return int(match.group(1))

    return None


def discover_files():
    """
    Recorre automáticamente todos los archivos del data lake raw.
    """

    discovered = []

    for file_path in RAW_DATA_PATH.rglob("*.xlsx"):

        filename = file_path.name

        category = detect_category(filename)

        year = extract_year(filename)

        # Caso especial graduados 2001-2017
        if "2001-2017" in filename:
            year = "2001_2017"

        discovered.append({
            "year": year,
            "category": category,
            "path": str(file_path),
            "filename": filename
        })

    return discovered


if __name__ == "__main__":

    files = discover_files()

    for file in files:
        print(file)