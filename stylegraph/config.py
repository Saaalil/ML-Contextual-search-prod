from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CATALOG_DIR = DATA_DIR / "catalog"
ATTR_DIR = DATA_DIR / "attrs"
INDEX_DIR = DATA_DIR / "index"
MODEL_DIR = ROOT_DIR / "models"
