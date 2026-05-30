from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CATALOG_DIR = DATA_DIR / "catalog"
ATTR_DIR = DATA_DIR / "attrs"
INDEX_DIR = DATA_DIR / "index"
MODEL_DIR = ROOT_DIR / "models"
DEMO_IMAGES_DIR = DATA_DIR / "demo_images"
EMBED_CACHE_DIR = DATA_DIR / "embed_cache"

# ── Gemini Embedding ─────────────────────────────────────────────────
GEMINI_EMBED_MODEL = "gemini-embedding-2"
GEMINI_EMBED_DIM = 768
GEMINI_EMBED_TASK_DOC = "RETRIEVAL_DOCUMENT"
GEMINI_EMBED_TASK_QUERY = "RETRIEVAL_QUERY"

# ── Gemini LLM (query parsing & re-ranking) ──────────────────────────
GEMINI_LLM_MODEL = "gemini-2.0-flash"

# ── Rate limiting ────────────────────────────────────────────────────
GEMINI_RPM_LIMIT = 15
GEMINI_BATCH_DELAY = 4.1

# ── Retrieval ────────────────────────────────────────────────────────
TOP_K = 20
RERANK_CANDIDATES = 60
