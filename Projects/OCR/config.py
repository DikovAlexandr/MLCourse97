"""
config.py — все настройки проекта в одном месте.
"""
from pathlib import Path

# ── Пути ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
YOLO_MODEL = MODELS_DIR / "best.pt"

# ── PDF → Image ────────────────────────────────────────────────────────────────
PDF_DPI       = 300
TARGET_WIDTH  = 1240
TARGET_HEIGHT = 1754

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_IMG_SIZE = 1280
YOLO_CONF     = 0.25
CLASS_NAMES   = ["formula", "picture", "text"]

# ── TrOCR (текст) ─────────────────────────────────────────────────────────────
TROCR_MODEL_DIR = MODELS_DIR / "trocr_finetuned"
TROCR_NUM_BEAMS = 4

# ── Агрегация / рендер ────────────────────────────────────────────────────────
PICTURE_EMBED = True    # True = base64 в Markdown, False = ссылка на файл