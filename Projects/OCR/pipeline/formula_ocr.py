"""
pipeline/formula_ocr.py
Шаг 4b: формулы не распознаются — кроп вставляется как изображение.
"""
from pathlib import Path
from typing import List


def load_formula_model():
    """Модель не нужна — возвращаем None."""
    return None


def recognize_formula_regions(
    model,
    crop_paths: List[Path],
) -> List[str]:
    """
    Возвращает пути к кропам формул как строки.
    aggregator.py вставит их как изображения.
    """
    return [str(p) for p in crop_paths]