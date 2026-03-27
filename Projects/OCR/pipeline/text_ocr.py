"""
pipeline/text_ocr.py
Шаг 4a: распознавание текстовых регионов через дообученный TrOCR.
Модель: models/trocr_finetuned/ (safetensors формат)
"""
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from config import MODELS_DIR

TROCR_MODEL_DIR = MODELS_DIR / "trocr_finetuned"


def load_text_model() -> Tuple:
    """
    Загружает дообученный TrOCR из models/trocr_finetuned/.
    Вызывается один раз при старте приложения.

    Returns:
        Кортеж (processor, model, device)
    """
    print(f"[TextOCR] Загрузка модели из {TROCR_MODEL_DIR} ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = TrOCRProcessor.from_pretrained(str(TROCR_MODEL_DIR))
    model = VisionEncoderDecoderModel.from_pretrained(
        str(TROCR_MODEL_DIR),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    print(f"[TextOCR] Готово. Device: {device}")
    return processor, model, device


def recognize_text_regions(
    text_model: Tuple,
    crop_paths: List[Path],
    num_beams: int = 4,
    max_new_tokens: int = 128,
) -> List[str]:
    """
    Распознаёт текст на каждом кропе.

    Args:
        text_model:     кортеж (processor, model, device) из load_text_model()
        crop_paths:     список путей к PNG файлам текстовых регионов
        num_beams:      beam search width (4 лучше greedy)
        max_new_tokens: максимальная длина генерации

    Returns:
        Список строк — по одной на каждый кроп (в том же порядке).
        Если файл не найден или ошибка — возвращает пустую строку.
    """
    if not crop_paths:
        return []

    processor, model, device = text_model
    results = []

    for crop_path in crop_paths:
        try:
            image = Image.open(crop_path).convert("RGB")
            pixel_values = processor(
                images=image, return_tensors="pt"
            ).pixel_values.to(device)

            if device.type == "cuda":
                pixel_values = pixel_values.half()

            with torch.inference_mode():
                generated_ids = model.generate(
                    pixel_values,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    no_repeat_ngram_size=3,  # убирает повторы "ананан"
                    early_stopping=True,
                )

            text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            results.append(text)

        except Exception as e:
            print(f"[TextOCR] Ошибка на {crop_path}: {e}")
            results.append("")

    return results


def deduplicate_texts(texts: List[str], ngram_size: int = 4) -> List[str]:
    """
    Удаляет дублирующиеся текстовые регионы на основе n-gram перекрытия.

    Логика:
    - Строим n-граммы для каждого текста
    - Если n-граммы одного текста полностью входят в n-граммы другого
      (то есть один является подмножеством другого) — оставляем БОЛЕЕ КОРОТКИЙ
    - Дубликат помечается пустой строкой

    Args:
        texts:      список распознанных строк
        ngram_size: размер n-граммы (4 = баланс точности/полноты)

    Returns:
        Список строк с удалёнными дубликатами (дубликат → "")
    """
    def get_ngrams(text: str, n: int):
        words = text.split()
        if len(words) < n:
            # Для коротких текстов используем биграммы или сам текст
            return set(zip(*[words[i:] for i in range(min(n, len(words)))]))
        return set(zip(*[words[i:] for i in range(n)]))

    result = list(texts)
    n = len(texts)

    for i in range(n):
        if not result[i]:
            continue
        ngrams_i = get_ngrams(result[i], ngram_size)
        if not ngrams_i:
            continue

        for j in range(n):
            if i == j or not result[j]:
                continue
            ngrams_j = get_ngrams(result[j], ngram_size)
            if not ngrams_j:
                continue

            # Проверяем является ли один набор подмножеством другого
            i_in_j = ngrams_i.issubset(ngrams_j)
            j_in_i = ngrams_j.issubset(ngrams_i)

            if i_in_j or j_in_i:
                # Оставляем более короткий текст
                if len(result[i]) <= len(result[j]):
                    result[j] = ""  # j длиннее — удаляем j
                else:
                    result[i] = ""  # i длиннее — удаляем i
                    break           # i удалён, переходим к следующему i

    return result