"""
pipeline/orchestrator.py
Главный пайплайн — соединяет все шаги.
Временные файлы создаются через tempfile и удаляются автоматически.
"""
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image

from pipeline.pdf_to_images  import pdf_to_images
from pipeline.detector        import load_detector, detect_page
from pipeline.crop_regions    import crop_page_regions
from pipeline.text_ocr        import load_text_model, recognize_text_regions, deduplicate_texts
from pipeline.formula_ocr     import load_formula_model, recognize_formula_regions
from pipeline.aggregator      import aggregate_page, aggregate_document


class OCRPipeline:
    """Загружает модели один раз, обрабатывает любое число PDF."""

    def __init__(self):
        print("[Pipeline] Загрузка моделей...")
        self.detector      = load_detector()
        self.text_model    = load_text_model()
        self.formula_model = load_formula_model()
        print("[Pipeline] Готово.")

    def process(
        self,
        pdf_path: str | Path,
        page_nums: Optional[List[int]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[List[Image.Image], str]:
        """
        Полный пайплайн: PDF → (список PNG страниц, Markdown).

        Args:
            pdf_path:          путь к PDF
            page_nums:         список страниц (None = все)
            progress_callback: fn(current, total, message)

        Returns:
            (rendered_pages, full_markdown)
        """
        def _progress(cur, total, msg):
            if progress_callback:
                progress_callback(cur, total, msg)
            print(f"[{cur}/{total}] {msg}")

        # Шаг 1 — PDF → изображения
        _progress(0, 1, "Конвертация PDF...")
        images = pdf_to_images(pdf_path, page_nums)
        total  = len(images)
        _progress(1, total, f"Загружено страниц: {total}")

        rendered_pages: List[Image.Image] = []
        page_markdowns: List[str]         = []

        # Одна общая временная директория на весь документ —
        # удаляется автоматически после выхода из with-блока
        with tempfile.TemporaryDirectory(prefix="ocr_") as tmp_root:
            tmp_path = Path(tmp_root)

            for idx, image in enumerate(images):
                page_no  = idx + 1
                page_dir = tmp_path / f"page_{idx:03d}"
                page_dir.mkdir()

                _progress(idx, total, f"Страница {page_no}/{total}: детекция...")
                detections = detect_page(self.detector, image, idx)

                _progress(idx, total, f"Страница {page_no}/{total}: вырезка регионов...")
                crop_paths = crop_page_regions(image, detections, page_dir)

                _progress(idx, total, f"Страница {page_no}/{total}: распознавание текста...")
                raw_texts    = recognize_text_regions(self.text_model, crop_paths["text"])
                deduped      = deduplicate_texts(raw_texts, ngram_size=4)
                text_results = {i: t for i, t in enumerate(deduped)}

                _progress(idx, total, f"Страница {page_no}/{total}: обработка формул...")
                formula_list    = recognize_formula_regions(self.formula_model, crop_paths["formula"])
                formula_results = {i: f for i, f in enumerate(formula_list)}

                _progress(idx, total, f"Страница {page_no}/{total}: рендер...")
                rendered, md = aggregate_page(
                    idx, detections, text_results, formula_results, crop_paths
                )
                rendered_pages.append(rendered)
                page_markdowns.append(md)

        # Шаг 6 — сборка документа (tmp уже удалён, но rendered_pages в памяти)
        _progress(total, total, "Сборка документа...")
        pages, full_md = aggregate_document(rendered_pages, page_markdowns)

        return pages, full_md