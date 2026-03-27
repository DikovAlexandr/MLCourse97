"""
pipeline/crop_regions.py
Шаг 3: вырезка bbox регионов из страницы.
Кропы сохраняются во временную директорию переданную из orchestrator.
Никаких постоянных папок на диске не создаётся.
"""
from pathlib import Path
from typing import Dict, List

from PIL import Image

from pipeline.detector import PageDetections


def crop_page_regions(
    image: Image.Image,
    detections: PageDetections,
    base_dir: Path,
    padding: int = 6,
) -> Dict[str, List[Path]]:
    """
    Вырезает регионы из страницы и сохраняет в base_dir.

    Args:
        image:      PIL изображение страницы
        detections: результат детекции
        base_dir:   временная директория (из tempfile.TemporaryDirectory)
        padding:    отступ в пикселях вокруг bbox

    Returns:
        {"text": [path, ...], "formula": [...], "picture": [...]}
    """
    W, H = image.size
    result: Dict[str, List[Path]] = {"text": [], "formula": [], "picture": []}

    groups = {
        "text":    detections.texts,
        "formula": detections.formulas,
        "picture": detections.pictures,
    }

    for label, regions in groups.items():
        label_dir = base_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for idx, region in enumerate(regions):
            x0, y0, x1, y1 = region.bbox
            x0 = max(0, x0 - padding)
            y0 = max(0, y0 - padding)
            x1 = min(W, x1 + padding)
            y1 = min(H, y1 + padding)

            crop      = image.crop((x0, y0, x1, y1))
            out_path  = label_dir / f"region_{idx:02d}.png"
            crop.save(out_path, "PNG")
            result[label].append(out_path)

    return result