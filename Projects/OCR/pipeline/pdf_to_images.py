"""
pipeline/pdf_to_images.py
Шаг 1: PDF → список PIL.Image одинакового размера и качества.
"""
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageOps
from pdf2image import convert_from_path

from config import PDF_DPI, TARGET_WIDTH, TARGET_HEIGHT


def pdf_to_images(
    pdf_path: str | Path,
    page_nums: Optional[List[int]] = None,
    dpi: int = PDF_DPI,
) -> List[Image.Image]:
    all_images = convert_from_path(str(pdf_path), dpi=dpi)
    total = len(all_images)

    indices = (
        [p - 1 for p in page_nums if 1 <= p <= total]
        if page_nums else list(range(total))
    )

    return [_normalize_image(all_images[i]) for i in indices]


def _normalize_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)

    canvas = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (255, 255, 255))
    offset = (
        (TARGET_WIDTH  - img.width)  // 2,
        (TARGET_HEIGHT - img.height) // 2,
    )
    canvas.paste(img, offset)
    return ImageOps.autocontrast(canvas, cutoff=1)
