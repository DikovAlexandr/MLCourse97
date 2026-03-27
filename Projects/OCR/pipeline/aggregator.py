"""
pipeline/aggregator.py
Шаг 5: рендер страницы с сохранением оригинального расположения регионов.

Каждый элемент рисуется по координатам bbox из YOLO детекции —
текст/формула вписывается в bbox, картинка масштабируется в bbox.
"""
import base64
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from config import TARGET_WIDTH, TARGET_HEIGHT, PICTURE_EMBED
from pipeline.detector import PageDetections, DetectedRegion


# ── Константы рендера ─────────────────────────────────────────────────────────
FONT_SIZE_TEXT    = 18
FONT_SIZE_FORMULA = 16
LINE_HEIGHT       = 22
TEXT_COLOR        = (20, 20, 20)
FORMULA_COLOR     = (10, 60, 160)
BG_COLOR          = (255, 255, 255)

# Полупрозрачные рамки для отладки (False = без рамок в продакшене)
DEBUG_BOXES       = False
DEBUG_COLORS = {
    "text":    (33,  150, 243),
    "formula": (255, 87,  34),
    "picture": (76,  175, 80),
}


def aggregate_page(
    page_idx: int,
    detections: PageDetections,
    text_results:    Dict[int, str],
    formula_results: Dict[int, str],
    crop_paths:      Dict[str, List[Path]],
) -> Tuple[Image.Image, str]:
    """
    Рендерит страницу: каждый регион рисуется по своим bbox координатам.

    Returns:
        (PIL.Image rendered page, markdown string)
    """
    canvas = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    font_text, font_formula = _load_fonts()
    md_lines = [f"\n\n---\n\n## Страница {page_idx + 1}\n"]

    # Счётчики для индексации результатов
    text_idx    = 0
    formula_idx = 0
    picture_idx = 0

    # Рендерим в порядке сверху вниз (all_regions уже отсортированы по y0)
    for region in detections.all_regions:
        x0, y0, x1, y1 = region.bbox
        bbox_w = x1 - x0
        bbox_h = y1 - y0

        # Отладочные рамки
        if DEBUG_BOXES:
            color = DEBUG_COLORS.get(region.label, (200, 200, 200))
            draw.rectangle([x0, y0, x1, y1],
                           outline=color, width=2)

        if region.label == "text":
            content = text_results.get(text_idx, "")
            text_idx += 1
            if content:
                _render_text_in_bbox(draw, content, x0, y0, bbox_w, bbox_h, font_text)
                md_lines.append(content + "\n")

        elif region.label == "formula":
            img_path = formula_results.get(formula_idx, "")
            formula_idx += 1
            if img_path and Path(img_path).exists():
                _render_picture_in_bbox(canvas, Path(img_path), x0, y0, bbox_w, bbox_h)
                md_lines.append(_md_image(Path(img_path), f"formula_{formula_idx}"))

        elif region.label == "picture":
            paths = crop_paths.get("picture", [])
            if picture_idx < len(paths):
                img_path = paths[picture_idx]
                _render_picture_in_bbox(canvas, img_path, x0, y0, bbox_w, bbox_h)
                md_lines.append(_md_image(img_path, f"picture_{picture_idx}"))
            picture_idx += 1

    return canvas, "\n".join(md_lines)


def aggregate_document(
    rendered_pages: List[Image.Image],
    page_markdowns: List[str],
) -> Tuple[List[Image.Image], str]:
    full_md = "# Оцифрованный документ\n" + "\n".join(page_markdowns)
    return rendered_pages, full_md


# ── Рендер по bbox ────────────────────────────────────────────────────────────

def _render_text_in_bbox(
    draw: ImageDraw.Draw,
    text: str,
    x0: int, y0: int,
    bbox_w: int, bbox_h: int,
    font,
) -> None:
    """Вписывает текст в bbox с автопереносом строк."""
    # Оцениваем сколько символов влезает по ширине
    char_w = max(1, FONT_SIZE_TEXT // 2 + 1)
    chars_per_line = max(1, bbox_w // char_w)

    lines = []
    for paragraph in text.split("\n"):
        wrapped = textwrap.wrap(paragraph, width=chars_per_line)
        lines += wrapped if wrapped else [""]

    y = y0
    for line in lines:
        if y + LINE_HEIGHT > y0 + bbox_h:
            break
        draw.text((x0, y), line, font=font, fill=TEXT_COLOR)
        y += LINE_HEIGHT


def _render_formula_in_bbox(
    draw: ImageDraw.Draw,
    latex: str,
    x0: int, y0: int,
    bbox_w: int,
    font,
) -> None:
    """Рисует LaTeX строку формулы в bbox."""
    clean = latex.strip().strip("$").strip()
    # Усекаем если не влезает
    max_chars = max(1, bbox_w // (FONT_SIZE_FORMULA // 2 + 1))
    if len(clean) > max_chars:
        clean = clean[:max_chars - 1] + "…"
    draw.text((x0, y0), clean, font=font, fill=FORMULA_COLOR)


def _render_picture_in_bbox(
    canvas: Image.Image,
    img_path: Path,
    x0: int, y0: int,
    bbox_w: int, bbox_h: int,
) -> None:
    """Масштабирует кроп картинки и вставляет точно в bbox."""
    try:
        pic = Image.open(img_path).convert("RGB")
        # Масштабируем с сохранением пропорций в пределах bbox
        pic.thumbnail((bbox_w, bbox_h), Image.LANCZOS)
        canvas.paste(pic, (x0, y0))
    except Exception as e:
        print(f"[Aggregator] Ошибка вставки картинки {img_path}: {e}")


# ── Вспомогательные ───────────────────────────────────────────────────────────

def _load_fonts():
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    font_text = font_formula = None
    for path in font_paths:
        try:
            font_text    = ImageFont.truetype(path, FONT_SIZE_TEXT)
            font_formula = ImageFont.truetype(path, FONT_SIZE_FORMULA)
            break
        except Exception:
            continue
    if font_text is None:
        font_text    = ImageFont.load_default()
        font_formula = ImageFont.load_default()
    return font_text, font_formula


def _md_image(img_path: Path, label: str = None) -> str:
    tag = label or img_path.stem
    if PICTURE_EMBED:
        try:
            data = base64.b64encode(img_path.read_bytes()).decode()
            return f"\n![{tag}](data:image/png;base64,{data})\n"
        except Exception:
            pass
    return f"\n![{tag}]({img_path})\n"