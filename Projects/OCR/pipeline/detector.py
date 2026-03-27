"""
pipeline/detector.py
Шаг 2: YOLO детектор регионов на странице.
Все списки регионов отсортированы по y0 (сверху вниз).
"""
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from PIL import Image
from ultralytics import YOLO

from config import YOLO_MODEL, YOLO_IMG_SIZE, YOLO_CONF, CLASS_NAMES


@dataclass
class DetectedRegion:
    label:      str
    bbox:       tuple       # (x0, y0, x1, y1) в пикселях
    confidence: float
    page_idx:   int


@dataclass
class PageDetections:
    page_idx: int
    texts:    List[DetectedRegion] = field(default_factory=list)
    formulas: List[DetectedRegion] = field(default_factory=list)
    pictures: List[DetectedRegion] = field(default_factory=list)

    def sort_by_position(self):
        """Сортирует каждый класс по y0 — гарантирует совпадение
        порядка регионов и порядка crop файлов на диске."""
        self.texts    = sorted(self.texts,    key=lambda r: r.bbox[1])
        self.formulas = sorted(self.formulas, key=lambda r: r.bbox[1])
        self.pictures = sorted(self.pictures, key=lambda r: r.bbox[1])

    @property
    def all_regions(self) -> List[DetectedRegion]:
        """Все регионы отсортированные сверху вниз по y0."""
        all_r = self.texts + self.formulas + self.pictures
        return sorted(all_r, key=lambda r: r.bbox[1])


def load_detector() -> YOLO:
    return YOLO(str(YOLO_MODEL))


def detect_page(model: YOLO, image: Image.Image, page_idx: int) -> PageDetections:
    detections = PageDetections(page_idx=page_idx)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    image.save(tmp_path)

    pred = model.predict(tmp_path, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF, verbose=False)[0]
    Path(tmp_path).unlink(missing_ok=True)

    for box in pred.boxes:
        cls   = int(box.cls[0])
        conf  = float(box.conf[0])
        bbox  = tuple(map(int, box.xyxy[0].tolist()))
        label = CLASS_NAMES[cls]

        region = DetectedRegion(
            label=label, bbox=bbox, confidence=conf, page_idx=page_idx
        )
        if label == "text":
            detections.texts.append(region)
        elif label == "formula":
            detections.formulas.append(region)
        elif label == "picture":
            detections.pictures.append(region)

    # Сортируем каждый класс по y0 — region_00 всегда самый верхний
    detections.sort_by_position()

    return detections