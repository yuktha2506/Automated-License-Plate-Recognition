from __future__ import annotations

from typing import Iterable

BBox = tuple[int, int, int, int]


def normalize_plate_text(text: str) -> str:
    if not text:
        return ""
    return "".join(c for c in text.upper().strip() if c.isalnum())

def looks_like_plate(text: str, min_length: int = 5, max_length: int = 10) -> bool:
    normalized = normalize_plate_text(text)
    if not (min_length <= len(normalized) <= max_length):
        return False
    if not any(character.isalpha() for character in normalized):
        return False
    if not any(character.isdigit() for character in normalized):
        return False
    return True


def plate_text_score(text: str, confidence: float) -> float:
    normalized = normalize_plate_text(text)
    if not normalized:
        return 0.0

    target_length = 8
    length_score = max(0.0, 1.0 - (abs(len(normalized) - target_length) / target_length))
    mixed_content_bonus = 0.15 if any(char.isalpha() for char in normalized) and any(char.isdigit() for char in normalized) else 0.0
    return confidence + (0.25 * length_score) + mixed_content_bonus


def bbox_iou(first: BBox, second: BBox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[0] + first[2], second[0] + second[2])
    y2 = min(first[1] + first[3], second[1] + second[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0:
        return 0.0

    first_area = first[2] * first[3]
    second_area = second[2] * second[3]
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area if union_area else 0.0


def expand_bbox(bbox: BBox, frame_width: int, frame_height: int, padding_ratio: float) -> BBox:
    x, y, width, height = bbox
    padding_x = int(width * padding_ratio)
    padding_y = int(height * padding_ratio)

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(frame_width, x + width + padding_x)
    y2 = min(frame_height, y + height + padding_y)
    return x1, y1, x2 - x1, y2 - y1


def deduplicate_bboxes(boxes: Iterable[BBox], iou_threshold: float) -> list[BBox]:
    deduplicated: list[BBox] = []
    for box in boxes:
        if any(bbox_iou(box, existing) >= iou_threshold for existing in deduplicated):
            continue
        deduplicated.append(box)
    return deduplicated

