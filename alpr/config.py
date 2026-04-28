from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ALPRConfig:
    min_plate_ratio: float = 2.0
    max_plate_ratio: float = 6.5
    preferred_plate_ratio: float = 4.0
    min_plate_area: int = 1200
    max_candidates: int = 10
    candidate_iou_threshold: float = 0.35
    ocr_allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    min_text_length: int = 6
    max_text_length: int = 10
    min_ocr_confidence: float = 0.35
    crop_padding_ratio: float = 0.08
    min_character_regions: int = 3
    max_character_regions: int = 12
    min_crop_visual_score: float = 0.18
