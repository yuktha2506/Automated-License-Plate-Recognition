from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ALPRConfig
from .utils import BBox, bbox_iou, expand_bbox


@dataclass(slots=True)
class PlateCandidate:
    bbox: BBox
    image: np.ndarray
    method: str
    score: float


class PlateDetector:
    def __init__(self, config: ALPRConfig) -> None:
        self.config = config
        self._cascade: cv2.CascadeClassifier | None = None

    def detect(self, frame: np.ndarray) -> list[PlateCandidate]:
        candidates = self._detect_with_cascade(frame) + self._detect_with_contours(frame)
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return self._deduplicate_candidates(candidates)[: self.config.max_candidates]

    def _detect_with_cascade(self, frame: np.ndarray) -> list[PlateCandidate]:
        cascade = self._load_cascade()
        if cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(80, 25),
        )

        candidates: list[PlateCandidate] = []
        for x, y, width, height in detections:
            ratio = width / max(height, 1)
            if not (self.config.min_plate_ratio <= ratio <= self.config.max_plate_ratio):
                continue
            bbox = self._expand_and_clip_bbox(frame, (x, y, width, height))
            crop = self._crop(frame, bbox)
            if crop.size == 0:
                continue

            visual_score = self._score_crop_visual_quality(crop)
            if visual_score < self.config.min_crop_visual_score:
                continue
            area_score = min((width * height) / 12000.0, 1.0)
            candidates.append(
                PlateCandidate(
                    bbox=bbox,
                    image=crop,
                    method="haar_cascade",
                    score=1.0 + area_score + (0.6 * visual_score),
                )
            )
        return candidates

    def _detect_with_contours(self, frame: np.ndarray) -> list[PlateCandidate]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

        gradient_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
        gradient_x = np.absolute(gradient_x)
        max_value = gradient_x.max()
        if max_value > 0:
            gradient_x = np.uint8(255 * gradient_x / max_value)
        else:
            gradient_x = np.zeros_like(gray)

        gradient_x = cv2.GaussianBlur(gradient_x, (5, 5), 0)
        closed = cv2.morphologyEx(gradient_x, cv2.MORPH_CLOSE, rect_kernel)
        thresholded = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresholded = cv2.erode(thresholded, None, iterations=1)
        thresholded = cv2.dilate(thresholded, None, iterations=2)

        contours, _ = cv2.findContours(
            thresholded,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        image_area = frame.shape[0] * frame.shape[1]
        candidates: list[PlateCandidate] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:50]:
            x, y, width, height = cv2.boundingRect(contour)
            bbox_area = width * height
            ratio = width / max(height, 1)
            contour_area = cv2.contourArea(contour)
            fill_ratio = contour_area / bbox_area if bbox_area else 0.0

            if bbox_area < self.config.min_plate_area:
                continue
            if not (self.config.min_plate_ratio <= ratio <= self.config.max_plate_ratio):
                continue
            if fill_ratio < 0.35:
                continue

            bbox = self._expand_and_clip_bbox(frame, (x, y, width, height))
            crop = self._crop(frame, bbox)
            if crop.size == 0:
                continue

            visual_score = self._score_crop_visual_quality(crop)
            if visual_score < self.config.min_crop_visual_score:
                continue
            size_score = min(bbox_area / max(image_area * 0.08, 1), 1.0)
            ratio_score = 1.0 - min(
                abs(ratio - self.config.preferred_plate_ratio) / self.config.preferred_plate_ratio,
                1.0,
            )
            score = (
                (0.7 * fill_ratio)
                + (0.6 * size_score)
                + (0.4 * ratio_score)
                + (0.8 * visual_score)
            )

            candidates.append(
                PlateCandidate(
                    bbox=bbox,
                    image=crop,
                    method="contour",
                    score=score,
                )
            )

        return candidates

    def _load_cascade(self) -> cv2.CascadeClassifier | None:
        if self._cascade is not None:
            return self._cascade

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_russian_plate_number.xml"
        if not cascade_path.exists():
            return None

        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return None

        self._cascade = cascade
        return self._cascade

    def _expand_and_clip_bbox(self, frame: np.ndarray, bbox: BBox) -> BBox:
        return expand_bbox(
            bbox,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            padding_ratio=self.config.crop_padding_ratio,
        )

    @staticmethod
    def _crop(frame: np.ndarray, bbox: BBox) -> np.ndarray:
        x, y, width, height = bbox
        return frame[y : y + height, x : x + width]

    def _deduplicate_candidates(self, candidates: list[PlateCandidate]) -> list[PlateCandidate]:
        deduplicated: list[PlateCandidate] = []
        for candidate in candidates:
            if any(
                bbox_iou(candidate.bbox, existing.bbox) >= self.config.candidate_iou_threshold
                for existing in deduplicated
            ):
                continue
            deduplicated.append(candidate)
        return deduplicated

    def _score_crop_visual_quality(self, crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
        if min(gray.shape[:2]) < 16:
            return 0.0

        resized = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)

        edge_map = cv2.Canny(enhanced, 100, 200)
        edge_density = float(np.count_nonzero(edge_map)) / max(edge_map.size, 1)
        edge_score = min(edge_density / 0.12, 1.0)

        thresholded = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )[1]
        thresholded = cv2.morphologyEx(
            thresholded,
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
        )

        contour_count = 0
        crop_height, crop_width = thresholded.shape
        min_region_area = max(20, int(crop_height * crop_width * 0.002))
        max_region_area = int(crop_height * crop_width * 0.18)

        contours, _ = cv2.findContours(
            thresholded,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            region_area = width * height
            if region_area < min_region_area or region_area > max_region_area:
                continue
            if not (0.02 * crop_width <= width <= 0.22 * crop_width):
                continue
            if not (0.30 * crop_height <= height <= 0.95 * crop_height):
                continue
            if height <= width:
                continue
            contour_count += 1

        if contour_count < self.config.min_character_regions:
            char_score = contour_count / max(self.config.min_character_regions, 1) * 0.5
        elif contour_count > self.config.max_character_regions:
            char_score = max(0.0, 1.0 - ((contour_count - self.config.max_character_regions) / 10.0))
        else:
            ideal_count = 6
            char_score = 1.0 - min(abs(contour_count - ideal_count) / ideal_count, 1.0) * 0.4

        contrast_score = min(float(enhanced.std()) / 64.0, 1.0)
        return (0.45 * char_score) + (0.35 * edge_score) + (0.20 * contrast_score)
