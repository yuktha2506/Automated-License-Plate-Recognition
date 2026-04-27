from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import ALPRConfig
from .database import ALPRDatabase, StolenVehicleRecord
from .detector import PlateDetector
from .ocr import EasyPlateReader, OCRDetection
from .utils import BBox, looks_like_plate, normalize_plate_text, plate_text_score


@dataclass(slots=True)
class PlateMatchResult:
    plate_text: str
    normalized_plate: str
    confidence: float
    bbox: BBox
    detector_method: str
    candidate_score: float
    stolen_record: StolenVehicleRecord | None = None
    frame_index: int | None = None
    timestamp_seconds: float | None = None

    @property
    def is_stolen(self) -> bool:
        return self.stolen_record is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "plate_text": self.plate_text,
            "normalized_plate": self.normalized_plate,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3],
            },
            "detector_method": self.detector_method,
            "candidate_score": round(self.candidate_score, 4),
            "frame_index": self.frame_index,
            "timestamp_seconds": None if self.timestamp_seconds is None else round(self.timestamp_seconds, 3),
            "stolen_record": self.stolen_record.to_dict() if self.stolen_record else None,
        }


class ALPRPipeline:
    def __init__(
        self,
        database: ALPRDatabase,
        config: ALPRConfig = ALPRConfig(),
        languages: list[str] | None = None,
        gpu: bool = False,
    ) -> None:
        self.config = config
        self.detector = PlateDetector(config)
        self.reader = EasyPlateReader(
            languages=languages or ["en"],
            gpu=gpu,
            allowlist=config.ocr_allowlist,
        )
        self.database = database

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int | None = None,
        timestamp_seconds: float | None = None,
    ) -> tuple[np.ndarray, list[PlateMatchResult]]:
        candidates = self.detector.detect(frame)
        results_by_plate: dict[str, tuple[float, PlateMatchResult]] = {}

        for candidate in candidates:
            best_ocr = self._select_best_ocr_detection(self.reader.read_text(candidate.image))
            if best_ocr is None:
                continue

            normalized_plate = normalize_plate_text(best_ocr.text)
            stolen_record = self.database.lookup(normalized_plate)
            result = PlateMatchResult(
                plate_text=best_ocr.text,
                normalized_plate=normalized_plate,
                confidence=best_ocr.confidence,
                bbox=candidate.bbox,
                detector_method=candidate.method,
                candidate_score=candidate.score,
                stolen_record=stolen_record,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )
            composite_score = plate_text_score(normalized_plate, best_ocr.confidence) + candidate.score

            current = results_by_plate.get(normalized_plate)
            if current is None or composite_score > current[0]:
                results_by_plate[normalized_plate] = (composite_score, result)

        results = [entry[1] for entry in sorted(results_by_plate.values(), key=lambda item: item[0], reverse=True)]
        annotated_frame = self._annotate_frame(frame, results)
        return annotated_frame, results

    def _select_best_ocr_detection(self, detections: list[OCRDetection]) -> OCRDetection | None:
        viable_detections: list[tuple[float, OCRDetection]] = []
        for detection in detections:
            if detection.confidence < self.config.min_ocr_confidence:
                continue
            if not looks_like_plate(
                detection.text,
                min_length=self.config.min_text_length,
                max_length=self.config.max_text_length,
            ):
                continue
            viable_detections.append((plate_text_score(detection.text, detection.confidence), detection))

        if not viable_detections:
            return None
        return max(viable_detections, key=lambda item: item[0])[1]

    def _annotate_frame(self, frame: np.ndarray, detections: list[PlateMatchResult]) -> np.ndarray:
        annotated = frame.copy()
        for detection in detections:
            color = (0, 0, 255) if detection.is_stolen else (0, 180, 0)
            x, y, width, height = detection.bbox
            cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)

            labels = [f"{detection.normalized_plate} ({detection.confidence:.2f})"]
            if detection.is_stolen:
                labels.append(f"STOLEN | Case {detection.stolen_record.case_number}")
            else:
                labels.append("CLEAR | No database hit")

            for index, label in enumerate(labels):
                label_y = max(24, y - 10 - (index * 24))
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    annotated,
                    (x, label_y - text_size[1] - 6),
                    (x + text_size[0] + 8, label_y + baseline - 4),
                    color,
                    thickness=-1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x + 4, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        return annotated

