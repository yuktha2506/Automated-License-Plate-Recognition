from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import normalize_plate_text


@dataclass(frozen=True, slots=True)
class OCRDetection:
    text: str
    confidence: float
    bbox: tuple[tuple[int, int], ...] | None = None


class EasyPlateReader:
    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
        allowlist: str | None = None,
    ) -> None:
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.allowlist = allowlist
        self._reader = None

    def read_text(self, plate_image: np.ndarray) -> list[OCRDetection]:
        if plate_image.size == 0:
            return []

        reader = self._get_reader()
        detections_by_text: dict[str, OCRDetection] = {}

        for variant in self._prepare_variants(plate_image):
            results = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                decoder="greedy",
                allowlist=self.allowlist,
            )
            for bbox, text, confidence in results:
                normalized = normalize_plate_text(text)
                if not normalized:
                    continue

                detection = OCRDetection(
                    text=normalized,
                    confidence=float(confidence),
                    bbox=self._normalize_bbox(bbox),
                )
                best = detections_by_text.get(normalized)
                if best is None or detection.confidence > best.confidence:
                    detections_by_text[normalized] = detection

        return sorted(detections_by_text.values(), key=lambda detection: detection.confidence, reverse=True)

    def _get_reader(self):
        if self._reader is not None:
            return self._reader

        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "EasyOCR or one of its compiled dependencies failed to import. "
                "Use the project virtual environment at `D:\\LICENSE_DETECTION\\.venv` and reinstall "
                "the pinned dependencies from `requirements.txt`."
            ) from exc

        try:
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        except Exception as exc:
            raise RuntimeError(
                "EasyOCR could not initialize. This usually means the environment has incompatible "
                "binary packages such as NumPy/SciPy/Torch. Recreate the project `.venv` and "
                "reinstall `requirements.txt`."
            ) from exc
        return self._reader

    @staticmethod
    def _prepare_variants(plate_image: np.ndarray) -> list[np.ndarray]:
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()

        scale = max(1.0, 96 / max(gray.shape[0], 1))
        upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        filtered = cv2.bilateralFilter(upscaled, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(filtered)
        sharpened = cv2.filter2D(
            clahe,
            -1,
            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
        )
        otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        inverse_otsu = cv2.threshold(
            sharpened,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )[1]
        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return [gray, upscaled, clahe, sharpened, otsu, inverse_otsu, adaptive]

    @staticmethod
    def _normalize_bbox(bbox: list[list[float]] | tuple[tuple[float, float], ...]) -> tuple[tuple[int, int], ...]:
        return tuple((int(point[0]), int(point[1])) for point in bbox)
