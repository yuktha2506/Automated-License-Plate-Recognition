from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from alpr.config import ALPRConfig
from alpr.pipeline import ALPRPipeline, PlateMatchResult

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mov", ".mp4", ".mkv", ".mpeg", ".mpg", ".wmv"}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Detect vehicle number plates and match them against a stolen-vehicle database."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image/video file or a webcam index such as 0.",
    )
    parser.add_argument(
        "--db",
        default=str(project_root / "data" / "stolen_vehicles.db"),
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--seed",
        default=str(project_root / "data" / "stolen_vehicles_seed.csv"),
        help="Path to the CSV file used to initialize the database when empty.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "outputs"),
        help="Directory where annotated files and reports will be stored.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame for videos or webcam streams.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated output in a live OpenCV window.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU mode for EasyOCR if a supported PyTorch GPU setup is available.",
    )
    parser.add_argument(
        "--langs",
        default="en",
        help="Comma-separated EasyOCR language list. Default: en",
    )
    return parser.parse_args()


def resolve_source(source_value: str) -> tuple[int | Path, str]:
    source_value = source_value.strip()
    if source_value.isdigit():
        return int(source_value), "camera"

    source_path = Path(source_value).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return source_path, "image"
    if suffix in VIDEO_SUFFIXES:
        return source_path, "video"
    raise ValueError(f"Unsupported source type for {source_path.name}")


def detection_sort_key(detection: PlateMatchResult) -> tuple[float, float]:
    return (detection.confidence, detection.candidate_score)


def merge_unique_detections(
    best_by_plate: dict[str, PlateMatchResult], detections: list[PlateMatchResult]
) -> None:
    for detection in detections:
        existing = best_by_plate.get(detection.normalized_plate)
        if existing is None or detection_sort_key(detection) > detection_sort_key(existing):
            best_by_plate[detection.normalized_plate] = detection


def build_report_payload(source: int | Path, detections: list[PlateMatchResult]) -> dict[str, Any]:
    return {
        "source": str(source),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_unique_plates": len(detections),
        "stolen_hits": sum(1 for detection in detections if detection.is_stolen),
        "detections": [detection.to_dict() for detection in detections],
    }


def save_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_image_source(
    pipeline: ALPRPipeline,
    source_path: Path,
    output_dir: Path,
    show: bool,
) -> tuple[Path, Path, list[PlateMatchResult]]:
    frame = cv2.imread(str(source_path))
    if frame is None:
        raise RuntimeError(f"Unable to read image: {source_path}")

    annotated_frame, detections = pipeline.process_frame(frame)

    output_image_path = output_dir / f"{source_path.stem}_annotated{source_path.suffix}"
    report_path = output_dir / f"{source_path.stem}_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_image_path), annotated_frame):
        raise RuntimeError(f"Unable to save annotated image: {output_image_path}")

    save_report(report_path, build_report_payload(source_path, detections))

    if show:
        cv2.imshow("ALPR", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_image_path, report_path, detections


def process_stream_source(
    pipeline: ALPRPipeline,
    source: int | Path,
    source_kind: str,
    output_dir: Path,
    frame_stride: int,
    show: bool,
) -> tuple[Path, Path, list[PlateMatchResult]]:
    capture_source = str(source) if isinstance(source, Path) else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open stream source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if source_kind == "camera":
        output_video_path = output_dir / f"camera_{source}_{timestamp}.mp4"
        report_path = output_dir / f"camera_{source}_{timestamp}_report.json"
    else:
        source_path = Path(source)
        output_video_path = output_dir / f"{source_path.stem}_annotated.mp4"
        report_path = output_dir / f"{source_path.stem}_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create output video: {output_video_path}")

    unique_detections: dict[str, PlateMatchResult] = {}
    frame_index = 0
    safe_stride = max(frame_stride, 1)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % safe_stride == 0:
                timestamp_seconds = frame_index / fps if fps else None
                annotated_frame, detections = pipeline.process_frame(
                    frame,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                )
                merge_unique_detections(unique_detections, detections)
            else:
                annotated_frame = frame.copy()

            writer.write(annotated_frame)

            if show:
                cv2.imshow("ALPR", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break

            frame_index += 1
    finally:
        capture.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    detections = sorted(unique_detections.values(), key=detection_sort_key, reverse=True)
    save_report(report_path, build_report_payload(source, detections))
    return output_video_path, report_path, detections


def print_summary(output_path: Path, report_path: Path, detections: list[PlateMatchResult]) -> None:
    stolen_hits = [detection for detection in detections if detection.is_stolen]
    print(f"Saved annotated output to: {output_path}")
    print(f"Saved JSON report to: {report_path}")
    print(f"Detected unique plates: {len(detections)}")
    print(f"Stolen-vehicle hits: {len(stolen_hits)}")
    for detection in stolen_hits:
        print(
            f"[ALERT] {detection.normalized_plate} matches case "
            f"{detection.stolen_record.case_number}"
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    config = ALPRConfig()
    pipeline = ALPRPipeline(
        config=config,
        db_path=Path(args.db).resolve(),
        seed_csv_path=Path(args.seed).resolve(),
        languages=[language.strip() for language in args.langs.split(",") if language.strip()],
        gpu=args.gpu,
    )

    source, source_kind = resolve_source(args.source)
    if source_kind == "image":
        output_path, report_path, detections = process_image_source(
            pipeline,
            source,
            output_dir=output_dir,
            show=args.show,
        )
    else:
        output_path, report_path, detections = process_stream_source(
            pipeline,
            source=source,
            source_kind=source_kind,
            output_dir=output_dir,
            frame_stride=args.frame_stride,
            show=args.show,
        )

    print_summary(output_path, report_path, detections)


if __name__ == "__main__":
    main()
