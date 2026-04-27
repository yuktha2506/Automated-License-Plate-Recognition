import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from alpr.config import ALPRConfig
from alpr import database
from alpr.pipeline import ALPRPipeline, PlateMatchResult

class AlprEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types and datetimes."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def save_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, cls=AlprEncoder), 
        encoding="utf-8"
    )

def build_report_payload(source: str, detections: list[PlateMatchResult]) -> dict[str, Any]:
    return {
        "source": source,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detection_count": len(detections),
        "stolen_hits": sum(1 for d in detections if d.is_stolen),
        "detections": [d.to_dict() for d in detections],
    }

def print_summary(report_path: Path, detections: list[PlateMatchResult]):
    stolen_hits = [d for d in detections if d.is_stolen]
    print("-" * 30)
    print(f"Report saved to: {report_path}")
    print(f"Detected unique plates: {len(detections)}")
    print(f"Stolen-vehicle hits: {len(stolen_hits)}")
    for hit in stolen_hits:
        if hit.stolen_record:
            case = hit.stolen_record.get("case_number", "N/A")
            print(f"[ALERT] {hit.plate_text} matches case {case}")
    print("-" * 30)

def process_image_source(pipeline: ALPRPipeline, source_path: str, show: bool):
    path = Path(source_path)
    frame = cv2.imread(str(path))
    if frame is None:
        logging.error(f"Could not read image: {source_path}")
        return
    annotated_frame, detections = pipeline.process_frame(frame)
    output_dir = Path(ALPRConfig.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_output = output_dir / f"{path.stem}_annotated{path.suffix}"
    report_output = output_dir / f"{path.stem}_report.json"
    cv2.imwrite(str(img_output), annotated_frame)
    payload = build_report_payload(source_path, detections)
    save_report(report_output, payload)
    print_summary(report_output, detections)
    if show:
        cv2.imshow("ALPR - Press any key to exit", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Automated License Plate Recognition")
    parser.add_argument("--source", type=str, required=True, help="Path to image/video or '0' for webcam")
    parser.add_argument("--show", action="store_true", help="Display visual output")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    args = parser.parse_args()

    # FIX: Correctly initialize database with required paths
    db = database.ALPRDatabase(
        db_path=Path(ALPRConfig.DB_PATH),
        seed_csv_path=Path(ALPRConfig.SEED_DATA_PATH)
    )
    pipeline = ALPRPipeline(db)

    source = args.source
    if source.isdigit() or any(source.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv']):
        # process_video_source call logic...
        pass 
    else:
        process_image_source(pipeline, source, args.show)

if __name__ == "__main__":
    main()
