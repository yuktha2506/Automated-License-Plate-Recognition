# Automated License Plate Recognition (ALPR)

This project uses OpenCV for license plate candidate detection, EasyOCR for plate text extraction, and a local SQLite database to flag vehicles reported as stolen. It is designed to work without any custom model training.

## Features

- Detects number plate regions with classic OpenCV techniques and an OpenCV cascade fallback.
- Reads plate text with EasyOCR.
- Normalizes plate text before database lookup to reduce OCR formatting noise.
- Creates a local `sqlite3` database automatically from a CSV seed file.
- Supports image files, video files, and live webcam input.
- Saves annotated output plus a JSON report for each run.

## Project Structure

```text
LICENSE_DETECTION/
|-- alpr/
|   |-- config.py
|   |-- database.py
|   |-- detector.py
|   |-- ocr.py
|   |-- pipeline.py
|   `-- utils.py
|-- data/
|   `-- stolen_vehicles_seed.csv
|-- outputs/
|-- tests/
|-- main.py
`-- requirements.txt
```

## Setup on Windows

1. Open PowerShell in `D:\LICENSE_DETECTION`.
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

If `easyocr` or `torch` installation fails on your machine, install the recommended CPU build of PyTorch first, then rerun `pip install -r requirements.txt`.

## Run Examples

### Image

```powershell
python main.py --source path\to\car.jpg --show
```

### Video

```powershell
python main.py --source path\to\traffic.mp4 --frame-stride 2
```

### Webcam

```powershell
python main.py --source 0 --show
```

## Outputs

Every run stores results in `outputs/`:

- Annotated image or video with detected plates highlighted
- JSON report containing detected plate text, confidence, and stolen-vehicle matches

## Database

The application auto-creates `data/stolen_vehicles.db` from `data/stolen_vehicles_seed.csv` when the database is missing or empty.

Update the CSV file with your real stolen-vehicle records using these columns:

```text
plate_text,vehicle_make,vehicle_model,vehicle_color,owner_name,case_number,reported_date,notes
```

## Notes

- No training is required. The system relies on OpenCV preprocessing plus EasyOCR's pretrained OCR model.
- Classic computer-vision plate detection works best with clear, front-facing, well-lit plates.
- OCR accuracy improves when the plate crop is not blurry and the plate occupies a reasonable portion of the frame.

