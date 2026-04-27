from __future__ import annotations

import csv
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from .utils import normalize_plate_text


@dataclass(frozen=True, slots=True)
class StolenVehicleRecord:
    plate_text: str
    vehicle_make: str
    vehicle_model: str
    vehicle_color: str
    owner_name: str
    case_number: str
    reported_date: str
    notes: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class ALPRDatabase:
    def __init__(self, db_path: Path, seed_csv_path: Path | None = None) -> None:
        self.db_path = db_path
        self.seed_csv_path = seed_csv_path
        self._initialized = False

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS stolen_vehicles (
                    plate_text TEXT PRIMARY KEY,
                    vehicle_make TEXT NOT NULL,
                    vehicle_model TEXT NOT NULL,
                    vehicle_color TEXT NOT NULL,
                    owner_name TEXT NOT NULL,
                    case_number TEXT NOT NULL,
                    reported_date TEXT NOT NULL,
                    notes TEXT NOT NULL
                )
                """
            )
            row_count = connection.execute("SELECT COUNT(*) FROM stolen_vehicles").fetchone()[0]
            if row_count == 0 and self.seed_csv_path and self.seed_csv_path.exists():
                self._seed_from_csv(connection)
            connection.commit()
        self._initialized = True

    def lookup(self, plate_text: str) -> StolenVehicleRecord | None:
        if not self._initialized:
            self.initialize()

        normalized_plate = normalize_plate_text(plate_text)
        if not normalized_plate:
            return None

        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    plate_text,
                    vehicle_make,
                    vehicle_model,
                    vehicle_color,
                    owner_name,
                    case_number,
                    reported_date,
                    notes
                FROM stolen_vehicles
                WHERE plate_text = ?
                """,
                (normalized_plate,),
            ).fetchone()

        if row is None:
            return None
        # FIX: Changed from ALPRDatabase(*row) to StolenVehicleRecord(*row)
        return StolenVehicleRecord(*row)

    def _seed_from_csv(self, connection: sqlite3.Connection) -> None:
        assert self.seed_csv_path is not None
        with self.seed_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = [
                (
                    normalize_plate_text(row["plate_text"]),
                    row["vehicle_make"].strip(),
                    row["vehicle_model"].strip(),
                    row["vehicle_color"].strip(),
                    row["owner_name"].strip(),
                    row["case_number"].strip(),
                    row["reported_date"].strip(),
                    row["notes"].strip(),
                )
                for row in reader
                if row.get("plate_text")
            ]

        connection.executemany(
            """
            INSERT OR REPLACE INTO stolen_vehicles (
                plate_text,
                vehicle_make,
                vehicle_model,
                vehicle_color,
                owner_name,
                case_number,
                reported_date,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
