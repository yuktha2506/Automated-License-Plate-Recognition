from __future__ import annotations

import csv
import shutil
import unittest
import uuid
from pathlib import Path

from alpr.database import StolenVehicleDatabase


class StolenVehicleDatabaseTests(unittest.TestCase):
    def test_database_initializes_from_seed_and_normalizes_lookup(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / "tests" / ".tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_path = temp_root / f"db_test_{uuid.uuid4().hex}"
        temp_path.mkdir(parents=True, exist_ok=True)

        try:
            seed_path = temp_path / "seed.csv"
            db_path = temp_path / "stolen_vehicles.db"

            with seed_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "plate_text",
                        "vehicle_make",
                        "vehicle_model",
                        "vehicle_color",
                        "owner_name",
                        "case_number",
                        "reported_date",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "plate_text": "mh-12 ab 1234",
                        "vehicle_make": "Tata",
                        "vehicle_model": "Nexon",
                        "vehicle_color": "Blue",
                        "owner_name": "Rahul Patil",
                        "case_number": "FIR-2026-001",
                        "reported_date": "2026-01-14",
                        "notes": "Test record",
                    }
                )

            database = StolenVehicleDatabase(db_path=db_path, seed_csv_path=seed_path)
            database.initialize()
            record = database.lookup("MH12 AB1234")
            non_exact_record = database.lookup("MH12AB123")

            self.assertIsNotNone(record)
            self.assertEqual(record.plate_text, "MH12AB1234")
            self.assertEqual(record.case_number, "FIR-2026-001")
            self.assertIsNone(non_exact_record)
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
