import os
import sqlite3
from datetime import datetime


class ViolationDB:
    def __init__(self, db_path: str = "data/violations.db", schema_path: str = "src/database/schema.sql"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db(schema_path)

    def _init_db(self, schema_path: str):
        """Creates SQLite table if it does not exist."""
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = f.read()
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(schema)

    def insert_violation(
        self,
        track_id: int,
        speed_kmh: float,
        plate_text: str = "UNKNOWN",
        ocr_confidence: float = 0.0,
        vehicle_image_path: str = "",
        plate_image_path: str = ""
    ) -> int:
        """Inserts a speeding violation record into SQLite DB."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
            INSERT INTO violations 
            (track_id, plate_text, ocr_confidence, speed_kmh, timestamp, vehicle_image_path, plate_image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                query, 
                (track_id, plate_text, ocr_confidence, speed_kmh, timestamp, vehicle_image_path, plate_image_path)
            )
            conn.commit()
            return cursor.lastrowid

    def get_all_violations(self) -> list[dict]:
        """Queries all recorded speeding violations."""
        query = "SELECT * FROM violations ORDER BY id DESC"
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]