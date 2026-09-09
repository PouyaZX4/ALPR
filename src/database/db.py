import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any
from contextlib import contextmanager


class ViolationDB:
    def __init__(self, db_path: str = "data/violations.db", schema_path: str = "src/database/schema.sql"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db(schema_path)
        self._migrate_schema()

    @contextmanager
    def _get_connection(self):
        """Ensures connections and file locks are explicitly closed."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self, schema_path: str):
        if os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = f.read()
            with self._get_connection() as conn:
                conn.executescript(schema)
                conn.commit()

    def _migrate_schema(self):
        """Ensures all columns exist if upgrading from an older DB file."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(violations)")
            columns = [col[1] for col in cursor.fetchall()]
            if "video_time" not in columns:
                cursor.execute("ALTER TABLE violations ADD COLUMN video_time TEXT DEFAULT '00:00:00'")
                conn.commit()

    def insert_violation(
        self,
        track_id: int,
        speed_kmh: float,
        plate_text: str = "UNKNOWN",
        ocr_confidence: float = 0.0,
        video_time: str = "00:00:00",
        vehicle_image_path: str = "",
        plate_image_path: str = ""
    ) -> int:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = """
            INSERT INTO violations 
            (track_id, plate_text, ocr_confidence, speed_kmh, timestamp, video_time, vehicle_image_path, plate_image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query, 
                (track_id, plate_text, ocr_confidence, speed_kmh, timestamp, video_time, vehicle_image_path, plate_image_path)
            )
            conn.commit()
            return cursor.lastrowid

    def get_all_violations(self) -> List[Dict[str, Any]]:
        query = "SELECT * FROM violations ORDER BY id DESC"
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.get_all_violations()[:limit]