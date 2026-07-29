CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,
    plate_text TEXT,
    ocr_confidence REAL,
    speed_kmh REAL,
    timestamp TEXT,
    vehicle_image_path TEXT,
    plate_image_path TEXT
);
