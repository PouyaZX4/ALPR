CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    plate_text TEXT DEFAULT 'UNKNOWN',
    ocr_confidence REAL DEFAULT 0.0,
    speed_kmh REAL NOT NULL,
    timestamp TEXT NOT NULL,
    video_time TEXT DEFAULT '00:00:00',
    vehicle_image_path TEXT,
    plate_image_path TEXT
);