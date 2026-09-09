from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    ground_point: Tuple[float, float]        # (x, y) bottom-center contact point


@dataclass
class SpeedRecord:
    track_id: int
    speed_kmh: float
    frame_idx: int
    video_time_str: str = "00:00.00"
    timestamp_seconds: float = 0.0


@dataclass
class ViolationRecord:
    id: Optional[int]
    track_id: int
    speed_kmh: float
    speed_limit: float
    plate_text: str
    ocr_confidence: float
    video_time: str
    timestamp: str
    vehicle_image_path: str
    plate_image_path: str