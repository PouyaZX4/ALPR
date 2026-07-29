from dataclasses import dataclass

@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int

@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    ground_point: tuple[float, float]   # bottom-center of bbox, pixel coords

@dataclass
class SpeedRecord:
    track_id: int
    speed_kmh: float
    frame_idx: int

@dataclass
class PlateResult:
    plate_text: str
    confidence: float
    plate_crop_path: str
