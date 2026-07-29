from src.detection.vehicle_detector import VehicleDetector
from src.detection.plate_detector import PlateDetector
from src.tracking.tracker import VehicleTracker
from src.speed.speed_estimator import SpeedEstimator
from src.ocr.infer import PlateOCR
from src.database.db import ViolationDB

class SpeedALPRPipeline:
    def __init__(self, config: dict):
        self.config = config
        # Initializing modules will be populated here
        self.flagged_track_ids = set()

    def process_frame(self, frame, frame_idx: int):
        """
        Orchestrates object detection, tracking, speed estimation, plate detection, OCR, and DB recording per frame.
        """
        pass
