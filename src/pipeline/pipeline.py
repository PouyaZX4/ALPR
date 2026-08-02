import os
import cv2
import numpy as np

from src.tracking.tracker import VehicleTracker
from src.speed.calibration import load_homography
from src.speed.speed_estimator import SpeedEstimator
from src.detection.plate_detector import PlateDetector
from src.ocr.hazar_ocr import PlateOCR
from src.database.db import ViolationDB


class SpeedALPRPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.speed_threshold = config['thresholds']['speed_kmh']

        # 1. Load Calibration Matrix
        homography_path = config['paths']['homography']
        if os.path.exists(homography_path):
            self.H = load_homography(homography_path)
        else:
            print(f"[WARNING] '{homography_path}' not found! Using identity matrix fallback.")
            self.H = np.eye(3, dtype=np.float32)

        # 2. Initialize Submodules
        self.tracker = VehicleTracker(
            weights_path=config['paths']['vehicle_detector_weights'],
            imgsz=1280,
            conf_threshold=config['thresholds']['vehicle_conf']
        )

        self.speed_estimator = SpeedEstimator(
            homography_matrix=self.H,
            fps=30.0,
            window_size=config['speed']['smoothing_window']
        )

        self.plate_detector = PlateDetector(
            local_weights_dir="models/plate_detector",
            conf_threshold=config['thresholds']['plate_conf']
        )

        self.ocr_engine = PlateOCR(
            model_name="hezarai/crnn-fa-license-plate-recognition-v2"
        )

        self.db = ViolationDB(db_path=config['paths']['database'])

        # Track already-flagged vehicles (avoids duplicate DB writes per track_id)
        self.flagged_track_ids = set()

    def process_frame(self, frame, frame_idx: int):
        h_frame, w_frame, _ = frame.shape

        # 1. Update Tracking & Speed
        tracks = self.tracker.update(frame)
        speed_records = self.speed_estimator.update(tracks, frame_idx)
        speed_map = {rec.track_id: rec.speed_kmh for rec in speed_records}

        # 2. Check Speeding Violations
        for trk in tracks:
            x1, y1, x2, y2 = map(int, trk.bbox)
            tid = trk.track_id
            speed = speed_map.get(tid, 0.0)

            # SPEED TRIGGER: Speed > Limit AND not flagged yet
            if speed > self.speed_threshold and tid not in self.flagged_track_ids:
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w_frame, x2), min(h_frame, y2)

                vehicle_crop = frame[cy1:cy2, cx1:cx2].copy()

                if vehicle_crop.size > 0:
                    # Detect Plate inside Vehicle Crop
                    plate_det = self.plate_detector.detect(vehicle_crop, imgsz=1280)

                    if plate_det:
                        px1, py1, px2, py2 = map(int, plate_det.bbox)
                        vh, vw, _ = vehicle_crop.shape
                        px1, py1 = max(0, px1), max(0, py1)
                        px2, py2 = min(vw, px2), min(vh, py2)

                        plate_crop = vehicle_crop[py1:py2, px1:px2].copy()

                        if plate_crop.size > 0:
                            self.flagged_track_ids.add(tid)

                            # Read Persian OCR Text
                            ocr_result = self.ocr_engine.read(plate_crop)

                            # Save Lossless PNG Crops
                            crop_dir = "data/violations"
                            os.makedirs(crop_dir, exist_ok=True)
                            veh_crop_path = os.path.join(crop_dir, f"veh_track_{tid}_{frame_idx}.png")
                            plate_crop_path = os.path.join(crop_dir, f"plate_track_{tid}_{frame_idx}.png")

                            cv2.imwrite(veh_crop_path, vehicle_crop)
                            cv2.imwrite(plate_crop_path, plate_crop)

                            # Log Violation to SQLite Database
                            self.db.insert_violation(
                                track_id=tid,
                                speed_kmh=speed,
                                plate_text=ocr_result.plate_text,
                                ocr_confidence=float(ocr_result.confidence),
                                vehicle_image_path=veh_crop_path,
                                plate_image_path=plate_crop_path
                            )

                            print(f"🚨 [VIOLATION LOGGED TO DB] Track ID: {tid} | Speed: {speed:.1f} km/h | Plate: '{ocr_result.plate_text}'")

        return tracks, speed_records