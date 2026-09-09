import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from src.tracking.tracker import VehicleTracker
from src.speed.calibration import load_homography
from src.speed.speed_estimator import SpeedEstimator, format_video_time
from src.detection.plate_detector import PlateDetector
from src.ocr.hazar_ocr import PlateOCR
from src.database.db import ViolationDB
from src.utils.data_types import Track, SpeedRecord, ViolationRecord


class SpeedALPRPipeline:
    def __init__(self, config: dict, fps: float = 30.0):
        self.config = config
        self.speed_threshold = float(config['thresholds']['speed_kmh'])
        self.fps = fps if fps > 0 else 30.0

        # 1. Load Calibration Matrix
        homography_path = config['paths'].get('homography', 'data/calibration/homography.npy')
        if os.path.exists(homography_path):
            self.H = load_homography(homography_path)
        else:
            print(f"[WARNING] '{homography_path}' not found! Using identity matrix fallback.")
            self.H = np.eye(3, dtype=np.float32)

        # 2. Initialize Submodules
        self.tracker = VehicleTracker(
            weights_path=config['paths'].get('vehicle_detector_weights', 'yolo11s.pt'),
            imgsz=1280,
            conf_threshold=config['thresholds'].get('vehicle_conf', 0.3)
        )

        self.speed_estimator = SpeedEstimator(
            homography_matrix=self.H,
            fps=self.fps,
            window_size=config['speed'].get('smoothing_window', 8)
        )

        self.plate_detector = PlateDetector(
            local_weights_dir="models/plate_detector",
            conf_threshold=config['thresholds'].get('plate_conf', 0.3)
        )

        self.ocr_engine = PlateOCR(
            model_name="hezarai/crnn-fa-license-plate-recognition-v2"
        )

        self.db = ViolationDB(db_path=config['paths'].get('database', 'data/violations.db'))

        # Track management:
        # confirmed_violations: set of track_ids that have been written to DB
        self.confirmed_violations = set()
        # candidate_tracking: buffer best candidate per speeding track {track_id: {best_score, veh_crop, plate_crop, ...}}
        self.speeding_candidates: Dict[int, Dict[str, Any]] = {}

    def set_fps(self, fps: float):
        if fps and fps > 0:
            self.fps = float(fps)
            self.speed_estimator.set_fps(fps)

    def set_homography(self, H: np.ndarray):
        if H is not None:
            self.H = H
            self.speed_estimator.set_homography(H)

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[Track], List[SpeedRecord], List[ViolationRecord]]:
        """
        Processes a single frame.
        Returns:
            tracks: active vehicle tracks
            speed_records: speeds of active tracks
            new_violations: any violation records finalized in this frame
        """
        h_frame, w_frame, _ = frame.shape
        new_violations: List[ViolationRecord] = []

        # 1. Update Tracking & Speed
        tracks = self.tracker.update(frame)
        speed_records = self.speed_estimator.update(tracks, frame_idx)
        speed_map = {rec.track_id: rec.speed_kmh for rec in speed_records}
        time_str_map = {rec.track_id: rec.video_time_str for rec in speed_records}

        active_track_ids = {trk.track_id for trk in tracks}

        # 2. Check for Speeding Violations & Best-Shot Plate Capture
        for trk in tracks:
            tid = trk.track_id
            speed = speed_map.get(tid, 0.0)

            # Check if speeding and not yet confirmed
            if speed > self.speed_threshold and tid not in self.confirmed_violations:
                x1, y1, x2, y2 = map(int, trk.bbox)
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w_frame, x2), min(h_frame, y2)

                if cx2 <= cx1 or cy2 <= cy1:
                    continue

                vehicle_crop = frame[cy1:cy2, cx1:cx2].copy()
                if vehicle_crop.size == 0:
                    continue

                # Run Plate Detection on Vehicle Crop
                plate_det = self.plate_detector.detect(vehicle_crop, imgsz=1280)
                
                # If plate detected, evaluate candidate quality
                if plate_det:
                    px1, py1, px2, py2 = map(int, plate_det.bbox)
                    vh, vw, _ = vehicle_crop.shape
                    px1, py1 = max(0, px1), max(0, py1)
                    px2, py2 = min(vw, px2), min(vh, py2)

                    if px2 > px1 and py2 > py1:
                        plate_crop = vehicle_crop[py1:py2, px1:px2].copy()
                        plate_area = (px2 - px1) * (py2 - py1)
                        quality_score = float(plate_det.confidence) * 1000 + plate_area

                        if tid not in self.speeding_candidates:
                            self.speeding_candidates[tid] = {
                                "frames_seen": 0,
                                "quality_score": quality_score,
                                "speed_kmh": speed,
                                "vehicle_crop": vehicle_crop,
                                "plate_crop": plate_crop,
                                "frame_idx": frame_idx,
                                "video_time": time_str_map.get(tid, format_video_time(frame_idx, self.fps)[0])
                            }
                        else:
                            cand = self.speeding_candidates[tid]
                            cand["frames_seen"] += 1
                            cand["speed_kmh"] = max(cand["speed_kmh"], speed)
                            if quality_score > cand["quality_score"]:
                                cand["quality_score"] = quality_score
                                cand["vehicle_crop"] = vehicle_crop
                                cand["plate_crop"] = plate_crop
                                cand["frame_idx"] = frame_idx
                                cand["video_time"] = time_str_map.get(tid, cand["video_time"])

        # 3. Finalize candidates that reached sufficient quality (or tracked >= 3 frames)
        for tid in list(self.speeding_candidates.keys()):
            cand = self.speeding_candidates[tid]
            # Finalize if seen for at least 3 frames or vehicle left the frame
            if cand["frames_seen"] >= 2 or tid not in active_track_ids:
                ocr_result = self.ocr_engine.read(cand["plate_crop"])
                
                # Save lossless images to disk
                crop_dir = os.path.join("data", "violations")
                os.makedirs(crop_dir, exist_ok=True)
                
                f_idx = cand["frame_idx"]
                veh_crop_path = os.path.join(crop_dir, f"veh_track_{tid}_{f_idx}.png")
                plate_crop_path = os.path.join(crop_dir, f"plate_track_{tid}_{f_idx}.png")

                cv2.imwrite(veh_crop_path, cand["vehicle_crop"])
                cv2.imwrite(plate_crop_path, cand["plate_crop"])

                # Insert into DB
                row_id = self.db.insert_violation(
                    track_id=tid,
                    speed_kmh=cand["speed_kmh"],
                    plate_text=ocr_result.plate_text,
                    ocr_confidence=float(ocr_result.confidence),
                    video_time=cand["video_time"],
                    vehicle_image_path=veh_crop_path.replace("\\", "/"),
                    plate_image_path=plate_crop_path.replace("\\", "/")
                )

                self.confirmed_violations.add(tid)
                del self.speeding_candidates[tid]

                violation_rec = ViolationRecord(
                    id=row_id,
                    track_id=tid,
                    speed_kmh=cand["speed_kmh"],
                    speed_limit=self.speed_threshold,
                    plate_text=ocr_result.plate_text,
                    ocr_confidence=float(ocr_result.confidence),
                    video_time=cand["video_time"],
                    timestamp=cand["video_time"],
                    vehicle_image_path=veh_crop_path.replace("\\", "/"),
                    plate_image_path=plate_crop_path.replace("\\", "/")
                )
                new_violations.append(violation_rec)

                print(f"🚨 [VIOLATION CONFIRMED] ID: {row_id} | Track: {tid} | Speed: {cand['speed_kmh']:.1f} km/h | Video Time: {cand['video_time']} | Plate: '{ocr_result.plate_text}'")

        return tracks, speed_records, new_violations