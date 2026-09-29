import os
import cv2
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from src.tracking.tracker import VehicleTracker
from src.speed.calibration import load_homography
from src.speed.speed_estimator import SpeedEstimator, format_video_time
from src.detection.plate_detector import PlateDetector
from src.ocr.hazar_ocr import PlateOCR
from src.database.db import ViolationDB
from src.utils.data_types import Track, SpeedRecord, ViolationRecord


def compute_sharpness(gray_img: np.ndarray) -> float:
    """Computes Laplacian variance to measure sharpness."""
    if gray_img is None or gray_img.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())


class SpeedALPRPipeline:
    def __init__(self, config: dict, fps: float = 30.0):
        self.config = config
        self.speed_threshold = float(config.get('thresholds', {}).get('speed_kmh', 60.0))
        self.fps = fps if fps > 0 else 30.0

        self.calib_points: Optional[List[List[float]]] = None
        self.custom_gate_line: Optional[List[List[float]]] = None

        # Load Points from calibration_points.json if available
        pts_path = "data/calibration/calibration_points.json"
        if os.path.exists(pts_path):
            try:
                with open(pts_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.calib_points = data.get("calibration_points", None)
                    self.custom_gate_line = data.get("custom_gate_line", None)
            except Exception:
                pass

        # 1. Load Calibration Matrix
        homography_path = config.get('paths', {}).get('homography', 'data/calibration/homography.npy')
        if os.path.exists(homography_path):
            self.H = load_homography(homography_path)
        else:
            self.H = np.eye(3, dtype=np.float32)

        # 2. Submodules
        self.tracker = VehicleTracker(
            weights_path=config.get('paths', {}).get('vehicle_detector_weights', 'yolo11s.pt'),
            imgsz=1280,
            conf_threshold=config.get('thresholds', {}).get('vehicle_conf', 0.3)
        )

        self.speed_estimator = SpeedEstimator(
            homography_matrix=self.H,
            fps=self.fps,
            window_size=config.get('speed', {}).get('smoothing_window', 8)
        )

        self.plate_detector = PlateDetector(
            local_weights_dir="models/plate_detector",
            conf_threshold=config.get('thresholds', {}).get('plate_conf', 0.25)
        )

        self.ocr_engine = PlateOCR(
            model_name="hezarai/crnn-fa-license-plate-recognition-v2"
        )

        self.db = ViolationDB(db_path=config.get('paths', {}).get('database', 'data/violations.db'))

        self.confirmed_violations = set()
        self.candidates: Dict[int, Dict[str, Any]] = {}

    def set_fps(self, fps: float):
        if fps and fps > 0:
            self.fps = float(fps)
            self.speed_estimator.set_fps(fps)

    def set_homography(self, H: np.ndarray):
        if H is not None:
            self.H = H
            self.speed_estimator.set_homography(H)

    def set_calibration_data(self, calib_points: Optional[List[List[float]]], custom_gate_line: Optional[List[List[float]]]):
        self.calib_points = calib_points
        self.custom_gate_line = custom_gate_line

    def get_capture_line_endpoints(self, w_frame: int, h_frame: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Determines the 2 endpoints of the active trigger line.
        1. If custom_gate_line is set: returns those 2 points.
        2. Else if 4 road dots exist: returns line between Dot 3 (P3) and Dot 4 (P4).
        3. Else fallback: returns horizontal line at 70% frame height.
        """
        if self.custom_gate_line and len(self.custom_gate_line) == 2:
            p1 = (int(self.custom_gate_line[0][0]), int(self.custom_gate_line[0][1]))
            p2 = (int(self.custom_gate_line[1][0]), int(self.custom_gate_line[1][1]))
            return p1, p2
        elif self.calib_points and len(self.calib_points) == 4:
            # P4 (Bottom-Left) -> P3 (Bottom-Right)
            p4 = (int(self.calib_points[3][0]), int(self.calib_points[3][1]))
            p3 = (int(self.calib_points[2][0]), int(self.calib_points[2][1]))
            return p4, p3
        else:
            default_y = int(h_frame * 0.70)
            return (0, default_y), (w_frame, default_y)

    def get_capture_y_at_x(self, x: float, w_frame: int, h_frame: int) -> int:
        """Calculates expected line Y for a given vehicle X contact coordinate."""
        (x1, y1), (x2, y2) = self.get_capture_line_endpoints(w_frame, h_frame)
        if abs(x2 - x1) < 1e-4:
            return int((y1 + y2) / 2)
        slope = (y2 - y1) / (x2 - x1)
        y = y1 + slope * (x - x1)
        return int(np.clip(y, 10, h_frame - 10))

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[Track], List[SpeedRecord], List[ViolationRecord]]:
        h_frame, w_frame, _ = frame.shape
        gate_margin = int(h_frame * 0.12)

        new_violations: List[ViolationRecord] = []

        # 1. Update Tracking and Speed
        tracks = self.tracker.update(frame)
        speed_records = self.speed_estimator.update(tracks, frame_idx)
        speed_map = {rec.track_id: rec.speed_kmh for rec in speed_records}
        time_str_map = {rec.track_id: rec.video_time_str for rec in speed_records}

        active_track_ids = {trk.track_id for trk in tracks}

        # 2. Track & Buffer Candidates
        for trk in tracks:
            tid = trk.track_id
            speed = speed_map.get(tid, 0.0)
            x1, y1, x2, y2 = map(int, trk.bbox)
            ground_x = float(trk.ground_point[0])
            ground_y = int(trk.ground_point[1])

            # Get target line Y specifically for this vehicle's current lane position
            target_line_y = self.get_capture_y_at_x(ground_x, w_frame, h_frame)

            if speed > self.speed_threshold and tid not in self.confirmed_violations:
                if tid not in self.candidates:
                    self.candidates[tid] = {
                        "max_speed": speed,
                        "best_score": -1.0,
                        "best_veh_crop": None,
                        "best_plate_crop": None,
                        "frame_idx": frame_idx,
                        "video_time": time_str_map.get(tid, format_video_time(frame_idx, self.fps)[0]),
                        "in_gate_frames": 0,
                        "last_ground_y": ground_y
                    }
                else:
                    self.candidates[tid]["max_speed"] = max(self.candidates[tid]["max_speed"], speed)
                    self.candidates[tid]["last_ground_y"] = ground_y

            # Sample near the trigger line
            if tid in self.candidates:
                cand = self.candidates[tid]
                is_in_gate = (target_line_y - gate_margin) <= ground_y <= (target_line_y + gate_margin)

                if is_in_gate:
                    cand["in_gate_frames"] += 1

                    cx1, cy1 = max(0, x1), max(0, y1)
                    cx2, cy2 = min(w_frame, x2), min(h_frame, y2)
                    if cx2 > cx1 and cy2 > cy1:
                        veh_crop = frame[cy1:cy2, cx1:cx2].copy()
                        vh, vw = veh_crop.shape[:2]

                        plate_det = self.plate_detector.detect(veh_crop, imgsz=640)
                        if plate_det:
                            px1, py1, px2, py2 = map(int, plate_det.bbox)
                            pw = px2 - px1
                            ph = py2 - py1

                            pad_x = int(pw * 0.10)
                            pad_y = int(ph * 0.12)

                            c_px1 = max(0, px1 - pad_x)
                            c_py1 = max(0, py1 - pad_y)
                            c_px2 = min(vw, px2 + pad_x)
                            c_py2 = min(vh, py2 + pad_y)

                            plate_crop = veh_crop[c_py1:c_py2, c_px1:c_px2].copy()
                            if plate_crop.size > 0:
                                gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                sharpness = compute_sharpness(gray_plate)
                                area = pw * ph

                                score = (area * 2.0) + (sharpness * 1.5) + (plate_det.confidence * 400)

                                if score > cand["best_score"]:
                                    cand["best_score"] = score
                                    cand["best_veh_crop"] = veh_crop
                                    cand["best_plate_crop"] = plate_crop
                                    cand["frame_idx"] = frame_idx
                                    cand["video_time"] = time_str_map.get(tid, cand["video_time"])

        # 3. Finalize candidate when crossing the line
        for tid in list(self.candidates.keys()):
            cand = self.candidates[tid]
            current_track = next((t for t in tracks if t.track_id == tid), None)

            if current_track is not None:
                curr_ground_x = float(current_track.ground_point[0])
                curr_ground_y = int(current_track.ground_point[1])
                target_line_y = self.get_capture_y_at_x(curr_ground_x, w_frame, h_frame)
                crossed_line = (curr_ground_y >= target_line_y)
            else:
                crossed_line = True

            left_frame = (tid not in active_track_ids)
            enough_samples = (cand["in_gate_frames"] >= 8)

            if (crossed_line or left_frame or enough_samples) and cand["best_plate_crop"] is not None:
                ocr_result = self.ocr_engine.read(cand["best_plate_crop"])

                crop_dir = os.path.join("data", "violations")
                os.makedirs(crop_dir, exist_ok=True)

                f_idx = cand["frame_idx"]
                veh_crop_path = os.path.join(crop_dir, f"veh_track_{tid}_{f_idx}.png")
                plate_crop_path = os.path.join(crop_dir, f"plate_track_{tid}_{f_idx}.png")

                cv2.imwrite(veh_crop_path, cand["best_veh_crop"])
                cv2.imwrite(plate_crop_path, cand["best_plate_crop"])

                row_id = self.db.insert_violation(
                    track_id=tid,
                    speed_kmh=cand["max_speed"],
                    plate_text=ocr_result.plate_text,
                    ocr_confidence=float(ocr_result.confidence),
                    video_time=cand["video_time"],
                    vehicle_image_path=veh_crop_path.replace("\\", "/"),
                    plate_image_path=plate_crop_path.replace("\\", "/")
                )

                self.confirmed_violations.add(tid)
                del self.candidates[tid]

                violation_rec = ViolationRecord(
                    id=row_id,
                    track_id=tid,
                    speed_kmh=cand["max_speed"],
                    speed_limit=self.speed_threshold,
                    plate_text=ocr_result.plate_text,
                    ocr_confidence=float(ocr_result.confidence),
                    video_time=cand["video_time"],
                    timestamp=cand["video_time"],
                    vehicle_image_path=veh_crop_path.replace("\\", "/"),
                    plate_image_path=plate_crop_path.replace("\\", "/")
                )
                new_violations.append(violation_rec)

                print(f"🚨 [VIOLATION CONFIRMED] ID: {row_id} | Track: {tid} | Speed: {cand['max_speed']:.1f} km/h | Plate: '{ocr_result.plate_text}'")

            elif left_frame:
                del self.candidates[tid]

        return tracks, speed_records, new_violations