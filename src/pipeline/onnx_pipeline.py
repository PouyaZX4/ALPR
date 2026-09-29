import os
import cv2
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from src.detection.onnx_yolo import OnnxYOLO
from src.tracking.onnx_tracker import OnnxVehicleTracker
from src.ocr.onnx_ocr import OnnxPlateOCR
from src.speed.calibration import load_homography
from src.speed.speed_estimator import SpeedEstimator, format_video_time
from src.database.db import ViolationDB
from src.utils.data_types import Track, SpeedRecord, ViolationRecord


def compute_sharpness(gray_img: np.ndarray) -> float:
    """Computes the Laplacian variance to evaluate image focus and sharpness."""
    if gray_img is None or gray_img.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())


def enhance_plate_contrast(plate_img: np.ndarray) -> np.ndarray:
    """Applies unsharp masking to enhance Persian plate character edges."""
    if plate_img is None or plate_img.size == 0:
        return plate_img
    gaussian = cv2.GaussianBlur(plate_img, (0, 0), 2.0)
    unsharp = cv2.addWeighted(plate_img, 1.35, gaussian, -0.35, 0)
    return unsharp


class OnnxSpeedALPRPipeline:
    def __init__(self, config: dict, fps: float = 30.0):
        self.config = config
        self.speed_threshold = float(config.get('thresholds', {}).get('speed_kmh', 60.0))
        self.fps = fps if fps > 0 else 30.0

        self.calib_points: Optional[List[List[float]]] = None
        self.custom_gate_line: Optional[List[List[float]]] = None

        # Load calibration points from disk if available
        pts_path = config.get('paths', {}).get('calibration_points', "data/calibration/calibration_points.json")
        if os.path.exists(pts_path):
            try:
                with open(pts_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.calib_points = data.get("calibration_points", None)
                    self.custom_gate_line = data.get("custom_gate_line", None)
            except Exception:
                pass

        # Load Homography Matrix
        homography_path = config.get('paths', {}).get('homography', 'data/calibration/homography.npy')
        self.H = load_homography(homography_path) if os.path.exists(homography_path) else np.eye(3, dtype=np.float32)

        print("[INFO] Initializing Clean-Crop ONNX Engines...")
        veh_model = config.get('paths', {}).get('vehicle_detector', "models/onnx/vehicle_detector.onnx")
        self.vehicle_detector = OnnxYOLO(
            veh_model, 
            imgsz=1280, 
            conf_thresh=config.get('thresholds', {}).get('vehicle_conf', 0.25)
        )
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        self.tracker = OnnxVehicleTracker(iou_threshold=0.20, max_misses=15)
        self.speed_estimator = SpeedEstimator(
            self.H, 
            fps=self.fps, 
            window_size=config.get('speed', {}).get('smoothing_window', 8)
        )

        plate_model = config.get('paths', {}).get('plate_detector', "models/onnx/plate_detector.onnx")
        self.plate_detector = OnnxYOLO(
            plate_model, 
            imgsz=640, 
            conf_thresh=config.get('thresholds', {}).get('plate_conf', 0.20)
        )
        
        ocr_model = config.get('paths', {}).get('ocr_model', "models/onnx/crnn_plate_ocr.onnx")
        ocr_vocab = config.get('paths', {}).get('ocr_vocab', "models/onnx/ocr_vocab.json")
        self.ocr_engine = OnnxPlateOCR(ocr_model, ocr_vocab)
        
        db_path = config.get('paths', {}).get('database', 'data/violations.db')
        self.db = ViolationDB(db_path=db_path)

        # Violation Deduplication & Candidate State
        self.confirmed_violations = set()          # Set of confirmed track IDs
        self.candidates: Dict[int, Dict[str, Any]] = {}
        self.recent_plates: Dict[str, int] = {}    # {plate_str: frame_idx}
        self.plate_cooldown_frames = int(self.fps * 12)  # 12-second temporal cooldown

        self.detect_interval = 2
        print("✅ ONNX Speed & Persian ALPR Pipeline Successfully Initialized!")

    def set_fps(self, fps: float):
        if fps and fps > 0:
            self.fps = float(fps)
            self.speed_estimator.set_fps(fps)
            self.plate_cooldown_frames = int(self.fps * 12)

    def set_homography(self, H: np.ndarray):
        if H is not None:
            self.H = H
            self.speed_estimator.set_homography(H)

    def set_calibration_data(self, calib_points: Optional[List[List[float]]], custom_gate_line: Optional[List[List[float]]]):
        self.calib_points = calib_points
        self.custom_gate_line = custom_gate_line

    def get_capture_line_endpoints(self, w_frame: int, h_frame: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if self.custom_gate_line and len(self.custom_gate_line) == 2:
            p1 = (int(self.custom_gate_line[0][0]), int(self.custom_gate_line[0][1]))
            p2 = (int(self.custom_gate_line[1][0]), int(self.custom_gate_line[1][1]))
            return p1, p2
        elif self.calib_points and len(self.calib_points) == 4:
            p4 = (int(self.calib_points[3][0]), int(self.calib_points[3][1]))
            p3 = (int(self.calib_points[2][0]), int(self.calib_points[2][1]))
            return p4, p3
        else:
            default_y = int(h_frame * 0.70)
            return (0, default_y), (w_frame, default_y)

    def get_capture_y_at_x(self, x: float, w_frame: int, h_frame: int) -> int:
        (x1, y1), (x2, y2) = self.get_capture_line_endpoints(w_frame, h_frame)
        if abs(x2 - x1) < 1e-4:
            return int((y1 + y2) / 2)
        slope = (y2 - y1) / (x2 - x1)
        y = y1 + slope * (x - x1)
        return int(np.clip(y, 10, h_frame - 10))

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[List[Track], List[SpeedRecord], List[ViolationRecord]]:
        h_frame, w_frame, _ = frame.shape
        gate_margin = int(h_frame * 0.15)
        new_violations: List[ViolationRecord] = []

        # 1. Detection and Tracking
        if frame_idx % self.detect_interval == 0:
            detections = self.vehicle_detector.detect(frame, target_classes=self.vehicle_classes)
            tracks = self.tracker.update(detections)
        else:
            tracks = self.tracker.predict_only()

        # 2. Compensate Bumper Parallax on Ground Points BEFORE Speed Estimation
        for trk in tracks:
            x1, y1, x2, y2 = map(int, trk.bbox)
            box_h = y2 - y1
            # Lift ground_point off the protruding bumper as it nears the lens
            parallax_lift = int(box_h * 0.12) if box_h > (h_frame * 0.18) else 0
            trk.ground_point = (float((x1 + x2) / 2.0), float(y2 - parallax_lift))

        # 3. Velocity Estimation
        speed_records = self.speed_estimator.update(tracks, frame_idx)
        speed_map = {rec.track_id: rec.speed_kmh for rec in speed_records}
        time_str_map = {rec.track_id: rec.video_time_str for rec in speed_records}
        active_track_ids = {trk.track_id for trk in tracks}

        # 4. Dynamic Enforcement Gate & Speed Locking
        for trk in tracks:
            tid = trk.track_id
            raw_speed = speed_map.get(tid, 0.0)
            ground_x, ground_y = trk.ground_point
            target_line_y = self.get_capture_y_at_x(ground_x, w_frame, h_frame)

            # Near-gate distance check: within 12% of the gate line
            is_near_gate = abs(ground_y - target_line_y) <= (h_frame * 0.12)

            # Candidate registration
            if raw_speed > self.speed_threshold and tid not in self.confirmed_violations:
                if tid not in self.candidates:
                    self.candidates[tid] = {
                        "stable_speeds": [raw_speed],
                        "final_speed": raw_speed,
                        "best_score": -1.0,
                        "best_veh_crop": None,
                        "best_plate_crop": None,
                        "frame_idx": frame_idx,
                        "video_time": time_str_map.get(tid, format_video_time(frame_idx, self.fps)[0]),
                        "in_gate_frames": 0,
                    }
                else:
                    cand = self.candidates[tid]
                    # Only collect speed samples in the stable upstream zone (ignore near-gate distortion)
                    if not is_near_gate and raw_speed > (self.speed_threshold * 0.8):
                        cand["stable_speeds"].append(raw_speed)
                        # Clean median speed
                        sorted_s = sorted(cand["stable_speeds"])
                        cand["final_speed"] = sorted_s[len(sorted_s) // 2]

            # Sample best license plate crop while inside gate
            if tid in self.candidates:
                cand = self.candidates[tid]
                is_in_gate = (target_line_y - gate_margin) <= ground_y <= (target_line_y + gate_margin)

                if is_in_gate:
                    cand["in_gate_frames"] += 1
                    x1, y1, x2, y2 = map(int, trk.bbox)
                    bw, bh = x2 - x1, y2 - y1
                    pad_vw, pad_vh = int(bw * 0.10), int(bh * 0.12)

                    cx1, cy1 = max(0, x1 - pad_vw), max(0, y1 - pad_vh)
                    cx2, cy2 = min(w_frame, x2 + pad_vw), min(h_frame, y2 + pad_vh)

                    if cx2 > cx1 and cy2 > cy1:
                        veh_crop = frame[cy1:cy2, cx1:cx2].copy()
                        vh, vw = veh_crop.shape[:2]

                        plate_dets = self.plate_detector.detect(veh_crop)
                        if plate_dets:
                            plate_det = max(plate_dets, key=lambda d: d.confidence)
                            px1, py1, px2, py2 = map(int, plate_det.bbox)
                            pw, ph = px2 - px1, py2 - py1

                            c_px1 = max(0, px1 - int(pw * 0.15))
                            c_py1 = max(0, py1 - int(ph * 0.18))
                            c_px2 = min(vw, px2 + int(pw * 0.15))
                            c_py2 = min(vh, py2 + int(ph * 0.18))

                            plate_crop = veh_crop[c_py1:c_py2, c_px1:c_px2].copy()
                            if plate_crop.size > 0:
                                plate_crop_enhanced = enhance_plate_contrast(plate_crop)
                                gray_plate = cv2.cvtColor(plate_crop_enhanced, cv2.COLOR_BGR2GRAY)
                                sharpness = compute_sharpness(gray_plate)
                                score = ((pw * ph) * 2.0) + (sharpness * 1.5) + (plate_det.confidence * 400)

                                if score > cand["best_score"]:
                                    cand["best_score"] = score
                                    cand["best_veh_crop"] = veh_crop
                                    cand["best_plate_crop"] = plate_crop_enhanced
                                    cand["frame_idx"] = frame_idx
                                    cand["video_time"] = time_str_map.get(tid, cand["video_time"])

        # 5. Trigger OCR & Confirm Violation
        for tid in list(self.candidates.keys()):
            cand = self.candidates[tid]
            current_track = next((t for t in tracks if t.track_id == tid), None)

            if current_track is not None:
                curr_gx, curr_gy = current_track.ground_point
                target_line_y = self.get_capture_y_at_x(curr_gx, w_frame, h_frame)
                crossed_line = (curr_gy >= target_line_y)
            else:
                crossed_line = True

            left_frame = (tid not in active_track_ids)
            enough_samples = (cand["in_gate_frames"] >= 6)

            if (crossed_line or left_frame or enough_samples) and cand["best_plate_crop"] is not None:
                ocr_result = self.ocr_engine.read(cand["best_plate_crop"])
                final_display_plate = ocr_result.formatted_text
                normalized_plate = final_display_plate.strip().replace(" ", "")

                is_duplicate = False
                if normalized_plate in self.recent_plates:
                    if (frame_idx - self.recent_plates[normalized_plate]) < self.plate_cooldown_frames:
                        is_duplicate = True

                crop_dir = os.path.join("data", "violations")
                os.makedirs(crop_dir, exist_ok=True)
                f_idx = cand["frame_idx"]
                veh_crop_path = os.path.join(crop_dir, f"veh_track_{tid}_{f_idx}.png")
                plate_crop_path = os.path.join(crop_dir, f"plate_track_{tid}_{f_idx}.png")

                cv2.imwrite(veh_crop_path, cand["best_veh_crop"])
                cv2.imwrite(plate_crop_path, cand["best_plate_crop"])

                # Use the clean, locked upstream speed
                final_speed = cand["final_speed"]

                if not is_duplicate and normalized_plate != "UNKNOWN":
                    row_id = self.db.insert_violation(
                        track_id=tid,
                        speed_kmh=final_speed,
                        plate_text=final_display_plate,
                        ocr_confidence=float(ocr_result.confidence),
                        video_time=cand["video_time"],
                        vehicle_image_path=veh_crop_path.replace("\\", "/"),
                        plate_image_path=plate_crop_path.replace("\\", "/")
                    )

                    self.confirmed_violations.add(tid)
                    self.recent_plates[normalized_plate] = frame_idx

                    violation_rec = ViolationRecord(
                        id=row_id,
                        track_id=tid,
                        speed_kmh=final_speed,
                        speed_limit=self.speed_threshold,
                        plate_text=final_display_plate,
                        ocr_confidence=float(ocr_result.confidence),
                        video_time=cand["video_time"],
                        timestamp=cand["video_time"],
                        vehicle_image_path=veh_crop_path.replace("\\", "/"),
                        plate_image_path=plate_crop_path.replace("\\", "/")
                    )
                    new_violations.append(violation_rec)
                    print(f"🚨 [VIOLATION LOGGED] #{row_id} | Track: {tid} | Speed: {final_speed:.1f} km/h | Plate: {final_display_plate}")
                else:
                    self.confirmed_violations.add(tid)

                del self.candidates[tid]

            elif left_frame:
                del self.candidates[tid]

        return tracks, speed_records, new_violations