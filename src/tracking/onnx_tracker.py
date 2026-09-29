import numpy as np
from typing import List, Tuple
from src.utils.data_types import Track, Detection


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0.0 else 0.0


class VelocityTrack:
    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float]):
        self.track_id = track_id
        self.bbox = [float(b) for b in bbox]
        self.vx = 0.0
        self.vy = 0.0
        self.misses = 0
        self.hits = 1

    def predict(self):
        self.bbox[0] += self.vx
        self.bbox[1] += self.vy
        self.bbox[2] += self.vx
        self.bbox[3] += self.vy

    def update(self, new_bbox: Tuple[float, float, float, float]):
        # Calculate velocity vector
        dx = new_bbox[0] - self.bbox[0]
        dy = new_bbox[1] - self.bbox[1]

        self.vx = 0.7 * self.vx + 0.3 * dx
        self.vy = 0.7 * self.vy + 0.3 * dy

        # Directly snap to measured detection box to eliminate bounding box lag
        self.bbox = [float(b) for b in new_bbox]
        self.misses = 0
        self.hits += 1


class OnnxVehicleTracker:
    def __init__(self, iou_threshold: float = 0.20, max_misses: int = 15):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.next_id = 1
        self.tracks: List[VelocityTrack] = []

    def predict_only(self) -> List[Track]:
        active_tracks: List[Track] = []
        for trk in self.tracks:
            trk.predict()
            x1, y1, x2, y2 = trk.bbox
            ground_x = (x1 + x2) / 2.0
            ground_y = y2
            active_tracks.append(Track(
                track_id=trk.track_id,
                bbox=(x1, y1, x2, y2),
                ground_point=(ground_x, ground_y)
            ))
        return active_tracks

    def update(self, detections: List[Detection]) -> List[Track]:
        for trk in self.tracks:
            trk.predict()

        active_tracks: List[Track] = []
        det_boxes = [d.bbox for d in detections]
        used_detections = set()

        # Match tracks to fresh detections
        for trk in self.tracks:
            best_iou = 0.0
            best_idx = -1
            for idx, box in enumerate(det_boxes):
                if idx in used_detections:
                    continue
                iou = calculate_iou(trk.bbox, box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= self.iou_threshold and best_idx != -1:
                trk.update(det_boxes[best_idx])
                used_detections.add(best_idx)
            else:
                trk.misses += 1

        # Drop dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        # Register new tracks
        for idx, box in enumerate(det_boxes):
            if idx not in used_detections:
                new_trk = VelocityTrack(self.next_id, box)
                self.next_id += 1
                self.tracks.append(new_trk)

        # Build output tracks
        for trk in self.tracks:
            if trk.misses <= 1:
                x1, y1, x2, y2 = trk.bbox
                ground_x = (x1 + x2) / 2.0
                ground_y = y2
                active_tracks.append(Track(
                    track_id=trk.track_id,
                    bbox=(x1, y1, x2, y2),
                    ground_point=(ground_x, ground_y)
                ))

        return active_tracks