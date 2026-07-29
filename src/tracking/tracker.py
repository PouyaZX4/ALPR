import torch
from ultralytics import YOLO
from src.utils.data_types import Detection, Track

VEHICLE_CLASS_IDS = [2, 3, 5, 7]

class VehicleTracker:
    def __init__(
        self, 
        weights_path: str = "yolo11s.pt", 
        conf_threshold: float = 0.3,
        imgsz: int = 1280,
        tracker_type: str = "bytetrack.yaml"
    ):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.tracker_type = tracker_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def update(self, frame) -> list[Track]:
        """
        Runs object tracking directly using Ultralytics ByteTrack integration.
        Returns clean list of `Track` dataclasses.
        """
        results = self.model.track(
            source=frame,
            persist=True,  # Maintains state between consecutive frames
            tracker=self.tracker_type,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            classes=VEHICLE_CLASS_IDS,
            device=self.device,
            verbose=False
        )[0]

        tracks = []
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(float, box)
                
                # Bottom-center point of the bounding box (Ground contact point)
                ground_x = (x1 + x2) / 2.0
                ground_y = y2

                tracks.append(Track(
                    track_id=int(track_id),
                    bbox=(x1, y1, x2, y2),
                    ground_point=(ground_x, ground_y)
                ))

        return tracks