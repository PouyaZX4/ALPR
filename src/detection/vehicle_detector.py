import torch
from ultralytics import YOLO
from src.utils.data_types import Detection

# COCO Class IDs: 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

class VehicleDetector:
    def __init__(
        self, 
        weights_path: str = "yolo11s.pt", 
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        imgsz: int = 1280,  # Increase resolution for distant/small cars
        device: str = None
    ):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        
        # Auto-select GPU if available
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def detect(self, frame) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=VEHICLE_CLASS_IDS,  # YOLO filters directly on hardware
            device=self.device,
            verbose=False
        )[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id
                ))
        return detections