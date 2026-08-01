import os
import shutil
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from src.utils.data_types import Detection


class PlateDetector:
    def __init__(
        self,
        repo_id: str = "morsetechlab/yolov11-license-plate-detection",
        filename: str = "license-plate-finetune-v1n.pt",
        local_weights_dir: str = "models/plate_detector",
        conf_threshold: float = 0.3
    ):
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(local_weights_dir, exist_ok=True)
        local_weights_path = os.path.join(local_weights_dir, "best.pt")

        # Download from Hugging Face automatically if local file doesn't exist
        if not os.path.exists(local_weights_path):
            print(f"[INFO] Downloading plate detector from Hugging Face ({repo_id})...")
            try:
                hf_file = hf_hub_download(repo_id=repo_id, filename=filename)
                shutil.copy(hf_file, local_weights_path)
                print(f"[SUCCESS] Saved model weights to '{local_weights_path}'")
            except Exception as e:
                print(f"[ERROR] Failed to download from Hugging Face: {e}")
                raise e

        # Load YOLO model
        self.model = YOLO(local_weights_path)

    def detect(self, vehicle_crop) -> Detection | None:
        """
        Takes a cropped vehicle image (numpy array).
        Returns Detection object for the highest-confidence license plate, or None.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        results = self.model.predict(
            source=vehicle_crop,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return None

        # Pick box with highest confidence
        best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        conf = float(best_box.conf[0])
        cls_id = int(best_box.cls[0])

        return Detection(
            bbox=(x1, y1, x2, y2),
            confidence=conf,
            class_id=cls_id
        )