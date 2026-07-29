from src.utils.data_types import Detection

class PlateDetector:
    def __init__(self, weights_path: str, conf_threshold: float = 0.5):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold

    def detect(self, vehicle_crop) -> Detection | None:
        """
        Detect license plate in a cropped vehicle image.
        """
        pass
