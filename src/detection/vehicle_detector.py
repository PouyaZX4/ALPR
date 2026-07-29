from src.utils.data_types import Detection

class VehicleDetector:
    def __init__(self, weights_path: str, conf_threshold: float = 0.4):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold

    def detect(self, frame) -> list[Detection]:
        """
        Detect vehicles (car, truck, bus) in a frame.
        """
        pass
