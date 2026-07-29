from src.utils.data_types import Detection, Track

class VehicleTracker:
    def __init__(self):
        pass

    def update(self, detections: list[Detection], frame) -> list[Track]:
        """
        Update tracker with new detections and return tracks assigned with track_ids.
        """
        pass
