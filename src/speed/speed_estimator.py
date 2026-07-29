import numpy as np
from src.utils.data_types import Track, SpeedRecord

class SpeedEstimator:
    def __init__(self, homography: np.ndarray, fps: float, window_size: int = 8):
        self.homography = homography
        self.fps = fps
        self.window_size = window_size

    def update(self, tracks: list[Track], frame_idx: int) -> list[SpeedRecord]:
        """
        Updates tracking history and estimates speeds per track.
        """
        pass
