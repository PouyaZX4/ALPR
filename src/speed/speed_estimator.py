import numpy as np
from collections import defaultdict, deque
from src.speed.calibration import pixel_to_world, load_homography
from src.utils.data_types import Track, SpeedRecord


class SpeedEstimator:
    def __init__(
        self, 
        homography_matrix: np.ndarray, 
        fps: float = 30.0, 
        window_size: int = 8
    ):
        self.H = homography_matrix
        self.fps = fps
        self.window_size = window_size
        
        # Stores track history: {track_id: deque([(world_x, world_y, frame_idx), ...])}
        self.track_history = defaultdict(lambda: deque(maxlen=self.window_size))

    def update(self, tracks: list[Track], frame_idx: int) -> list[SpeedRecord]:
        speed_records = []

        for track in tracks:
            # 1. Convert ground pixel point to real-world meters
            world_x, world_y = pixel_to_world(track.ground_point, self.H)

            # 2. Add to track history
            history = self.track_history[track.track_id]
            history.append((world_x, world_y, frame_idx))

            # 3. We need at least 2 points in history to calculate speed
            if len(history) >= 2:
                # Compare oldest point in window to newest point
                x_start, y_start, frame_start = history[0]
                x_end, y_end, frame_end = history[-1]

                # Euclidean distance in meters
                distance_m = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)

                # Time elapsed in seconds
                frame_diff = frame_end - frame_start
                if frame_diff == 0:
                    continue
                time_s = frame_diff / self.fps

                # Speed calculation: (meters / seconds) * 3.6 = km/h
                speed_mps = distance_m / time_s
                speed_kmh = speed_mps * 3.6

                speed_records.append(SpeedRecord(
                    track_id=track.track_id,
                    speed_kmh=round(speed_kmh, 1),
                    frame_idx=frame_idx
                ))

        return speed_records