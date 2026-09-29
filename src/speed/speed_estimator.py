import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple
from src.speed.calibration import pixel_to_world
from src.utils.data_types import Track, SpeedRecord


def format_video_time(frame_idx: int, fps: float) -> Tuple[str, float]:
    """Converts frame index and FPS to MM:SS.ss string and total seconds."""
    if fps <= 0:
        fps = 30.0
    total_seconds = frame_idx / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    time_str = f"{minutes:02d}:{seconds:05.2f}"
    return time_str, total_seconds


class SpeedEstimator:
    def __init__(
        self, 
        homography_matrix: np.ndarray, 
        fps: float = 30.0, 
        window_size: int = 8
    ):
        self.H = homography_matrix
        self.fps = fps if fps > 0 else 30.0
        self.window_size = window_size
        
        # Stores track history: {track_id: deque([(world_x, world_y, frame_idx), ...])}
        self.track_history = defaultdict(lambda: deque(maxlen=self.window_size))

    def set_fps(self, fps: float):
        if fps and fps > 0:
            self.fps = float(fps)

    def set_homography(self, H: np.ndarray):
        if H is not None and H.shape == (3, 3):
            self.H = H

    def update(self, tracks: List[Track], frame_idx: int) -> List[SpeedRecord]:
        speed_records = []
        video_time_str, timestamp_seconds = format_video_time(frame_idx, self.fps)

        for track in tracks:
            # 1. Convert ground pixel point to real-world meters
            world_x, world_y = pixel_to_world(track.ground_point, self.H)

            # 2. Add to track history
            history = self.track_history[track.track_id]
            history.append((world_x, world_y, frame_idx))

            # 3. Need at least 2 apoints to compute delta
            if len(history) >= 2:
                x_start, y_start, frame_start = history[0]
                x_end, y_end, frame_end = history[-1]

                distance_m = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
                frame_diff = frame_end - frame_start
                if frame_diff == 0:
                    continue

                time_s = frame_diff / self.fps
                speed_mps = distance_m / time_s
                speed_kmh = speed_mps * 3.6

                speed_records.append(SpeedRecord(
                    track_id=track.track_id,
                    speed_kmh=round(speed_kmh, 1),
                    frame_idx=frame_idx,
                    video_time_str=video_time_str,
                    timestamp_seconds=timestamp_seconds
                ))

        return speed_records