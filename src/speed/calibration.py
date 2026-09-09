import os
import cv2
import numpy as np
from typing import List, Tuple, Optional


def compute_homography(
    pixel_points: List[Tuple[float, float]],
    world_points: List[Tuple[float, float]]
) -> np.ndarray:
    """Computes 3x3 Homography Matrix (H) from 4 points."""
    if len(pixel_points) != 4 or len(world_points) != 4:
        raise ValueError("Exactly 4 points are required for homography calculation.")
    pts_src = np.array(pixel_points, dtype=np.float32)
    pts_dst = np.array(world_points, dtype=np.float32)
    H, status = cv2.findHomography(pts_src, pts_dst)
    if H is None:
        raise ValueError("Could not compute valid homography matrix with given points.")
    return H


def pixel_to_world(point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """Transforms a pixel coordinate (x, y) to real-world meters (X, Y)."""
    pt = np.array([point[0], point[1], 1.0], dtype=np.float32)
    world_pt = np.dot(H, pt)
    if abs(world_pt[2]) < 1e-6:
        return 0.0, 0.0
    world_x = world_pt[0] / world_pt[2]
    world_y = world_pt[1] / world_pt[2]
    return float(world_x), float(world_y)


def save_homography(H: np.ndarray, path: str = "data/calibration/homography.npy") -> None:
    """Saves matrix to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, H)


def load_homography(path: str = "data/calibration/homography.npy") -> np.ndarray:
    """Loads matrix from disk."""
    if not os.path.exists(path):
        return np.eye(3, dtype=np.float32)
    return np.load(path)


def extract_reference_frame(video_path: str, frame_number: int = 0) -> Optional[np.ndarray]:
    """Extracts a single frame from video for calibration visual setup."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None