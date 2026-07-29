import numpy as np

def compute_homography(pixel_points: list[tuple[float, float]], real_world_points: list[tuple[float, float]]) -> np.ndarray:
    """
    Computes a 3x3 homography matrix mapping image pixel points to real-world coordinates.
    """
    pass

def load_homography(path: str) -> np.ndarray:
    """
    Loads homography matrix from a .npy file.
    """
    pass

def pixel_to_world(point: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """
    Transforms pixel coordinate (x, y) to real-world coordinate (X, Y) using matrix H.
    """
    pass
