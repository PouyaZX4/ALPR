import os
import cv2
import numpy as np


def compute_homography(
    pixel_points: list[tuple[float, float]],
    world_points: list[tuple[float, float]]
) -> np.ndarray:
    """Computes 3x3 Homography Matrix (H) from 4 points."""
    pts_src = np.array(pixel_points, dtype=np.float32)
    pts_dst = np.array(world_points, dtype=np.float32)
    H, _ = cv2.findHomography(pts_src, pts_dst)
    return H


def pixel_to_world(point: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """Transforms a pixel coordinate (x, y) to real-world meters (X, Y)."""
    pt = np.array([point[0], point[1], 1.0], dtype=np.float32)
    world_pt = np.dot(H, pt)
    world_x = world_pt[0] / world_pt[2]
    world_y = world_pt[1] / world_pt[2]
    return float(world_x), float(world_y)


def save_homography(H: np.ndarray, path: str = "data/calibration/homography.npy") -> None:
    """Saves matrix to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, H)


def load_homography(path: str = "data/calibration/homography.npy") -> np.ndarray:
    """Loads matrix from disk."""
    return np.load(path)


# --- Interactive Calibration Tool (Runs when executing this script directly) ---

click_points = []

def _mouse_callback(event, x, y, flags, param):
    global click_points
    if event == cv2.EVENT_LBUTTONDOWN and len(click_points) < 4:
        click_points.append((x, y))
        print(f"Point {len(click_points)} selected: ({x}, {y})")


def run_calibration_gui(video_path: str, output_path: str):
    global click_points
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Could not read video at {video_path}")
        return

    clone = frame.copy()
    cv2.namedWindow("Calibrate Camera")
    cv2.setMouseCallback("Calibrate Camera", _mouse_callback)

    print("\n--- CALIBRATION INSTRUCTIONS ---")
    print("Click 4 points on the road in order:")
    print(" 1. Top-Left | 2. Top-Right | 3. Bottom-Right | 4. Bottom-Left")
    print("Press 'r' to reset points. Press 'c' when done.\n")

    while True:
        display_frame = clone.copy()

        for i, pt in enumerate(click_points):
            cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(display_frame, f"P{i+1}", (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(click_points) > 1:
            for i in range(len(click_points) - 1):
                cv2.line(display_frame, click_points[i], click_points[i + 1], (255, 0, 0), 2)
        if len(click_points) == 4:
            cv2.line(display_frame, click_points[3], click_points[0], (255, 0, 0), 2)

        cv2.imshow("Calibrate Camera", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            click_points = []
            print("Points reset.")
        elif key == ord('c'):
            if len(click_points) == 4:
                break
            else:
                print("Click 4 points first!")

    cv2.destroyAllWindows()

    print("\n--- REAL-WORLD DISTANCES ---")
    width_m = float(input("Enter real-world WIDTH of this box in meters (e.g. 3.5 for a lane): "))
    length_m = float(input("Enter real-world LENGTH of this box in meters (e.g. 15.0): "))

    world_points = [
        (0.0, 0.0),
        (width_m, 0.0),
        (width_m, length_m),
        (0.0, length_m)
    ]

    H = compute_homography(click_points, world_points)
    save_homography(H, output_path)
    print(f"\n[SUCCESS] Homography matrix saved to {output_path}")


if __name__ == "__main__":
    # Allows running this file directly to calibrate
    video_input = r"G:\AILPR\data\raw_videos\Cars Moving On Road Stock Footage - Free Download.mp4"
    save_output = "data/calibration/homography.npy"
    run_calibration_gui(video_input, save_output)