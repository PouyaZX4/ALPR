import argparse
import cv2
import yaml
import numpy as np
from src.pipeline.pipeline import SpeedALPRPipeline


def main():
    parser = argparse.ArgumentParser(description="Speed-Triggered ALPR Pipeline CLI")
    parser.add_argument("--video", required=True, help="Path to input traffic video")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize Pipeline
    pipeline = SpeedALPRPipeline(config)

    print(f"=== RUNNING SPEED ALPR MAIN PIPELINE ===")
    print(f"Video: {args.video}")
    print(f"Speed Limit: {config['thresholds']['speed_kmh']} km/h\n")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Process Frame
        tracks, speed_records = pipeline.process_frame(frame, frame_idx)

        # Draw Overlay Boxes
        speed_map = {rec.track_id: rec.speed_kmh for rec in speed_records}
        for trk in tracks:
            x1, y1, x2, y2 = map(int, trk.bbox)
            tid = trk.track_id
            speed = speed_map.get(tid, 0.0)

            is_speeding = speed > config['thresholds']['speed_kmh']
            color = (0, 0, 255) if is_speeding else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {tid} | {speed:.1f} km/h"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Speed ALPR Main Pipeline", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Processing complete. Violations stored in '{config['paths']['database']}'")


if __name__ == "__main__":
    main()