import os
import argparse
import cv2
import yaml
import numpy as np
from src.pipeline.pipeline import SpeedALPRPipeline
from src.speed.speed_estimator import format_video_time


def main():
    parser = argparse.ArgumentParser(description="Speed-Triggered ALPR Pipeline CLI")
    parser.add_argument("--video", required=True, help="Path to input traffic video")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without GUI window")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video file '{args.video}' does not exist.")
        return

    # Load Config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pipeline = SpeedALPRPipeline(config, fps=fps)

    print("\n" + "=" * 60)
    print("=== SPEED-TRIGGERED ALPR PIPELINE RUNNING ===")
    print(f"Video File:     {args.video}")
    print(f"Frame Rate:     {fps:.2f} FPS")
    print(f"Total Frames:   {total_frames}")
    print(f"Speed Limit:    {config['thresholds']['speed_kmh']} km/h")
    print("=" * 60 + "\n")

    frame_idx = 0
    speed_limit = float(config['thresholds']['speed_kmh'])

    window_name = "Speed ALPR Main Pipeline"
    if not args.no_display:
        # Create a resizable window with standard 1280x720 aspect ratio
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h_frame, w_frame = frame.shape[:2]

        # Process Frame
        tracks, speed_records, new_violations = pipeline.process_frame(frame, frame_idx)

        # Print new violations
        for v in new_violations:
            print(f"\n🚨 [SPEEDING VIOLATION RECORDED]")
            print(f"   ├─ Track ID:      #{v.track_id}")
            print(f"   ├─ Speed:         {v.speed_kmh:.1f} km/h (Limit: {v.speed_limit:.0f} km/h)")
            print(f"   ├─ Plate Text:    {v.plate_text} (Conf: {v.ocr_confidence * 100:.1f}%)")
            print(f"   ├─ Video Time:    {v.video_time}")
            print(f"   ├─ Vehicle Crop:  {v.vehicle_image_path}")
            print(f"   └─ Plate Crop:    {v.plate_image_path}\n")

        # Visualization
        if not args.no_display:
            speed_map = {rec.track_id: rec.speed_kmh for rec in speed_records}

            # Draw Virtual Capture Gate Line
            capture_line_y = int(h_frame * pipeline.capture_line_ratio)
            cv2.line(frame, (0, capture_line_y), (w_frame, capture_line_y), (255, 105, 180), 3)
            cv2.putText(frame, "VIRTUAL ENFORCEMENT & OCR GATE", (25, capture_line_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 105, 180), 2)

            for trk in tracks:
                x1, y1, x2, y2 = map(int, trk.bbox)
                tid = trk.track_id
                speed = speed_map.get(tid, 0.0)

                is_speeding = speed > speed_limit
                color = (0, 0, 255) if is_speeding else (0, 230, 118)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Draw ground contact tracking circle
                cv2.circle(frame, (int(trk.ground_point[0]), int(trk.ground_point[1])), 5, (0, 255, 255), -1)

                label = f"ID: {tid} | {speed:.1f} km/h"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Header info
            time_str, _ = format_video_time(frame_idx, fps)
            header = f"TIME: {time_str} | FRAME: {frame_idx}/{total_frames} | LIMIT: {speed_limit:.0f} km/h"
            cv2.rectangle(frame, (10, 10), (620, 45), (0, 0, 0), -1)
            cv2.putText(frame, header, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 2)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] User requested stop.")
                break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"\n[DONE] Processing complete. Database updated: '{config['paths']['database']}'")


if __name__ == "__main__":
    main()