import os
import argparse
import time
import cv2
import yaml
import numpy as np
from src.pipeline.onnx_pipeline import OnnxSpeedALPRPipeline
from src.speed.speed_estimator import format_video_time


def main():
    parser = argparse.ArgumentParser(description="Speed-Triggered ALPR Pipeline (High Quality ONNX)")
    parser.add_argument("--video", required=True, help="Path to input traffic video")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without GUI window")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video file '{args.video}' does not exist.")
        return

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipeline = OnnxSpeedALPRPipeline(config, fps=fps)

    print("\n" + "=" * 60)
    print("=== SPEED-TRIGGERED ALPR PIPELINE (HIGH QUALITY ONNX) ===")
    print(f"Video File:     {args.video} ({orig_w}x{orig_h})")
    print(f"Frame Rate:     {fps:.2f} FPS")
    print(f"Total Frames:   {total_frames}")
    print(f"Speed Limit:    {config['thresholds']['speed_kmh']} km/h")
    print("=" * 60 + "\n")

    frame_idx = 0
    speed_limit = float(config['thresholds']['speed_kmh'])

    window_name = "Speed ALPR High-Quality Stream"
    if not args.no_display:
        # Calculate aspect ratio correctly to avoid stretching
        display_w = 1280
        display_h = int(display_w * (orig_h / orig_w))
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_w, display_h)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h_frame, w_frame = frame.shape[:2]

        tracks, speed_records, new_violations = pipeline.process_frame(frame, frame_idx)

        for v in new_violations:
            print(f"\n🚨 [SPEEDING VIOLATION CONFIRMED]")
            print(f"   ├─ Track ID:      #{v.track_id}")
            print(f"   ├─ Speed:         {v.speed_kmh:.1f} km/h (Limit: {v.speed_limit:.0f} km/h)")
            print(f"   ├─ Plate Text:    {v.plate_text} (Conf: {v.ocr_confidence * 100:.1f}%)")
            print(f"   ├─ Video Time:    {v.video_time}")
            print(f"   ├─ Vehicle Crop:  {v.vehicle_image_path}")
            print(f"   └─ Plate Crop:    {v.plate_image_path}\n")

        if not args.no_display:
            speed_map = {rec.track_id: rec.speed_kmh for rec in speed_records}

            # Draw Anti-Aliased High-Contrast Gate Line
            line_p1, line_p2 = pipeline.get_capture_line_endpoints(w_frame, h_frame)
            cv2.line(frame, line_p1, line_p2, (255, 0, 127), 3, cv2.LINE_AA)
            cv2.putText(frame, "ENFORCEMENT GATE", (max(15, line_p1[0]), max(25, line_p1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 127), 2, cv2.LINE_AA)

            for trk in tracks:
                x1, y1, x2, y2 = map(int, trk.bbox)
                tid = trk.track_id
                speed = speed_map.get(tid, 0.0)

                is_speeding = speed > speed_limit
                color = (0, 0, 255) if is_speeding else (0, 230, 118)

                # Draw crisp boxes with anti-aliasing
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.circle(frame, (int(trk.ground_point[0]), int(trk.ground_point[1])), 5, (0, 255, 255), -1, cv2.LINE_AA)

                label = f"ID:{tid} | {speed:.1f} km/h"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 24), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            time_str, _ = format_video_time(frame_idx, fps)
            header = f"TIME: {time_str} | FRAME: {frame_idx}/{total_frames} | LIMIT: {speed_limit:.0f} km/h"
            cv2.rectangle(frame, (10, 10), (620, 44), (0, 0, 0), -1)
            cv2.putText(frame, header, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 2, cv2.LINE_AA)

            # Display maintaining clean scaling
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"\n[DONE] Processing complete. Database updated: '{config['paths']['database']}'")


if __name__ == "__main__":
    main()