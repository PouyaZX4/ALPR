import os
import argparse
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
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="Rotate video if recorded sideways (90, 180, 270)")
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

    # Adjust dimensions if rotated
    if args.rotate in (90, 270):
        eff_w, eff_h = orig_h, orig_w
    else:
        eff_w, eff_h = orig_w, orig_h

    pipeline = OnnxSpeedALPRPipeline(config, fps=fps)

    print("\n" + "=" * 65)
    print("=== SPEED-TRIGGERED ALPR PIPELINE (HIGH QUALITY ONNX) ===")
    print(f"Video File:     {args.video} ({orig_w}x{orig_h})")
    print(f"Frame Rate:     {fps:.2f} FPS")
    print(f"Total Frames:   {total_frames}")
    print(f"Speed Limit:    {config['thresholds']['speed_kmh']} km/h")
    if args.rotate > 0:
        print(f"Rotation:       {args.rotate}°")
    print("=" * 65 + "\n")

    frame_idx = 0
    speed_limit = float(config['thresholds']['speed_kmh'])

    window_name = "Speed ALPR Monitor"
    if not args.no_display:
        # Smart bounding box scaling: fits cleanly on any monitor (max 1100w x 680h)
        max_w, max_h = 1100, 680
        scale = min(max_w / max(1, eff_w), max_h / max(1, eff_h))
        disp_w = int(eff_w * scale)
        disp_h = int(eff_h * scale)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, disp_w, disp_h)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Optional manual rotation if video was recorded sideways on mobile
        if args.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif args.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

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

            # Draw Gate Line
            line_p1, line_p2 = pipeline.get_capture_line_endpoints(w_frame, h_frame)
            cv2.line(frame, line_p1, line_p2, (255, 0, 127), 3, cv2.LINE_AA)
            cv2.putText(frame, "ENFORCEMENT GATE", (max(15, line_p1[0]), max(25, line_p1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 127), 2, cv2.LINE_AA)

            for trk in tracks:
                x1, y1, x2, y2 = map(int, trk.bbox)
                tid = trk.track_id
                speed = speed_map.get(tid, 0.0)

                is_speeding = speed > speed_limit
                color = (0, 0, 255) if is_speeding else (0, 230, 118)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                cv2.circle(frame, (int(trk.ground_point[0]), int(trk.ground_point[1])), 5, (0, 255, 255), -1, cv2.LINE_AA)

                label = f"ID:{tid} | {speed:.1f} km/h"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

            time_str, _ = format_video_time(frame_idx, fps)
            header = f"TIME: {time_str} | FRAME: {frame_idx}/{total_frames} | LIMIT: {speed_limit:.0f} km/h"
            cv2.rectangle(frame, (10, 10), (580, 42), (0, 0, 0), -1)
            cv2.putText(frame, header, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 2, cv2.LINE_AA)

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