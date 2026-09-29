import sys
import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import yaml
import shutil
import asyncio
import numpy as np
from typing import List, Optional
from tqdm import tqdm

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Body, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse

from src.database.db import ViolationDB
from src.pipeline.onnx_pipeline import OnnxSpeedALPRPipeline
from src.speed.calibration import compute_homography, save_homography, load_homography, extract_reference_frame

CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CALIBRATION_PATH = os.path.join(DATA_DIR, "calibration", "homography.npy")
CALIB_PTS_PATH = os.path.join(DATA_DIR, "calibration", "calibration_points.json")
REF_FRAME_PATH = os.path.join(DATA_DIR, "calibration", "reference_frame.jpg")

os.makedirs(os.path.join(DATA_DIR, "raw_videos"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "violations"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "calibration"), exist_ok=True)

# -------------------------------------------------------------
# Ensure static directory & genuine placeholder.png exist on disk
# -------------------------------------------------------------
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

placeholder_file = os.path.join(static_dir, "placeholder.png")
if not os.path.exists(placeholder_file):
    dummy_img = np.full((180, 320, 3), 18, dtype=np.uint8)
    cv2.putText(dummy_img, "NO PREVIEW", (90, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.imwrite(placeholder_file, dummy_img)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()


class LiveStreamState:
    def __init__(self):
        self.current_video_path: str = ""
        self.is_streaming: bool = False
        self.pipeline: Optional[OnnxSpeedALPRPipeline] = None
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.current_frame: int = 0
        self.should_stop: bool = False
        self.rotation_angle: int = 0  # 0, 90, 180, 270

stream_state = LiveStreamState()


def get_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "paths": {"database": os.path.join(DATA_DIR, "violations.db")},
        "thresholds": {"speed_kmh": 60.0},
        "speed": {"smoothing_window": 8}
    }


def get_or_init_pipeline(fps: float = 30.0) -> OnnxSpeedALPRPipeline:
    if stream_state.pipeline is None:
        config = get_config()
        stream_state.pipeline = OnnxSpeedALPRPipeline(config, fps=fps)
    else:
        stream_state.pipeline.set_fps(fps)
    return stream_state.pipeline


def apply_rotation(frame: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


app = FastAPI(title="Speed-Triggered Persian ALPR System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount asset directories
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>Error: static/index.html not found!</h2>")


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


def _save_uploaded_file_with_tqdm(file: UploadFile, dest_path: str):
    file.file.seek(0, 2)
    total_size = file.file.tell()
    file.file.seek(0)

    print(f"\n[INFO] Receiving file: {file.filename} ({total_size / (1024 * 1024):.1f} MB)")

    chunk_size = 1024 * 1024
    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=f"📥 Uploading {file.filename[:18]}") as pbar:
        with open(dest_path, "wb") as buffer:
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
                pbar.update(len(chunk))


def _probe_video_sync(file_path: str, rotation_angle: int = 0):
    ref_frame = extract_reference_frame(file_path, 25)
    if ref_frame is None:
        ref_frame = extract_reference_frame(file_path, 0)
    if ref_frame is not None:
        ref_frame = apply_rotation(ref_frame, rotation_angle)
        cv2.imwrite(REF_FRAME_PATH, ref_frame)

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    return fps, total_frames


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    save_dir = os.path.join(DATA_DIR, "raw_videos")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file.filename)

    await run_in_threadpool(_save_uploaded_file_with_tqdm, file, file_path)
    fps, total_frames = await run_in_threadpool(_probe_video_sync, file_path, stream_state.rotation_angle)

    stream_state.current_video_path = file_path
    stream_state.fps = fps
    stream_state.total_frames = total_frames
    stream_state.current_frame = 0
    stream_state.should_stop = False

    print("[INFO] Initializing pipeline with uploaded video...")
    await run_in_threadpool(get_or_init_pipeline, fps)
    print("🚀 [READY] Video loaded successfully!\n")

    return {
        "status": "success",
        "filename": file.filename,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "reference_frame_url": "/data/calibration/reference_frame.jpg"
    }


@app.post("/api/video/rotate")
async def rotate_video(payload: dict = Body(...)):
    angle = int(payload.get("angle", 0))
    if angle not in (0, 90, 180, 270):
        raise HTTPException(status_code=400, detail="Angle must be 0, 90, 180, or 270")
    
    stream_state.rotation_angle = angle
    print(f"🔄 [ORIENTATION] Angle set to {angle}°")

    if stream_state.current_video_path and os.path.exists(stream_state.current_video_path):
        # Force stop active stream so next play picks up new orientation immediately
        stream_state.should_stop = True
        await asyncio.sleep(0.05)
        stream_state.should_stop = False
        await run_in_threadpool(_probe_video_sync, stream_state.current_video_path, angle)

    return {"status": "success", "rotation_angle": angle}


def _process_stream_frame(pipeline, frame, frame_idx, speed_limit, total_frames, fps, rotation_angle):
    if rotation_angle > 0:
        frame = apply_rotation(frame, rotation_angle)

    h_f, w_f = frame.shape[:2]
    tracks, speed_records, new_violations = pipeline.process_frame(frame, frame_idx)
    speed_map = {rec.track_id: rec.speed_kmh for rec in speed_records}

    # Draw Current Trigger Gate Line
    line_p1, line_p2 = pipeline.get_capture_line_endpoints(w_f, h_f)
    cv2.line(frame, line_p1, line_p2, (255, 0, 127), 3, cv2.LINE_AA)
    cv2.putText(frame, "RADAR GATE", (max(10, line_p1[0]), max(25, line_p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 127), 2, cv2.LINE_AA)

    for trk in tracks:
        x1, y1, x2, y2 = map(int, trk.bbox)
        tid = trk.track_id
        speed = speed_map.get(tid, 0.0)

        is_speeding = speed > speed_limit
        color = (0, 0, 255) if is_speeding else (0, 230, 118)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.circle(frame, (int(trk.ground_point[0]), int(trk.ground_point[1])), 4, (0, 255, 255), -1, cv2.LINE_AA)

        label = f"ID:{tid} | {speed:.1f} km/h"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + text_w, max(22, y1)), color, -1)
        cv2.putText(frame, label, (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    s = frame_idx / fps
    m = int(s // 60)
    sec = s % 60
    time_str = f"{m:02d}:{sec:05.2f}"
    header_text = f"LIMIT: {speed_limit:.0f} km/h | {time_str} | F:{frame_idx}/{total_frames}"
    cv2.rectangle(frame, (10, 10), (min(w_f - 10, 480), 38), (0, 0, 0), -1)
    cv2.putText(frame, header_text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 2, cv2.LINE_AA)

    # Maintain strictly proportional aspect ratio regardless of portrait / landscape
    max_dim = 960
    if max(w_f, h_f) > max_dim:
        scale = max_dim / float(max(w_f, h_f))
        disp_w = int(w_f * scale)
        disp_h = int(h_f * scale)
        stream_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        stream_frame = frame

    _, buffer = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buffer.tobytes(), new_violations

@app.get("/api/stream/video")
async def stream_video():
    if not stream_state.current_video_path or not os.path.exists(stream_state.current_video_path):
        raise HTTPException(status_code=400, detail="No video loaded. Upload a video first.")

    pipeline = get_or_init_pipeline(stream_state.fps)

    async def frame_generator():
        cap = cv2.VideoCapture(stream_state.current_video_path)
        frame_idx = 0
        stream_state.is_streaming = True
        stream_state.should_stop = False

        config = get_config()
        speed_limit = float(config.get("thresholds", {}).get("speed_kmh", 60.0))

        try:
            while cap.isOpened() and not stream_state.should_stop:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                stream_state.current_frame = frame_idx

                frame_bytes, new_violations = await run_in_threadpool(
                    _process_stream_frame,
                    pipeline,
                    frame,
                    frame_idx,
                    speed_limit,
                    stream_state.total_frames,
                    stream_state.fps,
                    stream_state.rotation_angle
                )

                for v in new_violations:
                    payload = {
                        "type": "NEW_VIOLATION",
                        "data": {
                            "id": v.id,
                            "track_id": v.track_id,
                            "speed_kmh": v.speed_kmh,
                            "speed_limit": v.speed_limit,
                            "plate_text": v.plate_text,
                            "ocr_confidence": round(v.ocr_confidence, 2),
                            "video_time": v.video_time,
                            "timestamp": v.timestamp,
                            "vehicle_image_url": f"/{v.vehicle_image_path}",
                            "plate_image_url": f"/{v.plate_image_path}"
                        }
                    }
                    asyncio.create_task(manager.broadcast(payload))

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                await asyncio.sleep(0.005)

        finally:
            cap.release()
            stream_state.is_streaming = False

    response = StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.post("/api/stream/stop")
async def stop_stream():
    stream_state.should_stop = True
    return {"status": "success", "message": "Stream stopped."}


@app.get("/api/calibration/reference-frame")
async def get_reference_frame():
    if os.path.exists(REF_FRAME_PATH):
        return FileResponse(REF_FRAME_PATH, headers={"Cache-Control": "no-cache"})
    return Response(status_code=204)


@app.post("/api/calibration/save")
async def save_calibration(payload: dict = Body(...)):
    pixel_points = payload.get("pixel_points", [])
    custom_gate_line = payload.get("custom_gate_line", None)
    road_width_m = float(payload.get("road_width_m", 3.5))
    road_length_m = float(payload.get("road_length_m", 20.0))

    if len(pixel_points) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 reference points are required.")

    world_points = [
        (0.0, 0.0),
        (road_width_m, 0.0),
        (road_width_m, road_length_m),
        (0.0, road_length_m)
    ]

    try:
        H = compute_homography(pixel_points, world_points)
        save_homography(H, CALIBRATION_PATH)

        pts_data = {
            "calibration_points": pixel_points,
            "custom_gate_line": custom_gate_line
        }
        with open(CALIB_PTS_PATH, "w", encoding="utf-8") as f:
            json.dump(pts_data, f)

        if stream_state.pipeline is not None:
            stream_state.pipeline.set_calibration_data(pixel_points, custom_gate_line)
            stream_state.pipeline.set_homography(H)

        return {
            "status": "success",
            "message": "Road calibration updated.",
            "homography": H.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Homography error: {str(e)}")


@app.get("/api/calibration/current")
async def get_current_calibration():
    H = load_homography(CALIBRATION_PATH)
    pts_data = {}
    if os.path.exists(CALIB_PTS_PATH):
        try:
            with open(CALIB_PTS_PATH, "r", encoding="utf-8") as f:
                pts_data = json.load(f)
        except Exception:
            pass

    return {
        "status": "success",
        "has_custom_calibration": os.path.exists(CALIBRATION_PATH),
        "homography": H.tolist(),
        "calibration_points": pts_data.get("calibration_points", []),
        "custom_gate_line": pts_data.get("custom_gate_line", None)
    }


@app.get("/api/violations")
async def get_violations(limit: int = 100):
    config = get_config()
    db_path = config.get("paths", {}).get("database", os.path.join(DATA_DIR, "violations.db"))
    db = ViolationDB(db_path)
    return {"violations": db.get_recent(limit)}


@app.get("/api/stats")
async def get_stats():
    config = get_config()
    db_path = config.get("paths", {}).get("database", os.path.join(DATA_DIR, "violations.db"))
    db = ViolationDB(db_path)
    records = db.get_all_violations()

    if not records:
        return {"total_violations": 0, "max_speed": 0, "avg_speed": 0}

    speeds = [r.get("speed_kmh", 0) for r in records if r.get("speed_kmh")]
    return {
        "total_violations": len(records),
        "max_speed": round(max(speeds), 1) if speeds else 0,
        "avg_speed": round(sum(speeds) / len(speeds), 1) if speeds else 0
    }


@app.get("/api/config")
async def get_system_config():
    config = get_config()
    return {"thresholds": config.get("thresholds", {})}


@app.post("/api/config/threshold")
async def update_threshold(payload: dict = Body(...)):
    speed_kmh = payload.get("speed_kmh")
    if speed_kmh is None:
        raise HTTPException(status_code=400, detail="Missing speed_kmh")

    config = get_config()
    config.setdefault("thresholds", {})["speed_kmh"] = float(speed_kmh)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    if stream_state.pipeline is not None:
        stream_state.pipeline.speed_threshold = float(speed_kmh)

    return {"status": "success", "speed_kmh": float(speed_kmh)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)