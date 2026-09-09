import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import yaml
import shutil
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse

from src.database.db import ViolationDB
from src.pipeline.pipeline import SpeedALPRPipeline
from src.speed.calibration import compute_homography, save_homography, load_homography, extract_reference_frame

app = FastAPI(title="Speed-Triggered ALPR System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CALIBRATION_PATH = os.path.join(DATA_DIR, "calibration", "homography.npy")
REF_FRAME_PATH = os.path.join(DATA_DIR, "calibration", "reference_frame.jpg")

os.makedirs(os.path.join(DATA_DIR, "raw_videos"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "violations"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "calibration"), exist_ok=True)

# ----------------------------------------------------
# WEBSOCKET CONNECTION MANAGER
# ----------------------------------------------------
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

# ----------------------------------------------------
# GLOBAL PIPELINE STATE
# ----------------------------------------------------
class LiveStreamState:
    def __init__(self):
        self.current_video_path: str = ""
        self.is_streaming: bool = False
        self.pipeline: Optional[SpeedALPRPipeline] = None
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.current_frame: int = 0
        self.should_stop: bool = False

stream_state = LiveStreamState()

def get_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "paths": {
            "vehicle_detector_weights": "yolo11s.pt",
            "homography": CALIBRATION_PATH,
            "database": os.path.join(DATA_DIR, "violations.db")
        },
        "thresholds": {"speed_kmh": 60, "vehicle_conf": 0.3, "plate_conf": 0.3},
        "speed": {"smoothing_window": 8}
    }

def get_or_init_pipeline(fps: float = 30.0) -> SpeedALPRPipeline:
    if stream_state.pipeline is None:
        config = get_config()
        stream_state.pipeline = SpeedALPRPipeline(config, fps=fps)
    else:
        stream_state.pipeline.set_fps(fps)
    return stream_state.pipeline

# Mount static and data directories
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# ----------------------------------------------------
# REST API ENDPOINTS
# ----------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>ALPR Dashboard Static Asset Loading...</h2>")

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep-alive heartbeat & client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    save_dir = os.path.join(DATA_DIR, "raw_videos")
    file_path = os.path.join(save_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract reference frame for calibration preview
    ref_frame = extract_reference_frame(file_path, 0)
    if ref_frame is not None:
        cv2.imwrite(REF_FRAME_PATH, ref_frame)

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    stream_state.current_video_path = file_path
    stream_state.fps = fps
    stream_state.total_frames = total_frames
    stream_state.current_frame = 0
    stream_state.should_stop = False

    # Initialize/update pipeline with new video FPS
    get_or_init_pipeline(fps)

    return {
        "status": "success",
        "filename": file.filename,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "reference_frame_url": "/data/calibration/reference_frame.jpg"
    }

@app.get("/api/stream/video")
async def stream_video():
    """Streams the live annotated video with bounding boxes, speeds, and timestamps."""
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

                # 1. Process Frame through Pipeline
                tracks, speed_records, new_violations = pipeline.process_frame(frame, frame_idx)
                speed_map = {rec.track_id: rec.speed_kmh for rec in speed_records}

                # 2. Broadcast new violations immediately via WebSocket
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

                # 3. Draw On-Screen Overlays (Boxes, Speeds, Timestamp)
                for trk in tracks:
                    x1, y1, x2, y2 = map(int, trk.bbox)
                    tid = trk.track_id
                    speed = speed_map.get(tid, 0.0)

                    is_speeding = speed > speed_limit
                    color = (0, 0, 255) if is_speeding else (0, 230, 118)  # Red if speeding, Green if normal

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{tid} | {speed:.1f} km/h"
                    
                    # Background text banner
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, y1 - 22), (x1 + text_w, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # 4. Draw Header Status Banner
                video_time_str = format_time_simple(frame_idx, stream_state.fps)
                header_text = f"SPEED LIMIT: {speed_limit:.0f} km/h | TIME: {video_time_str} | FRAME: {frame_idx}/{stream_state.total_frames}"
                cv2.rectangle(frame, (10, 10), (550, 42), (0, 0, 0), -1)
                cv2.putText(frame, header_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 255), 2)

                # 5. Encode JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Yield control to event loop for 30 FPS timing
                await asyncio.sleep(0.001)

        finally:
            cap.release()
            stream_state.is_streaming = False

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


def format_time_simple(frame_idx: int, fps: float) -> str:
    fps = fps if fps > 0 else 30.0
    s = frame_idx / fps
    m = int(s // 60)
    sec = s % 60
    return f"{m:02d}:{sec:05.2f}"


@app.post("/api/stream/stop")
async def stop_stream():
    stream_state.should_stop = True
    return {"status": "success", "message": "Stream stop requested."}


# ----------------------------------------------------
# CALIBRATION ENDPOINTS
# ----------------------------------------------------

@app.get("/api/calibration/reference-frame")
async def get_reference_frame():
    if os.path.exists(REF_FRAME_PATH):
        return FileResponse(REF_FRAME_PATH)
    raise HTTPException(status_code=404, detail="No reference frame available. Upload a video first.")


@app.post("/api/calibration/save")
async def save_calibration(payload: dict = Body(...)):
    """
    Accepts 4 pixel points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    and real-world dimensions (road_width_m, road_length_m).
    Computes homography and updates active pipeline in real time.
    """
    pixel_points = payload.get("pixel_points", [])
    road_width_m = float(payload.get("road_width_m", 3.5))
    road_length_m = float(payload.get("road_length_m", 20.0))

    if len(pixel_points) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 reference points (Top-Left, Top-Right, Bottom-Right, Bottom-Left) are required.")

    world_points = [
        (0.0, 0.0),
        (road_width_m, 0.0),
        (road_width_m, road_length_m),
        (0.0, road_length_m)
    ]

    try:
        H = compute_homography(pixel_points, world_points)
        save_homography(H, CALIBRATION_PATH)

        # Update active pipeline instance immediately
        if stream_state.pipeline is not None:
            stream_state.pipeline.set_homography(H)

        return {
            "status": "success",
            "message": "Road calibration updated and active.",
            "homography": H.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to calculate calibration: {str(e)}")


@app.get("/api/calibration/current")
async def get_current_calibration():
    H = load_homography(CALIBRATION_PATH)
    return {
        "status": "success",
        "has_custom_calibration": os.path.exists(CALIBRATION_PATH),
        "homography": H.tolist()
    }


# ----------------------------------------------------
# STATS & VIOLATIONS ENDPOINTS
# ----------------------------------------------------

@app.get("/api/violations")
async def get_violations(limit: int = 50):
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
        raise HTTPException(status_code=400, detail="Missing speed_kmh value")

    config = get_config()
    config.setdefault("thresholds", {})["speed_kmh"] = float(speed_kmh)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    if stream_state.pipeline is not None:
        stream_state.pipeline.speed_threshold = float(speed_kmh)

    return {"status": "success", "speed_kmh": float(speed_kmh)}