import sys
import os

# Add project root (G:\AILPR) to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import yaml
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from src.database.db import ViolationDB
from src.pipeline.pipeline import SpeedALPRPipeline

app = FastAPI(title="Speed-Triggered ALPR Web Application")

# 1. ENABLE CORS (Fixes "Failed to fetch" browser errors)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")

# Global status dictionary
pipeline_status = {
    "status": "idle",       # "idle", "processing", "completed", "error"
    "progress": 0,          # 0 to 100 %
    "current_frame": 0,
    "total_frames": 0,
    "current_video": "",
    "error": ""
}

def get_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "paths": {
            "database": os.path.join(PROJECT_ROOT, "data", "violations.db"), 
            "homography": os.path.join(PROJECT_ROOT, "data", "calibration", "homography.npy")
        },
        "thresholds": {"speed_kmh": 60, "vehicle_conf": 0.4, "plate_conf": 0.5},
        "speed": {"smoothing_window": 8}
    }

# Mount static web assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount data folder to serve cropped violation images to the web browser
data_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "data"))
os.makedirs(data_dir, exist_ok=True)
app.mount("/data", StaticFiles(directory=data_dir), name="data")

# ----------------------------------------------------
# BACKGROUND ALPR PIPELINE WORKER
# ----------------------------------------------------
def run_pipeline_background(file_path: str):
    global pipeline_status
    filename = os.path.basename(file_path)
    
    pipeline_status.update({
        "status": "processing",
        "progress": 0,
        "current_frame": 0,
        "total_frames": 0,
        "current_video": filename,
        "error": ""
    })
    
    try:
        config = get_config()
        pipeline = SpeedALPRPipeline(config)
        
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pipeline_status["total_frames"] = total_frames
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            pipeline.process_frame(frame, frame_idx)
            
            if total_frames > 0:
                pipeline_status["current_frame"] = frame_idx
                pipeline_status["progress"] = int((frame_idx / total_frames) * 100)

        cap.release()
        pipeline_status["status"] = "completed"
        pipeline_status["progress"] = 100
        print(f"✅ [APP PIPELINE COMPLETE] Processed '{filename}' successfully!")

    except Exception as e:
        print(f"❌ [APP PIPELINE ERROR] {e}")
        pipeline_status["status"] = "error"
        pipeline_status["error"] = str(e)

# ----------------------------------------------------
# REST API ENDPOINTS
# ----------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>ALPR Web Application</h1><p>Static index.html file missing in app/static/.</p>")

@app.post("/api/upload")
async def upload_and_process(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads video file and starts ALPR processing in background."""
    save_dir = os.path.join(PROJECT_ROOT, "data", "raw_videos")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Start background processing
    background_tasks.add_task(run_pipeline_background, file_path)

    return {
        "status": "success",
        "message": "Video uploaded. Background ALPR processing started!",
        "filename": file.filename,
        "file_path": file_path
    }

@app.get("/api/status")
async def get_processing_status():
    """Returns live processing status and % progress bar."""
    return pipeline_status

@app.get("/api/violations")
async def get_violations(limit: int = 50):
    """Fetches speeding violation records from SQLite DB."""
    config = get_config()
    db_path = config.get("paths", {}).get("database", os.path.join(PROJECT_ROOT, "data", "violations.db"))
    if not os.path.exists(db_path):
        return {"violations": []}

    db = ViolationDB(db_path)
    records = db.get_all_violations()[:limit]
    return {"violations": records}

@app.get("/api/stats")
async def get_stats():
    """Fetches summary statistics for web metrics cards."""
    config = get_config()
    db_path = config.get("paths", {}).get("database", os.path.join(PROJECT_ROOT, "data", "violations.db"))
    if not os.path.exists(db_path):
        return {"total_violations": 0, "max_speed": 0, "avg_speed": 0}

    db = ViolationDB(db_path)
    records = db.get_all_violations()
    if not records:
        return {"total_violations": 0, "max_speed": 0, "avg_speed": 0}

    speeds = [r.get("speed_kmh", 0) for r in records if r.get("speed_kmh")]
    total = len(records)
    max_s = max(speeds) if speeds else 0
    avg_s = sum(speeds) / len(speeds) if speeds else 0

    return {
        "total_violations": total,
        "max_speed": round(max_s, 1),
        "avg_speed": round(avg_s, 1)
    }

@app.get("/api/config")
async def get_system_config():
    config = get_config()
    thresholds = config.get("thresholds", {"speed_kmh": 60, "vehicle_conf": 0.4, "plate_conf": 0.5})
    return {"thresholds": thresholds}

@app.post("/api/config/threshold")
async def update_threshold(payload: dict):
    speed_kmh = payload.get("speed_kmh")
    if speed_kmh is None or not isinstance(speed_kmh, (int, float)):
        raise HTTPException(status_code=400, detail="Invalid speed threshold")

    config = get_config()
    if "thresholds" not in config:
        config["thresholds"] = {}
    config["thresholds"]["speed_kmh"] = float(speed_kmh)

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    return {"status": "success", "speed_kmh": float(speed_kmh)}


if __name__ == "__main__":
    import uvicorn
    print("=== STARTING ALPR WEB APPLICATION ===")
    print("Open http://127.0.0.1:8000 in your browser!")
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)