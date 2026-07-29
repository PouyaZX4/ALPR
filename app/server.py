import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import yaml

from src.database.db import ViolationDB
from src.pipeline.pipeline import SpeedALPRPipeline

app = FastAPI(title="Speed-Triggered ALPR Dashboard")

CONFIG_PATH = "configs/config.yaml"

def get_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

# Mount static web assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Mount data folder for serving cropped images/videos
data_dir = os.path.abspath("data")
os.makedirs(data_dir, exist_ok=True)
app.mount("/data", StaticFiles(directory=data_dir), name="data")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>ALPR Web Dashboard</h1><p>Index file missing.</p>")

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    save_dir = "data/raw_videos"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"status": "success", "filename": file.filename, "file_path": file_path}

@app.get("/api/violations")
async def get_violations(limit: int = 50):
    config = get_config()
    db_path = config.get("paths", {}).get("database", "data/violations.db")
    if not os.path.exists(db_path):
        return {"violations": []}
    
    db = ViolationDB(db_path)
    records = db.get_recent(limit=limit)
    return {"violations": records}

@app.get("/api/stats")
async def get_stats():
    config = get_config()
    db_path = config.get("paths", {}).get("database", "data/violations.db")
    if not os.path.exists(db_path):
        return {"total_violations": 0, "max_speed": 0, "avg_speed": 0}
    
    db = ViolationDB(db_path)
    records = db.get_all()
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
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)
