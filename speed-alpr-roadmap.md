# Speed-Triggered ALPR System — Full Architecture & Roadmap

## 0. High-level idea

```
Video → Vehicle Detection → Tracking → Speed Estimation
                                              │
                                    (speed > threshold?)
                                              │
                                      Crop vehicle image
                                              │
                                       Plate Detection
                                              │
                                        Crop plate
                                              │
                                             OCR
                                              │
                                        SQLite DB write
```

Every stage is its own module with a clean input/output contract, so you can swap, retrain, or debug one piece without touching the others.

---

## 1. Project structure

```
speed-alpr/
├── configs/
│   └── config.yaml                 # all paths, thresholds, model configs in one place
│
├── data/
│   ├── raw_videos/                 # input test footage
│   ├── calibration/                # calibration points/homography per camera setup
│   └── training/                   # downloaded datasets (IR-LPR, Iranis) for fine-tuning
│
├── models/
│   ├── vehicle_detector/           # pretrained YOLO weights (car/truck)
│   ├── plate_detector/             # fine-tuned YOLO weights (plate class)
│   └── ocr/                        # trained CRNN weights + char vocabulary file
│
├── src/
│   ├── detection/
│   │   ├── vehicle_detector.py     # wraps YOLO for car detection
│   │   └── plate_detector.py       # wraps fine-tuned YOLO for plate detection
│   │
│   ├── tracking/
│   │   └── tracker.py              # ByteTrack wrapper, assigns track_id
│   │
│   ├── speed/
│   │   ├── calibration.py          # builds/loads homography matrix
│   │   └── speed_estimator.py      # per-track speed history + smoothing
│   │
│   ├── ocr/
│   │   ├── model.py                # CRNN architecture definition
│   │   ├── infer.py                # loads ONNX model, runs inference
│   │   └── decode.py               # CTC decode + char vocabulary mapping
│   │
│   ├── database/
│   │   ├── db.py                   # SQLite connection + insert/query helpers
│   │   └── schema.sql              # table definitions
│   │
│   ├── pipeline/
│   │   └── pipeline.py             # orchestrates all modules per frame
│   │
│   └── utils/
│       ├── video_io.py             # frame reading, FPS handling
│       ├── image_utils.py          # cropping, resizing, drawing overlays
│       └── data_types.py           # shared dataclasses (Detection, Track, PlateResult...)
│
├── training/
│   ├── prepare_plate_dataset.py    # converts IR-LPR annotations → YOLO format
│   ├── train_plate_detector.py     # fine-tunes YOLO on plates
│   ├── prepare_ocr_dataset.py      # converts Iranis/IR-LPR crops → CRNN training format
│   └── train_ocr.py                # trains CRNN with CTC loss
│
├── app/
│   └── streamlit_app.py            # frontend: upload video, run pipeline, browse results
│
├── main.py                         # CLI entrypoint: python main.py --video path.mp4
├── requirements.txt
└── README.md
```

**Why this layout:** each top-level folder under `src/` is a self-contained module with one job. `pipeline.py` is the only file that imports from all of them — every other module only knows about `utils/data_types.py`. That means you can rip out ByteTrack for a different tracker, or swap the CRNN for something else, by rewriting one file and never touching the rest.

---

## 2. Shared data contracts (`src/utils/data_types.py`)

Define these once, use everywhere — this is what keeps modules decoupled:

```python
from dataclasses import dataclass

@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int

@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    ground_point: tuple[float, float]   # bottom-center of bbox, pixel coords

@dataclass
class SpeedRecord:
    track_id: int
    speed_kmh: float
    frame_idx: int

@dataclass
class PlateResult:
    plate_text: str
    confidence: float
    plate_crop_path: str
```

Every module takes/returns these, never raw tuples/dicts. Makes testing each stage in isolation trivial.

---

## 3. Module-by-module breakdown

### 3.1 Vehicle Detection — `src/detection/vehicle_detector.py`

- **Model:** YOLOv8n or YOLO11n, pretrained on COCO (car, truck, bus classes already included).
- **Fine-tuning needed:** none to start. Only fine-tune later if your specific camera angle/footage causes missed detections.
- **Interface:**
```python
class VehicleDetector:
    def __init__(self, weights_path, conf_threshold=0.4): ...
    def detect(self, frame) -> list[Detection]: ...
```

### 3.2 Tracking — `src/tracking/tracker.py`

- **Model:** ByteTrack (no training needed, motion-based).
- **Interface:**
```python
class VehicleTracker:
    def __init__(self): ...
    def update(self, detections: list[Detection], frame) -> list[Track]: ...
```
Internally this is a thin wrapper around Ultralytics' built-in tracker (`model.track(persist=True, tracker="bytetrack.yaml")`), reshaped to return your own `Track` dataclass so the rest of the code doesn't depend on Ultralytics' internal format.

### 3.3 Speed Calibration — `src/speed/calibration.py`

- **One-time setup per camera position**, not a trained model — a geometry step.
- You mark 4+ reference points in a sample frame with known real-world distances (measure physically, or use lane-width standards), then compute a homography matrix.
- Save the matrix to `data/calibration/homography.npy` so the pipeline just loads it at runtime.
```python
def compute_homography(pixel_points, real_world_points) -> np.ndarray: ...
def load_homography(path) -> np.ndarray: ...
def pixel_to_world(point, H) -> tuple[float, float]: ...
```

### 3.4 Speed Estimation — `src/speed/speed_estimator.py`

- Maintains a rolling history of each `track_id`'s ground point over the last N frames.
- Converts to real-world coords via the homography, computes distance/time, smooths with a moving average.
```python
class SpeedEstimator:
    def __init__(self, homography, fps, window_size=8): ...
    def update(self, tracks: list[Track], frame_idx: int) -> list[SpeedRecord]: ...
```

### 3.5 Plate Detection — `src/detection/plate_detector.py`

- **Model:** YOLOv8n/YOLO11n, single class (`plate`), fine-tuned.
- **Fine-tuning needed:** yes — this is your first real training job.
  - Dataset: **IR-LPR** (mut-deep) — ~21k car images with plate bounding-box annotations.
  - `training/prepare_plate_dataset.py` converts IR-LPR's annotation format into YOLO `.txt` label format (`class x_center y_center width height`, normalized).
  - `training/train_plate_detector.py` runs `YOLO("yolo11n.pt").train(data=..., epochs=..., imgsz=640)`.
  - Runs **only on the cropped vehicle image**, not the full frame — smaller search space, higher accuracy, cheaper compute.
- **Interface:**
```python
class PlateDetector:
    def __init__(self, weights_path): ...
    def detect(self, vehicle_crop) -> Detection | None: ...
```

### 3.6 OCR — `src/ocr/`

- **Model:** small CRNN (CNN backbone + BiLSTM + CTC loss) — reads the whole plate as a character sequence, no per-character segmentation needed.
- **Fine-tuning needed:** yes, trained from scratch (or a light pretrained backbone) since this is domain-specific.
  - Datasets: **Iranis** (character-level) and **IR-LPR** (plate-level crops with ground-truth text).
  - `training/prepare_ocr_dataset.py` builds (plate_crop_image, ground_truth_text) pairs and a character vocabulary file (Persian digits/letters + Latin if needed).
  - `training/train_ocr.py` trains with CTC loss, exports to ONNX at the end (`torch.onnx.export`).
- **Interface:**
```python
class PlateOCR:
    def __init__(self, onnx_path, vocab_path): ...
    def read(self, plate_crop) -> PlateResult: ...
```
- `decode.py` handles CTC greedy/beam decoding and maps model output indices back to Persian/Latin characters.

### 3.7 Database — `src/database/`

`schema.sql`:
```sql
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,
    plate_text TEXT,
    ocr_confidence REAL,
    speed_kmh REAL,
    timestamp TEXT,
    vehicle_image_path TEXT,
    plate_image_path TEXT
);
```

`db.py`:
```python
class ViolationDB:
    def __init__(self, db_path): ...
    def insert(self, record: dict): ...
    def get_all(self) -> list[dict]: ...
    def get_recent(self, limit=50) -> list[dict]: ...
```

### 3.8 Pipeline orchestrator — `src/pipeline/pipeline.py`

This is the only file that ties everything together:

```python
class SpeedALPRPipeline:
    def __init__(self, config):
        self.vehicle_detector = VehicleDetector(...)
        self.tracker = VehicleTracker()
        self.speed_estimator = SpeedEstimator(...)
        self.plate_detector = PlateDetector(...)
        self.ocr = PlateOCR(...)
        self.db = ViolationDB(...)
        self.flagged_track_ids = set()   # avoid duplicate DB writes per track

    def process_frame(self, frame, frame_idx):
        detections = self.vehicle_detector.detect(frame)
        tracks = self.tracker.update(detections, frame)
        speed_records = self.speed_estimator.update(tracks, frame_idx)

        for record in speed_records:
            if record.speed_kmh > self.threshold and record.track_id not in self.flagged_track_ids:
                self.flagged_track_ids.add(record.track_id)
                vehicle_crop = crop(frame, track.bbox)
                plate_det = self.plate_detector.detect(vehicle_crop)
                if plate_det:
                    plate_crop = crop(vehicle_crop, plate_det.bbox)
                    result = self.ocr.read(plate_crop)
                    self.db.insert({...})
        return tracks, speed_records
```

### 3.9 Config — `configs/config.yaml`

```yaml
paths:
  vehicle_detector_weights: models/vehicle_detector/yolo11n.pt
  plate_detector_weights: models/plate_detector/best.pt
  ocr_model: models/ocr/crnn.onnx
  ocr_vocab: models/ocr/vocab.json
  homography: data/calibration/homography.npy
  database: data/violations.db

thresholds:
  speed_kmh: 60
  vehicle_conf: 0.4
  plate_conf: 0.5

speed:
  smoothing_window: 8
```

Everything reads from this file — no hardcoded paths anywhere in `src/`.

---

## 4. Frontend — Custom Web Interface

A modern custom web dashboard built with FastAPI backend and a custom responsive HTML/CSS/JS frontend (dark mode, glassmorphism UI, keyframe animations, live metrics grid, drag-and-drop video upload, and interactive violation log table).

`app/`:
- `server.py`: FastAPI server serving static UI assets & REST endpoints (`/api/upload`, `/api/violations`, `/api/stats`).
- `static/index.html`: Dashboard structure with control cards, live status, metric counters, and search filter.
- `static/css/style.css`: Custom design system with glassmorphism, responsive grid layout, and dark mode theme.
- `static/js/app.js`: Interactive client logic for file dropzone, REST API integration, modal plate viewer, and real-time updates.


---

## 5. Roadmap (step-by-step, ~12–14 weeks)

**Week 1 — Scaffolding**
- Create the folder structure above.
- Write `data_types.py`, `config.yaml`, empty module stubs.
- Get a test video, install dependencies (`ultralytics`, `opencv-python`, `onnxruntime-gpu`, `streamlit`).

**Week 2 — Vehicle detection + tracking**
- Implement `vehicle_detector.py`, `tracker.py`.
- Write a small script to run both on your test video and visualize boxes + track IDs.

**Week 3 — Calibration**
- Implement `calibration.py`.
- Manually mark reference points on a frame of your actual camera footage, compute and save the homography.

**Week 4 — Speed estimation**
- Implement `speed_estimator.py`.
- Validate against a known-speed test pass if you can (film your own car at a known GPS speed).

**Weeks 5–6 — Plate detector fine-tuning**
- Download IR-LPR, write `prepare_plate_dataset.py`.
- Fine-tune YOLO nano on plates, evaluate mAP, export weights.
- Implement `plate_detector.py` using the trained weights.

**Weeks 7–9 — OCR**
- Download Iranis + IR-LPR character data, write `prepare_ocr_dataset.py`.
- Build and train the CRNN (`model.py`, `train_ocr.py`).
- Export to ONNX, implement `infer.py` + `decode.py`.

**Week 10 — Database + pipeline integration**
- Implement `db.py`, `schema.sql`.
- Write `pipeline.py` tying every module together.
- Run end-to-end on test video, confirm rows land in SQLite correctly.

**Week 11 — Frontend**
- Build `streamlit_app.py`: upload, run, preview, results table.

**Week 12 — Optimization**
- Export vehicle + plate YOLO models to ONNX, swap into the detectors.
- Tune frame-skip interval, confirm smoothing window still gives stable speeds.
- Benchmark end-to-end FPS.

**Weeks 13–14 — Evaluation + report**
- Metrics: detection mAP, tracking ID-switch rate, speed error %, OCR character/full-plate accuracy, end-to-end FPS.
- Write up results, prep demo.

---

## 6. `main.py` — the "one command to run it all" entrypoint

```python
import argparse
from src.pipeline.pipeline import SpeedALPRPipeline
from src.utils.video_io import read_frames
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    pipeline = SpeedALPRPipeline(config)

    for frame_idx, frame in enumerate(read_frames(args.video)):
        pipeline.process_frame(frame, frame_idx)

    print("Done. Results in", config["paths"]["database"])

if __name__ == "__main__":
    main()
```

Run with: `python main.py --video data/raw_videos/street.mp4`
Or launch the UI with: `streamlit run app/streamlit_app.py`
