# Speed-Triggered ALPR System

Speed-triggered Automatic License Plate Recognition (ALPR) system for vehicle detection, tracking, speed estimation, plate detection, and OCR.

## Project Structure

- `configs/`: Configuration parameters (YAML)
- `data/`: Raw video footage, camera calibration, and training datasets
- `models/`: Weights for vehicle detection, plate detection, and OCR models
- `src/`: Modular core codebase (detection, tracking, speed, OCR, database, pipeline, utils)
- `training/`: Fine-tuning scripts for plate detection and CRNN OCR
- `app/`: Custom web dashboard (FastAPI backend + Vanilla HTML/CSS/JS interface)
- `main.py`: CLI pipeline entrypoint

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run CLI pipeline:
   ```bash
   python main.py --video data/raw_videos/street.mp4
   ```

3. Launch Custom Web Dashboard:
   ```bash
   python -m app.server
   ```
   Or:
   ```bash
   uvicorn app.server:app --reload
   ```
   Then open `http://127.0.0.1:8000` in your browser.
