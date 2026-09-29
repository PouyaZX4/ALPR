@echo off
setlocal enabledelayedexpansion
title Persian Speed ALPR Dashboard

echo ================================================================
echo           PERSIAN SPEED-TRIGGERED ALPR SYSTEM
echo ================================================================

:: 1. Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

:: 2. Create virtual environment if missing
if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Creating Python virtual environment in .venv...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
)

:: 3. Activate Virtual Environment
call .venv\Scripts\activate.bat

:: 4. Verify GPU-enabled ONNX Runtime is installed
python -c "import onnxruntime as ort; assert any('Dml' in p or 'CUDA' in p for p in ort.get_available_providers())" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [HARDWARE] Configuring GPU execution providers...
    python -m pip install --upgrade pip
    :: Remove CPU-only package if present
    python -m pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml >nul 2>nul
    
    :: Install DirectML (Universal GPU support for Windows)
    echo [INFO] Installing onnxruntime-directml for hardware acceleration...
    python -m pip install onnxruntime-directml
    python -m pip install -r requirements.txt
    
    echo GPU_CONFIGURED > ".venv\.gpu_ready"
    echo [SUCCESS] GPU packages installed successfully!
) else (
    echo [INFO] Hardware-accelerated environment verified.
)

:: 5. Create required directories
if not exist "data\raw_videos" mkdir "data\raw_videos"
if not exist "data\violations" mkdir "data\violations"
if not exist "data\calibration" mkdir "data\calibration"

:: 6. Launch Browser and Server
echo.
echo ================================================================
echo [STARTING] Launching SpeedALPR Dashboard on GPU...
echo [SERVER] http://127.0.0.1:8000
echo ================================================================
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000

pause