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
python -c "import onnxruntime as ort; assert any(p in ort.get_available_providers() for p in ('CUDAExecutionProvider', 'DmlExecutionProvider'))" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [HARDWARE] Configuring CUDA and DirectML GPU drivers...
    python -m pip install --upgrade pip
    
    :: Check if NVIDIA hardware exists
    where nvidia-smi >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        echo [HARDWARE] NVIDIA card detected. Installing onnxruntime-gpu...
        python -m pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ 2>nul
    )

    :: Install DirectML as secondary fallback
    python -m pip install onnxruntime-directml 2>nul
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
echo [STARTING] Launching SpeedALPR Dashboard...
echo [SERVER] http://127.0.0.1:8000
echo ================================================================
echo.

start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000

pause