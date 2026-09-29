@echo off
setlocal enabledelayedexpansion
title Persian Speed ALPR Dashboard

echo ================================================================
echo           PERSIAN SPEED-TRIGGERED ALPR SYSTEM [GPU]
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

:: 4. Verify all critical packages + DirectML GPU are present
python -c "import uvicorn, fastapi, cv2, yaml, onnxruntime as ort; assert 'DmlExecutionProvider' in ort.get_available_providers()" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [INFO] Missing dependencies detected. Installing requirements...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    echo [SUCCESS] Environment successfully configured!
) else (
    echo [INFO] All dependencies verified - DirectML GPU ready.
)

:: 5. Create required directories
if not exist "data\raw_videos" mkdir "data\raw_videos"
if not exist "data\violations" mkdir "data\violations"
if not exist "data\calibration" mkdir "data\calibration"

:: 6. Launch Browser and Server
echo.
echo ================================================================
echo [STARTING] Loading models into GPU and launching dashboard...
echo [SERVER] http://127.0.0.1:8000
echo ================================================================
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000"
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000

pause