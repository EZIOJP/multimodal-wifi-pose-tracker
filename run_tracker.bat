@echo off
title Multimodal Activity & 3D Spatial Digital Twin Launcher
color 0B
cd /d "%~dp0"

echo ======================================================================
echo    MULTIMODAL HUMAN ACTIVITY, POSTURE ^& 3D DIGITAL TWIN SYSTEM
echo    Webcam Vision + RuView WiFi CSI + 3D Virtual Room Environment
echo ======================================================================
echo.

:: 1. Verify Python
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo [ERROR] Python is not found in PATH!
    echo Please ensure Python 3.10+ is installed.
    pause
    exit /b 1
)

:: 2. Ensure AI Models are ready
echo [1/2] Verifying AI Detection Models...
python activity_tracker\download_models.py
if errorlevel 1 (
    echo [Warning] Failed to verify models. Proceeding with caution...
)

echo.
echo [2/2] System Ready! Choose launch mode:
echo.
echo   [1] RUN EVERYTHING TOGETHER (Desktop 3D Split-Screen + WebGL 3D Browser)
echo   [2] OPEN GRAPHICAL CONTROL CENTER (GUI Window with Run Buttons)
echo   [3] Desktop 3D Split-Screen App Only
echo   [4] Run 3-Step Calibration Wizard (Sit / Stand / Gaze)
echo   [5] WebGL 3D Browser Server Only (http://localhost:8000)
echo   [Q] Quit
echo.
echo Automatic launch in 5 seconds (Default: 1 - Run Everything)...

choice /c 12345Q /n /t 5 /d 1 /m "Select option [1-5, Q]: "

if errorlevel 6 goto EXIT
if errorlevel 5 goto RUN_WEB
if errorlevel 4 goto RUN_CAL
if errorlevel 3 goto RUN_DESKTOP
if errorlevel 2 goto RUN_GUI
if errorlevel 1 goto RUN_ALL

:RUN_ALL
cls
echo ======================================================================
echo  Starting WebGL 3D Server and Desktop 3D Spatial Tracker...
echo ======================================================================
start "RuView 3D Web Server" /min python activity_tracker\server_3d.py
timeout /t 2 /nobreak >nul
start http://localhost:8000
python activity_tracker\app.py
goto EXIT

:RUN_GUI
cls
echo Starting Graphical Control Center...
python activity_tracker\launcher_gui.py
goto EXIT

:RUN_DESKTOP
cls
echo Starting Desktop 3D Split-Screen Tracker...
python activity_tracker\app.py
goto EXIT

:RUN_CAL
cls
echo Launching 3-Step Calibration Wizard...
python activity_tracker\calibrator.py
pause
goto EXIT

:RUN_WEB
cls
echo Starting WebGL 3D Server on http://localhost:8000...
start http://localhost:8000
python activity_tracker\server_3d.py
goto EXIT

:EXIT
echo.
echo System stopped.
exit /b 0
