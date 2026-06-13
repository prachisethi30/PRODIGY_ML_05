@echo off
title Food Calorie Estimator
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found. Run setup first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo Starting Food Calorie Estimator...
echo Browser will open at http://localhost:8501
echo Close this window to stop the app.
echo.
python main.py app
pause
