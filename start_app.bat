@echo off
cd /d "%~dp0"
if exist "dist\Personal Denoiser\Personal Denoiser.exe" (
    start "" "dist\Personal Denoiser\Personal Denoiser.exe"
    exit /b
)
if not exist ".venv\Scripts\pythonw.exe" (
    echo Run setup_app.bat first.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0run_app.py"
