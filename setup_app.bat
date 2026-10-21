@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r app\requirements.txt
if errorlevel 1 exit /b 1
echo App dependencies installed. Run start_app.bat.
echo Noise processing requires target\release\pdn_engine.dll from a release build.
pause
