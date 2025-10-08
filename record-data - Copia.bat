@echo off
REM === Change this to your Python path if needed ===
set PYTHON_EXE=python

REM === Run the predict.py in background ===
start "" %PYTHON_EXE% "%~dp0predict.py"

echo Waiting for Dash server to start on port 8050...
:wait_loop
timeout /t 2 >nul

REM === Check if port 8050 is open (server ready) ===
netstat -ano | findstr :8050 >nul
if errorlevel 1 (
    goto wait_loop
)

echo Server detected! Opening Chrome...
start "" "chrome" "http://127.0.0.1:8050/"

exit
