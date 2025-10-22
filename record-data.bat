@echo off
REM === Python executable (change if needed) ===
set PYTHON_EXE=py

REM === Step 1: Start upload.py (port 8051) ===
echo Starting upload.py...
start "" %PYTHON_EXE% "%~dp0upload.py"

echo Waiting for Upload server to start on port 8051...
:wait_upload
timeout /t 2 >nul
netstat -ano | findstr :8051 >nul
if errorlevel 1 (
    goto wait_upload
)

echo Upload server detected! Opening Chrome tab for 8051...
start "" "chrome" "http://127.0.0.1:8051/"

REM === Step 2: Start predict.py (port 8050) ===
echo Starting predict.py...
start "" %PYTHON_EXE% "%~dp0predict.py"

echo Waiting for Predict server to start on port 8050...
:wait_predict
timeout /t 2 >nul
netstat -ano | findstr :8050 >nul
if errorlevel 1 (
    goto wait_predict
)

echo Predict server detected! Opening Chrome tab for 8050...
start "" "chrome" "--new-tab" "http://127.0.0.1:8050/"

echo All servers started and browser tabs opened successfully.
exit
