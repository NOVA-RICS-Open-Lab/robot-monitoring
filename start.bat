@echo off
title Robot Monitor — localhost:5000
cd /d "%~dp0"
cls
echo ================================================================
echo   ROBOT MONITOR
echo   Monitor:  http://localhost:5000
echo   Setup:    http://localhost:5000/setup
echo ================================================================
echo.
echo Installing / checking dependencies...
pip install -r requirements.txt --quiet
echo.
echo Starting server...
timeout /t 1 /nobreak > nul
start "" http://localhost:5000
echo.
echo  Server running. Close this window to stop it.
echo ----------------------------------------------------------------
python web_monitor.py
pause
