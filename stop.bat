@echo off
title Stop Robot Monitor
cls
echo Stopping Robot Monitor...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000" 2^>nul') do (
    taskkill /pid %%a /f >nul 2>&1
)
echo Server stopped.
timeout /t 2 /nobreak > nul
