#!/usr/bin/env bash
# Robot Monitor — Linux/macOS launcher
set -e
cd "$(dirname "$0")"

echo "================================================================"
echo "  ROBOT MONITOR"
echo "  Monitor:  http://localhost:5000"
echo "  Setup:    http://localhost:5000/setup"
echo "================================================================"
echo
echo "Installing / checking dependencies..."
pip install -r requirements.txt --quiet

echo
echo "Starting server (Ctrl+C to stop)..."
python3 web_monitor.py
