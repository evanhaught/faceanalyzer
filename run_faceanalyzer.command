#!/bin/bash
# Double-click this file in Finder to launch the emotion overlay app.

cd "$(dirname "$0")"
source .venv/bin/activate
python faceanalyzer.py

echo ""
echo "Press any key to close this window..."
read -n 1
