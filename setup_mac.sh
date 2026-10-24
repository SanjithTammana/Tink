#!/usr/bin/env bash
# Tink setup for macOS (Apple Silicon or Intel). Run from the repo root: bash setup_mac.sh
set -euo pipefail

if ! command -v python3.11 >/dev/null; then
  echo "Python 3.11 is required (mediapipe 0.10.7 / envisionhgdetector). Install it with: brew install python@3.11"
  exit 1
fi
if command -v brew >/dev/null; then
  brew list libomp >/dev/null 2>&1 || brew install libomp   # OpenMP runtime LightGBM links against
else
  echo "Homebrew not found: install libomp yourself if LightGBM fails to import."
fi

python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps envisionhgdetector==3.0.8
.venv/bin/python -c "import video, audio_text, retrieval, tts; print('Tink imports OK')"

echo
echo "Done. Grant Camera and Microphone access to your terminal (or PyCharm) in"
echo "System Settings > Privacy & Security, then run:"
echo "  .venv/bin/python run.py --camera 0 --audio --no-yolo"
