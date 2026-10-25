#!/usr/bin/env bash
# Start Tink: ./run.sh   (extra options pass through, e.g. ./run.sh --no-audio)
ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$ROOT"
# First run: fetch the model files the backend needs (it never downloads them by itself).
if [ ! -f "$BACKEND/yolo11n-pose.pt" ] || [ ! -f "$BACKEND/models/pose_landmarker_full.task" ] || [ ! -f "$BACKEND/models/envision/cnn_default.npz" ]; then
  echo "Some model files are missing, downloading them now (one time)..."
  bash "$ROOT/prepare_models.sh" || exit 1
fi
cd "$ROOT/Tink-web" || exit 1
if [ -x "$BACKEND/.venv/bin/python" ]; then PY="$BACKEND/.venv/bin/python"; else PY="$(command -v python3 || command -v python)"; fi
exec "$PY" start.py "$@"
