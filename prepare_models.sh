#!/usr/bin/env bash
# Downloads the model files Tink needs, once. ./prepare_models.sh   (run.sh does this for you if they are missing)
# Needs internet and the root .venv (create it first with: bash setup_mac.sh)
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
[ -x "$PY" ] || { echo "No .venv yet. Run first: bash setup_mac.sh"; exit 1; }

echo "1/4  Envision gesture models"
"$PY" envision_port.py export

echo "2/4  MediaPipe pose, face and gesture models"
"$PY" -c "from video import MODELS, GESTURE_URL, FACE_URL, download_once; download_once(GESTURE_URL, MODELS/'gesture_recognizer.task'); download_once(FACE_URL, MODELS/'face_landmarker.task'); download_once('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task', MODELS/'pose_landmarker_full.task')"

echo "3/4  YOLO person detector"
"$PY" -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"

echo "4/4  Sound, fingerspelling and speech models (large; cached for later)"
"$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('alkiskoudounas/voc2vec-hubert-ls-pt'); snapshot_download('abdollahhh/asl-sign-language-efficientnet-b0'); snapshot_download('Systran/faster-whisper-base.en')"

echo "Done. Start Tink with ./run.sh"
