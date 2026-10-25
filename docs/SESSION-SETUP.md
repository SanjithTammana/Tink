# Session setup from a fresh clone

Use Python 3.11 and run commands from this repository's root. Install with
`setup_windows.ps1` on Windows or `setup_mac.sh` on macOS. The examples below use this
clone's Windows `.venv`; on macOS substitute `.venv/bin/python`. No sibling Tink checkout
is required. Existing local development guides may show the author's shared environment.
The root setup scripts also install `Tink-web/requirements.txt` for Django. Node.js is needed
once to build the supplied static frontend; Django remains the application server.

## Prepare the model assets once

The session modes deliberately require local weights. Weights, caches, personal recordings,
profiles and keys are not part of Git. These preparation commands download public checkpoints
but do not open a camera/microphone or call Gemini. Check the model publishers' licenses before
redistributing their weights.

Export Envision's packaged CNN and LightGBM assets (h5py is included in requirements):

```powershell
& .\.venv\Scripts\python.exe envision_port.py export
```

Download the three MediaPipe task bundles, using the existing atomic download helper:

```powershell
& .\.venv\Scripts\python.exe -c "from video import MODELS, GESTURE_URL, FACE_URL, download_once; download_once(GESTURE_URL, MODELS/'gesture_recognizer.task'); download_once(FACE_URL, MODELS/'face_landmarker.task'); download_once('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task', MODELS/'pose_landmarker_full.task')"
```

See Google's [Pose Landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
for task model requirements. Prepare the person detector in the repository root:

```powershell
& .\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
```

The filename is an official [YOLO11 pose checkpoint](https://docs.ultralytics.com/models/yolo11/).
At this point the mandatory files are:

```text
yolo11n-pose.pt
models/pose_landmarker_full.task
models/face_landmarker.task
models/gesture_recognizer.task
models/envision/cnn_default.npz
models/envision/lightgbm_default.txt
models/envision/lightgbm_default.json
```

For all local audio, fingerspelling and transcription layers, fill the standard Hugging Face
cache with the complete public model snapshots:

```powershell
& .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('alkiskoudounas/voc2vec-hubert-ls-pt'); snapshot_download('abdollahhh/asl-sign-language-efficientnet-b0'); snapshot_download('Systran/faster-whisper-base.en')"
```

Without cached ASL/Whisper weights those optional features report unavailable. Recorded assessment
can continue without local voc2vec embeddings; the default integrated learning lab requires them.
The runtime itself does not silently download missing checkpoints.

## Configure and run

Copy `.env.example` to ignored `.env`, then edit its empty `GEMINI_API_KEY` locally. No real key
or hard-coded demo credential is shipped. An existing local `demo_secrets.py` deliberately
overrides normal settings; it is also ignored. See [Gemini setup](GEMINI-SETUP.md).

```powershell
Copy-Item .env.example .env
& .\.venv\Scripts\python.exe tools\audio_layer_lab.py devices
& .\.venv\Scripts\python.exe tink_session.py --assessment
```

For the complete Django application, run `.\run.ps1` on Windows or `bash run.sh` on macOS
instead of the standalone OpenCV assessment command. The Django launcher builds missing
static assets from the included lockfile and opens the local site.

Choose `--device N` only if the default microphone is wrong. Device numbers are machine-specific.
`SPACE` records, `ENTER` stops/uploads the completed clip, `1-3` confirms/speaks, and `Q` quits.
Use `Y` to establish an ambiguous communicator and `N` to correct the selected person. ASL is
the previous alphabet/fingerspelling component, not full language translation. See
[recorded assessment](RECORDED-ASSESSMENT.md) for inference and source-attribution limits.

Offline checks open no sensors and make no real provider calls:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_*.py'
```

Fresh-clone source checks use the available Python environment; that does not certify a new
dependency installation or live hardware/cloud behavior on another machine.
