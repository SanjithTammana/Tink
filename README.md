# Tink

Assistive communication from voluntary signals. Video gives an initial distribution over intents, audio
updates it (Bayesian), and the posterior ranks phrases for the user to confirm and speak.

## Setup (Python 3.11)
**macOS** (Apple Silicon or Intel; needs Homebrew):
```bash
bash setup_mac.sh
```
Then allow Camera and Microphone for your terminal or PyCharm in System Settings → Privacy & Security.

**Windows** (clone into a short path such as `C:\code\Tink`):
```powershell
.\setup_windows.ps1
```

Both scripts install `requirements.txt` and then `envisionhgdetector` with `--no-deps`. Tink loads only its
LightGBM model, so TensorFlow is not needed. Model files download on first run.

## Run
```bash
.venv/bin/python run.py --camera 0 --audio          # macOS
.venv\Scripts\python.exe run.py --camera 0 --audio  # Windows
```
Keys: **1–3** confirm an interpretation and speak it, **0** rejects all, **Q** quits. `--video file.mp4`
replaces the camera; `--headless --max-frames N` runs without a window.

## Layout
| File | Role |
|---|---|
| `video.py` | EnvisionHGdetector LightGBM gesture probability, MediaPipe gestures and face, YOLO people |
| `audio_text.py` | voc2vec vocalization classifier, mic pipeline, `BayesianFusion` |
| `transcribe.py` | faster-whisper transcription |
| `retrieval.py` | phrase database, MiniLM embeddings, phrase ranker |
| `tts.py` | offline text-to-speech |
| `run.py` | live loop and panel |
| `config/` | likelihood tables (placeholders) and the phrase database |
| `docs/` | plan and research notes |

Phrase data includes messages from Vertanen & Kristensson (2011), CC BY 4.0. See `data/aac_corpus/README.md`.
