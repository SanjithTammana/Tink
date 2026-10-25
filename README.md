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
Optional conversation model (dialogue corpora are downloaded, not committed: DailyDialog is CC BY-NC-SA 4.0):
```bash
python conversation.py fetch
python conversation.py build
```

Teach Tink your signs (each sign recorded 4 times; personal recordings stay in `data/signs/`, not in git):
```bash
python signs.py enroll --signs YES NO HELP MORE WATER BATHROOM
python signs.py check
```
These are the user's own enrolled keyword signs, not ASL interpretation (see `docs/RESEARCH-asl.md`).

Dialogue-act model (Switchboard, CC BY-NC-SA 3.0, downloaded not committed):
```bash
python acts.py fetch
python acts.py train
python conversation.py soften
python conversation.py fit
```

Keys: **1–3** confirm an interpretation and speak it, **0** rejects all, **N** switches which person is the user, **Q** quits.
Fingerspelling: **A** adds the stable letter, **SPACE** a space, **D** deletes, **C** clears, **S** speaks the spelled text. `--video file.mp4`
replaces the camera; `--headless --max-frames N` runs without a window.

## Layout
| File | Role |
|---|---|
| `video.py` | person tracking (YOLO11n-pose + ByteTrack: boxes, IDs, wrist kinematics), Envision LightGBM gesture probability and MediaPipe gestures/face on the selected user |
| `audio_text.py` | voc2vec vocalization classifier, mic pipeline, `BayesianFusion` |
| `transcribe.py` | faster-whisper transcription |
| `retrieval.py` | phrase database, MiniLM embeddings, phrase ranker |
| `tts.py` | offline text-to-speech |
| `meaning.py` | meaning layer: partner-question context, referents from pointing, mixed-signal clarify, sentence composer |
| `conversation.py` | conversation model: embedded (partner turn -> reply) pairs give P(reply type \| partner) and real replies |
| `asl.py` | ASL fingerspelling (A-Z): EfficientNet-B0 classifier + stability voting; standalone app and the in-pipeline reader |
| `signs.py` | enrolled keyword signs: guided recording, DTW spotting with a no-sign option, calibrated probabilities |
| `acts.py` | Switchboard dialogue acts as probabilities: P(act \| text), act transitions, act -> intent map |
| `simulate.py` | simulator and evaluations (synthetic signals, real held-out dialogues, AAC corpus coverage) |
| `run.py` | live loop and panel |
| `config/` | likelihood tables (placeholders) and the phrase database |
| `docs/` | plan and research notes |

Phrase data includes messages from Vertanen & Kristensson (2011), CC BY 4.0. See `data/aac_corpus/README.md`.
