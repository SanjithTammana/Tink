# Tink website (Django)

One command runs everything:

```bash
cd Tink-web
python start.py          # use the Python that has the backend's packages (see below)
```

`start.py` builds the website if needed (needs Node.js once), starts Django at http://127.0.0.1:8000 and opens it.
Opening the **Normal** or **ASL** page starts the engine automatically. Nothing else to run.

## One-time setup for the backend
The engine imports the backend modules from the repository root. Run the root
`setup_mac.sh` or `setup_windows.ps1` once to create `.venv` and install backend and Django
packages, then prepare the models using `docs/SESSION-SETUP.md`. Django finds that `.venv`
automatically; set `TINK_PYTHON` to use another compatible backend environment.
Gemini needs a key in the repository root's ignored `.env` (`GEMINI_API_KEY=...`, see `docs/GEMINI-SETUP.md`). Without one the
Normal page still records and shows the error, and **Try the same recording again** works once a key is added.

## How the pieces connect
```
browser (React)  ->  Django  /api/*  ->  engine/server.py  ->  backend modules
```
- **Normal page = recorded assessment.** The engine opens the camera and microphone (`ClipAssessmentService`,
  `ClipRecorder`, `RecordedAudioCapture`). The page streams the camera with Tink's tracking drawn on. **Record** starts a
  clip (up to 30 s), **Stop** saves it and the engine builds the evidence packet and asks Gemini; the page then shows the
  three possibilities. Choosing one confirms it in the backend (`service.confirm`) and your browser speaks it.
  Camera off releases the camera and feeds black frames, with a warning that accuracy will be lower.
  **How Tink decided** shows the notes, transcript, partner context and limits, and downloads the full trace JSON
  (capture info, evidence packet, Gemini assessment).
- **ASL page** is separate: `asl.py`'s `LetterClassifier` and `LetterStabilizer` on one tracked hand. It returns a ranked
  list of possible letters (no percentage gate); you add any of them by click or keys 1 to 8.
- Sessions run on the engine's main thread (MediaPipe crashes on macOS if started in a background thread). If the engine
  process crashes (e.g. a segmentation fault), the site stays up and shows the engine log. Engine output is also in
  `Tink-web/engine.log`.

## Options
`python start.py --port 8000 --no-browser --no-build --no-audio`. Environment: `TINK_PYTHON` (Python for the engine),
`TINK_BACKEND_DIR` (another backend folder), `TINK_CAMERA` (camera index), `TINK_MIC` (microphone device index),
`TINK_AUDIO=0`, `TINK_DEBUG=1`. `python manage.py test` runs the site's tests.

The supplied browser source is in `../web/frontend`; Django serves its built static assets.
For optional Vite development on :5173, explicitly set `TINK_ALLOWED_ORIGINS=http://localhost:5173`
before starting Django. The default only permits the Django site's own origin.
