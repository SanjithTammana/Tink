# Frontend ↔ backend

The site has three screens: Home, **Normal** and **ASL**. It is served by the Django project in `Tink-web/`, which starts and
supervises the engine (`Tink-backend-v1/server.py`) in a separate process and forwards `/api` to it. The pages call it through `src/lib/api.js`.

| Page | Backend code it drives | What the page does |
| --- | --- | --- |
| Normal | `run.Pipeline` (same class `run.py` uses) | Shows the camera stream with detections, a camera on/off switch (warns that accuracy drops when off), suggested phrases to confirm and speak, and an evidence trace (downloadable JSON). |
| ASL | `asl.ASLRecognizer` (same class `asl.py` uses) | Shows the camera stream with the hand outlined, the candidate letter and its stability, and the typed message (add, space, delete, clear, speak). |

Endpoints: `POST /api/session/start|stop`, `GET /api/state`, `GET /api/stream` (MJPEG), `GET /api/trace`,
`POST /api/camera|confirm|reject|switch-user`, `POST /api/asl/add|space|delete|clear`.
The server owns the camera and microphone and only accepts requests from this computer. One page's session runs at
a time because both use the camera. Speech playback uses the browser's own speech synthesis.
