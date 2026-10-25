# Tink

Home page plus two working pages, **Normal** (camera + voice → suggested phrases, with an evidence trace) and **ASL** (fingerspelling → text).

## Run it
```bash
cd Tink-web
../Tink-backend-v1/.venv/bin/python start.py
```
This builds this React site if needed, starts the Django site at http://127.0.0.1:8000, and starts the AI engine by itself when you open the Normal or ASL page. See `../Tink-web/README.md`. Allow camera and microphone for your terminal (macOS: System Settings → Privacy & Security). The first run downloads models and can take a few minutes.

For front-end development: `cd frontend && npm install && npm run dev` (proxies `/api` to Django on :8000).
See [`frontend/INTEGRATION.md`](frontend/INTEGRATION.md).
