# Django integration verification

Abhiram's supplied HackTX ZIP was integrated on `codex/abhiram-web-integration`,
on top of the recorded Gemini backend. Django remains the application server and
serves the supplied browser assets; no framework migration was performed.
Bundled environments, dependencies, credentials, private recordings, caches and
control-plane instructions were excluded from the import.

## Checks on Windows, 2026-10-25

- Backend offline suite: 106 tests run, 105 passed, one Windows symlink test skipped.
- Native engine contracts: 15 passed, including owner checks, stale confirmation
  rejection, queued-command cancellation, resource cleanup and supervisor tracking.
- Django checks: no issues; 10 route/security tests passed.
- Browser code: 21 tests passed; the production static build completed.
- Live Django site: camera preview and microphone opened; a short recording was
  saved and Gemini returned exactly three options. A confirmation and browser
  playback reservation/release also completed.
- ASL: live hand outline and ranked candidates appeared; adding and deleting a
  candidate changed the message. This checks integration, not letter accuracy.
- Windows process smoke test: force-stopping Django also exited the actual native
  engine and its virtual-environment launcher within three seconds.
- The full staged tree passed a credential/private-artifact scan, including
  fault-injection checks against the scanner.

Live tests used existing prepared models and compatible local Python packages.
This does not certify a new installation, other hardware, intent accuracy,
conversation attribution accuracy or production deployment readiness.
Real keys and test recordings stay in ignored local files.

## Run and repeat

Prepare the environment and model assets using [SESSION-SETUP.md](SESSION-SETUP.md),
configure an ignored `.env`, then run `./run.ps1` on Windows or `bash run.sh` on macOS.
Use the system default microphone first. If a different input is necessary, list
devices with `python tools/audio_layer_lab.py devices` and set `TINK_MIC` to an
input device index before launch; output-device indices cannot record.

Offline checks from the repository root:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tools -p 'test_*.py'
& .\.venv\Scripts\python.exe -m unittest discover -s Tink-web/tests -p 'test_*.py'
& .\.venv\Scripts\python.exe Tink-web/manage.py check
& .\.venv\Scripts\python.exe Tink-web/manage.py test tink
Push-Location web/frontend
npm.cmd ci
npm.cmd test
npm.cmd run build
Pop-Location
```
