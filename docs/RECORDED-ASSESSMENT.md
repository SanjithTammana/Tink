# Recorded assessment in the current environment

No Tkinter or Django scaffolding is introduced. The current OpenCV entry point wraps a UI-independent Python service, ready to be called by a future Django integration.

```powershell
cd C:\Users\sanji\PycharmProjects\tink-alt
& ..\tink\.venv\Scripts\python.exe tink_session.py --assessment --device 2
```

The test surface opens the camera and microphone for live preview. `SPACE` begins saving a clip; `ENTER` stops and sends that completed recording and its evidence packet to Gemini. Clips stop automatically after 30 seconds. `1`, `2`, `3` explicitly confirm one proposed utterance and speak it locally. `R` retries the same saved clip; `P` opens the JSON packet; `Q` closes the test. `Y` confirms a provisional communicator and `N` corrects the selected person. Establish the intended person before recording when several unfamiliar people are visible. ASL alphabet controls remain `A` append, `B` space, `D` delete and `S` confirm/speak spelling. Speech cannot start from a selection while recording.

For one automated smoke test after a five-second preview:

```powershell
& ..\tink\.venv\Scripts\python.exe tink_session.py --assessment --device 2 --auto-record-seconds 8
```

Device numbers are machine-specific. Device 2 was the built-in microphone array when this integration was tested; the default input was a Bluetooth headset. Omitting `--device` uses the current default.

One clip saves ignored local `video.mp4`, `audio.wav`, per-block timing files, full `features.json`, `capture.json`, `assessment-input.json` and the validated Gemini `assessment.json`. Video is a silent, uniformly sampled 8 fps visual track up to 640 by 480. Audio is 16 kHz mono on the same host monotonic origin, with explicit gaps padded by silence. Each delivered block has its own ADC/host timestamp: a startup overflow cannot silently erase all subsequent raw audio. An affected clip remains clock-uncertain. Camera exposure time and host-read time are not identical, and callback fallback timing has uncertain sensor latency.

Gemini receives the clip's complete recorded visual/audio tracks, compact measured features, local transcript, current uncalibrated belief, prior confirmed messages and bounded recent conversation. Full local measurements remain in `features.json`; transmitted measurements are evenly subsampled to respect a serialized JSON budget. Partner roles, topic/question inference and transcripts remain hypotheses. A room microphone does not identify the communicator; ambiguous identity and clock failures permit clarification options rather than asserted meanings. The result has exactly three alternatives and requires confirmation.

The backend lives in `assessment_service.py`: `ClipAssessmentService.prepare()`, `.observe(frame, at_s)`, `.packet(clip, capture)`, `.assess(clip, capture, packet)` and `.confirm(option, packet, at_s)`. `assessment_capture.py` owns media recording; `gemini_assessment.py` owns the validated cloud request. `assessment_session.py` is only the OpenCV test surface. No framework, package installation or publication is required.

The interpretation update labels speculative options `[hypothesis]` and asks for different
possible meanings, with evidence and caveats in the JSON reasons. Uncertain role/timing no
longer automatically forces three generic clarification requests; usable single-modality
evidence can support conditional hypotheses. Definite `message` eligibility stays stricter.
Generation itself never trains meanings; selection can ground measured events only under
valid ownership, clock and matching-person checks. Restart an already-running OpenCV process
to load an updated prompt. `R` retries within the version currently loaded by that process.

Credentials come from ignored, untracked `.env`; the client reloads them on an explicit assessment. The configured model is `gemini-3.8-flash`, which has a [free-tier offering](https://ai.google.dev/gemini-api/docs/pricing), subject to project eligibility and quotas. Choosing the model does not change the project's billing tier. HTTP 503 indicates temporary service unavailability; it does not establish a need for a paid model. Recordings are preserved after failure. See [packet/security boundaries](ASSESSMENT-PACKET.md) and [key setup](GEMINI-SETUP.md).

Sanjith requested a hard-coded local demo override: ignored `demo_secrets.py` now takes
precedence over `.env` and OS variables. Both the model and key are literal constants there;
remove that file to restore normal configuration. The ignored curl demonstration is:

```powershell
& C:\Users\sanji\PycharmProjects\tink-alt\.scratch_session\demo_gemini.ps1
```

Live verification selected `gemini-3.5-flash-lite` for the demonstration: 3.8 Flash returned
HTTP 503/high demand, while Flash-Lite returned HTTP 200 for both curl and native requests.
The saved 8.18-second corrected recording yielded a validated three-option assessment with
Flash-Lite. Its audio was near-silent and communicator ownership ambiguous, so the options
were clarifications, not claimed intent recognition. The demo constants and `.env` select
Flash-Lite; the general library default remains 3.8 Flash.

Provider-facing JSON schema removes expensive nested length constraints while retaining the
exactly-three requirement. The full response is still checked locally against strict string,
array, identity, evidence, role-uncertainty and timestamp bounds. This avoids the provider's
"too many states" decoder error without relaxing accepted outputs.
