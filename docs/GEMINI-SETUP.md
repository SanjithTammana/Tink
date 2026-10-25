# Gemini setup and boundaries

For the explicitly requested local demonstration, `demo_secrets.py` contains hard-coded
Gemini constants and overrides `.env`/OS settings. That file is Git-ignored and untracked.
The loader accepts string constants only and never imports or executes the file. Delete it
after the demonstration to restore the normal configuration order below. The ignored
`.scratch_session/demo_gemini.ps1` provides a one-command curl check without environment setup.

Put your Gemini Developer API key in this local file:

`C:\Users\sanji\PycharmProjects\tink-alt\.env`

```dotenv
GEMINI_API_KEY=replace_with_your_key
TINK_GEMINI_MODEL=gemini-3.8-flash
```

Create the key in [Google AI Studio](https://aistudio.google.com/api-keys). Paste it directly into your local editor; do not paste it into chat, source files, screenshots, or command arguments. The model line is optional. OS environment variables with these exact names override the file, including an explicitly empty value. Only these two names are loaded; values are literal, with no interpolation or execution. A symlinked or outside-project `.env` is refused.

The integration uses Gemini only when you explicitly enable the runtime's `--gemini` flag. A configured key alone does not start cloud requests. Without Gemini, local layers remain available for individual testing. Never put a real key in `.env.example`. `.env` and secret variants must remain ignored and untracked; Git ignore rules do not remove secrets already committed. If a key was committed or shared, revoke it and create a replacement.

The default is Gemini 3.8 Flash, using the official native [Interactions API](https://ai.google.dev/api/interactions-api), [structured JSON output](https://ai.google.dev/gemini-api/docs/structured-output), and [low thinking level](https://ai.google.dev/gemini-api/docs/thinking). The client sends the key in `x-goog-api-key`, never a URL. It uses a fixed Google HTTPS destination, refuses redirects, bypasses environment proxies, times out after eight seconds, caps request/response size, and does not retry automatically. It exposes only a boolean key-availability check. Errors omit response bodies, credentials, and private conversation text.

Enabling `--gemini` sends the compact evidence packet to Google: recent transcript text with speaker/source flags, confirmed messages, event references, motif/posterior summaries, candidate phrases, spelling, and quality flags. That text-suggestion mode sends no raw camera frames or microphone recordings. The separate `--gemini-asr` flag explicitly enables raw audio upload for speaker-aware transcription: up to eight seconds of 16 kHz mono audio per request, inline in memory, using the documented [audio understanding API](https://ai.google.dev/gemini-api/docs/audio). It does not require enabling text suggestions. Runtime suppresses audio uploads during TTS playback. Labels are uncertain and local to each chunk; they do not identify a person persistently. The client preserves the whole chunk's time window rather than fabricating precise word timing. Local Whisper captions leave speakers unknown.

People in the conversation should understand and agree to the enabled cloud modes. Check the actual account's service terms before using sensitive conversation data. Under Google's [Gemini API terms](https://ai.google.dev/gemini-api/terms), unpaid services can use submitted content and generated responses to improve Google products; standard paid-service treatment differs. Regional provisions and optional data-sharing/contributor arrangements can change that treatment. Billing and privacy terms are separate from a Gemini consumer subscription.

Each request sets `store:false` and sends a bounded local snapshot rather than a server conversation ID. This opts out of stored Interaction history, as documented in the [Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview); it is not a blanket claim about all provider processing or retention obligations.

Suggestions are hypotheses that require user confirmation. The client accepts at most four distinct, short message/clarification options with valid local evidence references. During cold start, message suggestions must exactly match an explicit candidate phrase or the user's spelling; ungrounded motifs never acquire semantic labels from the LLM. A candidate message also needs an owned, clock-valid communicator event. Context alone produces clarification. Fewer than four options are valid when evidence is insufficient. No LLM number is presented as a calibrated posterior, and the client never speaks or executes actions.

Syntactic checks cannot prove that a fluent grounded paraphrase contains no invented meaning. Prompt constraints, bounded candidates at cold start, explicit provenance, stale-result checks in the runtime, and user confirmation address different parts of that risk. Transcript mentions remain mentions rather than inferred facts. Unknown speakers stay unknown; chunk-local Meta labels are not persistent identities. ASR timing is approximate and overlap detection is not supplied by Whisper alone.

Offline verification, with no real API key or cloud upload:

```powershell
& ..\tink\.venv\Scripts\python.exe tools\test_gemini_context.py
```

The suite exercises mocked native requests, output validation, configuration boundaries, conversation revisions/TTS echo exclusion, and structured ASR. It does not certify a live Google account, quota, latency, diarization accuracy, or communication accuracy.
