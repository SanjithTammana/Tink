# Optional Muse Voice Transcribe

The existing recorded Django/OpenCV assessment can optionally obtain a separate
speech transcript from Meta before asking Gemini for its three interpretations.
This worktree adds an opt-in adapter; local Whisper remains the default. Nothing
is enabled merely by installing the code, and no Meta key or real audio was used
to verify this change.

## Fit and verified API contract

Meta's [speech-to-text guide](https://dev.meta.ai/docs/speech-to-text) and
[recording API reference](https://dev.meta.ai/docs/api-reference/voice/transcribe)
describe `muse-voice-transcribe-1.0` at
`POST https://api.meta.ai/v1/asr/transcribe`. The request is multipart: JSON
settings plus a mono, 16-bit PCM WAV; authentication is a bearer Model API key.
The `DIARIZATION` response contains speech turns and millisecond boundaries.
Existing Tink recordings are already mono at 16 kHz, an accepted input rate.

This is a good match for the current Record/Stop/Assess flow. The code does not
need a new capture system, package, or server. The existing legacy Meta caption
class was upgraded rather than adding a competing provider implementation.

The adapter uploads at most 30 seconds in one request, asks for buffered JSON,
and validates bounded responses. It refuses redirects and environment proxies,
does not expose provider errors or credentials, and makes no automatic retries.
The official endpoint allows longer recordings, but this integration deliberately
keeps Tink's shorter recording limit.

## Configuration and test

In the root of the checkout you intend to run, edit the ignored `.env` locally:

```dotenv
TINK_TRANSCRIPTION_PROVIDER=meta
MODEL_API_KEY=replace-locally-with-a-meta-model-api-key
```

Keep `GEMINI_API_KEY` and `TINK_GEMINI_MODEL` configured separately for assessment.
The Meta key is not a Gemini key. It stays server-side and is never included in
the browser bundle, evidence packet, logs, or the example file. OS environment
values override `.env`, including an explicitly empty key. Only literal values
are parsed; configuration files are never executed.

Restart the app after changing `TINK_TRANSCRIPTION_PROVIDER`, then record a new
clip. Pressing Stop/Assess sends that clip's audio to Meta in addition to the
existing Gemini assessment upload. Obtain participants' agreement before using
the shared microphone with either cloud service. The existing **How Tink decided**
notes report when Muse ASR is enabled. Set the provider back to `local` and restart
to disable it.

The packet retains `conversation.local_transcript` and adds `cloud_transcript`
plus `cloud_transcription` provenance. Speaker labels remain unverified and
recording-local; they are never mapped automatically to a visible person, partner,
or communicator. Estimated speech-turn times are offset onto the recording's
clock. Turns crossing an audio gap are marked uncertain and excluded from rolling
conversation context. An uncertain capture clock prevents the Muse upload.

Each recorded packet makes at most one Muse attempt. Retrying the Gemini result
reuses the saved transcription status, even after a Meta failure, so it cannot
silently rebill the audio. After adding/correcting a key, make a new recording or
explicitly retry a saved WAV using the standalone command below. Failure or an
absent key leaves the existing local transcript and Gemini raw-audio path usable.
Local packet creation and model preparation never call Meta.

For an explicit saved-file smoke test (this uploads the named file to Meta):

```powershell
& .\.venv\Scripts\python.exe transcribe.py path\to\short-clip.wav --meta
```

Use a recording no longer than 30 seconds. The legacy CLI retains its Whisper
fallback and prints which backend supplied the result. The recorded-service path
uses only its already-prepared local fallback, without new model downloads.

## Cost, data policy and limits

The [official pricing page](https://dev.meta.ai/docs/pricing-rate-limits), checked
2026-10-25, lists $0.18 per processed audio hour, equivalent to $3 per 1,000 minutes.
A 30-second processed clip is approximately $0.0015 for transcription, separate
from Gemini usage. This is a paid optional provider, not a free model setting.

Do not infer zero retention from the endpoint or price. Meta's
[ZDR help page](https://dev.meta.ai/help/policies-and-privacy/zero-data-retention)
says eligibility and enablement are organization-specific, with exceptions and
feature restrictions. No retention guarantee or ZDR request flag is invented by
this adapter. Confirm the relevant account/model policy before using private
conversations. Local transcripts and recordings remain in ignored session files.

Meta also documents a [realtime WebSocket endpoint](https://dev.meta.ai/docs/speech-to-text).
Continuous streaming is deferred: it needs a session lifecycle, cumulative
partial replacement, pacing, reconnect handling and capture-clock/TTS echo guards.
This file adapter does not claim streaming latency, persistent speaker identity,
word-level timing, confidence scores, intent recognition or accuracy on atypical
speech. Those require separate evaluation with consented recordings.

## Offline verification

```powershell
& .\.venv\Scripts\python.exe -m unittest tools.test_meta_transcribe tools.test_gemini_context tools.test_assessment_service
```

Tests use synthetic arrays, temporary settings files and mocked HTTP responses.
They cover multipart encoding, timing/speaker scope, invalid input and responses,
credential/error handling, opt-in behavior, no upload during packet creation,
cached attempts, local fallback and capture-gap/clock handling. A real request
still requires your Meta key and account access; no live accuracy claim is made.

## Local connection check, 2026-10-25

After the user saved a key in the ignored worktree `.env`, one generated-speech
test returned HTTP 402. After the user added billing and requested a retry, the
same 5.42-second, locally generated WAV returned HTTP 200 and one timed transcript
matching the test phrase. The speaker role remained unknown. No microphone,
private recording or key value was logged or sent as test content. This confirms
the file endpoint, credentials and response parsing for synthetic speech; it does
not validate recognition of atypical speech or real conversational attribution.
