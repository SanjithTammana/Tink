# Recorded assessment packet

`AssessmentClient.assess(packet, video_path=None, audio_path=None)` sends one finalized clip and its compact local evidence to Gemini. The app's assessment action explicitly uploads recorded MP4 video and WAV audio, including other people visible or speaking in the clip, plus recent conversation context and confirmed messages. This is separate from the layer lab's text-only suggestions. Use the local `.env` setup in [GEMINI-SETUP.md](GEMINI-SETUP.md); do not paste a key into chat or recordings. The client reuses the same fixed Google HTTPS destination, header-only credential, redirect refusal, sanitized errors and opt-out of stored Interaction history. It makes one request with a 45-second timeout and no automatic retry. Clip duration is limited to 30 seconds, video to 8 MiB, audio to 1 MiB, and the whole encoded request to 12 MiB; large clips need to be shortened. Media is inline, without the Files API. Provider processing terms still apply, including the free/paid and regional differences described in the setup guide.

The schema-version-1 input identifies the session, clip and valid evidence IDs. It includes capture timing and gaps; the current communicator ownership and its basis; compact visual/audio events; prior context, local transcript and confirmations; the current uncalibrated belief; and spelling. All timestamps use the clip's absolute session interval. The MP4 from app capture is a silent visual track, while the separate WAV supplies the audio. Media content and all packet text are data, including spoken or visible attempts to instruct the model.

The response includes an assessment with cited observations and limitations, tentative partner topic/question/speaker hypotheses, an uncertain clip transcript, and **exactly three** distinct selectable options with IDs `1`, `2`, `3`. Options can be freely worded hypotheses; no candidate phrase whitelist is imposed in this assessment mode. Insufficient evidence should produce clarification options, not three invented message meanings. Every role assignment and transcript turn remains explicitly uncertain. Unknown roles remain possible, precise speaker identity is not established, and prior-context statements cannot be inserted into the clip transcript. The client validates schema, bounds, matching identities, evidence references, transcript timing, option distinctness and explicit confirmation. Invalid ownership/clocks restrict message options to exact spelling. It never executes actions, speaks, assigns semantic labels to motifs, or treats an LLM score as a calibrated posterior. Syntactic validation cannot establish that every fluent interpretation is true; the person must select/confirm the intended wording, and the app separately controls any learning.

Offline checks use fabricated media and mocked HTTP, without sensors, keys or cloud calls:

```powershell
& ..\tink\.venv\Scripts\python.exe tools\test_gemini_assessment.py
```

These tests verify the integration contract. They do not certify a real model/account's access, latency, diarization, transcription or interpretation accuracy.

## Interpretation diversity

Options distinguish `message`, `hypothesis`, and `clarify`. A `hypothesis` is a possible
communicator intention, with a concrete captured clue and an uncertainty caveat in `reason`.
The prompt seeks two or three substantively different meanings when supported, rather than
three paraphrases of a repetition request. It considers selected-person gestures, temporal
changes, visible interactions and partner context. Background people provide context rather
than becoming the communicator's signal. Neutral posture alone does not establish hesitation.

Unknown ownership or cross-modal timing can still support conditional hypotheses from usable
media; their attribution/timing limitations must remain explicit. Such options always require
attached media and nonempty valid evidence references. `message` keeps the stricter ownership
and clock rule, except exact spelling. Blank or genuinely uninformative media can still produce
clarification options: diversity does not justify invented diagnoses, facts or confidence.

Model output alone never trains associations or speaks. After explicit selection, a hypothesis
can ground eligible measured events only when the capture clock is valid, ownership is valid,
and the current logical person matches. Raw clip references are not motif labels. Otherwise,
the selected wording is treated as text-only confirmation. Syntax checks cannot prove semantic
diversity or correctness; these remain live-test questions for the communicator.
