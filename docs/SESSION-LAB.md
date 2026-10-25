# Integrated layer test

Run from `C:\Users\sanji\PycharmProjects\tink-alt` with the existing environment:

```powershell
& ..\tink\.venv\Scripts\python.exe tink_session.py --record
```

This opens one live feature window: tracked people, selected communicator's face/pose/hands, raw measurements, audio level/event state, anonymous motifs, ASL alphabet state, transcript context and up to four possibilities. Capture runs independently of inference. The visible landmarks belong to the analyzed frame; the preview can lag capture on this CPU. Recorded raw video has host-read timestamps in `capture_times.jsonl`; MP4 frame rate alone is not the capture clock. Start with a quiet room for the first second of audio calibration and a neutral face for the first 1.5 seconds of visible face measurements.

One visible person initializes automatically under a stated sole-person assumption. With several unfamiliar people the candidate remains provisional: `Y` confirms which person is the communicator, `N` corrects it. The saved local appearance signature supports automatic reacquisition after tracking-ID changes and short exits. Overlapping boxes pause feature evidence without erasing that signature. Clothing colors are a heuristic, not reliable identity across similar clothes, lighting/room changes or long absences. `--profile Alice` uses separate ignored state; use different profiles for different communicators. `N` starts a new logical identity so previous person's grounded meanings cannot transfer silently.

No automatic method can know which unfamiliar person is the communicator solely from a room image. This implementation minimizes repeated selection while preserving an explicit ambiguous state; it does not infer a person's disability or role from appearance.

Try each layer before fusion:

```powershell
# Visual tracking and old ASL reader; no microphone or transcription.
& ..\tink\.venv\Scripts\python.exe tink_session.py --no-audio --no-asr --no-tts
# Audio alone: previous independent waveform/template lab remains available.
& ..\tink\.venv\Scripts\python.exe tools\audio_layer_lab.py live --record
# Full local session, with speech disabled while inspecting results.
& ..\tink\.venv\Scripts\python.exe tink_session.py --record --no-tts
```

The old audio lab's named-template enrollment is still a diagnostic baseline. The new session requires no word/sound enrollment: it discovers recurring anonymous sound/visual patterns, then learns message associations from explicit confirmations. Explore varied sounds and gestures, rest between trials, and include background speech/other people as negative-control conditions. Repeating similar inputs tests repeatability; random inputs alone do not establish that a pattern carries a particular meaning. Existing seed phrases are candidate wording, never labels inferred from a sound. Use `F` to type a new message and confirm it when the candidate pool lacks what you mean.

A shared room microphone stays speaker-unknown. It contributes transcript context and anonymous audio discovery, but cannot train communicator meanings or fuse with the selected person's visual events. `--user-mic` is an explicit operator assertion that a dedicated input carries only the communicator; use `--device N` to choose it. Do not enable that flag for an ordinary shared microphone. It is not automatic diarization, source separation or voice identification. Even under that assertion, visual ownership must cover the whole audio interval; tracking gaps/overlap, clock loss and TTS windows reject pairing. Fusion uses measured interval overlap; no cross-modal semantic alignment is claimed before confirmed examples.

Local Whisper provides timestamped, speaker-unknown six-second chunks. The optional Gemini audio mode provides uncertain speaker labels local to each chunk; those labels do not persistently identify people or establish which speaker is the communicator. Transcript revisions/finalization, mentions and confirmed communicator statements are represented separately. Nothing said by an unknown speaker becomes a confirmed communicator message. Chunk boundaries may split speech; this lab does not yet provide continuous cross-chunk diarization or robust simultaneous-speaker separation.

Controls:

| Key | Action |
|---|---|
| `R` | Request a fresh choice set; existing choices otherwise stay fixed |
| `1`–`4` | Confirm the visible message, learn eligible evidence, then speak it |
| `0` | Reject wording without training a negative intent label |
| `U` | Undo the latest learned association; cannot undo spoken audio |
| `F` | Freeze eligible evidence and type an exact message; Enter confirms/speaks, Esc cancels |
| `A` | Append the stable ASL letter from the selected person's single visible hand |
| Space / `D` / `C` | Add a space / delete / clear spelling |
| `S` | Confirm/speak the spelling buffer |
| `Y` / `N` | Confirm provisional communicator / correct to next person |
| `Q` | Close and save local learning |

The reused ASL component is the previous EfficientNet A–Z fingerspelling classifier and stability gate, not a full ASL language translator. Dynamic signs and J/Z motion are not validated. Appending letters is explicit to avoid repeats. Selection loss/person changes clear stability. Spelling can provide exact wording independently of motif learning.

Choose `F` while the intended event is recent; its evidence stays attached while you type. The learner permits confirmation for up to 120 seconds, then treats the exact typed message as text-only rather than training stale evidence. Changing communicators clears the entry. Confirming displayed options also retains their original evidence references.

Gemini and API-key instructions: [GEMINI-SETUP.md](GEMINI-SETUP.md). To enable text suggestions add `--gemini`; raw audio upload separately requires `--gemini-asr`. No cloud flag means no Gemini requests. Suggestions never speak themselves. TTS enqueue/playback plus a trailing guard suppresses audio embedding/transcription/learning to avoid feedback; headphones still improve practical echo control. The guard is not acoustic echo cancellation.

Private output is under `.scratch_session/<profile>/`: local person signature, motif/confirmation JSON, per-session manifest, and optional raw recordings/evidence. These contain personal data and are Git-ignored, not encrypted. Keep device/account access controlled. `.env` and `.env.*` are ignored; `.env.example` contains empty placeholders only. No key is written to evidence packets, recordings or error messages.

Offline replay opens no sensors:

```powershell
& ..\tink\.venv\Scripts\python.exe tink_session.py --video ..\tink\.scratch_video\two_people.mp4 --headless --max-frames 48 --no-asl --no-asr --no-tts --profile replay
```

Optional `--wav path.wav --audio-offset 0` aligns a WAV to source-video seconds using the supplied offset; it cannot recover missing synchronization metadata. Use a separate replay profile. Replay validates code/model execution, not live latency or user communication accuracy.

Next test gates: verify identity recovery with two people and overlap; inspect each channel's actual landmarks; repeat gestures/sounds and confirm intended messages; check that different people/ordinary conversation do not train the communicator; test exact fingerspelling and confirmed speech; then compare held-out local retrieval with Gemini ranking. Measure the intended message's presence among four choices, corrections, abstentions and latency. The count-based posterior remains explicitly uncalibrated until evaluated on held-out examples.
