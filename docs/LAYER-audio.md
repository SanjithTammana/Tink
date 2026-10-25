# Layer: nonverbal audio evidence (draft, 2026-10-25)

Sanjith's diagram: voc2vec → audio preprocessing → acoustic embeddings → vocalization classification →
nonverbal audio evidence → Bayesian update. This note breaks that branch into steps, fixes what each step
hands the next, says how each step is measured on its own, and records the baseline. Numbers come from
`tools/bench_audio.py` on this laptop (i7-1355U, Windows, torch 2.14 CPU).

## 1. Output contract (what the layer hands fusion)

One `AudioEvidence` per detected sound event, emitted once, when the event ends:

```python
{'source': 'vocalization',
 'event_id': 'voc-000123',            # fusion applies each id once (A6)
 't_start': 812.43, 't_end': 813.11,  # capture clock: sample index of the ring buffer / SR, not wall time at publish
 'person_id': 'user' | None,          # None when attribution fails (see step 3); fusion skips None
 'probs': {'<user class 1>': p, ..., 'speech': p, 'other': p},   # sums to 1; open set is explicit
 'quality': {'dur_s': 0.68, 'snr_db': 21.0, 'speech_max': 0.12},
}
```

- Labels are the **user's own** sound classes (enrolled, like `signs.py`), plus two negative labels the
  model always has: `speech` (anyone talking, including the partner) and `other` (background, cough, noise).
- Label → intent is a separate, user-editable table (`P(label | intent)`), built the same way
  `BayesianFusion.add_sign_source` builds the sign table: the mapped intent gets `peak`, other intents 1,
  `speech`/`other` rows flat. Fusion then needs no change beyond reading the event once.
- Expiry: an event is evidence for the ~0.5 s after `t_end` (cross-layer contract); after that it is gone,
  not repeated.
- Silence produces **no event**, not an event that votes "(no message)" (A3: quiet channels must not veto).

## 2. Steps, the handover between them, and how each is measured

| # | Step | Hands to next | Measured by (on its own) |
|---|---|---|---|
| 1 | Capture | continuous 16 kHz mono ring buffer + sample clock | samples lost while inference runs (target 0) |
| 2 | Segmentation (event detection) | `(t_start, t_end, snr)` per sound | recall of planted sounds; false segments per minute of background |
| 3 | Attribution / speech gate | segment + `speech_max`; drop if partner speech | AUROC vocalization vs speech |
| 4 | Embedding | 768-d vector of **the segment only** | held-out-speaker UAR of a linear probe; ms per segment |
| 5 | Classification (open set) | `probs` over user classes + speech + other | held-out-speaker UAR; k-shot accuracy on a new person; false events per hour |
| 6 | Evidence | `AudioEvidence` above + label→intent table | end to end: correct events per vocalization, false events per hour |

STATUS: baseline and alternatives being measured.

## 3. Recovered results and test boundary (2026-10-25 continuation)

Sanjith will test each layer independently and decide next steps. No audio implementation or
fusion was changed in this continuation. Existing jobs were allowed to finish.

- CNVVE held-out-speaker few-shot accuracy, averaged over speakers: layer 5, population + user
  examples weighted x20, k=1 .961 (20 speakers), k=2 .990 (18 speakers); last layer 12 .921/.949.
  These are exploratory results on six vocalization classes, not intent accuracy or an AAC-user
  validation. Layer selection and k-shot adaptation need an independent final test set.
- `stream --layer 5` completed: 280 synthetic four-second mixtures, including 160 positives from
  15 held-out CNVVE speakers. The full segment+negative-class candidate emitted zero false
  events on each of noisy/quiet/synthesized-speech negative subsets (30/30/60 chunks). Its correct
  top-1 rates were .75 noisy, .68 noisy+speech, .93 quiet, .53 quiet+speech. Emission rates were
  .85/.70/1.00/.57, so rejection/missed events must be counted alongside accuracy.
- The **segmenter alone** still emitted 1.33 false events per synthesized-speech chunk. Do not
  attribute that figure to the full candidate. Zero false events in 60 four-second synthesized
  speech chunks is only four minutes of this synthetic condition, not a real-world zero/hour rate.
- Mean measured processing cost was 1627 ms per four-second chunk for baseline and 865 ms for
  candidate under the then-active workload. This is not live microphone capture/drop validation.
- The final CLAP comparison **failed** with `ValueError: Found array with dim 5, while dim <= 2 is
  required by StandardScaler`. Earlier processor-keyword trouble was already corrected. The
  shell's grep pipeline ended with status 0 despite this Python traceback. CLAP has no valid
  comparison score; its saved embedding shape/cache needs diagnosis before a rerun.
- ReCANVo's final redownload verified MD5 `228c89ee9e70b3d45cde9eb6467283b3` and completed unpacking.
  Its new evaluation command has **not** been run. Earlier corrupt-download errors were superseded.
  The current within-participant random-clip split may share recording/session acoustics between
  train and test. Group by original recording/session before treating that as generalization.

At that recovery checkpoint, `tools/bench_audio.py` was only an independent offline evaluation
harness and a live, user-enrolled audio event layer was not yet ready. The existing live app does not
automatically acquire the measured layer-5 classifier/segmenter. Do not integrate or silently
train one from these exploratory comparisons before Sanjith's next instruction.

Evidence: the readable audio transcript is preserved in
`ClaudeOS/projects/hacktx/transcripts/claude/2026-10-25_f6d33f9e_tink-audio-layer.md`; original task
outputs `bac3ifv41.output`, `bjwr9hzqe.output`, `bnbtmfu7q.output` remain in that Claude session's
temporary tasks folder. A failed comparison is not a successful model result.

## 4. Standalone microphone test lab (added after Sanjith requested audio readiness)

This is an independent test surface: `audio_layer.py` and `tools/audio_layer_lab.py`. It does not
change `audio_text.py`, `run.py`, the benchmark harness, video selection, fusion or TTS. Its output
is sound-type template matches and explicit rejection, not communication intent. **A single
microphone cannot identify who made a sound.** The panel and every event say speaker/person
unknown. Another person making a similar enrolled sound may match the same class. Selecting a
person in the video lab does not attribute microphone events to that person.

### Run and enroll

In PowerShell:

```powershell
Set-Location C:\Users\sanji\PycharmProjects\tink-alt
& ..\tink\.venv\Scripts\python.exe tools\audio_layer_lab.py devices
& ..\tink\.venv\Scripts\python.exe tools\audio_layer_lab.py enroll --labels hum ah --takes 3
& ..\tink\.venv\Scripts\python.exe tools\audio_layer_lab.py live --record
```

`hum` and `ah` are example sound names: choose your own distinct labels, not presumed intentions.
Names begin with a letter and contain letters/numbers/underscore/hyphen, up to 32 characters.
The enrollment window starts with one second of quiet-room calibration. SPACE or Enter starts
each two-second take; make that sound, then rest before the next take. The lab trims room silence
around the sound before embedding. It records at least three examples for each personal class,
then requires three **speech** negatives (ordinary sentences) and three **other** negatives
(room sounds, cough, keyboard or quiet). Missing examples do not become a completed profile.
The microphone keeps capturing while embeddings are calculated.

`--device N` selects the input index shown by `devices`. `--profile name` keeps separate enrollment
sets. Existing profiles are preserved unless enrollment explicitly includes `--replace`. By
default enrollment saves embeddings only; add `--record` to retain the individual take WAVs.
Profiles and recordings stay under ignored `.scratch_audio_layer/` folders.

### What the live window shows

- Waveform, short-window RMS level in dBFS, measured noise floor and listening/active status.
- One result per segmented sound, with the closest personal/negative classes, raw cosine scores
  and a rejection reason. These are **not calibrated probabilities**. No sound means no event.
- Event age at publication and embedding latency, so a delayed result is not mistaken for a
  newly captured sound. Original sample-clock start/end times remain unchanged through inference.
- Delivered-sample capture clock, queue-drop counts, device status flags and inference drops.
  Audio capture runs in the device callback independently of the embedding worker. The bounded
  queues expose overload rather than silently merging across missing audio. Device-side loss
  has unknown duration: after a device status flag, events are rejected with unknown capture
  start/end times and the GUI asks you to quit/restart. Delivered-sample offsets remain in
  separate diagnostic fields; they are not claimed to be valid elapsed capture time.
- Q quits. `--record` stores the continuous local WAV plus `events.jsonl` and a summary. Missing
  callback blocks are represented by zero-filled spans in that optional WAV, with drop counts
  reported; hardware overflow flags cannot reveal how many upstream samples the device lost.

The first second of `live` is quiet-room calibration. The user test should include rest, each
enrolled sound, ordinary speech and nearby-person sounds separately. Report missed sounds,
wrong class, another person's sound accepted, duplicated events, or queue drops before choosing
any next change. No speaker verification, diarization or audiovisual attribution is claimed.

### Candidate defaults and exact feature boundary

Audio is 16 kHz mono. A 20 ms RMS envelope opens at 9 dB above the room floor and remains active
above 5 dB; minimum activity is 120 ms, release wait 260 ms and pre/trailing padding 80 ms. The
four-second maximum emits one incomplete event, then waits for rest instead of recounting a
long sound. A queue gap discards a partial segment; an unfinished sound at Q is counted as
discarded. `--noise-db -55` supplies an explicit initial floor instead of startup calibration.

The encoder loads **cached files only**, no network download: voc2vec
`alkiskoudounas/voc2vec-hubert-ls-pt`, exact hidden-state index 5, temporal mean, 768 values. The
complete model is retained rather than assuming that truncating it preserves every architecture's
hidden-state semantics. The same extractor normalization as the prior benchmark is used; sounds
shorter than 0.5 s are padded before embedding. The encoder runs in evaluation/no-gradient mode.

Classification uses the maximum cosine match within each enrolled class. Personal matches must
exceed a candidate cosine threshold of .75 and exceed the next class by .05; speech/other winners
are rejected. Low score or small margin yields unknown. Change these explicitly with
`--min-score` and `--min-margin` only as a subsequent measured comparison. Profiles validate
model ID, layer, sample rate, embedding dimension, labels and negative-example counts. These
defaults are neither tuned to this user nor an accuracy claim. Schema 2 profiles also pin the
resolved local Hugging Face snapshot revision. Extractor and model load from that same snapshot;
an enrollment from another revision is rejected rather than silently reused.

Each event preserves a unique ID, segment start/end from sample indices, completion status,
raw class scores, rejection reason and embedding time. `person_id` is always null and
`calibrated` is false. Events do not enter the existing application or fusion.

### Replay and verification

Replay a local WAV through the same segmentation/encoding/classification path:

```powershell
& ..\tink\.venv\Scripts\python.exe tools\audio_layer_lab.py wav path\to\capture.wav --headless
& ..\tink\.venv\Scripts\python.exe tools\test_audio_layer.py
```

Replay does not open the microphone. Without a profile, `--without-profile` tests segmentation
and embedding but always reports unknown class. `--save-preview` saves the waveform/feature panel.
WAV input is downmixed/resampled to 16 kHz; live capture requests that rate from PortAudio. A device
that cannot provide it fails visibly; select a supported input rather than changing rates silently.

Verified here: 15 deterministic synthetic/fake-encoder checks pass, covering brief sounds that
four-second RMS would discard, silence, merging/release, one event per long sound, gap handling,
callback sample clocks/overflow/drop accounting, negatives/unknown/margin rejection, profile schema,
sample-clock event times, overflow invalidation, model-revision mismatch and unknown person
attribution. CLI help and microphone-device listing also execute without opening a device.

Actual local cached-model smoke: the existing CNVVE
`ahem_0088d688_19-29_female_home_Q_1_0.wav` (8192 samples after 16 kHz loading, 0.512 s) completed
the WAV path with one 768-dimensional layer-5 embedding and no capture gap. It ended while sound
was active, so its event was correctly incomplete/unknown at EOF. No invented personal profile
was used and no classification-accuracy claim follows. This short-clip smoke supplies an explicit
noise floor (otherwise the one-second quiet calibration would consume this 0.512-second file).
The physical microphone, human enrollment
interaction and live GUI remain for Sanjith's test. No microphone was opened during verification.
