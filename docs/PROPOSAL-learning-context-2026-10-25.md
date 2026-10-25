# Proposal: lower-friction user tracking, learned signals and contextual suggestions

2026-10-25. Research/design result, not an implemented replacement. Sanjith asked to reconsider
repeated manual target selection, explore unlabeled audiovisual learning, accumulate conversation
context, compare Gemini models for four suggestions, and finish with text-to-speech. He says a
fixed location/context cannot be assumed and wants minimum friction with good results. No cloud
request, model installation, sensor capture or production-code change was made for this proposal.

## 1. What the current audio test actually tests

`audio_layer_lab.py live --record` matches segmented sounds against an existing enrolled acoustic
profile and saves the input locally. The quiet calibration estimates the room's noise floor; it
does not learn meaning. The names `hum` and `ah` are sound-class tags, not words or intentions.
Random sounds, talking, coughs and background noise are useful tests of segmentation and rejection.
An arbitrary sound matching an enrolled class is not evidence that the app understood a message.
`live --without-profile --record` exposes segmentation/encoding while every class stays unknown;
it does not secretly enable unsupervised learning.

The present code is a measurement baseline. Keep it for comparison; remove compulsory named
sound enrollment as the only path in the proposed product. Current event JSON saves scores and
embedding dimension, not every raw embedding vector; the shared recorder should persist local
embedding references/arrays if future clustering is intended. Recorded WAVs can be re-encoded.

## 2. Automatic targeting without assuming a fixed seat

Separate two jobs: track the same physical person, and establish that person's communicator role.
ByteTrack IDs solve neither role assignment nor persistent identity after every occlusion.
The present lab is deliberately conservative: any overlapping box clears the user and asks for
another click. That is too much friction for the desired product, even though it avoids an
obvious source-mixing failure. Do not restore the original app's largest-person fallback.

Proposed lifecycle:

1. Automatically initialize a provisional target when only one plausible user is visible or an
   existing, locally established user profile supplies a sufficiently strong match. Clearly
   display which person is being followed and allow a quick correction.
2. Preserve a logical communicator identity separately from short-lived tracker IDs. Keep its
   appearance/motion evidence through temporary losses, camera movement and changes of room.
3. During overlap/occlusion, retain the identity hypothesis while withholding ambiguous features.
   On separation, automatically rebind only when the match is sufficiently unambiguous. Preserve
   calibration only when identity continuity is supported; reset temporal windows across gaps.
4. Ask for a quick recovery cue only when genuinely needed. Do not ask on every frame or box
   overlap, and do not impose a fixed-seat requirement. Cross-session remembered identity would
   be an explicit local profile feature, not a claim that body/clothing appearance never changes.

A zero-setup system cannot always infer the intended role when an entirely new scene starts
with multiple equally plausible people. Active-speaker detection identifies who is speaking,
often the partner, not who is the nonverbal communicator. Never define the user as the quietest
person or infer a disability from appearance. The unresolved initial-role cue should be made as
small as possible, rather than hidden as an unreliable automatic guess.

Candidate comparison: existing ByteTrack versus BoT-SORT with appearance-based re-identification
and a persistent role layer. The official tracker documentation describes optional ReID and
camera-motion compensation; local CPU cost, actual installed-version support, available weights
and identity-switch rates must be checked before switching. This is a proposal, not an assertion
that enabling one YAML flag solves user identity. [Ultralytics tracking](https://docs.ultralytics.com/modes/track/).

## 3. Unsupervised discovery versus semantic grounding

The useful direction is self-supervised discovery of repeated audiovisual motifs, with lightweight
grounding through the user's later choices. It reduces up-front naming/training burden.

Counterexample: two recurring patterns could mean yes/no, no/yes, or attention/discomfort. Swapping
their meaning leaves the unlabeled recordings unchanged. A cluster identity is not an intent.
Conversation supplies a useful prior, but cannot establish that an expected answer was intended.
The user must still be able to contradict what context makes likely.

Start with frozen voc2vec embeddings and normalized temporal visual measurements: head-angle
trajectories, blendshape changes, relative wrists, hand geometry and quality. Compare simple
clustering/regularized mixture baselines and small alignment projections before training another
large network. Fit normalization/dimensionality reduction only on training sessions. Retain
unknown/noise groups and modality absence. Some valid signals are audio-only or video-only;
requiring synchrony for all signals would exclude those users/events.

Cross-modal clustering and synchrony are supported research directions, but their published
benchmarks do not validate personal AAC intent recovery from a few minutes of room data:

- [Cross-Modal Deep Clustering](https://papers.nips.cc/paper/2020/hash/6f2268bd1d3d3ebaabb04d6b5d099425-Abstract.html)
  uses one modality's clusters to supervise representation learning in the other on large corpora.
- [GestSync](https://www.robots.ox.ac.uk/~vgg/research/gestsync/) learns correspondence between
  speech and body motion. That relationship is looser than lip-sync, and its setting differs
  from atypical voluntary nonverbal signals.
- [Light-ASD](https://github.com/Junhua-Liao/Light-ASD) is a candidate partner-speech attribution
  baseline. It is not a validated hard gate for a nonverbal person's vocalizations.

Discover a motif such as `m07` without naming a word. Later confirmed/rejected suggestions can
gradually ground some motifs. Never train on an unconfirmed LLM suggestion as though it were
ground truth. Rejection of wording is not necessarily rejection of the underlying intent; retain
correction/Undo and explicit provenance when updating a personal model.

The [voc2vec paper](https://arxiv.org/html/2502.16298v1) lists ReCANVo in its pretraining corpus.
ReCANVo is therefore an in-domain probe, not automatically a clean unseen-recording test of the
checkpoint. New held-out recordings/sessions/rooms are necessary for the proposed comparison.

## 4. The prerequisite: shared capture and source ownership

The current separate lab commands have different relative time origins. Their JSON timestamps
cannot simply be joined to create positive audiovisual pairs. A shared observation/session
recorder is the next technical prerequisite, not a larger LLM or more random sound trials.

Record one monotonic session origin plus audio device/read-time offsets, sample indices, video
capture times, event intervals, clock-validity/gap flags, modality quality and local person/speaker
hypotheses. Keep raw private media optional and local. Preserve observations across model delays;
count inference drops. A device overflow must not become a valid aligned training example.

Positive pair candidates need temporal overlap and plausible ownership. A partner talking while
the communicator moves is an important *negative/confounded* example, not automatically a
matching audio/video signal. Compare true pairs against time shifts, wrong-person pairs and
shuffled pairs. Similarity alone cannot establish which person's signal was captured.

For simultaneous voices/nonverbal sounds, diarization is not source separation. Overlapping
mixtures can contaminate voc2vec embeddings even when a transcript has speaker labels. Expose
overlap/unknown and abstain; compare separation or a nearer user microphone only if this proves
to be the limiting condition. Do not claim that a speech VAD isolates every nonverbal signal.

## 5. Conversation context and transcription

Current `transcribe.py` collapses Whisper segments to a string. Its Meta adapter strips diarized
prefixes. The main runner appends each new transcript as a partner turn and supplies only a small
history to the composer. These interfaces must change if multiple people contribute context.

Return structured turns with IDs, source-time intervals, speaker hypotheses/unknown/overlap,
text, final-versus-interim status and revision links. Speaker labels must persist across chunks
or remain explicitly local/uncertain. Build a rolling context state from finalized turns and
confirmed user messages: current question/topic, mentioned referents and unresolved ambiguities.
Keep verbatim turns so a compressed summary cannot become an invented fact.

Partner speech informs context; it is not evidence that the user expressed those words. The
alt checkout still contains an older fused-transcript path, so it must not be assumed to inherit
the fixes on the original checkout. Proposed suggestions must not enter confirmed conversation
history before selection. Tag app-generated TTS and avoid learning its echo as new user/partner
evidence while preserving the possibility of a genuine overlapping interruption.

Official Gemini options checked 2026-10-25:

- [`gemini-3.5-transcribe-live`](https://ai.google.dev/gemini-api/docs/live-api/live-transcribe)
  provides incremental streaming transcription and utterance timing, but currently no streaming
  diarization or word timestamps; sessions also need renewal. It is not a complete answer to
  the multi-speaker requirement by itself.
- [`gemini-3.5-transcribe`](https://ai.google.dev/gemini-api/docs/transcribe)
  supports file transcription with diarization and word timestamps. Using overlapping chunks
  adds latency and still needs cross-chunk reconciliation; overlapping speech is not magically
  separated. Preserve verbatim signals rather than enabling disfluency removal by default.

Compare these against a continuous local Whisper path plus an independently evaluated diarizer;
do not replace all local components merely because the phrase generator uses Gemini.

## 6. Bayesian belief, retrieval, Gemini and four suggestions

The modalities must refer to a shared latent variable before their distributions can be fused.
Audio cluster 2 and video class 2 are not corresponding meanings. Raw cosine scores and the 52
face coefficients are not likelihoods or mutually exclusive categorical probabilities.

For a grounded act `I`, user `u` and prior conversation context `C`, the target is:

```
P(I | A, V, C, u) proportional to P(A, V | I, C, u) * P(I | C, u)
```

Factoring the joint likelihood into independent audio and video terms is a model assumption to
test, not a consequence of calling the calculation Bayesian. Multiplying two posteriors without
correcting their shared prior also overcounts evidence. Until semantic grounding exists, report
a belief over anonymous motifs, not a fabricated calibrated posterior over intentions.

Use a hybrid comparison: local retrieval remains a baseline and fallback; Gemini can generate
or rerank contextually appropriate candidate utterances from a compact, provenance-bearing packet.
The present `llm.py` already sends a JSON-like frame but hard-codes three options; it is not yet
the proposed four-option pipeline. Its configured Gemini profile is not the active provider.

Proposed input fields:

```json
{
  "session_id": "...",
  "event_ids": ["..."],
  "user_role": {"logical_id": "...", "ownership_status": "uncertain"},
  "belief": {"kind": "ungrounded_motifs", "calibrated": false, "items": []},
  "quality": {"clock_valid": true, "overlap": false},
  "conversation": {"final_turns": [], "confirmed_user_turns": [], "pending_question": null},
  "candidate_pool": [],
  "rejected_option_ids": []
}
```

This example is a schema sketch, not actual measurements. If a learned intent posterior exists,
change `belief.kind` explicitly and supply it. Do not dump 768 anonymous floats into an LLM and
assume it understands their acoustic meaning. Use local embedding references, learned motif
associations and observable summaries, with confidence type/validity stated honestly.

Request four distinct, grounded suggestions with evidence/event references, plus an external
None-of-these/clarify control. Do not force unsupported filler when fewer useful suggestions are
available. LLM rankings or self-reported confidence are not calibrated posterior probabilities.
Trigger a request on a new completed communication event or materially revised finalized context,
not every video frame. Cache/deduplicate by the evidence/context version and discard stale replies.
Use [native structured output](https://ai.google.dev/gemini-api/docs/structured-output)
and application validation: count, uniqueness, supported facts, current context revision, candidate
age and invalid-output fallback. Context may condition both belief and verbalization; do not
multiply a second context-derived model score as if it were independent fresh evidence.

User confirmation remains before speech. A wrong or stale suggestion must not become a voiced
statement merely because its model score is high. Existing local `pyttsx3` TTS can serve this last
stage without a new cloud dependency. Add the fourth selection/display path where needed.

[SpeakFaster](https://www.nature.com/articles/s41467-024-53873-3) supports context-aware candidate
generation, user selection and TTS. Its grounding included user-entered abbreviations/keywords;
it does not show that unlabeled nonverbal signals alone recover personal intentions.

## 7. Gemini model shortlist

These are API options verified in official docs, not a latency or AAC-quality benchmark. My
recommendation is to compare Flash 3.8 with Flash-Lite 3.5 on identical captured packets; use
Argon only as a stronger offline comparison if the smaller models demonstrably fail.

| Role in the comparison | Model | Reason |
|---|---|---|
| Main quality/latency candidate | [`gemini-3.8-flash`](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) | Current stable Flash, structured output; request supported low thinking and measure full response latency |
| Lower-cost/latency baseline | [`gemini-3.5-flash-lite`](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite) | Stable throughput-oriented model, already represented in the current configuration |
| Offline capability comparison | [`gemini-4-argon`](https://ai.google.dev/gemini-api/docs/models/gemini-4-argon) | Current flagship reasoning option; no assumption that additional capacity improves this AAC task |

Standard paid text input/output prices per million tokens currently listed: Flash-Lite $0.30/$2.50;
Flash $0.75/$3.75 through 2026-12-31; Argon $2/$10 through 2026-11-30. Thinking output is billed.
Promotional prices change afterward. Google's free-tier content-use terms differ from standard
paid use; real conversations need an appropriate choice before any upload. No account quota,
availability or request latency was tested. [Official pricing](https://ai.google.dev/gemini-api/docs/pricing).

## 8. Proposed next work, sequenced and testable

| Step | Smallest output | Test that changes the next decision |
|---|---|---|
| 1. Shared recording + persistent role | One session/clock and ownership contract, automatic tracking/recovery prototype | Identity-switch errors, recovery prompts per minute, aligned-time error, withheld-event coverage; single person, crossings, re-entry and moving camera |
| 2. Unlabeled motif discovery | Frozen audio embeddings + temporal visual features + unknown groups, separate from semantic labels | Same-signal stability across held-out sessions/rooms; false activations during idle; true-pair versus shifted/wrong-person alignment |
| 3. Timed conversation state | Speaker-aware finalized/revised turns, overlap/unknown, confirmed history and TTS provenance | Wrong-speaker context contamination, transcription error, turn latency and revision handling |
| 4. Lightweight semantic grounding | Confirmation-linked evidence records, not compulsory per-word enrollment | Amount of feedback needed, correction/Undo behavior, held-out intent/phrase recall; compare audio-only/video-only/joint/context |
| 5. Candidate comparison and confirmed speech | Same packets into retrieval, Flash-Lite, Flash; four-choice UI + existing TTS | Top-4 intended-phrase coverage, unsupported suggestions, None-of-these rate, user effort and capture-to-usable-suggestion p50/p95 |

Collect realistic positives, ordinary movement, unrelated speech, overlapping activity and
deliberately mismatched pairs. Split by recording/session/room before fitting normalizers or
selecting models. Keep held-out ground truth for evaluation even if representation learning uses
no semantic labels. Do not label arbitrary daily movement as a communicative negative without
user evidence, and do not use synthetic-only results as real-user communication accuracy.

## 9. Scope boundaries

Likely seams: `person_selection.py` / the tracker adapter for persistent role recovery; a new shared
observation recorder rather than merging two unrelated lab logs; an optional discovery module
over saved features; `transcribe.py` and conversation state for structured turns; the fusion
interface only after grounded likelihoods exist; `llm.py`, configuration and selection UI for four
suggestions; reuse `tts.py`. Implement these as bounded comparisons, not an all-at-once rewrite.

This document is the reviewable proposal. Existing test apps, model weights, enrollment profiles,
fusion, cloud configuration and recordings remain unchanged. No Gemini request has been made.
