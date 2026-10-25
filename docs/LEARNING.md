# Personal motifs and confirmed meaning

`motif_learning.py` discovers repeated feature patterns without word enrollment.
It does not infer what an unlabelled gesture or vocalization means. The user can
choose or edit an utterance after an occurrence; that explicit confirmation can
ground the preceding evidence. Gemini suggestions are not training labels.

For exploratory testing, produce a variety of natural movements and sounds,
including ordinary inactivity and background speech. Inspect the measurements,
source ownership, and motif IDs first. Repeating an intentionally similar pattern
should eventually reuse an ID. A different pattern can form a new provisional ID.
Those are tests of feature consistency, not evidence of communication accuracy.
Do not assign random sounds to words just to make the system produce answers.

## Input contract

```python
from motif_learning import MotifLearner

learner = MotifLearner('.scratch-learning/personal-motifs.json')
discovery = learner.observe({
    'event_id': 'unique-window-id',
    'session_id': 'new-id-for-each-capture-session',
    'modality': 'audio',                 # audio, video, or internally paired
    't_start': 2.0, 't_end': 2.6,        # common session-clock seconds
    'person_id': 'confirmed-logical-user',
    'ownership_valid': True,            # never assume microphone speech is this user
    'clock_valid': True,
    'features': audio_features,         # finite nonzero vector; actual audio: 768
    'feature_schema': 'encoder-name-revision-and-transform-v1',
    'quality': {},
    'context_id': 0,
})
```

Feature vectors must be one-dimensional, finite, nonzero, and 2–4096 values long.
The first valid vector fixes dimension and feature schema separately for each
modality in a profile. Change the encoder, feature ordering, or transformation:
start a separate profile. This module applies L2 normalization. The visual
extractor must normalize mixed geometric units before calling it; audio amplitude
and other desired distinctions must be represented in the incoming features.
There is no learned metric or deep self-supervised training here.

`quality` accepts Boolean `tts`, `overlap`, `tracking_lost`, and `source_lost`
flags. Any such flag, invalid clocks, or `complete=False` excludes an observation
from discovery, pairing, and grounding. `is_communicative` being absent means
unknown. A negative example is excluded only when `quality.confirmed_by_user=True`
and `negative=True` or `is_communicative=False`. Ordinary idle video is not
automatically labelled meaningless. Unknown-owner audio can discover anonymous
patterns in a separate feature space; it cannot acquire personal meanings.

`observe` returns a motif ID or `None`, a status, and memberships with raw cosine
similarity scores. A new cluster is `provisional`; default maturity requires three
distinct observations. Similarities are neither probabilities nor calibrated
confidence. Counts alone do not establish that a motif carries intent. Duplicate
event IDs are idempotent; reusing an ID with changed content raises an error.

## Pairing and evidence independence

```python
paired = learner.pair(audio_event, visual_event)
if paired['pair_status'] == 'paired':
    pair_id = paired['event_id']
```

`pair` requires a shared capture-session ID, valid clocks, the same confirmed
logical person, confident ownership, complete sources, and no exclusion flags.
Default alignment requires at least 25% overlap of the shorter interval. A caller
can explicitly permit a nonoverlapping gap with `max_lag_s=0.25` or less.
Large shifts, another person's voice, unknown ownership, and TTS are rejected.
Zero-duration intervals do not meet the default overlap rule.

The method observes the sources and a joint vector made by concatenating the two
normalized modalities, giving them equal total norm. It returns the paired event
and source IDs for provenance. Repeated pairing of the same sources is idempotent;
a source cannot enter another retained pair. A paired observation supplied through
`observe` must reference retained, valid source events with matching features and
ownership. Temporal co-occurrence still does not prove a shared meaning.

At inference and confirmation, a pair suppresses its component votes. Explicit
groups and overlapping windows for the same person/session are also conservatively
collapsed. Two confirmations with overlapping time intervals are rejected as
correlated, even with different event IDs. Two independently repeated occurrences
are required before a confirmed utterance can enter the semantic posterior.

## Confirmations, corrections, and posterior

```python
learner.confirm([pair_id], 'I would like some water', at_s=3.0)
result = learner.posterior([pair_id], now_s=3.0)
learner.reject(['suggestion-2'], [pair_id], at_s=3.1)
learner.undo()
```

Only retained, clock-valid, attributable events ending within the preceding 120
seconds are eligible for confirmation. A confirmation may intentionally reference
past event IDs. It cannot combine people or capture sessions. Confirming the same
source again cannot increase support. To change its label, Undo its confirmation
and confirm again. `undo()` removes the most recent confirmation and returns a
Boolean. A rejected wording stores a correction for later prompt construction;
it does not label an entire intent, sound, or gesture negative.

`posterior` uses only requested eligible evidence from the preceding three seconds
by default. Missing, stale, unowned, provisional, or insufficiently grounded
evidence returns `kind='ungrounded_motifs'` and an explicit `(unknown)` outcome.
Context alone cannot force an utterance. After at least two independent
confirmations associate a mature current motif with an utterance, it can return
`kind='intent_posterior'`.

That posterior uses Dirichlet-smoothed empirical counts of
`P(motif | user-confirmed utterance)`, one shared utterance prior, and one
likelihood per effective independent group. It reserves an explicit unknown
model. Each item includes its probability, confirmation support, and current
evidence IDs. `calibrated=False` is always present. These probabilities are a
small personal count model, not validated AAC confidence or decoder accuracy.
The same logical user ID can carry confirmations across capture sessions; reuse
that ID only for the same person.

## Persistence and bounds

Pass a path to load an existing profile and autosave confirmations, corrections,
and Undo. Call `save()` on an orderly capture shutdown to retain new unconfirmed
motifs as well. Persistence uses bounded, atomic JSON, never pickle, and stores
derived features, IDs, timestamps, and actual confirmations. No raw media is
copied. Derived features and chosen messages remain personal data.

Inside a Git repository, profiles must live under `.scratch*` or `models`.
The profile's own directory receives a `*` `.gitignore`; existing ignore files
are never overwritten and must already contain an unconditional `*` rule with no
exceptions. Redirected/symlink directories are rejected. Keep profiles out of
shared source directories and verify the actual file with `git check-ignore`.
Ignore rules do not encrypt data or prevent a deliberate `git add -f`.

Defaults bound the history to 512 events, 64 prototypes, 512 confirmations,
32 sessions, and 128 wording corrections; persisted JSON is capped at 32 MB.
Evicted event times are retired per session to prevent old IDs being replayed as
fresh training. New sources outside history and new patterns beyond prototype
capacity abstain. Rotate to a deliberate new profile when its session capacity is
reached; no silent model/schema migration occurs.

## Verification and limits

Run the synthetic suite from the project root:

```powershell
& ..\tink\.venv\Scripts\python.exe -B tools\test_motif_learning.py
```

Tests cover repetition/novelty, missing ownership, nonfinite/schema errors,
duplicate IDs, alignment counterexamples, held-out motif associations,
correlated-vote suppression, reversible confirmations, stale evidence, capacity,
and JSON persistence. They do not establish real-person tracking, audio source
attribution, semantic correctness, clinical usefulness, or communication accuracy.
Those need separate layer tests with meaningful held-out examples, wrong-person
and time-shift controls, and the communicator's actual confirmation.
