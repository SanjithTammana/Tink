# Tink backend — implementation plan (v1, 2026-10-24)

Approach and logic: Sanjith's. This plan only turns his instructions plus the two papers' released code into
modules. Anything marked **[his call]** is a placeholder until he specifies it.

## Files
```
tink/
├── transcribe.py      # Transcriber: open CPU speech-to-text (faster-whisper int8, base.en default)
├── audio_text.py      # audio + Bayesian: VocalizationEncoder, VocalizationClassifier, AudioPipeline,
│                      #   BayesianFusion (imports Transcriber)
├── video.py           # VideoAnalyzer (YOLO + MediaPipe gesture/face, from message.txt) + EnvisionGesture
├── run.py             # argparse + capture loop + OpenCV panel (message.txt interface) + JSON packet
├── config/likelihoods.json   # phrases, prior, P(observation | phrase) tables, decision rule  [his call]
├── data/vocalizations/       # <label>/*.wav folders and/or manifest CSVs (path,label); any number of datasets
└── models/                   # downloaded .task files, trained vocalization classifier (git-ignored)
```
Environment: Python 3.11 `.venv` (envisionhgdetector pins tensorflow-cpu 2.15.1, mediapipe 0.10.7, pandas<2,
numpy<2; none install on 3.12). `constraints.txt` holds those pins; always `pip install -c constraints.txt`.

## What the papers change in the method
**voc2vec (Koudounas et al., ICASSP 2025).**
- It is an audio *encoder* (wav2vec2/HuBERT, 12 layers, 768-d, 16 kHz mono, ~20 ms frames). It does not
  turn text into audio. Its input is the raw waveform.
- The released checkpoints are **pre-trained only, with no classification head**. The README's
  `AutoModelForAudioClassification` snippet attaches a randomly initialized head, so its logits mean nothing
  until fine-tuned. The paper fine-tunes a linear head (batch 16, lr 1e-4, 10-fold CV, scored by UAR).
- With little data we use the same model frozen: mean-pooled last hidden state → 768-d embedding → light
  classifier (logistic regression, balanced class weights), scored by UAR under stratified k-fold as in
  the paper. Checkpoint: `alkiskoudounas/voc2vec-hubert-ls-pt` (best in the repo: UAR .696).
- Most relevant public data for Tink: **CNVVE** (6 non-verbal voice classes, healthy + dysarthric speakers,
  built for assistive HCI). ReCANVo (minimally speaking individuals) is in voc2vec's pre-training set.
  Datasets plug in as folders or manifests; nothing is hard-coded to one dataset.

**EnvisionHGdetector (Pouw, Shaikh et al., 2026 preprint).**
- MediaPipe Holistic → features → classifier. The **LightGBM** model is the real-time one (per-frame features,
  CPU). Its only labels are **Gesture vs NoGesture** (Move is merged into NoGesture and unreliable).
- Use it as "is the person producing a gesture right now", not "which gesture". Weak spots it reports:
  beat and symbolic (posture-like) gestures; it expects front view, one person, 25–30 fps. Out-of-sample
  F1 is about 0.6–0.7, and the paper itself says to smooth over time rather than trust single frames.
- Per-frame API we wrap: `LightGBMGestureModel.extract_features_from_frame(frame)` → `.predict(x)` →
  `label_encoder` (the same calls its own `RealtimeGestureDetector` makes inside its webcam loop). Our
  loop stays in run.py.

## Pipeline as built (Sanjith's diagram + agreed additions, 2026-10-24)
```
                      USER  →  MULTIMODAL INPUT
              ┌─────────────────┴──────────────────┐
         VIDEO (video.py)                    AUDIO (audio_text.py, transcribe.py)
   EnvisionHGdetector                   voc2vec (frozen)          Whisper (faster-whisper)
   Holistic → kinematic features        embeddings →              transcript keywords
   → LightGBM → P(Gesture), smoothed    vocalization classifier   [addition: in prototype]
   MediaPipe GestureRecognizer          → nonverbal evidence
   [addition: from prototype]
              │                                    │
        VISUAL EVIDENCE                      AUDIO EVIDENCE
              ▼                                    │
   INITIAL DISTRIBUTION over intents               │
   (BayesianFusion.visual_distribution)            │
              └─────────────────┬──────────────────┘
                     BAYESIAN UPDATE (BayesianFusion.update)
                                ▼
                 POSTERIOR DISTRIBUTION over intents
                                ▼
          SEMANTIC RETRIEVAL (retrieval.py)
          phrase database config/phrases.json, MiniLM vectors,
          P(intent | phrase) = tag (0.8) + embedding similarity (0.2)
                                ▼
          PHRASE RANKING: score = Σ_k P(k | evidence)·P(k | phrase)
          → classical ML ranker (logistic regression) once confirmations are logged
                                ▼
                 TOP INTERPRETATIONS (top 3 on the panel)
                                ▼
          USER CONFIRMATION: keys 1–3, 0 = none → logged to data/confirmations.jsonl
                                ▼
          TEXT-TO-SPEECH (tts.py, offline SAPI)  →  FINAL COMMUNICATION
```

## Target pipeline (Sanjith's diagram, 2026-10-24) and build status
| Stage | Module | Status |
|---|---|---|
| Video → Envision (Holistic → kinematic features → LightGBM → gesture detection) → visual evidence | `video.py` | built; tested on Envision's sample video |
| Visual evidence → **initial distribution** | `BayesianFusion.visual_distribution` | built |
| Audio → voc2vec (preprocess → embeddings → vocalization classification) → nonverbal audio evidence | `audio_text.py` | built; classifier needs real data and a label→distribution mapping |
| **Bayesian update** (initial × audio evidence) → **posterior distribution** | `BayesianFusion.update` / `infer` | built (posterior is order-invariant) |
| Semantic retrieval: phrase database + vector embeddings → phrase ranking → top interpretations | `retrieval.py` | built; 178 phrases |
| User confirmation | keys 1–3 / 0 in `run.py`; web board later | built (keys) |
| Text-to-speech | `tts.py` (offline SAPI) | built |
Not in the diagram but in the current code: MediaPipe GestureRecognizer (message.txt) as a video source, Whisper
transcript as an audio source. Both are kept until Sanjith says otherwise.

## Bayesian fusion — REVISED by Sanjith (supersedes the decision and learning parts below)
- Video gives the initial distribution P(intent | video), from the mp_gesture and envision sources.
- Audio updates it: posterior ∝ P(audio | intent) · P(intent | video), from the vocalization and transcript
  sources. (This was audio-first at one point; the posterior is the same in either order.)
- The output is the updated **distribution only**: no threshold, no abstain, no learn_likelihoods, and not
  self-correcting.
- The final answer comes from a separate **classical ML model** reading the distribution. In the built
  pipeline that is the phrase ranker in `retrieval.py`: a formula until confirmations are logged, then
  logistic regression.

## Bayesian fusion (original v1 text, kept for the math)
- Hypotheses: the phrase list (starts as message.txt's 8 phrases).
- Posterior: log P(phrase | E) = log prior(phrase) + Σ_sources log Σ_o P(o | phrase) · q_s(o), normalized.
  `q_s(o)` is each detector's own probability over its labels (soft/virtual evidence), so a 0.55 gesture
  score counts for less than a 0.95 one. A source with no data, or one that is stale, untrained or silent,
  contributes nothing (likelihood 1 for every phrase).
- Sources: `mp_gesture` (MediaPipe gesture labels), `envision` (Gesture/NoGesture), `vocalization` (voc2vec
  classifier labels), `transcript` (keywords in Whisper text).
- Tables live in `config/likelihoods.json`. Initial values convert message.txt's hand-coded weights w into
  likelihoods ∝ exp(w), so v1 ranks like the current prototype. They are labelled placeholders.
  Mappings for envision and vocalization labels are **neutral (equal for all phrases) until Sanjith gives them [his call]**.
- `learn_likelihoods(records)`: count-based estimate, with Laplace smoothing, from logged (observations,
  confirmed phrase) rows, so more data can replace the hand-coded tables.
- **Decision:** commit to the top phrase only if p ≥ `threshold` and p − p₂ ≥ `min_margin`. Otherwise
  **abstain → backup (manual board)**. Both numbers are in the JSON **[his call]**; defaults 0.6 / 0.15.

## Interfaces (the contract between files)
```python
# video.py
VideoAnalyzer(use_yolo=True, use_envision=True).analyze(frame_bgr, ts_ms) -> {
  'timestamp_ms': int, 'people': [{'bbox': [x1,y1,x2,y2], 'score': float}],
  'gestures': [{'label': str, 'score': float, 'alternatives': [{'label','score'}]}],
  'face_movements': {name: float}, 'hand_motion': str,
  'envision': {'status': 'ok'|'no pose'|'disabled'|'error: ...',
               'probs': {'Gesture': float, 'NoGesture': float},   # smoothed over a short window
               'label': 'Gesture'|'NoGesture'}}
VideoAnalyzer.close()

# audio_text.py
AudioPipeline(enabled, whisper_size='base.en', voc_classifier_path=...).start() / .get() / .close()
  .get() -> {'transcript': str, 'timestamp': float|None, 'status': str, 'rms': float,
             'vocalization': {'status': 'ok'|'untrained'|'silence'|'disabled'|'error: ...',
                              'label': str|None, 'probs': {label: float}}}
VocalizationEncoder().embed(wave_16k_mono: np.ndarray) -> np.ndarray[768]
VocalizationClassifier.train(dataset_dirs_or_manifests) / .save(path) / .load(path) / .predict_proba(emb)
BayesianFusion.from_json(path).infer(visual: dict, audio: dict) -> {
  'initial_distribution': {intent: p},                  # from video; sorted, sums to 1
  'posterior': {intent: p},                             # after the audio update; sorted, sums to 1
  'evidence': {source: {'used': bool, 'informative': bool, ...}}}
BayesianFusion.visual_distribution(visual) / .update(initial, audio)

# retrieval.py, tts.py
PhraseRanker(PhraseDatabase(), intents).rank(posterior, k=3) -> [{'text', 'intent', 'score', 'match'}]
PhraseRanker.log(posterior, shown, chosen)   # data/confirmations.jsonl, for train-ranker
Speaker().say(text)

# CLI
python audio_text.py train --data data/vocalizations [more dirs/manifests] --out models/voc_clf.joblib
python audio_text.py classify some.wav
python run.py --camera 0 --audio | --video sample.mp4 --no-audio [--no-yolo] [--no-envision]
```

## Open for Sanjith
- How envision Gesture/NoGesture should move each intent's probability (neutral until set).
- Which vocalization classes map to which intents, and which dataset(s) or ready-made model to use first.
- Live speed: Envision needs consecutive frames at 25–30 fps; the full pipeline runs about 1 fps on CPU.
