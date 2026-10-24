# Where the gap is, and Bayesian ways to close it (2026-10-24)

Research notes for Sanjith's decisions. Sources: the voc2vec paper and repo, the EnvisionHGdetector preprint and
source (`model_lightgbm.py`, `model_cnn.py`, `utils.py`, `config.py`), Hugging Face model configs, message.txt.
Nothing below has been run on our data yet unless it says so.

## 1. Trained versions
- **voc2vec:** none. All four checkpoints (`voc2vec`, `-as-pt`, `-ls-pt`, `-hubert-ls-pt`) are pre-trained
  encoders (`HubertModel` / `Wav2Vec2ForPreTraining`, no `id2label`). A Hugging Face base-model search finds zero
  fine-tunes of any of them.
- **Ready-made models that already output probabilities over vocal sounds:**
| Model | Labels | Notes |
|---|---|---|
| `MIT/ast-finetuned-audioset-10-10-0.4593` | 527 AudioSet; 53 vocal-related: Sigh, Groan, Grunt, Gasp, Humming, Whimper, Wail/moan, Laughter, Giggle, Screaming, Shout, Yell, Cough, Throat clearing, Sniff, Breathing, Pant, Babbling, Whispering, Speech… | BSD-3, ~87M params, 1.35M downloads. Multi-label sigmoid scores, so renormalize over the subset we use. |
| `padmalcom/wav2vec2-large-nonverbalvocalization-classification` | 16: sighing, moaning, screaming, crying, laughing, coughing, throat-clearing, tongue-clicking, lip-popping, lip-smacking, panting, yawning, sneezing, nose-blowing, teeth-chattering, teeth-grinding | Apache-2.0. Trained on the NonverbalVocalization set (800 clips, one of voc2vec's benchmarks). wav2vec2-large (~315M) is slower on CPU, and its custom head class needs their code. |
| CLAP zero-shot (`laion/clap-htsat-unfused`) | **Whatever text labels we write** ("a person humming", "a short grunt") | No training. A softmax over our own label set gives a distribution. Accuracy on subtle sounds is unknown, so it needs testing. |
| `laion/vocalburst-classifier-*` | vocal bursts | CC-BY-4.0, not a stock transformers model; format not checked yet. |

## 2. The gap, stated mathematically
Bayes: P(intent | e) ∝ P(e | intent) · P(intent).

voc2vec produces **e**, a 768-d point. It does not produce P(e | anything). The missing piece is the
**likelihood**. Bayesian inference consumes likelihoods; it cannot create them. Something must tell the model
that a region of embedding space belongs to class k. There are four places that knowledge can come from, and
none of them needs fine-tuning:
1. **Someone else's labels:** a pretrained head (AST, padmalcom). Fixed label set.
2. **Language:** CLAP zero-shot. The labels are text we write.
3. **A few examples per class (enrollment), Bayesian, no gradient training:**
   - *Gaussian discriminant:* reduce 768 → ~16–32 dims (PCA). Model each class as a Gaussian with shared
     covariance and a conjugate Normal–Inverse-Wishart prior. The posterior predictive is a Student-t. Each new
     confirmed example updates the class mean and covariance **in closed form**, instantly. This is the
     self-correcting part.
   - *Prototype:* likelihood ∝ exp(κ · cos(e, class mean)) (von Mises–Fisher). This works from 1–3 examples.
   - This matches Tink's user-defined signals: the user records 3–5 examples of their own sound.
4. **No labels:** an unsupervised mixture (GMM or Dirichlet-process mixture) finds recurring sound types. Each
   cluster still needs one confirmation to say what it means.

**What "self-correcting" requires.** Bayesian updating corrects itself only when it sees outcomes, such as a
board confirmation or rejection. Without feedback, updating on its own outputs makes it more confident in what
it already believed, and errors compound. In this design the board is the feedback channel that makes the loop
self-correcting.

## 3. Issues with plain naive Bayes over video
- **Correlated evidence.** MediaPipe gesture, Envision and face all read the same frames, and consecutive
  frames are near-duplicates. Multiplying them as if independent counts the same evidence many times, so the
  posterior becomes overconfident. Fix: a tempering exponent λ_s ∈ (0, 1] per source (tempered likelihood).
- **Time.** Use a **recursive Bayesian filter (HMM forward pass)**:
  b_t(k) ∝ P(o_t | k) · Σ_j T(j→k) · b_{t−1}(j), with a sticky transition (stay ≈ 0.9–0.97) and a "no intent"
  state. Evidence accumulates while a signal is held and decays after it ends, instead of multiplying 30
  frames per second forever. This is the principled form of "the video keeps updating the analysis".
- **Calibration.** Detector scores (MediaPipe gesture scores, LightGBM outputs) are not probabilities until
  checked. Use temperature or Platt scaling on a small labeled clip set, and plot reliability curves.

## 4. The decision on top of the distribution
- **Bayes decision rule (no training):** pick the action with the lowest expected cost. With cost c_wrong
  for a wrong phrase and c_abstain for falling back to the board, commit iff max p ≥ 1 − c_abstain / c_wrong.
  Example: if a wrong phrase is 5× worse than asking, the threshold is 0.8. The threshold follows from costs
  you choose.
- **Classical ML (Sanjith's idea):** logistic regression or boosting on features of the distribution (top p,
  margin, entropy, number of agreeing sources, time since signal onset) → P(top choice is correct). This needs
  labeled outcomes, and the board logs provide them. Start with the cost rule; switch once roughly 100+ logged
  decisions exist.

## 5. Envision: what is available beyond yes/no (from the source)
1. **Continuous P(Gesture) per frame.** LightGBM `predict()` returns probabilities; yes/no only appears after
   a 0.5 threshold. It's usable as a time series: peak, area, time above a level.
2. **The 100 window features, computed live** in `extract_sequence_features`: key-joint positions and
   velocities, wrist speeds, ranges and accelerations, smoothness, finger positions relative to the wrist,
   inter-finger distances, wrist height relative to the shoulders, wrist spread, arm extension, total motion,
   symmetry, visibility. These are continuous, mostly metric (meters, from world landmarks).
3. **CNN model (25 frames ≈ 1 s), a 3-way distribution:** sigmoid `has_motion` × softmax(Gesture, Move) →
   P(NoGesture), P(Gesture), P(Move). It's built for offline use, but the model is small. Live use on a rolling
   25-frame buffer is untested. Out-of-sample gesture F1 averaged 0.72, vs 0.61 for LightGBM.
4. **Per-segment kinematics** (`utils.compute_kinematic_features`): peak and mean speed, peak acceleration and
   jerk, submovement count and amplitude, hold count and time, McNeillian space zones, max height, volume,
   duration. Available once a gesture segment ends.
5. **DTW similarity** (`shapedtw`, used for its gesture-similarity space). This is the route from "a gesture"
   to "which gesture" without fine-tuning: enroll 3–5 examples of a user's movement, then use
   likelihood ∝ exp(−DTW(segment, template) / τ). It feeds Bayes directly.

**Continuous, regression-style use.** Two ways to turn continuous features into evidence:
- **Generative:** class-conditional Gaussians per feature or feature group (Gaussian naive Bayes, conjugate
  updates).
- **Discriminative:** logistic regression P(k | x) = softmax(Wx), which needs labels.

Visual: time series of P(Gesture), kinematic traces, and the phrase posterior as a stacked area over time.

## 6. Validating message.txt's gesture recognizer
What it gives per hand: scores for 8 classes (None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up,
Victory, ILoveYou), handedness, and 21 landmarks (image and world). The scores are not calibrated.

Ways to validate it:
1. **Geometric cross-check from the same landmarks:** finger extension (tip-to-wrist vs PIP-to-wrist), thumb
   direction vs vertical. Thumb_Up should mean thumb extended upward with the other fingers curled. Agreement
   raises the source's weight; disagreement lowers it. Envision's finger features measure the same kind of
   thing.
2. **Temporal stability:** a deliberate sign persists, and frame-level jitter doesn't. Require persistence, or
   let the HMM filter handle it.
3. **Cross-model coherence:** Pointing_Up + high Envision P(Gesture) + wrist above shoulder hang together. An
   Open_Palm at rest with Envision saying NoGesture is probably a resting hand. Caveat: Envision is weakest on
   static hand shapes, so low Envision does not refute a held thumbs-up.
4. **Calibration set:** 20–30 short clips per sign plus idle. Measure accuracy and a reliability curve, then fit
   a temperature. That turns the scores into q values that behave like probabilities.

## Decisions for Sanjith
- Source of vocal likelihoods: AST, padmalcom, CLAP, Bayesian enrollment on voc2vec, or a combination.
- Whether to replace frame-by-frame naive Bayes with the HMM filter plus per-source tempering.
- Decision layer: cost-derived threshold now, learned decision later?
- Which Envision outputs enter Bayes: P(Gesture) series, CNN 3-way, kinematic features, DTW templates.
- Gesture validation: geometric check, persistence, calibration set.
