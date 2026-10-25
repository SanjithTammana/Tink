# Improving interpretation: LLM layer vs. cheap math (2026-10-24)

Read-only research for Sanjith's decision. Nothing here was run on Tink's data. Tags: **[E]** evidence from a source
listed at the bottom, **[X]** extrapolation from a cited number, **[J]** my judgment. Numbered refs like [S3] point to Sources.

## 0. Diagnosis (from the repo)

- `PhraseRanker.rank` scores each phrase as `profile @ posterior`, with `embed_weight=0`, so a phrase's profile is its intent tag.
  On a flat posterior every phrase of every intent scores the same, and the top 3 become the first phrases in file order (plus the MMR term).
  That is the "weak suggestions" symptom. [E: retrieval.py]
- `config/likelihoods.json`: in `mp_gesture`, the rows for Help, Repeat, Want and Break are uniform (0.125 for every label). No gesture label
  can ever raise those four intents. `envision` and `vocalization` are neutral too. So from video alone the posterior can only move among
  Yes, No, Attention and Show; the other four move only through Whisper keywords. [E: likelihoods.json]
- `BayesianFusion.infer` is a memoryless snapshot: every call recomputes from the prior. Today that causes flicker, not double counting. Double
  counting appears the moment a recursive filter is added (section B). [E: audio_text.py]
- Output flatness is mostly an information problem. An LLM cannot recover intent the sensors did not capture. What an LLM does add is
  **conversation context** (what the partner just said), which is new information the pipeline does not use today. [J, supported by C1/C2 below]

## A. LLM-based interpretation

### A1. What the literature says

| Work | What it did | Finding relevant to Tink |
|---|---|---|
| Cai et al., Nat. Commun. 2024 (SpeakFaster) [S1] | Fine-tuned 64B LaMDA expands typed word initials into phrases, using prior dialogue turns including the partner's. 5 options per call; the user picks, adds letters, or types. | Simulated keystroke savings 0.657 vs 0.482 for a Gboard baseline. Two eye-gaze ALS users: entry rate 29-60% above baseline. Real-world savings (0.32-0.36) were far below simulation because of typos and misselections. Reviewing options costs seconds (about 12.7 s per scripted abbreviation-expansion round), more than the ~0.8 s model call. Without conversation context, single-call success roughly halved. The authors flag partner-speech ASR as a privacy concern. |
| Valencia et al., CHI 2023 "The less I type, the better" [S2] | 12 AAC users tried live LLM suggestions (extend a short reply, answer a biographical question, ask for help). | Users expected savings in time and effort but required phrases to match their own style. The paper flags a shift from composing to prompting-and-selecting (agency). |
| Valencia et al., CHI 2024 COMPA [S3] | Chrome extension for Google Meet: live transcript, user marks which turn they reply to, LLM generates starter phrases from that context plus a chosen intent (reply, opinion, question, yes/no). | Closest published analogue to Tink's "intent, then phrases" design. Details here come from a secondary summary; I did not read the paper. |
| AACessTalk, CHI 2025 [S4] | Tablet app for minimally verbal autistic children and parents. LLM proposes topic and action cards; emotion cards and core cards (Yes, No, I don't know) are fixed lists. | Safety pattern worth copying: the core vocabulary is not LLM-generated; at most 3 suggestions; a refresh button; suggestions are hidden until tapped. No quantitative accuracy of the generated cards was visible in the text I read. |
| Humor-in-AAC paper, CHI 2025 [S5] | AI-powered timely comments for AAC. | Authors warn that heavy reliance on AI auto-complete could reduce spontaneous, less predictable expression. A flagged risk, not a measurement. |

**Gap:** I found no 2024-2026 paper that infers intent from camera or gesture signals and passes it to an LLM for AAC. Searches returned only
adjacent HRI and virtual-human gesture work. Tink's combination is unpublished as far as I could find; that is an absence in my searches, not proof. [E, weak]

**Failure modes named in these papers [E]:** style mismatch with the user's own voice; agency shifting to the machine; overreliance; model
errors costing review time; losing conversation context; ASR errors in partner speech; privacy of partner speech. **Hallucinated intent** is
my extrapolation to Tink: an LLM given a flat posterior will confidently invent a plausible desire (for example "I'm thirsty"). [J]

**How they kept the user in control [E]:** the user always selects from a short list (3-5); manual fallback is always available; core
vocabulary is fixed; the user marks the context or intent rather than the system guessing it (COMPA).

### A2. Design that fits Tink (keeps the user's voice)

The LLM is a wording and re-ranking layer on top of the posterior, never an intent oracle. [J]

1. **Gate.** Call the LLM only when there is something to condition on: the partner's transcribed last utterance changed, or the posterior
   is peaked (top p >= ~0.4 or p(none) low). Never call it on a flat, no-evidence posterior; show "waiting" or the retrieval top-3.
2. **Prompt contents.** Partner's last 1-2 utterances (Whisper), top-3 intents with probabilities, which signals fired (for example
   "thumb up held 1.2 s, nod"), 10-20 of the user's own confirmed phrases as style exemplars, and the 8-12 retrieval candidates. [J]
3. **Constrain output.** Prefer **selection by id** over free text: `{"picks":[id,id,id]}` with `id` an enum over the retrieval candidates.
   This cannot hallucinate a new desire. Optionally allow one slot of free text, capped at about 10 words (SpeakFaster's limit is about ten
   words [S1]). Use a JSON schema: Ollama `format` takes a schema [S6]; llama.cpp converts a JSON-schema subset to a GBNF grammar and
   `llama-server` accepts `response_format` [S7]; a hosted model can use a forced tool call with the same schema. Set temperature 0. [E for the mechanisms]
4. **Presentation.** Slot 1 = LLM pick, slots 2-3 = the existing retrieval ranking (diversified), key 0 = none. Never speak without a key
   press. Label slot 1 as "context-aware". Log shown/chosen to `data/confirmations.jsonl`. This also gives the data for the logistic ranker.
5. **Failure fallback.** Timeout (about 2 s hosted, about 6 s local) means the retrieval-only result, silently. [J]

Sketch of the call: `system` (fixed rules, schema, "choose only from candidates, never infer needs not supported by the evidence, 'none' is valid")
+ `user` JSON `{partner_said, intents:[{name,p}], signals:[...], candidates:[{id,text}], my_style:[...]}`.
About 400-600 prompt tokens, about 30-50 output tokens. [J]

### A3. Local feasibility (laptop CPU / Apple Silicon)

No direct Qwen2.5 / Llama 3.2 laptop benchmark was found, so the speeds below are **extrapolations [X]** from the cited numbers.
Token generation is memory-bandwidth bound, so speed scales about inversely with weight bytes; prompt processing scales similarly.

Measured anchor: llama.cpp Apple Silicon table, LLaMA-7B Q4_0, base chips [S8]:

| Chip | Prompt processing (t/s) | Generation (t/s) |
|---|---|---|
| M1 (8 GPU cores) | 118 | 14.2 |
| M2 (10) | 180 | 21.9 |
| M3 (10) | 187 | 21.3 |

A 7B Q4_0 file is about 3.6 GB. Weights below are my estimates at Q4_K_M. [J]

| Model | Weights (est.) | M1 gen (est. t/s) | M2/M3 gen (est. t/s) | Est. latency for 500 in + 45 out, M1 / M2 |
|---|---|---|---|---|
| Qwen2.5-0.5B | ~0.4 GB | 100+ | 150+ | <1 s |
| Qwen2.5-1.5B / Llama 3.2 1B | 0.8-1.0 GB | ~50 | ~75 | ~2 s / ~1.5 s |
| Qwen2.5-3B / Llama 3.2 3B / Gemma 2B | 1.6-2.0 GB | ~25 | ~38 | ~4 s / ~3 s |
| Phi-3.5-mini / Phi-4-mini (3.8B) | ~2.4 GB | ~20 | ~30 | ~5 s / ~3.5 s |

- Windows x86 CPU: AMD reports up to 50.7 t/s for Llama 3.2 1B (4-bit) on a Ryzen AI 9 HX 375 via LM Studio's llama.cpp [S9] (vendor
  number; GPU offload added about 31%). A roundup of search results suggested roughly 15-50 t/s for 1B-class models on current x86 laptops;
  I did not find a primary source for that range. A 3B model is about 2-3x slower, so about 7-20 t/s and about 5-10 s per call on an
  ordinary laptop. [X]
- RAM: 2-3 GB for a 3B Q4 model plus KV cache; fine on 16 GB, tight next to Whisper + voc2vec (HuBERT-base, about 95M parameters) + YOLO + MediaPipe. [J]
- CPU contention is the real risk: Tink already runs the camera pipeline at low fps on CPU (PLAN.md says about 1 fps for the full pipeline). A
  3B decode competes with Whisper and MediaPipe. Run the LLM in its own process, event-triggered, lowest thread count (2-3), and
  expect the video fps to dip during a call. Same lesson as the earlier long-local-compute note. [J]
- Quality at 0.5B-1.5B is poor for context-aware phrasing; with selection-by-id the demands drop a lot, so 1.5B-3B is plausible. I have no
  benchmark for this task; test on 10 scripted partner utterances before committing. [J]
- **What `ollama list` shows here (read-only check):** `deepseek-v4.1-flash:cloud`, `glm-5.3-flash:cloud`, `glm-5.2:cloud`, `minimax-m2.7:cloud`,
  `gemma4:cloud`, `minimax-m3:cloud`. **No local models are pulled.** The `:cloud` tags run on Ollama's servers, so they are a hosted
  option with the same wifi and privacy risk, not a local one. Ollama's structured-output docs say cloud does not currently support structured
  outputs [S6], so these would need prompt-only JSON with validation. [E]

### A4. Hosted option (for example Claude Haiku)

| Item | Value |
|---|---|
| Price per the Anthropic pricing page [S10] | Haiku 4.5: $1 in / $5 out per MTok. Haiku 5.5: listed at $0.10 / $0.50 per MTok for prompts up to 100k tokens (the page lists this; check the live page and the model id before use). |
| Cost per call (500 in, 60 out) | Haiku 4.5: about $0.0008. Haiku 5.5: about $0.00008. A whole demo day is cents. [X] |
| Latency | Not measured here. Short-output calls are typically sub-second to ~1.5 s over good wifi. [J] |
| Risks | Venue wifi is the dominant risk (hackathon networks are congested); an API key in a demo laptop; sending the partner's transcribed speech to a third party needs consent from anyone speaking near the mic (SpeakFaster authors flag the same concern [S1]). Send text only, never video/audio. [J] |
| Mitigation | Pre-recorded fallback, 2 s timeout, retrieval-only fallback, and a visible "LLM on/off" toggle. Pre-compute LLM answers for the scripted demo partner utterances as a cache. [J] |

### A5. Verdict on A

- Highest-value use is **partner-speech context** (what was asked) plus wording, not decoding the weak sensor signal. [J, consistent with S1]
- Local 1.5-3B is feasible on M2/M3 and marginal on a typical Windows laptop while the vision/audio stack is running. Hosted Haiku-class is
  faster and better but fails when wifi fails. The most robust demo setup is hosted with the retrieval-only fallback, rehearsed offline mode
  on whichever machine will present. [J]

## B. Cheap math for detection and fusion

All of these are CPU-trivial; cost is engineering time.

| # | Technique | Cost (compute / time) | Evidence status |
|---|---|---|---|
| B1 | Recursive Bayes filter with "none" state | microseconds / ~1 h | Method standard [S11]; benefit for Tink [J] |
| B2 | Per-source likelihood tempering | none / 15 min + a fit | Domingos and Pazzani show naive-Bayes ranking can survive dependence while probabilities do not [S12]; the exponent trick is standard practice [J] |
| B3 | Head nod/shake from FaceLandmarker pose | <0.2 ms / ~2 h | Task established [S13, S14]; thresholds are mine |
| B4 | Hand-landmark geometry | <0.1 ms / ~1.5 h | Standard heuristics [J]; landmark layout is MediaPipe's [S15] |
| B5 | Hold (dwell) and periodicity | <0.1 ms / ~1 h | Hold time is a kinematic feature in the Envision paper per earlier notes in RESEARCH-bayes-gap.md; the rest [J] |
| B6 | Temperature / Platt calibration; count-based table fit | seconds / ~1 h + clips | Temperature scaling [S16]; enrollment counts [J] |
| B7 | Few-shot gestures via template matching / kNN | <1 ms / 3-4 h | $1 recognizer result [S17]; transfer to hand landmarks [J] |

### B1. Recursive filter (HMM forward pass) with a "none" state

States: the 8 intents plus `none` (no communicative act). With belief `b` (length 9) and observation likelihood vector `L_t(k)` from all fresh sources:

```
b_pred = T^T b_{t-1}                 # T[j,k] = P(k_t | j_{t-1})
b_t    = normalize(L_t * b_pred)
T      = s*I + (1-s) * row-normalized(prior mass, with extra mass into none)   # stay s = 0.90-0.97 per update step
```

- Evidence accumulates while a signal is held and fades after it stops (half-life = ln(0.5)/ln(s) steps; at s=0.95 and 5 Hz, about 2.7 s).
- `none` needs rows in each source table: it explains `None` gestures, `NoGesture`, silence. Gate suggestions on `b[none] < 0.5`; otherwise
  show nothing or "waiting". This directly addresses "flat posterior gives junk": a flat result becomes an explicit non-answer, and the LLM is not called.
- Add an **evidence-onset reset**: when a new strong observation disagrees with the belief, raise the escape probability (1-s) for one step so
  a new gesture can take over quickly. [J]
- Integration: `BayesianFusion` already builds a per-source log-likelihood vector; the filter replaces `_start(None)` with `T^T b_prev` and keeps
  `b` in `run.py` state. About 30 lines.

### B2. Likelihood tempering (stop double counting)

```
log L_t(k) = sum_s  lambda_s * log( sum_o P(o|k) q_s(o) )
lambda_s   = min(1, dt / tau_s)      # tau_s = evidence correlation time: ~0.5-1 s for mp_gesture, envision and face (same frames)
```

- Needed once B1 is on: at 5 Hz updates with 1 s correlation, lambda about 0.2-0.3, so one held second counts about as one observation.
- Fit `lambda_s` as one scalar per source by minimizing log-loss on a few labelled clips (the same procedure as temperature scaling with
  T = 1/lambda). Without clips, use the formula above and sanity-check that a held gesture takes about 1-2 s to reach p > 0.8. [J]
- Different modalities (voice vs hand) are closer to independent; give them lambda near 1 relative to their own update rate. [J]

### B3. Head nod / head shake as yes/no evidence

**Setup (cheap):** `video.py` already creates `FaceLandmarker` with `output_face_blendshapes=True`. Add
`output_facial_transformation_matrixes=True` (default False); the 4x4 matrix maps a canonical face model to the detected face [S18]. No new model.

**Algorithm [J]:**
1. Extract rotation `R` from the matrix; compute pitch and yaw (for example `yaw = atan2(R[0,2], R[2,2])`, `pitch = asin(-R[1,2])`, signs depend on the camera convention; verify by nodding at the webcam).
2. Subtract a slow baseline (running median over 2-3 s) so absolute posture and camera angle do not matter. [S19: developers asking for nod thresholds hit negative or offset angles for exactly this reason.]
3. Over a 1.5 s window: count reversals (sign changes of velocity with hysteresis delta). Nod = at least 2 pitch reversals with peak-to-peak >= ~4-6 degrees and yaw
   amplitude < half of pitch amplitude. Shake = same on yaw, amplitude >= ~6-8 degrees. Oscillation band 1-4 Hz; check with autocorrelation peak or a
   band-pass energy ratio as a second opinion.
4. Output a soft label `{nod, shake, still}` and add a `head` source to the config. Illustrative table (placeholders): `P(nod|Yes)=0.6, P(nod|No)=0.03, P(shake|No)=0.6, P(shake|Yes)=0.03, others near-neutral`.

**Literature [E]:**
- Kapoor and Picard (2001) built a real-time nod/shake detector from pupil tracking and HMMs and reported 78.46% real-time accuracy on their test set [S13].
- Morency et al. (2007, Artificial Intelligence) show that adding **dialog context** improves nod/shake recognition [S14], which supports using the partner's yes/no question as a prior for the `head` source. I did not read the paper's numbers.
- Kawato and Ohya (FG 2000) exists as "Real-time detection of nodding and head-shaking by directly detecting and tracking the between-eyes" [S20]; I did not retrieve its abstract.

**Threshold evidence is thin and informal [E, weak]:** a vendor demo uses a 5 degree yaw change inside 1500 ms for a "no" shake [S21]; a research
system used a frame-difference threshold of about 4 (units unstated) [S19]. My ranges above come from those plus judgment; **tune on 10 of your own clips per class.**

**Constraints:**
- Needs enough frames per second. The pipeline note says the full stack ran about 1 fps; head oscillation at 1-4 Hz needs at least 10-15 fps of face
  pose. `ThreadedVideoAnalyzer` already runs MediaPipe on its own worker; check the `mediapipe` rate on the panel and, if low, run FaceLandmarker alone at reduced resolution. [J]
- Failure cases: talking and chewing produce small pitch oscillations (require amplitude and 2+ reversals); users with involuntary head movement will
  false-trigger (user-calibrated thresholds, never decisive on its own); head-shake means "yes" in some cultures (config per user, since the table is editable). [J]

### B4. Hand-geometry validation of GestureRecognizer labels

Landmarks: 21 per hand, wrist=0, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20 [S15].

```
extended(f)  = |tip_f - wrist| > 1.15 * |pip_f - wrist|           # index/middle/ring/pinky, tip=8/12/16/20, pip=6/10/14/18
thumb_ext    = |tip4 - mcp5| > |ip3 - mcp5| * 1.1  and  angle(4-2 vs 5-0) > 30 deg
point_vec    = (8 - 5) / |8 - 5|     -> angle vs vertical; use world landmarks to avoid perspective
thumb_dir    = (4 - 2) / |4 - 2|     -> up if its angle to image-up < 45 deg, down if > 135 deg
palm_facing  = sign of (5-0) x (17-0) z-component with handedness
```

- **Rules:** Thumb_Up needs thumb extended and pointing up, others curled. Thumb_Down the reverse. Open_Palm needs five extended. Pointing_Up needs only the index extended.
- **Use as soft evidence:** compute a geometric label distribution `g` and combine with the recognizer label `r`. Agreement leaves the term; disagreement
  flattens `q` toward uniform (`q' = m*q + (1-m)/K`, with m about 0.4 on disagreement). Or promote it to its own `geom` source. [J]
- **Bonus:** geometry can give new, intent-bearing shapes the recognizer lacks: pinch (4 and 8 close), flat palm facing camera and raised (stop/break), fist, two-hands-raised (help). That opens evidence for Help/Break/Want, which currently have none. [J]

### B5. Hold (dwell) and periodicity

- **Hold:** wrist speed in shoulder-widths per second is already computed in `PersonTracker.kinematics`. Define `hold_time h` = seconds speed < ~0.15 sw/s with a
  constant gesture label. Use a ramp as the tempering exponent: `lambda_hold = min(1, h / 1.0 s)`. A passing hand shape gets low weight; a deliberate hold gets full weight. Require the hand
  to be in gesture space (`raised`, above hip), not resting on the lap. [J]
- **Periodicity (wave, knock, repeated tap):** autocorrelation of wrist x or y over 2 s; accept if normalized r > ~0.5 at a lag of 0.2-0.8 s (1.25-5 Hz) and amplitude > ~0.15 shoulder widths. Map to Attention/Show (waving) with a `periodic` source. [J]
- **Repetition as emphasis:** the same label held, released and re-held 2-3 times within 3 s increases confidence (second hold multiplies `lambda` by 1.5, cap 1). [J]
- Needs a steady wrist trace at 10+ fps; at 1 fps nothing periodic is detectable. [J]

### B6. Calibration and filling the placeholder tables

- **Temperature** (Guo et al., ICML 2017): `q = softmax(z / T)` with a single T per source fit by minimizing NLL on held-out labelled data [S16]. For binary Envision P(Gesture),
  Platt: `sigmoid(a * logit(p) + b)`, two parameters. Minimum practical data: about 10 clips per class (20-30 is better). Check with a reliability table in 5 bins. [E for method; sample sizes J]
- **Count-based enrollment (cheap, probably the biggest single fix):** ask the user to perform each intent 5 times (about 40 trials, 5-8 min). For each trial run the detectors and record the label
  distribution. Then `P(o | k) = (n_ko + a * P0(o | k)) / (n_k + a)`, where P0 is the current hand-set row and `a` is about 5. This is MAP with the hand-set table as a prior: it keeps your design where data is thin and
  replaces it where data speaks. It gives Help, Repeat, Want and Break non-uniform rows as soon as you choose a gesture for each. [J]
  **Note:** PLAN.md records that Sanjith removed `learn_likelihoods` from the pipeline ("no learn_likelihoods, not self-correcting"). A one-time enrollment is a calibration step, not online self-correction, but whether to allow it is his call.
- Calibrate at the venue: lighting and camera angle change score distributions more than the model does. [J]

### B7. Few-shot user-defined gestures

- **Static hand shapes:** feature = 21 landmarks, subtract wrist, divide by |0-9|, flatten (63-d, mirror left hands). Train kNN (k=3) or logistic regression on 10-20 frames per class (5 s of recording per class). Likelihood from softmax of
  negative distances or `predict_proba`. [J]
- **Dynamic gestures:** resample the wrist-relative trajectory to 32 frames, DTW or plain Euclidean, `L(k) ∝ exp(-d_k / tau)` where tau is the median leave-one-out within-class distance.
- Reference result: the $1 recognizer (2D strokes, not hand pose) reached over 97% accuracy with one template and 99% with three or more, close to DTW [S17]. Transfer to hand landmarks is a hypothesis. [E for strokes, J for transfer]
- Best use for a user whose gestures do not match MediaPipe's 8 labels, which is the AAC situation. In 24 h, do static shapes only. [J]

## C. Recommended path for <24 h

Order by expected impact per hour. Impact figures are judgment, not measured.

| Step | Do | Time | Expected impact | Risk |
|---|---|---|---|---|
| 1 | **Fix information, not the model.** Assign a gesture/signal to each of the 8 intents; run 5-trial enrollment (B6) to fill/replace the tables; add `none` state. Provide Help/Repeat/Want/Break with at least one handle (geometry shapes B4, keywords). | 2-3 h | Highest: removes the structurally flat cases. | Needs the owner's intent-to-gesture choices ("his call"). |
| 2 | **Head nod/shake source** (B3), with a pitch/yaw trace on the panel for debugging. | 2 h | High for Yes/No, the most frequent AAC intents; the signal is already computed by a running model. | fps too low; thresholds; user-specific. |
| 3 | **Recursive filter + tempering + hold ramp** (B1, B2, B5-hold). Suggestions suppressed when `none` dominates. | 2 h | Medium-high: stability, fewer flickering wrong suggestions; makes the demo look deliberate. | Over-sticky belief feels laggy; tune s and tau live. |
| 4 | **Hand geometry cross-check** (B4) and periodicity for wave (B5). | 1.5-2 h | Medium: filters false labels, adds Attention/Break handles. | Heuristics are brittle to angle; keep as soft evidence. |
| 5 | **LLM re-ranker/wording layer** (A2) over partner speech: hosted Haiku-class with cache + timeout + retrieval fallback; optional local 1.5-3B on an M-series Mac. | 3-4 h | High for demo storytelling when a partner asks a question (context-aware, conversational); low when the posterior is flat. | Wifi, partner-speech consent, latency, CPU contention, perceived agency loss. |
| 6 | Calibration on clips (B6 temperature) and DTW/kNN custom gestures (B7). | if time | Medium / nice-to-have. | Time. |

**Why this order:** steps 1-4 raise the information in the posterior and are deterministic and offline (venue-wifi-proof). Step 5 adds the
only source of *new* information the system lacks (what was said to the user) and the polish of natural phrasing. If time runs out, cut 6,
then 4, then 5; keep 1-3.

**Demo framing [J]:** say the LLM chooses and words among suggestions from evidence, and the user confirms every utterance. Do not claim it reads minds.
The published AAC work frames suggestions as accelerating the user's own message and keeping them in selection control [S1, S2].

**Metrics to log in the demo for credibility [J]:** top-1 and top-3 hit rate on 20 scripted trials per condition (retrieval-only vs +LLM), time to confirmed utterance, and number of "0 = none" presses.

## What I could not verify

- Local tokens/sec for Qwen2.5 / Llama 3.2 / Phi / Gemma on laptop CPU or M-series: no primary benchmark found. All per-model speeds are scaled from LLaMA-7B Q4_0 [S8] and one AMD vendor figure [S9]. Run `llama-bench` or `ollama run --verbose` on the actual machines.
- The "15-50 t/s for 1B on x86 laptops" range came from a search-result roundup, not a primary source.
- Haiku latency: not measured. Haiku 5.5 price comes from the pricing page as fetched; confirm the model id and price before depending on them.
- COMPA (CHI 2024) details are from secondary summaries; Morency 2007 and Kawato and Ohya 2000 were not read beyond abstracts or bibliography entries; the Kapoor and Picard accuracy comes from a search snippet of the MIT Media Lab page.
- Nod/shake thresholds: only informal sources (a vendor demo, a research-system snippet, and a forum question). The numbers in B3 are starting points.
- No published system combining gesture/camera intent inference with an LLM for AAC was found; absence in my searches only.
- Head-shake-means-yes cultural variation: general knowledge, not sourced here.
- The Apple Silicon numbers are for one llama.cpp commit and a 7B model, not current builds.
- The "frame-difference threshold of about 4" was a search-result snippet; I could not pin down which document it came from, so treat it as unsourced.

## Sources

- [S1] Cai et al., Using large language models to accelerate communication for eye gaze typing users with ALS, Nat. Commun. 15, 9449 (2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11530652/ ; summary at https://research.google/pubs/using-large-language-models-to-accelerate-communication-for-eye-gaze-typing-users-with-als/
- [S2] Valencia et al., "The less I type, the better", CHI 2023: https://research.google/pubs/the-less-i-type-the-better-how-ai-language-models-can-enhance-or-impede-communication-for-aac-users/ ; ACM full text https://dl.acm.org/doi/fullHtml/10.1145/3544548.3581560
- [S3] Valencia et al., COMPA, CHI 2024 (DOI 10.1145/3613904.3642762), secondary summaries: https://www.ruofeidu.com/cites/Valencia2024COMPA.html , https://a11y-paradise.onrender.com/reviews/69cf38af9a1ebca89784a0b3
- [S4] Choi et al., AACessTalk, CHI 2025: https://arxiv.org/html/2409.09641v4
- [S5] Why So Serious? Timely humorous comments in AAC through AI-powered interfaces: https://arxiv.org/pdf/2410.16634
- [S6] Ollama structured outputs: https://docs.ollama.com/capabilities/structured-outputs
- [S7] llama.cpp grammars and JSON-schema constrained output: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- [S8] llama.cpp Apple Silicon performance table (LLaMA-7B, PP512/TG128): https://github.com/ggml-org/llama.cpp/discussions/4167
- [S9] AMD, accelerating llama.cpp on Ryzen AI 300 (Llama 3.2 1B, 50.7 t/s, vendor): https://www.amd.com/en/blogs/2024/accelerating-llama-cpp-performance-in-consumer-llm.html
- [S10] Anthropic pricing: https://platform.claude.com/docs/en/about-claude/pricing
- [S11] Forward algorithm (HMM filtering): https://en.wikipedia.org/wiki/Forward_algorithm
- [S12] Domingos and Pazzani, On the optimality of the simple Bayesian classifier under zero-one loss (1997): https://dx.doi.org/10.1023/A:1007413511361 ; discussion https://cdn.aaai.org/FLAIRS/2004/Flairs04-097.pdf
- [S13] Kapoor and Picard, A real-time head nod and shake detector (2001): https://www.media.mit.edu/publications/a-real-time-head-nod-and-shake-detector-2
- [S14] Morency, Sidner, Lee, Darrell, Head gestures for perceptual interfaces: the role of context in improving recognition, Artificial Intelligence (2007): https://datalearner.com/academic/journal-papers/0004-3702/volumes-and-issues/319/paper-detail/68532 ; IUI 2006 version https://groups.csail.mit.edu/vision/vip/papers/morency_iui06.pdf
- [S15] MediaPipe Hand Landmarker (21-landmark layout; not fetched this session): https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- [S16] Guo, Pleiss, Sun, Weinberger, On calibration of modern neural networks, ICML 2017: https://arxiv.org/abs/1706.04599
- [S17] Wobbrock, Wilson, Li, Gestures without libraries, toolkits or training ($1 recognizer), UIST 2007: https://depts.washington.edu/acelab/proj/dollar/index.html ; https://unpaywall.org/10.1145%2F1294211.1294238
- [S18] MediaPipe Face Landmarker (facial transformation matrix option): https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- [S19] OpenCV forum: nod threshold from solvePnP angles: https://answers.opencv.org/questions/177130/revisions/ (the "about 4" frame-difference threshold mentioned in B3 is unsourced, see the last section)
- [S20] Kawato and Ohya, FG 2000 (bibliography entry only): https://dblp1.uni-trier.de/pid/43/2055.html
- [S21] Displayforce head-shake detector example (5 degree / 1500 ms demo threshold): https://support.displayforce.ai/hc/en-us/articles/32630500704541-Head-Shake-Detector-JavaScript-Designer-Example-Project
