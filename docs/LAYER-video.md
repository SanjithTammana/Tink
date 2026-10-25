# Video layer: independent test lab

2026-10-25 continuation of Claude session `4ba2d906-7751-435c-920a-59c189c936d2`.
Sanjith's boundary: test layers separately, then he decides next steps. This lab does not change
`run.py`, the live `video.py`, audio, fusion, mapping, phrase ranking or TTS. No environment upgrade.

## Test it

Run in PowerShell:

```powershell
Set-Location C:\Users\sanji\PycharmProjects\tink-alt
& ..\tink\.venv\Scripts\python.exe tools\video_layer_lab.py --camera 0 --record
```

This opens **live tracking while recording**: a body skeleton, face points/eye and mouth contours,
and hand bones on the measured video frame. A separate side panel shows head angles, blink/jaw/brow/
smile bars, hand-shape scores, pinch/finger geometry, arm geometry, tracking status and measured
preview FPS. Camera capture/recording runs independently while inference processes the latest
frame. The raw saved video includes frames skipped by inference; those skips are counted.

Everyone detected gets a numbered box. **Click the intended user's box, or press N to cycle users.**
Only that person's crop feeds the face, hand and pose models. The selected box is green; other
people retain boxes but do not contribute feature events. There is no automatic largest-person
fallback. A lost user withholds observations; overlapping boxes or a large track jump clear the
selection and ask you to select again after separation. These are local tracker IDs, not face
recognition or guaranteed identity through occlusion. Any detected box intersection is treated
conservatively as ambiguous, so crossed arms/bodies may pause the lab rather than mix sources.

Stay neutral for the first 2-3 seconds **after each selection**, then record separate repetitions
of a few movements you want to test. A target change/loss resets the pose windows, candidate
events, MediaPipe tracking state and neutral baseline; baselines are also saved by track/generation.
Press a number **before** starting a labeled interval, return to
`0` for rest/ordinary movement, and `Q` to finish. Labels are descriptions of your demonstrations:

| Key | Trial label |
|---|---|
| 0 | idle, including ordinary talking, looking around and fidgeting |
| 1 / 2 | thumb up / thumb down |
| 3 / 4 | nod / shake |
| 5 / 6 | eyebrow raise / mouth opening |
| 7 / 8 | wave / point |
| 9 | another voluntary signal |

On exit the live recorder prints the exact PowerShell command to analyze every frame of that clip,
if you want a complete temporal pass. By default, the
analysis runs pose, face and hands on every recorded frame. Analysis may take longer than the
clip's duration; its temporal windows use recorded capture times, not analysis speed. The label
and host-read timestamp sidecar stays alongside the MP4 and is read automatically. Sensor
exposure timestamps are unavailable; host-read times are the stated clock. MP4 playback uses a
nominal encoding FPS, so the sidecar, not playback duration, controls this test.

To inspect just one channel, append `--channels face`, `--channels pose` or `--channels hands` to
that printed analysis command. For a live **feature preview** of one channel:

```powershell
& ..\tink\.venv\Scripts\python.exe tools\video_layer_lab.py --camera 0 --channels face
```

All three channels are available live; use `--channels face`, `--channels pose` or `--channels hands`
to inspect one at a time with less inference work. Full-layer live inference can undersample the
pose stream: the panel says `warming / source gaps` rather than manufacturing an Envision score.
Recorded analysis remains the complete temporal test. Do not mistake preview FPS for camera FPS
or event accuracy. Optional `--record-only` still records without models or tracking overlays.

## What to inspect and report

- Whether the movement you performed changes the expected measured feature, and whether idle
  movements change it just as strongly. Repeat movements with a return to rest between them.
- Missing tracking, a signal that never activates, duplicate candidate events, and slow display.
- For pose, watch `motion` and `P(G)` after approximately one second of valid source history.
  `warming/gap` means there is not enough contiguous pose history; it is an abstention, not zero.
- Face preview shows the strongest coefficients. The first neutral interval supplies a baseline;
  if you move during it, rerun the trial. A saved neutral baseline is not probability calibration.
- A head turn is currently a **posture deviation** measurement, not a recognized nod or shake.
  Wave/point/head labels are ground truth you supply, not claims that these recognizers exist.
- Timing thresholds are lab defaults. Short spikes below 0.25 s are dropped. An event closes
  after 0.3 s below its release threshold. EOF/tracking loss closes it as incomplete. Held signals
  emit once at release, rather than repeatedly each frame. This delay is distinct from detection.

Outputs are private under `.scratch_video_layer/<trial>/` and additionally ignore themselves:

| File | Contents |
|---|---|
| `capture.mp4`, `capture_times.jsonl` | optional original capture and operator labels/times |
| `manifest.json` | source, clock, channels, attribution and assumptions |
| `neutral.json` | median neutral face coefficients/angles, if calibration completed |
| `neutral-track<ID>-g<generation>.json` | selected track's distinct neutral baseline |
| `observations.jsonl` | one observation per analyzed frame, plus completed/incomplete candidate events |
| `summary.json` | frame/event count and observed model costs; no accuracy claim |

Raw JSON includes image-space landmarks as well as the detailed features. The live side panel
is an inspection surface, not calibrated intent recognition. No automatic test
score is reported from unlabeled clips. Bring back the trial folder and your observations before
another model, threshold change, fusion change or integration is chosen.

## Actual outputs exposed

**Pose / Envision:** all 33 world landmarks; shoulder width; visibility; wrist height and relative
position in shoulder widths; elbow angles and extension ratios. Resample the first 23 landmarks
`(x,y,z,visibility)` to a 25 Hz grid and preserve the package's 5-frame/100-feature LightGBM input.
Expose the complete 100-value input vector, LightGBM `P(Gesture)`, and all three CNN outputs:

```
motion  = model score for motion
gesture = model score for Gesture conditional on motion
move    = model score for Move conditional on motion

unconditional model distribution:
NoGesture = 1-motion
Gesture   = motion*gesture
Move      = motion*move
```

These model probabilities are not calibrated for this user or equivalent to communication
intent. Preserve the conditional heads instead of reporting `gesture=.9` as unconditional .9.
Do not use the poor Move head as a dependable fidget detector. Do not change the port's feature
ordering, averaging or historical units under fixed weights. The new physical geometry has its
own names; it does not silently replace trained features.

**Face:** 52 named independent coefficients, scale-corrected pitch/yaw/roll, per-eye aspect ratio
with image-aspect correction, and coarse iris ratios. Coefficients do not sum to one and are not
probabilities of voluntary intent. A few sustained brow/eye/mouth shapes have candidate event
gates; all 52 raw values are exposed. Facial activation does not depend on arm/body activation.
The strongest face coefficients in ordinary speech are useful negative examples.

**Hands:** all returned canned gesture scores, both world-landmark sets, handedness scores,
thumb/index pinch normalized by palm width, ten joint angles and five extension ratios. The
combined event channel takes the strongest observed non-None shape; it is not hand-identity
tracking or temporal sign recognition. Handedness labels are retained, not used to assert
anatomical body ownership. Static shape recognition does not recognize waves or full ASL.

**Events:** session/target-generation-unique ID, selected local tracker ID, source, capture
start/end, emission time, sample count, mean/peak raw scores, explicit uncalibrated flag and
completion reason. `--person-id` is only an optional descriptive label. Actual event ownership
comes from click/N selection of a visible isolated person track. Full-frame image landmarks
are mapped back from the selected crop; world landmark geometry remains crop/model-native.
There is no biometric identification, audio-speaker attribution, intent mapping or calibrated
fusion likelihood here. Headless tests must specify `--target-id <visible tracker ID>`.

## Measurement evidence and limits

The operation shown in Sanjith's screenshot **finished** before the original session hit quota:

| Fresh-process model run | Median ms | p90 ms | Warm frames |
|---|---:|---:|---:|
| Pose lite | 94.4 | 116.2 | 568 |
| Pose full | 142.5 | 167.8 | 568 |
| Face | 121.7 | 243.9 | 568 |
| Hands | 162.4 | 269.5 | 568 |

Balanced power plan, battery status 2. Audio/research jobs were also active, so fresh process
does not imply an otherwise idle CPU. These are observed loaded-machine results. The research
report's 25-35 ms pose estimate was not a local measurement and must not replace them.

Recovered earlier same-clip Holistic/Pose comparison: Pose full CNN-motion correlation .990,
conditional Gesture correlation .899, LightGBM Gesture correlation .950 against Holistic.
Correlation on one unlabeled clip establishes neither accuracy nor threshold equivalence.

Continuation smoke test: all three channels completed 32 frames of the existing 20 FPS clip,
producing 12 model windows, each with 100 LightGBM features, 52 face coefficients, one detected
hand frame, and one incomplete head-posture event at EOF. This verifies execution and data shape,
not communication performance. Eighteen camera-free contract checks cover capture-time resampling,
tracking gaps/slow sampling, conditional heads, one event per hold, IDs, incomplete events,
rotation scale/invalidity, shoulder-relative geometry, a synthetic recorder round trip and
direct feature-vector parity against the installed package's pure NumPy method over 18
length/stride combinations, overlay alignment/clearing and continuous recording of frames skipped
by inference. A repeated 32-frame clip run checks the actual model-to-overlay path and saves an
annotated preview with `--save-preview`. The physical webcam interaction is
for Sanjith to test. macOS installation/runtime remains untested; the demo's legacy requirements
have not been migrated. The portable Envision classifiers use NumPy/LightGBM without importing
the Envision package or TensorFlow; this Windows environment's MediaPipe import can still load
its installed optional TensorFlow dependency and print startup warnings.

Linear interpolation preserves the model's temporal units; it cannot reconstruct fast motion
missed by a slow camera/model. Gaps over 0.2 s reset pose history. No raised FPS is claimed.

Multi-person continuation: a 48-frame run of the existing `two_people.mp4` fixture detected two
tracks on every frame; all observed features/events remained on explicitly selected track 1,
with 28 Envision windows. Overlay geometry was visually checked on the selected person's crop.
No manual switching or real camera/microphone interaction is claimed by that clip check. Tests
cover no fallback on loss, reacquisition generation changes, overlap including small strips,
large jumps and crop-coordinate mapping. The local YOLO person model is loaded from the original
checkout; `--person-model <path>` overrides it without a download. Additional tracking has a
CPU cost; per-frame timing is exposed rather than promising a particular live rate.

## Research to consult after the test

Detailed body/hand report and recovered face/head report are in the private ClaudeOS research
folder. The latter inventories all 52 coefficients, geometry, calibration/rejection choices,
known uncertainty and primary links. High-value next comparisons, **not implemented decisions**:
personal temporal nod/shake templates, held hand shapes, deliberate-versus-idle rejection,
finger-angle prototypes and coarse calibrated pointing. The user chooses after this test.

Primary references: [Envision implementation](https://github.com/WimPouw/envisionhgdetector),
[FaceLandmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker),
[GestureRecognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer).
The supplied Pouw/Shaikh preprint and installed 3.0.8 package were inspected by the recovered
session. Its reported random-sequence classifier parity did not independently prove the full
capture-to-feature path or Mac support.
