# ASL as an evidence source for Tink (2026-10-24)

Read-only research. Nothing was downloaded into the repo and no code was run on Tink data. I read two PDFs in a scratch folder (the PopSign NeurIPS paper and supplement, the YouTube-ASL and GloFE papers) and counted the PopSign word list myself.
Tags: **[E]** stated by a listed source, **[X]** a count or arithmetic I did from a fetched file, **[J]** my judgment. Refs like [S3] point to Sources. Builds on `RESEARCH-datasets.md` (WLASL counts are reused, not redone).

## 0. Bottom line

1. **Weekend target (the ladder's first rung): a closed-set isolated-sign classifier on MediaPipe landmarks with a "no sign" class, feeding the Bayesian layer as soft evidence.** Continuous ASL-to-English translation is not reachable on a laptop CPU in ~15 hours (section 6). [J]
2. **The one big landmark dataset, PopSign / the Google Kaggle ISLR data, has 7 of our 15 core signs exactly (YES, NO, DRINK, WATER, FINISH, PLEASE, THANK-YOU), 2 only as nearest child-register words (`potty` for BATHROOM, `owie` for PAIN), and lacks HELP, MORE, STOP, EAT, WANT, AGAIN.** Those six are the signs most useful for AAC and none is in its 250-word list [X, section 2]. A pretrained model restricted to our signs therefore cannot be the whole answer. [X + J]
3. **PopSign signs are deliberately the one-handed, phone-held variant** that five Deaf DPAN members agreed on, not the full range of ASL forms [E, S2]. That helps the single-hand webcam case and hurts any sign normally made with two hands. [J]
4. **Recommended path: few-shot enrollment (record each sign 3-5 times) as the primary recognizer, with a small GRU/1D-CNN trained on PopSign-style landmarks as an optional warm start for the signs it covers.** Enrollment is the only route that covers all of our core signs and is robust to this user's camera and signing style. [J]
5. **Present it as a keyword-sign pack for users who already sign, validated by a native signer, not as ASL recognition.** Deaf-led work argues strongly against the latter framing (section 7).

## 1. Comparison of approaches

All accuracy numbers are from the cited papers on their own benchmarks and test splits; none transfers to a new webcam, signer, or 15-sign vocabulary. CPU latency figures are **[J]** unless a source is given, because I found no published CPU timings for any of these models.

| Approach | Data needed | Reported accuracy | CPU latency (laptop) | Effort | License / release |
|---|---|---|---|---|---|
| **PopSign baseline: 2x BiLSTM(128), 60x63 hand landmarks, 250-way softmax** [S1, S2] | PopSign v1.0: ~210k examples, 47 Deaf signers, 31/8/8 train/val/test signers | 82.1% val, 84.2% test (250 signs; test signers unusually consistent per the authors) [E] | Tiny model; well under real-time [J] | Retrain: 30-60 min on 3-4 GPUs in the paper; CPU training feasible on a subset [J] | Dataset CC BY 4.0; their software MIT [E]. Trained weights: not found |
| **Kaggle ISLR 1st place: 1D CNN + Transformer on selected landmarks** [S4, S5] | Kaggle competition data (MediaPipe Holistic landmarks, PopSign) | Competition metric not read; a student report says ~80% val on held-out segments [E, secondary]; one paper says the top method reached 89.3% on 250 signs [E, secondary] | Built for TFLite on phone [E]; CPU real-time is plausible [J] | Notebook in repo; retraining needs a GPU [J] | Repo MIT [E]. Pretrained weights in repo: not stated |
| **Kaggle ISLR 2nd place: EfficientNet-B0 on 160x80x3 keypoint images, plus BERT/DeBERTa** [S6] | Same | CV 0.898 single fold, LB ~0.8, ~0.81 with transformers [E] | Not reported end-to-end; a custom TFLite depthwise conv ran 1.24 ms vs 3.70 ms built-in [E] | Heavy ensemble, not weekend-sized | Repo Apache-2.0; `common_weights.npy` / `bad_weights.npy` listed but purpose not explained [E] |
| **Pose-TGCN, WLASL** [S8] | WLASL (scraped YouTube clips) | WLASL100 top-1 55.43%, WLASL300 38.32% (as tabulated in SignBERT+, not re-checked) [E] | Light [J] | Needs dataset acquisition | WLASL C-UDA 1.0, academic/computational use only [S7, from datasets note] |
| **SPOTER (Bohacek, Hruz), WLASL pose transformer** [S9] | WLASL | WLASL100 top-1 63.18%, WLASL300 43.78% [E] | Small transformer; real-time plausible [J] | Retrain on subset | Code license not checked |
| **ASL Citizen supervised baselines** [S10] | 83,399 videos, 2,731 signs, 52 signers, webcam | 63% top-1 and 91% recall@10 on unseen signers (2,731-way; baseline model type not stated on the page I read) [E] | n/a | Needs download (size unknown) | Article CC BY 4.0; dataset terms ask for Deaf involvement and bar use to replace interpreters or teachers [S10, datasets note] |
| **(b) Own small classifier (GRU or 1D-CNN) on a vocabulary subset** | 10-35 WLASL clips/word (our counts) or ~700-880 PopSign clips/word where the word exists | No published number for a 15-word subset. WLASL100 results above (55-63%) are a pessimistic reference for noisy web video; PopSign's 84% over 250 words is the optimistic one [J] | ~1-3 ms per window [J] | 5-8 h incl. data wrangling | Inherits dataset licenses |
| **(c) Few-shot enrollment, kNN/DTW on normalized landmarks** | 3-5 recordings per sign from the actual user | No published figure for this exact setup; I found only proofs of concept and one blog saying "a few dozens" of videos per sign are needed for good DTW results [S11]. Treat any accuracy as unknown until measured | DTW over ~15 signs x 5 templates x 30-frame window: a few ms [J] | 3-5 h, no GPU | None; own data |
| **(a) Released pretrained model, softmax restricted to our signs** | none | Not applicable: no confirmed public weights; and 6 of our signs are absent from its vocabulary [X] | n/a | Blocked unless weights found | n/a |

## 2. Core-sign vocabulary coverage

PopSign v1.0 has 250 concepts, listed in its supplement Table 1; I parsed 245 of 250 rows from the PDF text and grepped the full text for the remaining check words. The Kaggle ISLR competition used PopSign data with 250 signs [S3, S12 discussion], but **I did not read the Kaggle `sign_to_prediction_index_map.json`**, so I assume the lists are identical. [X + assumption]

| Sign | PopSign / Kaggle ISLR (examples) | WLASL (instances, from datasets note) | ASL Citizen |
|---|---|---|---|
| YES | yes (851) | 22 | not verified |
| NO | no (881) | 22 | not verified |
| HELP | **absent** | 22 | not verified |
| MORE | **absent** | 15 | not verified |
| STOP | **absent** | 11 | not verified |
| EAT | **absent** | 19 | not verified |
| DRINK | drink (862) | 35 | not verified |
| WATER | water (636) | not counted | not verified |
| BATHROOM | **absent**; `potty` (805) is the toddler word and a different sign | bathroom 16, toilet 13 | not verified |
| FINISH / ALL-DONE | finish (792); `all` (534) exists, "done" absent | finish 21, done 7 | not verified |
| PAIN / HURT | **absent**; `owie` (693) is a child-register word, `sick` (844) exists | pain 10, hurt 12 | not verified |
| WANT | **absent** | 19 | not verified |
| PLEASE | please (883) | 15 | not verified |
| THANK-YOU | thankyou (864) | not a gloss | not verified |
| AGAIN | **absent** | 16 | not verified |
| Extras present in PopSign | wait (844), hungry (885), thirsty (869), sick (844), sleep (859), hello (860), bye, go | wait 18, hungry/thirsty 12/12, hello 13 | |

Reading the table [J]:
- **From PopSign alone you can train 7 of our 15 signs exactly** (YES, NO, DRINK, WATER, FINISH, PLEASE, THANK-YOU) with ~600-880 clips each, plus 2 nearest-form substitutes (POTTY, OWIE) that a native signer should judge before use. Not HELP, MORE, STOP, EAT, WANT, AGAIN. (WAIT, HUNGRY, THIRSTY, SICK, SLEEP, HELLO are extras that could be added to the vocabulary.)
- **WLASL has the six missing signs** but only 11-22 instances each, noisy labels, and dead links (see datasets note).
- **Extending beyond the shipped list requires enrollment (c).** Choose enrollment as the common mechanism so every sign is treated alike, then use PopSign/WLASL only as a warm start or sanity check.

PopSign caveats [E, S2]: collected on Pixel 4A selfie cameras held in one hand; left-handed signers were mirrored to right-handed; hands only were used for the paper's baseline (EYE vs TOOTH confusions were attributed to the lack of face-relative features); includes "variant" examples of other ways to sign a concept. The data card marks continuous sign recognition and sign-to-English translation as **unsuitable uses** [S2].

## 3. Isolated sign recognition from landmarks: details

- **Kaggle ISLR (Feb-May 2023, $100k)**: train a TFLite classifier on MediaPipe Holistic landmarks from PopSign so it runs on-device in the PopSign game [E, S3]. 1st place: "1DCNN combined with Transformer" by Hoyeol Sohn (handle Hoyso48); a student report says 118 landmarks and "3 1D-conv layers then 1 transformer layer" [E, secondary; S4, S5]. 2nd place: EfficientNet-B0 on keypoint images, CV 0.898 and LB ~0.8 [E, S6]. A frame embedding was present in all top entries and the top keypoint-only result is quoted as 89.3% on 250 signs [E, secondary; S13]. Takeaway: accuracy came from landmark subsetting (lips, pose, hands), heavy augmentation and ensembling, which are not weekend-sized. [J]
- **Model sizes and latency**: a GitHub project cites "< 40 MB" as a design target [E, secondary]. I could not confirm the competition's size and latency limits or any CPU timing. **Could not verify.**
- **Pretrained weights**: the 1st-place repo is MIT and holds a notebook; the 2nd-place repo is Apache-2.0 and holds weight files of unexplained purpose. Neither page says that a ready-to-run TFLite model is released [E]. A weekend plan must not depend on them.
- **Why landmarks alone are hard**: PopSign authors attribute errors to hand-only features lacking face reference and to near-homophone signs (WAKE/AWAKE confused 47% of the time) [E, S2]. A 2025 ASL Citizen study on 100 glosses found per-frame center-of-mass normalization (66.18%) beat nose-anchored (39.0%) and raw (50.29%) in a BiLSTM; final 68.5% top-1, overfit, no test set [E, S14]. So normalization choice matters more than architecture. [J]
- **MS-ASL**: not researched; coverage and license **not verified**.

## 4. Signs in a continuous stream

- **Datasets**: How2Sign (80+ h, multiview, CC BY-NC 4.0, sentence-level; datasets note [S15]). OpenASL: 98,417 translation pairs, 200+ signers, with 966 val and 975 test pairs, described as the largest public ASL translation set [E, S16]. YouTube-ASL: 984 h, 11,093 videos, >2,500 signers; only video IDs released [E, S17]. None has per-sign boundaries for our words, so none gives you a "no sign" negative class for free. [J]
- **Sign spotting / segmentation**: BSL-1K used weakly aligned subtitles plus a mouthing-based keyword spotter to find 1,000 signs in 1,000 h of BSL; spotting posteriors are per-frame with non-maximum suppression over a 0.6 s window [E, S18]. Moryossef et al. (EMNLP Findings 2023) segment signs and phrases with BIO tagging over pose (75 keypoints) using a BiLSTM; they found BIO tagging was needed for sign boundaries [E, S19]. I found no streaming or latency figures for it.
- **Co-articulation**: BSL-1K's point is that signs in continuous signing look different from isolated ones, and models trained on isolated clips transfer imperfectly [E, S18]. For us: a sign in a stream starts and ends mid-motion; expect lower accuracy than any isolated-sign number above. [J]
- **Practical scheme for a demo [J]**: sliding window of ~1 s (about 8-12 frames at our 8-12 fps; PopSign clips were capped at 60 frames), stride ~0.25 s; a "no sign" class built from rest hands, scratching, talking gestures, other signs, plus a motion/hand-presence gate so you skip windows with no hands or with hands below chest. Peak-pick over window posteriors with the 0.6 s NMS idea from BSL-1K.
- **Frame rate risk**: our MediaPipe pipeline runs ~5-10 fps per model. PopSign/Kaggle landmarks were not collected at that rate and sign motion is fast; fewer frames per sign will hurt motion-dependent signs (HELP, MORE, STOP, AGAIN). Measure on real webcam clips before trusting anything. [J]

## 5. Recommended path for this weekend

Ladder and time estimates are **[J]**; total ~13-15 h with slack, so cut item 6 first.

| # | Step | Time |
|---|---|---|
| 1 | Define the vocabulary: ~10 signs max (YES, NO, HELP, MORE, STOP, EAT, DRINK, BATHROOM, PLEASE, THANK-YOU, ALL-DONE). Have a Deaf/native signer or ASL teacher approve the exact sign forms and one-handed variants before recording. | 1 h + waiting on the signer |
| 2 | Landmark capture tool on top of `video.py`: record 3-5 takes per sign plus ~60 s of "no sign" activity; store raw hand (2x21x3), pose shoulders/elbows, nose and face-reference points per frame with timestamps. | 2 h |
| 3 | Normalization: translate to a body anchor (shoulder midpoint or per-frame center of mass), scale by shoulder width, mirror left-handers, interpolate dropped hand frames, resample to fixed length (e.g. 16). Keep face-relative features (hand-to-nose/mouth distances) because PopSign's own errors came from their absence. | 1.5 h |
| 4 | Recognizer v0 = kNN/DTW over enrolled templates (approach c). Leave-one-take-out accuracy report per sign and a confusion matrix. | 2 h |
| 5 | Window detector + "no sign" class + temperature calibration + smoothing (section 6). Test on 5 minutes of mixed live signing and idle movement. | 3 h |
| 6 | Optional: train a tiny GRU/1D-CNN on enrolled data augmented with PopSign clips for covered signs (YES, NO, DRINK, WATER, FINISH, PLEASE, THANK-YOU), compare with v0 on held-out takes. Skip if time slips. | 3 h |
| 7 | Integration into the fusion layer, a demo script, and honest on-screen labeling ("keyword signs, enrolled by the user"). | 2 h |

Risks to flag in the demo plan [J]: signer variation (the enrolling user is the only signer tested), camera angle and distance (PopSign was a phone selfie view, ours is a webcam), one- vs two-handed signs (HELP, MORE, and ALL-DONE are normally two-handed; PopSign's one-handed versions may not match what a signer would naturally do), missing face and body context (hands-only features confuse EYE/TOOTH-like pairs), and low fps.

**Ladder** (what each rung needs):
1. **This weekend**: enrolled keyword-sign spotting, ~10 signs, soft evidence into fusion.
2. **After the hackathon**: a landmark classifier pretrained on PopSign/Kaggle data, fine-tuned on enrollment data, then small-vocabulary expansion with a native-signer-validated set.
3. **Research-grade**: gloss-free continuous translation (section 7) needs ~1,000 h of sign video, GPU training, and still yields BLEU ~12. Not a Tink-scale project.

## 6. Output and calibration for the Bayesian layer

Input: per-frame landmarks from `video.py` (hands, face, pose). Output: `P(sign | window)` over `{core signs..., no_sign}` per hop. [J] throughout, except where cited.

1. **Include `no_sign` as a class** with its own training windows (idle hands, adjusting glasses, gesturing while speaking, off-vocabulary signs). Without it a softmax must pick a core sign for every window. PopSign's softmax has no such class, since it is designed around a user holding a button to aim [E, S2].
2. **Calibrate with temperature scaling** on held-out takes (Guo et al., ICML 2017; from prior knowledge, **not fetched**): fit one scalar T on validation log-loss, then check reliability on a separate take. With 3-5 enrollment takes there is no spare data, so use leave-one-take-out predictions to fit T. For kNN/DTW, convert distances to scores with `exp(-d/tau)` including a no-sign distance threshold, then fit tau the same way.
3. **Make it an observation model.** Fusion wants `P(observation | intent)`. Treat the window posterior as a likelihood by dividing by the classifier's training class prior, then map sign to intent through a small table (YES->Yes, MORE->Want/More, HELP->Help, ...) and floor the likelihood (e.g. max with 0.02) so one mistaken window cannot veto the other detectors. Apply a power (tempering) of <1 if the detector looks overconfident. (This follows the "calibrated soft evidence" framing of the existing architecture; check against how other detectors are weighted in `meaning.py`.)
4. **Temporal smoothing**: exponential moving average of log-probabilities over the last 3-5 hops, or a 2-3 state HMM (idle -> sign onset -> held) with a dwell time of ~0.5-1 s. Emit one evidence event per detected sign (peak-picked with a 0.6 s refractory period, borrowed from BSL-1K's NMS window [S18]) rather than every hop, so a held or repeated sign does not multiply the same evidence.
5. **Expose confidence honestly**: report the no-sign probability and entropy, and abstain when the top sign is under a threshold chosen on enrollment data.
6. **Metrics to track**: per-sign precision/recall, false-trigger rate per minute of idle video (more important than accuracy for a live demo), expected calibration error on held-out takes.

## 7. Continuous ASL-to-English translation

The coordinator asked for the state of the art, public availability, and a feasibility verdict.

**Published quality (exact numbers from papers I read):**

| System | Input | Training data | Test set | Result | Source |
|---|---|---|---|---|---|
| Alvarez et al. | CNN pretrained with gloss | How2Sign | How2Sign test | BLEU-1 17.40, B-2 7.69, B-3 3.97, **B-4 2.21** | [S20, S17] |
| GloFE-VN | video + concept anchors, gloss-free | How2Sign | How2Sign test | B-1 14.94, B-2 7.27, B-3 3.93, **B-4 2.24**, BLEURT 31.65 | [S20] |
| Tarres et al. | video | How2Sign | How2Sign test | B-1 34.01, B-2 19.30, B-3 12.18, **BLEU 8.03** | [S17] |
| GloFE-VN | video + concept anchors | OpenASL | OpenASL test (975 pairs) | ROUGE 21.75, B-1 21.56, B-2 12.74, B-3 9.05, **B-4 7.06**, BLEURT 36.35. Highest-scoring other row in the same table (label lost in PDF extraction): B-4 6.72, BLEURT 31.09 | [S20] |
| YouTube-ASL T5.1.1-Base on MediaPipe Holistic landmarks | landmarks | YouTube-ASL pretrain + How2Sign finetune | How2Sign test | **12.39 BLEU** (fine-tuned SOTA), **3.95 BLEU zero-shot** | [S17] |

- The authors call their own outputs "clearly not yet of usable quality" [S17]. At BLEU 4-12 most of the content is lost or wrong. [E + J]
- The domains are narrow: How2Sign is studio instructional videos, and YouTube-ASL's zero-shot jump to 3.95 shows poor transfer [E, S17].
- Newer work: RVLF (arXiv 2512.07273) reports YouTube-ASL BLEU-1..4 of 37.8/24.1/16.9/12.4 in one setting, but the snippet did not say which dataset or setting that row covers, so **I could not verify it** [S21]. Sign2GPT (adapter-based, large pretrained vision and language models) reports gains on PHOENIX-2014T and CSL-Daily (German and Chinese sign languages), **not ASL, and I could not read its numbers** [S22]. SignLLM: **not researched**.

**Latency / real time.** YouTube-ASL's pipeline is MediaPipe Holistic landmarks (their paper says 532 landmarks per frame) into a T5-Base encoder-decoder, and it was trained on TPUv4 [E, S17]. No paper I read reports CPU or streaming latency. These models translate whole clips with beam search (width 5), not a live stream with partial outputs [E]. Real-time at speech pace would also need online segmentation and incremental decoding, which none of them describes. [J]

**What is publicly usable now.**
- **SignGemma**: announced by Google at I/O in May 2025 as an open, on-device model "coming to the Gemma family later this year" [E, S23]. Secondary reports: best at ASL-to-English, trained on over 10,000 hours of annotated ASL video, with access through a free early-testing program [E, secondary, S24]. **I found no official model card, Hugging Face or Kaggle release, license, or hardware requirement.** The only Kaggle hit is a community notebook fine-tuning a 2B Gemma on SignWriting, a different thing. Status as of today is unverified. If a release exists, check Google's page first. [J]
- **YouTube-ASL**: only video IDs are released, and the model checkpoint is not stated as released in the paper I read [E, S17].
- **OpenASL / How2Sign**: datasets only; How2Sign is CC BY-NC 4.0 [S15]. No weights found.
- **Hardware**: GloFE trained 4 days on 4 GPUs (OpenASL) [E, S20]; YouTube-ASL trained on TPUs [E, S17]. Nothing here is a laptop-CPU job at training time.

**Verdict for a laptop-CPU demo in ~15 hours: not feasible.** [J] Reasons: (1) no public ASL-to-English weights confirmed; (2) the best published quality is BLEU 12.39 under favorable, in-domain finetuning, and 3.95 zero-shot; (3) no streaming implementation is described; (4) training needs 1,000 hours of video and TPU/GPU; (5) our 5-10 fps landmark stream is far below what these models were trained on; (6) Tink's users are not the studio signers in the benchmarks. A scripted "translation" demo would be misleading to a Deaf audience. If a hosted SignGemma-like model becomes available, put it behind Tink's fusion as a weak, clearly labeled signal, not the main recognizer.

## 8. Ethics and accuracy caveats

- Desai et al. (LREC-COLING workshop 2024; Deaf-led review of 101 recent sign-language AI papers) conclude that the field over-focuses on perceived communication barriers, uses unrepresentative datasets and annotations lacking linguistic grounding, and lacks meaningful Deaf stakeholder input; they call for Deaf leadership [E, S25].
- Bragg et al. (ASSETS 2019, Best Paper) came from a workshop with Deaf and Gallaudet researchers and calls for interdisciplinary work that includes cultural context [E, S26]. The PopSign paper cites Erard (The Atlantic, 2017) and Hill (Nature Electronics, 2020) for the point that technology made without Deaf input can be unwanted or harmful, and PopSign itself says the dataset "is not meant to be representative of ASL" [E, S2].
- ASL Citizen's terms ask that data not be used to replace interpreters or signing teachers (datasets note, S10). PopSign's data card lists continuous recognition and translation as unsuitable uses [E, S2]. A keyword-sign pack for an AAC user is a different purpose, but quote these terms in any write-up. [J]
- **Practices for the demo [J]**: (1) have a native signer approve each sign form and the framing before showing; (2) state that it recognizes ~10 enrolled keyword signs, not ASL, not sentences, not fingerspelling; (3) show per-sign accuracy and failure cases; (4) never present it as interpretation or a substitute for an interpreter; (5) obtain consent for the enrollment recordings and keep them local; (6) use ASL as a user-owned input, not an assumption about who the user is.
- I did not find a World Federation of the Deaf statement on sign recognition technology. **Could not verify.**

## Could not verify

- **Kaggle ISLR `sign_to_prediction_index_map.json`**: the 250-sign list of the competition (assumed to equal PopSign v1.0's list; I parsed 245 of 250 rows of that list from the supplement PDF, and grepped the rest of the text).
- **Kaggle competition rules**: model size limit (40 MB is secondary), latency limit (not confirmed), data license on Kaggle itself (PopSign v1.0 is CC BY 4.0 per its paper).
- **1st-place write-up details** (landmark list, augmentations, CV score, model size, inference time): Kaggle pages require JavaScript and the repo README is a pointer. The "118 landmarks, 3 conv layers + 1 transformer layer, ~80% val" claim is from a student report seen only through search.
- **Whether any trained ISLR weights or TFLite models are publicly released** (Kaggle code, Hugging Face): not found.
- **CPU latency** for any of these models: no published figure found; all latency numbers in section 1 are my estimates.
- **ASL Citizen core-word coverage** and its pose-baseline accuracy table; **MS-ASL** coverage, accuracy, and license.
- **Few-shot DTW accuracy with 3-5 examples** for MediaPipe sign recognition: only proofs of concept found.
- **SPOTER code license**, Pose-TGCN's original-paper numbers, **Sign2GPT** and **SignLLM** ASL numbers, **RVLF** table context.
- **SignGemma**: whether and where it was released, license, hardware requirements, latency, accuracy.
- **Temperature scaling reference** (Guo et al. 2017) is from my own knowledge and was not fetched.

## Sources

- [S1] PopSign ASL v1.0 (NeurIPS 2023 Datasets and Benchmarks): https://proceedings.neurips.cc/paper_files/paper/2023/hash/00dada608b8db212ea7d9d92b24c68de-Abstract.html ; PDF https://proceedings.neurips.cc/paper_files/paper/2023/file/00dada608b8db212ea7d9d92b24c68de-Paper-Datasets_and_Benchmarks.pdf
- [S2] PopSign supplement (word list with counts, data card, unsuitable uses): https://papers.neurips.cc/paper_files/paper/2023/file/00dada608b8db212ea7d9d92b24c68de-Supplemental-Datasets_and_Benchmarks.pdf ; dataset page https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/ (certificate error when fetched; license read from the paper)
- [S3] Kaggle competition page: https://www.kaggle.com/competitions/asl-signs (page text not rendered; facts from search summaries)
- [S4] 1st place discussion: https://www.kaggle.com/competitions/asl-signs/discussion/406684 ; [S5] repo (MIT): https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution
- [S6] 2nd place (Apache-2.0): https://github.com/ffs333/2nd_place_GISLR
- [S7] WLASL repo: https://github.com/dxli94/WLASL
- [S8] SignBERT+ comparison table (Pose-TGCN numbers): https://arxiv.org/pdf/2305.04868 ; Pose-based SLR with GCN and BERT: https://arxiv.org/pdf/2012.00781
- [S9] Bohacek and Hruz, SPOTER (WACVW 2022): https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.pdf
- [S10] ASL Citizen: https://arxiv.org/abs/2304.05934 ; datasheet https://www.microsoft.com/en-us/research/project/asl-citizen/datasheet/
- [S11] Theodo, sign recognition with MediaPipe and DTW: https://www.theodo.com/blog/sign-language-recognition-using-mediapipe ; MediaPipe+DTW project https://github.com/mammhoud/Sign-Language-Recognition--MediaPipe-DTW
- [S12] Sign-language-processing discussion on PopSign vs WLASL/MS-ASL sizes: https://github.com/sign-language-processing/spoken-to-signed-translation/issues/9
- [S13] Landmark-subset paper citing the 89.3% figure: https://arxiv.org/pdf/2510.24887 ; secondary notes on the winner: https://judicious-mastodon-287.notion.site/Kaggle-Competition-Isolated-Sign-Language-Recognition-390d371b694c40b1a7fd113002096426
- [S14] Segmentation and pose estimation for ISLR on ASL Citizen: https://arxiv.org/html/2512.14876v1
- [S15] How2Sign: https://how2sign.github.io/
- [S16] OpenASL (as described in GloFE): see [S20]
- [S17] YouTube-ASL: https://arxiv.org/abs/2306.15162 (PDF https://arxiv.org/pdf/2306.15162)
- [S18] BSL-1K: https://arxiv.org/pdf/2007.12131
- [S19] Moryossef et al., Linguistically Motivated Sign Language Segmentation: https://ar5iv.arxiv.org/html/2310.13960 ; https://github.com/sign-language-processing/segmentation
- [S20] GloFE, Gloss-Free End-to-End SLT (ACL 2023): https://arxiv.org/abs/2305.12876 (tables read from PDF)
- [S21] RVLF: https://arxiv.org/pdf/2512.07273
- [S22] Sign2GPT (abstract only): https://arxiv.org/abs/2405.04164
- [S23] Google Accessibility SignGemma announcement (X post, 2025-05-27): https://x.com/googleaccess/status/1927425916847685947
- [S24] SignGemma secondary coverage: https://multilingual.com/google-signgemma-on-device-asl-translation/ ; https://tech.slator.com/?p=106104
- [S25] Desai et al., Systemic Biases in Sign Language AI Research: https://arxiv.org/pdf/2403.02563
- [S26] Bragg et al., Sign Language Recognition, Generation, and Translation: An Interdisciplinary Perspective: https://arxiv.org/pdf/1908.08597
- Local files read: `docs/RESEARCH-datasets.md`, `video.py` (MediaPipe hand and face landmarkers, YOLO11n-pose).
