# Building the mapping layer: P(signal | intent) (2026-10-24)

Read-only research for Sanjith's decision; nothing here was run on Tink's data. Tags: **[E]** evidence from a listed source,
**[X]** arithmetic from a cited number, **[J]** my judgment. Refs like [S3] point to Sources. Prior research
(`RESEARCH-interpretation.md`: recursive filter, tempering, head pose, LLM layer, count-based enrollment in B6) is not repeated.

## 0. Bottom line

1. **Split the table in two.** Today `P(label | intent)` mixes two different things: what the person *does* for an intent (a
   personal, human-readable dictionary) and how often the detector *reports* what they did (a measurable confusion matrix).
   Write `P(reported | intent) = sum_p C[p -> reported] * D[intent, p]`, with `D` elicited from the person and `C` measured by
   a short enrollment. Multiply them at build time and write the same `likelihoods.json` schema, so `BayesianFusion` does not change. [J; the
   observer-error-versus-truth split is the Dawid-Skene model, S14]
2. **Keep the 8 intents as the backbone, add `none`, and put the precision in the referent.** The clinical standard taxonomy (Communication
   Matrix) has only 24 messages and they are generic ("requests an object", "directs attention to something") [S9]. Specific meaning
   ("the water") comes from the referent slot that `meaning.py` already has. [E for the taxonomy, J for the conclusion]
3. **No dataset gives usable P(signal | intent) for this user.** ReCANVo and CNVVE give vocalization base rates and class vocabularies; MIntRec2.0
   is scripted TV dialogue by typical speakers. Use them for class names and base rates, not for the numbers. [E, section 2]
4. **Numbers come from a verbal-anchor elicitation (about 15 min) smoothed with a Dirichlet pseudo-count, then updated with 5 enrollment trials per signal.**
   Five trials cannot certify reliability (rule of three: 0 failures in 5 leaves a 95% upper bound of about 60%), so the smoothing floor, not the count, is what protects the demo. [X]
5. **Total build is about 5-6 h**, mostly the enrollment screen. Defer hierarchical fitting, online updating, custom-gesture learning and noisy-OR.

## 1. Clinical AAC practice for idiosyncratic and unaided signals

| Practice | What it is | Format / procedure | Reliability handling | Source |
|---|---|---|---|---|
| Communication (gesture) dictionary | Person-specific list that lets unfamiliar partners recognise a signal, read it and respond | Three columns: communicative method, possible meaning, how to respond. Examples: "Joaquin grabs wrists when he wants to play"; "Ariella bangs her fists together when she wants more of something" | None stated. A commenter suggests repeating the gesture back so the learner can confirm or refute the reading (a comment, not guidance) | PrAACtical AAC [S1] |
| Dictionary procedure (Irish SLT service) | Same, longer procedure | Observe in different situations; describe every way the person communicates; check accuracy with the people who know them best; record the likely meaning in the first person; keep it with the person; review regularly | "Check with the people who know them best", then review. This is consensus, not a statistic | smh.ie guide, via search summary only; PDF unreadable here [S2] |
| Communication passport | Broader person-centred document carried by the person | People important to me, things I like to talk about, how I communicate, how you can help me communicate, things I don't like | None stated | Ace Centre [S3] |
| Triple C (Checklist of Communication Competence) | Standardised partner-completed checklist for people at unintentional-to-early-symbolic levels | Support worker completes it; five stages | 68 adults, two support workers each: item agreement about 81-87%, stage agreement kappa = 0.63 (moderate to high); KR-20 = 0.97 | Iacono et al. 2009 [S4] |
| ASHA assessment guidance | Assess current verbal and nonverbal means and their effectiveness; family and caregivers report consistent behaviours; assessment is ongoing with trial periods | Unaided AAC includes body language, facial expression, gestures, vocalizations; gestures may not carry the same meaning across languages and cultures | No interpretation-reliability statistics on the page | ASHA Practice Portal [S5] |
| Interpretation risk | Inference and intention both shape what partners read into behaviour; risk is highest when the partner cannot correct a misunderstanding | | Validate communication; ask key questions of those closest to the person | Porter et al. 2001 abstract [S6] |

**What this means for Tink**
- The dictionary is the clinically familiar artefact, so the mapping layer should be *displayable as one*: Signal | Means | Tink's response. That is
  also a good demo screen. [J]
- Clinical "reliability" is partner consensus plus confirmation, not a likelihood. No source gives a numeric threshold for admitting a
  signal to a dictionary. The closest honest analogue is enrollment hit rate (section 3.3) plus Tink's existing confirm key. [E for what exists, J for the analogue]
- Partners decode in context. Interpretation depends on the partner's knowledge of the situation (Porter [S6]), which is the argument for
  keeping partner-question context as an input to the intent prior (already in `meaning.py`).
- A 2026 CHI paper (AllyAAC) shows the same personalisation pattern in ML: participants chose their own 2-4 gestures and meanings, each recorded at least
  3 times plus idle/rest trials; the model used 3-20 annotated examples per category; the whole process took about 30 minutes per participant; F1 rose from about 60%
  (rule-based baseline) to over 85% (pretrained transformers) [S7]. The paper also says personalised modelling is needed and that atypical gestures can differ even for the same intent. [E]

## 2. Data sources for signal-to-meaning evidence

| Source | What it has | Labels | License | Use for Tink |
|---|---|---|---|---|
| ReCANVo [S8, S8b] | 7,077 vocalizations, 8 minimally speaking people aged 6-23 (autism, cerebral palsy, genetic disorders), 0.5-4 s clips, labelled live by a parent or familiar partner who labelled only when confident | Six shared labels: Frustration, Delight, Dysregulation, Self-talk, Request, Social; plus up to 4 custom labels per person from 25 presets (e.g. yes, no, bathroom, laughter, affectionate). Each person has 6-10 labels | Article CC BY 4.0; Zenodo record CC BY 3.0 US | Base rates and a vocabulary. **No inter-rater agreement is reported.** Labels are personal by design |
| CNVVE [S10, S10b] | 950 clips, 42 speakers, six non-verbal voice expressions for assistive HCI; 96.6% test accuracy in 5-fold CV with augmentation | "uh-huh/mm-hmm", "uh-uh/mm-mm", "hush/shh", "psst", "ahem", continuous humming | CC BY 4.0 (DaRUS page) | Conventional sounds with a strong default reading for typical speakers: affirmative, negative, quiet, get attention. Not from people with complex communication needs |
| MIntRec2.0 [S11, S11b] | 1,245 dialogues, 15,040 samples from three TV series (Superstore, The Big Bang Theory, Friends); text, video, audio | 30 intents in two groups. Express emotions or attitudes: doubt, acknowledge, refuse, warn, emphasize, complain, praise, apologize, thank, criticize, care, agree, oppose, taunt, flaunt, joke. Achieve goals: ask for opinions, confirm, explain, invite, plan, inform, advise, arrange, introduce, comfort, leave, prevent, greet, ask for help | CC BY-NC-SA 4.0 on Zenodo and arXiv; Hugging Face metadata says CC BY-SA 4.0. **Sources disagree.** Underlying video is copyrighted TV | Taxonomy reference only. Scripted sitcom speakers, no AAC users, so it cannot supply P(gesture | intent) |
| Emblem and head-gesture studies | Matsumoto and Hwang 2013 compared emblems across cultures: universal gestures tend to carry basic messages, culture-specific ones more complex messages; some universal-looking gestures differ in meaning (A-OK) [S12]. Head nod/shake: Andonova and Taylor 2012 found nod/shake compatibility effects were culture-specific (US yes, Bulgarian no); Bulgarian reversal attributed to McClave et al. 2007 [S13] | no lexicon with probabilities | | Justifies making the head row *per-user editable and enrolled*. Not a source of numbers |

**ReCANVo base rates** (my sums of the per-participant counts in Table 2 as extracted [S8]; they total 7,077) [X]:

| Label | Clips | Share |
|---|---|---|
| Self-talk | 1,885 | 26.6% |
| Frustrated | 1,536 | 21.7% |
| Delighted | 1,272 | 18.0% |
| Dysregulated | 704 | 9.9% |
| Social | 634 | 9.0% |
| Custom labels (combined) | 627 | 8.9% |
| Request | 419 | 5.9% |

Reading [J]: most vocalizations are affect or self-talk, and only about 6% are requests. So a vocalization is weak evidence for a *specific* intent and
the vocalization source needs a `none` row that absorbs self-talk. The ReCANVo labels do not line up with Tink's 8 intents (frustrated could be No, Break or Help),
which confirms that `P(vocal class | intent)` stays a hand-set or enrolled table. Personalised classifiers are the published approach (Narain et al. 2020: speaker-dependent SVM/random forest, recall 0.75, 0.53, 0.79 for three individuals on five classes, abstract only) [S15].

## 3. Statistical methods for P(signal | intent)

### 3.1 Two-stage model (separate meaning from detection)

```
D[k, p]   = P(person produces signal p | intent k)        # dictionary: personal, elicited
C_s[p, r] = P(detector s reports label r | signal p)      # confusion: measured at enrollment, per detector, per venue
T_s[k, r] = sum_p D_s[k, p] * C_s[p, r]                   # the table BayesianFusion already consumes
```

- `p` includes `rest` (no signal). `k` includes `none`. `rest` explains the detector's `None`, `NoGesture`, `still`, and `none` labels.
- Why bother [J]: (a) a detector change (new MediaPipe model, new camera, new venue lighting) re-measures `C` in minutes and leaves `D` alone; (b) `D` is the thing a clinician or
  the user can read and edit; (c) when the table is flat you can tell whether the *meaning* is missing (`D` flat) or the *detector* is noisy (`C` flat).
- Formal precedent: Dawid and Skene model each observer's confusion matrix separately from the latent true class, estimated by EM [S14]. Here `p` is observed during enrollment (the person is told what to do), so
  `C` is just counts, with no EM.
- Cost: `T` is precomputed offline. Runtime and the JSON schema are unchanged. [J]
- Noisy-OR (Pearl) cuts the number of elicited parameters when several causes can produce an effect [S16, secondary]. It matters for multi-intent frames, not for this single-intent table. **Defer.**

### 3.2 Getting D: verbal-anchor elicitation, Dirichlet-smoothed

- SHELF targets distributions over single continuous parameters (with copula extensions) and has no discrete conditional-table protocol [S17]. It is the wrong tool.
- For tables, the relevant literature (Druzdzel and van der Gaag) found standard decision-analysis elicitation does not scale and used verbal probability phrases with numerical anchors; a secondary account puts the rate at about 150-200 probabilities per hour [S16, secondary]. [E, weak]
- Proposed protocol [J]: for each intent, the person (or whoever knows them) lists the 1-3 signals they would use, tags each *Always / Often / Sometimes / Rare*, mapped to 0.7 / 0.4 / 0.15 / 0.03. Rows are
  normalised. The numeric mapping is mine; check it against the separation test below, not against any source.
- Dirichlet shrinkage: with prior mean `m_k` (the elicited row) and pseudo-count `alpha` (an equivalent sample size), the posterior mean after counts `n` is
  `(n_kp + alpha * m_kp) / (n_k + alpha)`. The equivalent-sample-size device is standard in Bayesian-network parameter learning; no general rule for choosing it exists and results can be sensitive to it [S18]. Use `alpha` about 5-10 for a hand-set row. [J for the value]
- Hierarchical/empirical Bayes would set `alpha` and the population mean by pooling many users (Dirichlet-multinomial marginal likelihood, Minka fixed point [S19]). There is no pool of AAC users here, so **defer**; the population prior is the design table itself. [J]
- **Separation test** (cheap, do it first) [J]: for every pair of intents (k, k') find the label with the largest `log(T[k,r]/T[k',r])`. If no label gives a likelihood ratio of about 4 or more,
  the sensors cannot tell those intents apart and the UI should say so. Today Help, Repeat, Want and Break rows are uniform (all 0.125) in `mp_gesture` and all of `envision`, `head` and `arm` are neutral, so this test fails for 4 of 8 intents. [E: likelihoods.json]

### 3.3 Getting C: enrollment, and how many trials

- Protocol: for each signal the user selected, 5 trials of a 2 s hold, with a `rest` set of 10. Record the detector's modal label per trial (frames within a trial are correlated, so count trials, not frames). [J]
- Uncertainty is binomial. After `n` trials with `x` correct, a uniform-prior Beta posterior has mean `(x+1)/(n+2)`; with zero failures, the 95% upper bound on the failure rate is about `3/n` (rule of three, Hanley and Lippman-Hand [S20, secondary]). [X]

| Trials n (all correct) | Posterior mean P(correct) | 95% upper bound on miss rate (3/n) |
|---|---|---|
| 5 | 0.857 | 0.60 |
| 10 | 0.917 | 0.30 |
| 20 | 0.955 | 0.15 |
| 30 | 0.969 | 0.10 |

- So 5 trials gives a usable point estimate and a useless guarantee. For a demo the target is "no zero cells and the hand-set row dominates where data is thin", which Dirichlet smoothing does. Do not report enrollment accuracy as a performance claim. [J]
- Admission rule (analogue of partner agreement) [J]: a signal enters the dictionary at full strength if at least 4 of 5 trials were detected as intended; otherwise it is kept but tempered (its row blended toward uniform) and flagged on the panel.
- Published enrollment sizes for comparison: AllyAAC 3-20 examples per gesture [S7]; the $1 recognizer's single-template result is in the earlier note. I found no published number for how many enrollment trials a gesture-to-intent table needs. [E, gap]
- Time: 8 signals x 5 trials x ~6 s (prompt, perform, release) + 10 rest trials is about 5 min of capture. Elicitation of `D` is about 15 min. First-time total about 20-25 min; re-enrollment at the venue about 5 min. [X on my per-trial assumption]
- PLAN.md records that Sanjith removed `learn_likelihoods` ("not self-correcting"). One-time enrollment is calibration, not online learning; **whether to allow it is his call** (as in the earlier note).

## 4. Intent taxonomy: is there a better backbone than 8 coarse intents?

| Taxonomy | Structure | Source |
|---|---|---|
| Communication Matrix (Rowland) | Four reasons: Refuse, Obtain, Engage (social interaction), Provide or seek information. 24 messages. Seven levels from pre-intentional behaviour to language. Built to accommodate any behaviour including gestures and eye gaze | [S9] |
| Matrix messages by level | I: expresses discomfort / comfort / interest in people. II: protests; continues an action; obtains more of something; attracts attention. III: refuses or rejects; requests more of an action; requests a new action; requests more of an object; makes choices; requests a new object; requests attention; shows affection. IV: greets; offers or shares; directs attention to something; polite social form; answers yes/no questions; asks questions. V: requests absent objects; names; comments | [S9] |
| Light (1988) four purposes | Wants and needs; information transfer; social closeness; social etiquette | [S21, secondary] |
| CSBS (Wetherby and Prizant) | Seven clusters including Communicative Function, Gestural/Vocal/Verbal means, Reciprocity, Social-Affective Signaling, Symbolic Behavior (cluster names only; the individual function categories were not verified) | [S22] |
| Pointing functions | Protoimperative (get an adult to fetch an object) versus protodeclarative (get an adult to attend to something), Bates and colleagues 1975; later critics call the split a false dichotomy | [S23, secondary] |
| MIntRec2.0 | 30 fine intents for typical-speaker TV dialogue | [S11] |

**Fit of the current 8 against the Matrix** (mapping is mine) [J]:

| Tink intent | Closest Matrix message(s) | Note |
|---|---|---|
| Yes | answers yes/no questions (IV) | Needs partner-question context |
| No | answers yes/no questions (IV); refuses or rejects (III); protests (II) | Two functions in one label. Answer-no needs a question; refuse-no does not. The question context already separates them |
| Attention | attracts attention (II); requests attention (III) | |
| Show | directs attention to something (IV); offers or shares | Same gesture (a point) can be Show or Want: protoimperative versus protodeclarative |
| Help | requests a new action (III) | Not a named Matrix item |
| Repeat | none | Conversational repair; no Matrix equivalent |
| Want | requests an object (III); requests absent objects (V) | |
| Break | refuses or rejects (III); requests a new action (III) | Not a named Matrix item |
| *(missing)* Greet, thanks | greets people (IV); polite social form (IV) | Light's social etiquette and closeness purposes are absent from the 8 |
| *(missing)* More / continue | continues an action (II); requests more of an action (III) | |
| *(missing)* None / self-talk | pre-intentional behaviour (I) | ReCANVo: self-talk is the largest class (26.6%) |

**Verdict** [J]:
- Do not swap the backbone. The 8 intents already sit at Matrix levels II-IV, so a taxonomy swap would change names, not coverage.
- Add `none` (required for the filter and for self-talk), and optionally `Greet` (wave has a natural home there) and `More/again`. Each added intent needs a signal in `D` and tagged phrases in the 178-phrase database, so add one only if you will demo it.
- "Precise, complex meaning" is not more intents. It is `(function, referent)`. Even the clinical standard stops at 24 generic messages; "the water" is the argument. `meaning.py` already factorises this way (`NEEDS_REFERENT`, templates, `ReferentFilter`). Labelling each intent with its Matrix reason (Refuse / Obtain / Engage / Inform) gives a shared vocabulary for the write-up.

## 5. Recommended mapping layer for the demo

**Backbone.** Keep the 8 intents + `none` (+ `Greet` if wave is demoed). Tag each with its Matrix reason in the config. Referents stay in `meaning.py`.

**Data flow.**

```
config/dictionary.json   D: per source, per intent -> {signal: tier}, plus rest/none rows      (elicited, ~15 min)
data/enrollment.json     C: per detector, per signal -> counts of reported labels (+ rest)       (enrolled, ~5 min)
build_tables.py          T = D x C, Dirichlet smoothing (alpha 5-10), separation test, admission rule
config/likelihoods.json  same schema as today                                                    (BayesianFusion unchanged)
dictionary view          Signal | Means | Tink says  (generated from D; the demo "passport" screen)
```

| Decision | Choice | Basis |
|---|---|---|
| Tables from data or experts | Experts (the user) for `D`, measurement for `C` | No dataset covers this user [E]; separation of the two is standard [E: S14]; the specific recipe is [J] |
| Detector-error layer | Yes, build-time only, for `mp_gesture` and `head`. Skip for keywords and vocalization | Gesture and head detectors are the ones whose noise you can enroll directly. Vocal and keyword tables stay hand-set [J] |
| Vocalization table | Few classes; one `none/self-talk` row sized by ReCANVo-style base rates; conventional sounds (affirmative, negative, attention) only if the classifier has those classes | ReCANVo shares [X]; CNVVE vocabulary [E] |
| Head table | Enrolled per user, not hard-coded nod=yes | Cultural reversal documented [E: S13]; enrollment handles it for free |
| Pointing | Same likelihood for Want and Show; decide by partner question and follow-up signals | Imperative versus declarative pointing [E: S23 secondary] |
| Partner-question context | Keep `CONTEXT_BOOST`, but state it as a prior shift P(intent \| question type), listed in the dictionary view | Matrix lists answering yes/no questions as an explicit message [E: S9] |
| Strength of evidence | Elicit tiers; report `T` as likelihood ratios; check against separation test (LR about 4+ for at least one label per intent pair) | [J] |
| Enrollment | 5 trials per chosen signal + 10 rest, once at the venue | Cost and counts in 3.3 |

**Time (estimates, [J]).** Dictionary schema + elicitation: 45 min. `build_tables.py` (product, smoothing, separation test, admission): 1-1.5 h. Enrollment screen on top of the existing detectors: 2-3 h.
Dictionary view: 30 min. Adding `none` (and `Greet`): 30-60 min including phrase tags. Total about 5-6 h.

**Defer:** hierarchical or empirical-Bayes fit of `alpha` (no user pool); online updating from confirmations (his earlier decision); noisy-OR / multi-intent frames; few-shot custom gesture learning (prior note B7); fitting `T` against real AAC users' data; taxonomy expansion beyond one or two intents.

**Evidence-backed versus judgment.**

| Evidence-backed | Judgment |
|---|---|
| Clinical dictionary format and its absence of numeric reliability [S1-S5] | The D x C factorisation as the recipe |
| Triple C agreement numbers [S4] | Tier-to-number map (0.7 / 0.4 / 0.15 / 0.03) |
| AllyAAC personalisation sizes: 3-20 examples, 30 min [S7] | alpha 5-10, 4-of-5 admission rule, LR >= 4 separation threshold |
| ReCANVo label set, counts, licenses [S8] | Mapping ReCANVo/CNVVE classes onto Tink intents |
| Matrix reasons and 24 messages [S9] | Mapping the 8 intents onto the Matrix |
| Culture-specific nod/shake [S13] | Which two to three intents to add |
| Rule-of-three arithmetic [S20] | Enrollment trial count of 5 |

## Could not verify

- **Beukelman and Light's AAC textbook** (and Mirenda): not accessed; no claim here depends on it. The ASHA page was read to about 100k of 137k characters.
- **Communication Matrix psychometrics:** the "Data and Research Basis" PDF did not extract; no reliability numbers for the Matrix. A search summary said the Matrix aligns with Light (1988); the Matrix page itself did not mention Light.
- **Smh.ie communication-dictionary PDF and the Oxford Health "Personal Communication Dictionary" guide:** contents not read; the procedure in section 1 is from a search summary.
- **Light 1988, Bates et al. 1975, Matsumoto and Hwang 2013, Moretti et al., McClave et al. 2007, Dawid and Skene 1979, Spiegelhalter and Lauritzen 1990, Hanley and Lippman-Hand 1983:** originals not retrieved; all cited via secondary pages or search summaries.
- **CSBS function categories:** only the cluster names were confirmed.
- **MIntRec2.0:** license conflict (Zenodo/arXiv CC BY-NC-SA 4.0 versus Hugging Face CC BY-SA 4.0); the 30 class names come from the Hugging Face README as extracted by a summariser (16 + 14 = 30, consistent with the paper's stated 30); the paper PDF did not extract. Inter-annotator agreement not obtained.
- **ReCANVo:** no inter-rater agreement is reported in the article; per-label totals are my sums of Table 2 as extracted by a summariser, so recheck against the paper before quoting. Article license (CC BY 4.0) and Zenodo record (CC BY 3.0 US) differ. The full Zenodo label list was not inspected.
- **CNVVE:** class names and license come from the DaRUS page via a summariser; the Interspeech abstract names neither.
- **AllyAAC:** first 100k of 173k characters read; per-participant results (Section 6) not read.
- **Narain et al. 2020:** abstract only.
- **Druzdzel and van der Gaag, SHELF-for-tables, hierarchical empirical-Bayes naive Bayes paper (arXiv 2608.11162):** only search summaries; the 150-200 probabilities per hour figure is secondary.
- **No published figure** was found for how many enrollment trials a gesture-to-intent likelihood needs, nor any gesture-to-intent likelihood table for AAC users.
- **voc2vec class list** was not checked, so the vocalization mapping depends on what the trained classifier actually outputs.
- Numerical choices in sections 3 and 5 (tier values, alpha, thresholds, time estimates) are judgment and untested on Tink.

## Sources

- [S1] PrAACtical AAC, On the Same Page (gesture dictionaries): https://praacticalaac.org/praactical/on-the-same-page-helping-team-members-recognize-and-respond-to-unconventional-communication-signals/
- [S2] Communication dictionaries (Irish SLT service PDF; unreadable here, summary only): https://smh.ie/assets/files/pdf/communication_dictionaries.pdf ; Oxford Health resources page listing the guide: https://www.oxfordhealth.nhs.uk/oxtc/resources/
- [S3] Ace Centre, communication passports: https://docs.acecentre.org.uk/aac-policy/transition/communication-passports
- [S4] Iacono, West, Bloomberg, Johnson (2009), revised Triple C, J. Intellect. Disabil. Res. 53(1):44-53: https://research.monash.edu/en/publications/reliability-and-validity-of-the-revised-triple-c-checklist-of-com/ ; fact sheet https://shop.scopeaust.org.au/wp-content/uploads/2015/05/5.Triple-C.pdf
- [S5] ASHA Practice Portal, AAC: https://www.asha.org/practice-portal/professional-issues/augmentative-and-alternative-communication/
- [S6] Porter, Ouvry, Morgan, Downs (2001), interpreting communication of people with profound and multiple learning difficulties, DOI 10.1046/j.1468-3156.2001.00083.x: https://anhoriga.se/Artiklar-och-rapporter/2001/interpreting-the-communication-of-people-with-profound-and-multiple-learning-difficulties (abstract as summarised by search)
- [S7] Kabir et al., Giving Meaning to Movements (AllyAAC), CHI 2026: https://arxiv.org/abs/2602.22131v1 ; full text https://arxiv.org/html/2602.22131v1
- [S8] Johnson, Narain, et al., ReCANVo, Sci. Data 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10404278/ ; [S8b] Zenodo dataset (CC BY 3.0 US): https://zenodo.org/records/5786860
- [S9] Communication Matrix research basis (four reasons, 24 messages, seven levels): https://communicationmatrix.org/Matrix/Pages/ResearchBasis
- [S10] Hedeshy, Menges, Staab, CNVVE, Interspeech 2023: https://www.isca-archive.org/interspeech_2023/hedeshy23_interspeech.html ; [S10b] dataset (classes, CC BY 4.0): https://doi.org/10.18419/darus-3898
- [S11] MIntRec2.0, Zhang et al., ICLR 2024: https://arxiv.org/abs/2403.10943 ; Zenodo https://zenodo.org/record/8041676 ; [S11b] Hugging Face README (class names): https://huggingface.co/datasets/THU-IAR/MIntRec2.0/blob/main/README.md
- [S12] Matsumoto and Hwang (2013), emblematic gestures, J. Nonverbal Behav., via blog summary: https://www.humintell.com/2013/01/gestures-emblems-cultural-variances/
- [S13] Moretti et al., Head nod and shake compatibility effect: cultural or universal? (abstract; cites Andonova and Taylor 2012, McClave et al. 2007): https://rivisteweb.it/doi/10.1422/93575
- [S14] Dawid and Skene (1979), J. R. Stat. Soc. C 28(1):20-28: https://ideas.repec.org/a/bla/jorssc/v28y1979i1p20-28.html ; implementation notes https://crowd-kit.readthedocs.io/en/latest/classification/
- [S15] Narain, Johnson, et al., Personalized modeling of real-world vocalizations from nonverbal individuals, ICMI 2020: https://www.media.mit.edu/publications/personalized-modeling-of-real-world-vocalizations-from-nonverbal-individuals/
- [S16] How to elicit many probabilities (UAI; verbal-numerical scale; noisy-OR context): https://arxiv.org/pdf/1301.6745 ; summaries via search results listing van der Gaag's elicitation work
- [S17] SHELF R package description: https://est.colpos.mx/web/packages/SHELF/index.html
- [S18] Equivalent sample size sensitivity (BDeu): https://arxiv.org/pdf/1206.5293 ; Learning Bayesian networks, Dirichlet priors: https://arxiv.org/pdf/1302.4957
- [S19] Dirichlet-multinomial empirical Bayes, Minka fixed point: https://bayesianbandits.readthedocs.io/en/latest/_sources/math/dirichlet-eb.rst.txt
- [S20] Rule of three (Hanley and Lippman-Hand 1983, JAMA 249:1743-5): https://en.wikipedia.org/wiki/Rule_of_three_(statistics) ; https://jamanetwork.com/journals/jama/article-abstract/389714
- [S21] Light's four purposes (secondary summaries): https://praacticalaac.org/?wpfb_dl=51 ; https://thinksmartbox.com/news/getting-started-with-aac-competencies/
- [S22] CSBS (Wetherby and Prizant), cluster names: https://txautism.net/evaluations/communication-and-symbolic-behavior-scales
- [S23] Imperative versus declarative pointing (Bates, Camaioni, Volterra 1975, via secondary): https://pmc.ncbi.nlm.nih.gov/articles/PMC2043155 ; https://www.eva.mpg.de/documents/Wiley-Blackwell/Tomasello_New_ChildDev_2007_1554823.pdf
- Local files read: `config/likelihoods.json`, `audio_text.py` (BayesianFusion), `meaning.py` (MeaningComposer), `docs/PLAN.md`, `docs/RESEARCH-interpretation.md`.
