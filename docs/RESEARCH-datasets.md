# Datasets that fit Tink's users and its conversational layer (2026-10-24)

Read-only research; nothing here was run on Tink's data and no dataset was downloaded. One exception: I pulled the
WLASL metadata JSON (a 12 MB list of glosses and video URLs, no video) into a scratch folder to count which core words it has.
Builds on `RESEARCH-mapping.md` (ReCANVo, CNVVE, MIntRec2.0, Communication Matrix are not redone). Tags: **[E]** stated by a listed
source, **[X]** arithmetic or a count I made from a fetched file, **[J]** my judgment. Refs like [S3] point to Sources.

## 0. Bottom line

1. **No open dataset links raw gestures or vocalizations to intended meaning for AAC or minimally verbal users.** ReCANVo (already
   covered) is still the only one found. The clinical gesture-coded corpora (AphasiaBank, DementiaBank) are behind a membership
   process that needs a faculty sponsor, and their gesture coding records gesture form and its relation to speech, not "this
   gesture means YES" [S1, S2, S3]. Not usable this weekend. [E + J]
2. **The conversation layer is where new data can help, and it can help today.** The Switchboard Dialog Act corpus (SwDA) labels
   yes answers, no answers, agree, reject, backchannel, signal-non-understanding, repeat-phrase and requests, with counts, and the
   annotated transcripts are a direct download [S20, S21]. That gives `P(reply act | partner act)` with acts that map onto Tink's
   Yes, No, Repeat and Help intents. DailyDialog's 4 acts (inform, question, directive, commissive) are too coarse for that [S27]. [E + J]
3. **For keyword signs, WLASL covers most of the core AAC words but with 7 to 35 videos per word and only as ASL.** It is usable
   for a small, optional "signs you already know" pack and for evaluation, not as a general recognizer for idiosyncratic gestures. [X + J]
4. **Dysarthric speech datasets are audio for ASR. None is conversational, and the two best (Speech Accessibility Project, UASpeech)
   need an application that does not finish this weekend.** Euphonia is closed. [E]
5. **L2 corpora are interview- or exam-style, with no act labels.** NICT JLE is the only one that is a direct, openly licensed
   download. It is low value for Tink. [E + J]

## 1. Summary table

| Dataset | Population | Modalities | Meaning / intent labels? | Size | License / access | Usable this weekend? | Use in Tink |
|---|---|---|---|---|---|---|---|
| **SwDA** (Switchboard Dialog Act) | Typical adults, telephone chat between strangers | Text (Switchboard-1 transcripts) + DA tags | Yes: 200+ tags, collapsed to 44 in the manual; includes Yes answers, No answers, Agree/Accept, Reject, Acknowledge, Signal-non-understanding, Repeat-phrase, Action-directive | 221,616 utterances [S21]; 1,155 conversations / 205k utterances in the Stolcke et al. paper [S22] | Annotations tarball is a direct download from Jurafsky's page with no stated terms [S20]; Potts's `swda.zip` + code is on GitHub (code GPL-2.0) [S21, S21b]. Underlying Switchboard-1 is an LDC corpus; terms for the text are unclear (see 3.5) | **Yes, minutes** | Conversation layer: `P(reply act \| partner act)`; act-conditioned reply phrases |
| **ICSI MRDA** (via Edinburgh AMI site) | Typical adults, research meetings | Text + DA coding (audio optional, large) | Yes: DA tags (SWBD-DAMSL-derived), 180,000+ tags over ~72 h / 75 meetings [S24] | Annotation archives 19-53 MB [S25] | CC BY 4.0, no registration mentioned on the download page [S25] | **Yes** | Second DA source with a clean license; cross-check of SwDA transition rates. Label mapping not verified (see 3.5) |
| **AMI meeting corpus** | Typical adults, scenario and real meetings | Audio, video, text, DA + addressee | Yes: 15 DA types incl. BACKCHANNEL, INFORM, ELICIT-INFORM, SUGGEST, OFFER, ASSESS, BE-POSITIVE, BE-NEGATIVE [S26] | Not verified | CC BY 4.0 for signals, transcripts and some annotations; NXT-format download, one source says registration [S23, S26b] | Possible but format cost | Not recommended: no yes/no-answer tag (answers fall under ASSESS), NXT parsing overhead |
| **DailyDialog** (already in use) | Typical adults, written ESL-practice dialogues | Text | 4 acts (inform, question, directive, commissive) + emotion | 13,118 dialogues [S27] | CC BY-NC-SA 4.0 on HF/ConvLab cards; a Kaggle mirror says CC0 [S27] | Already used | Keep as reply-phrase source; do not use its acts as the act model |
| **AphasiaBank** | People with aphasia (post-stroke) + controls | Video + CHAT transcripts; gesture coding in ELAN for lab studies | Partial: gesture type, function relative to speech (e.g. redundant, supplementary), onset/offset; not gesture-to-intent. Codes only unambiguous, non-elicited gestures [S3] | Not verified (per-study samples of 46-98 people) | Password-protected, consortium members; students need a faculty advisor to request; "no response to student-only requests"; repository "under review" notice [S1] | **No** (days to weeks, needs sponsor) | Reference for gesture-function codes. Evaluation only if access arrives |
| **DementiaBank** | People with dementia | Audio/video + CHAT | Not a gesture-meaning resource (not checked in depth) | Not verified | Same membership process as AphasiaBank [S2] | **No** | Not usable |
| **ASDBank** | Autistic children and adults | Multimedia + CHAT | Not verified which corpora code gesture or function | Not verified | Registration with email + password and email confirmation for most data; a few controlled items need established-researcher verification [S4] | Registration likely same day, but no corpus identified that helps | Browse the corpus index once; low expected value |
| **CHILDES Rowe corpus** | 47 typically developing infants (10-18 months) with caregivers | Video (media link) + CHAT with `%gpx`/`%gcd` gesture tiers | Yes, for gestures: category (deictic / conventional / representational), form, **gloss** (conventional gestures glossed with their meaning, e.g. "yes") and relation to speech [S5, S6] | ~218 recordings of 15-min play sessions [S5] | Zip download and browsable transcripts via TalkBank; must cite listed papers; registration not mentioned [S5] | Yes, but wrong population | Taxonomy reference (deictic / conventional / representational; reinforcing / supplementary / disambiguating / gesture only). Not a prior for Tink users |
| **CHILDES INCA-A `%spa`** (New England corpus etc.) | Typical children + caregivers | Transcripts with speech-act codes | Yes: 67 speech-act types (directives, questions, statements, markings, ...) [S7] | Not verified | CHILDES / TalkBank terms | Possible | Vocabulary of communicative acts only. Not AAC |
| **ANAWC** (AAC and Non-AAC Workplace Corpus) | 8 focal participants, AAC users and non-AAC users at work | Spoken interaction transcripts | Not verified | 200+ h, ~1M words [S8] | Not found; contact authors | **No** (no access route found) | Would be the best match for AAC-mediated conversation if released |
| **Vertanen "comm2" AAC-like test set** (already in use) | Crowd workers imitating AAC messages | Text | No | 1,506 unique communications from 10 hypothetical situations [S9] | Public | Already used | The authors note no public set of real AAC communications existed [S9] |
| **TORGO** | 8 dysarthric (CP, ALS) + 7 control speakers | Audio (16 kHz) + 3D articulatory (EMA); video excluded | No intent labels. Prompts: non-words, words, restricted and unrestricted sentences (photo descriptions) | ~23 h; 18 GB [S10, S11] | Free for academic non-profit use, direct download; also on LDC as LDC2012S02 [S10, S11] | Downloadable, **not useful** | ASR evaluation only; not conversational; 18 GB |
| **UASpeech** | 16 speakers with CP-related dysarthria in the current release + controls | Audio, isolated words | No | 765-word vocabulary [S12] | Email request with institution, intended use, period; academic and government labs only; no redistribution [S12] | **No** (request/approval time unknown) | Isolated-word recognition evaluation. Not conversational |
| **Speech Accessibility Project** | Parkinson's; since mid-2024 also ALS, CP, Down syndrome, stroke | Audio + transcripts; some perceptual ratings | No intent labels. Includes spontaneous answers to questions plus read commands and sentences [S13] | ~1,090 h, ~1,130 participants (project page, via search summary) [S13] | Signed data use agreement + one-page proposal; receipt confirmed within a week, DUA execution "a few more weeks" [S13, S14] | **No** | ASR robustness, later. Not conversation pairs |
| **Project Euphonia** | Speakers with speech impairments | Audio | No | >1.5 M utterances, ~3,000 speakers (Feb 2025 figure) [S15] | Not available to outside teams [S15] | **No** | Not usable |
| **Trinity Lancaster Corpus** | L2 English speakers (B1-C2) in GESE oral exams with L1 examiners | Transcripts of examiner-candidate interaction (audio not mentioned) | No act labels found | 4.2 M words, 2,000+ L2 speakers [S16] | Free via password-protected TLC Hub; password printed on the Lancaster page; license for the data and raw download not stated [S16] | Browse only | Phrasing of L2 replies to examiner prompts; hub is a query tool, bulk export unverified |
| **ICNALE Spoken Dialogue** | Asian L2 English learners, interviewed | Transcripts (+ audio on request not verified) | No act labels | 425 participants, 4,250 samples, ~1.6 M words [S17] | Download after agreeing to terms; **redistribution prohibited and analysis of interviewer speech prohibited**; no license named [S17] | Technically yes, **but interviewer turns are off-limits** | Blocked: Tink needs partner-turn to reply pairs |
| **NICT JLE** | 1,281 Japanese learners in ACTFL-ALC oral proficiency interviews | Transcripts (TXT), 30+ basic tags incl. speaker turns; error tags on 167 files | No act labels | 1.2 M words, ~300 h audio source, 7 MB download [S18] | CC BY-SA 3.0, direct download, no application [S18] | **Yes** | Weak: interview-style Q to answer pairs from L2 speakers; marginal |
| **WLASL** | ASL signers, 119 signers in the metadata [X] | Isolated-sign video (YouTube-sourced) | Gloss labels (word meaning of the sign) | 2,000 glosses, 21,083 instances in the metadata [X]; many source videos may be dead | C-UDA 1.0: academic and computational use only, no commercial use [S19]. You scrape videos with `yt-dlp`; missing ones by request form, ~7 days [S19] | Partly (download of ~100-200 target-word videos feasible; completeness uncertain) | Optional keyword-sign pack; evaluation clips for MediaPipe-based sign recognizers |
| **ASL Citizen** | 52 ASL signers (community-sourced, consented) | Isolated-sign video, webcam | Gloss labels from ASL-LEX vocabulary | 83,399 videos, 2,731 signs, ~30.5 videos per sign [S28, S29] | Free research license, wget from Microsoft Download Center; zip size not stated; do not use to replace interpreters or signing teachers [S29, S30] | Probably (download size unknown) | Cleaner, more videos per sign than WLASL. Core-word coverage not verified |
| **How2Sign** | ASL signers | Continuous signing, multiview video, depth, keypoints, gloss, English translation | Sentence-level translations | 80+ hours | CC BY-NC 4.0; Google Drive links, download page "under construction" [S31] | Not useful | Continuous signing, not isolated core words |
| **HaGRID** (adjacent; not asked) | 65,977 adults | RGB images + MediaPipe hand landmarks in a separate annotations download | Gesture-shape class labels, no intent | 1,086,158 images; 33 gesture classes + `no_gesture`; 1.5 TB full [S32] | "Variant of CC BY-SA 4.0" with a license PDF [S32] | Landmark annotations only: possibly | Pretraining/evaluating static hand-shape classes (like, dislike, stop, ok, palm, fist, point, mute); images not needed |
| **Makaton / key-word-sign video sets** | | | | | None found; one 2023 paper trains MediaPipe-based Makaton recognition on its own custom videos, release not stated [S33] | **No** | Gap |

## 2. Bottom-line coverage of the five questions

| Question | Answer |
|---|---|
| 1. Partially/minimally verbal users with signal-to-meaning labels | No open dataset. ReCANVo stays the only one; Rowe CHILDES has glossed gestures but for typical infants |
| 2. Dysarthric speech | Audio for ASR only; two need applications, one is closed, TORGO is open but not useful |
| 3. L2 conversation | Interview/exam style, no act labels; NICT JLE open, ICNALE blocks interviewer turns |
| 4. Signs for core AAC words | WLASL covers most (section 3.4); recognizer from landmarks is feasible only for a small keyword-sign pack |
| 5. Dialogue acts | SwDA is the best direct fit; ICSI MRDA is the license-clean second source |

## 3. Notes per dataset

### 3.1 AphasiaBank, DementiaBank, ASDBank, CHILDES (TalkBank) [S1-S7]

- **Access.** AphasiaBank and DementiaBank data are password protected and limited to consortium members. Applicants register with TalkBank,
  read the Ground Rules, and email talkbank@cmu.edu with affiliation and intended use. Students and lab workers must have a faculty
  advisor submit their own membership request and the student request with start and end dates; requests from students alone get no
  answer [S1, S2]. Both pages carry a note that the repository is under review for modification, so confirm status before relying on it.
  The Ground Rules say TalkBank data is CC BY-NC-SA and the license precludes use in commercial products, including systems such as large
  language models; they also bar uploading TalkBank data to web services that keep it [S2]. That last clause matters if any pipeline step
  calls a hosted LLM on transcripts. [E]
- **ASDBank** is the open-door option: most data needs only email-and-password registration with email confirmation; a few materials need
  verification as an established researcher [S4]. The index page I could read does not list corpora, so I could not tell which ones
  (if any) include minimally verbal participants or gesture coding.
- **What AphasiaBank gesture coding is.** The NEURAL lab manual codes gesture type (e.g. deictic, pantomime), usage (e.g. emphasis,
  replacement for speech), onset/offset and gesture space in ELAN, plus an informational relationship to speech (Redundant, Supplement, and others).
  It says to code only clearly codable, unambiguous, non-elicited gestures; its example of redundancy is "yes" said with head nods [S3]. So
  it labels how a gesture relates to speech, not what a gesture means for an AAC message. The only gesture coding available on an open
  page is 20 ELAN files by Ferré, which need the member-only videos [S1b].
- **Rowe CHILDES corpus** is the one clearly open, clearly gesture-glossed corpus: `%gcd` lines hold category (`$D`, `$C`, `$R`), form, a `#gloss`
  (for conventional gestures, the agreed meaning, e.g. yes or be quiet) and a relation to speech (reinforcing, supplementary,
  disambiguating, gesture and vocalization, gesture only) [S5, S6]. Participants are 47 typically developing infants, so
  the numbers say nothing about Tink's users. Its value is the coding vocabulary. Note its definition: conventional gestures should be
  understandable "in isolation" with one-to-one meaning, while deictic gestures depend on the referent [S6]. That matches the mapping note
  that a point is Show or Want depending on context. [J]
- **INCA-A speech act codes** (`%spa`, 67 speech act types) are the communicative-function annotation in CHILDES; the New England corpus is
  described as the largest child-caregiver dataset annotated for speech acts [S7]. Typical children; use it as taxonomy only.
- **Communication-function annotated AAC interactions.** I found no open dataset of AAC users' interactions coded with request, protest or comment.
  The Communication Matrix is an assessment instrument, not a corpus (already covered).

### 3.2 AAC-mediated conversation corpora

- **ANAWC** is the closest match: 200+ hours (~1M words) of workplace interaction with eight focal participants, comparing AAC and non-AAC users [S8].
  I found no download or access statement; it would be an email to the authors, not a weekend task.
- **Vertanen's comm2** test set (already in use) is crowd-written. Its authors state that to their knowledge no such public test set of
  actual AAC communications existed [S9].

### 3.3 Dysarthric and atypical speech [S10-S15]

- **TORGO.** 8 speakers with CP or ALS and 7 controls, about 23 hours, 16 kHz audio with 3D articulography. Prompts are non-words, short
  words, restricted sentences and spontaneous photo descriptions. Video is excluded. Free for academic non-profit use with citation of one of three listed papers;
  four archives total 18 GB [S10, S11]. Not conversational and nothing here maps to intent. Use only to test a speech recognizer on dysarthric audio, and the
  18 GB download is not worth it unless Tink's partial-speech path has a recognizer to test.
- **UASpeech.** 765 isolated words; 19 original speakers with CP-related dysarthria, 16 in the current distribution after three
  withdrew or were corrupted. Access by email (name, institution, intended use, institutional email, a cloud account to deliver the data, access period) and limited to government and academic research labs; commercial use and redistribution not allowed [S12]. Turnaround is not stated.
- **Speech Accessibility Project.** About 1,090 hours from about 1,130 participants (ALS, CP, Down syndrome, Parkinson's, stroke), with transcripts of what was said;
  releases before June 2024 are Parkinson's only [S13, S14]. Prompt types are computer commands, novel sentences and spontaneous answers to
  questions about daily life. Access needs a signed DUA and a one-page proposal; receipt is confirmed within a week and DUA execution takes a few more weeks [S14]. Not usable this weekend; a
  reasonable follow-up if partial-speech recognition becomes a goal.
- **Project Euphonia.** Corpus is not available to outside teams; Google points to the Speech Accessibility Project [S15].

### 3.4 Sign and key-word-sign data [S19, S28-S33]

**WLASL core-word coverage (counted from the metadata JSON, [X]).** 2,000 glosses, 21,083 instances, 119 signers, 14,289/3,916/2,878 train/val/test. Instances per gloss
(rank by instance count in parentheses):

| Word | Instances | Word | Instances |
|---|---|---|---|
| yes | 22 (18th) | help | 22 (13th) |
| no | 22 (14th) | stop | 11 |
| more | 15 | eat | 19 |
| drink | 35 (2nd) | bathroom | 16 |
| toilet | 13 | finish | 21 |
| done | 7 | again | 16 |
| want | 19 | wait | 18 |
| please | 15 | hungry / thirsty | 12 / 12 |
| hello | 13 | bye | 7 |
| pain | 10 | hurt | 12 |
| thanks | not a gloss | all | 21 |

- "ALL DONE" is not a gloss; `finish` (21) and `done` (7) are the closest. "Thank you" appears under another gloss or not at all (only `thanks` was checked).
- These are instance counts in the metadata. Many source URLs are dead, and a secondhand mirror held about 12k videos instead of 21k [S19b]; the maintainers will send missing links on request within about a week [S19]. Expect fewer than the table.
- A 2022 paper found no one-to-one match between sign productions and gloss labels in WLASL and provided consistent annotations for 19,672 examples [S19b]. Treat the labels as noisy. [E]

**ASL Citizen.** 83,399 videos of 2,731 signs by 52 signers, about 30.5 videos per sign, collected by webcam with consent, vocabulary from ASL-LEX [S28, S29]. Each
sign has more and cleaner examples than WLASL. I could not confirm which of Tink's core words it contains. Terms ask for Deaf community involvement and say the data should not be used for technology that replaces
interpreters, signing educators or accommodations; it is meant for isolated-sign recognition [S29]. A keyword recognizer for an AAC user is a different use but the restriction should be quoted in any write-up. Download size is not stated.

**How2Sign.** 80+ hours of continuous ASL, CC BY-NC 4.0, download page marked under construction [S31]. Sentence-level, so not useful for isolated core words.

**Makaton / key-word signing / home signs.** No public video set found. Makaton's core vocabulary is about 450 concepts per a search summary, used alongside speech [S33]. Home signs are by definition idiosyncratic. This is a gap that
the per-user enrollment in `RESEARCH-mapping.md` already addresses. [E + J]

**Could a small recognizer be trained from MediaPipe landmarks?** [J]
- Feasible only as a closed set of about 8 to 12 signs, with 10 to 35 clips per sign. Landmark extraction on a few hundred clips is minutes to an hour; the bottleneck is acquiring the clips.
- Many ASL signs depend on motion, location relative to the face, and two hands (HELP, MORE, STOP differ mainly in motion and placement). Single-frame static classifiers will confuse them; it needs a short sequence model, which adds time.
- ASL-trained recognition tells you nothing about Tink's idiosyncratic or motor-impaired users, whose gestures differ even for the same intent (AllyAAC, cited in the mapping note). The right framing is "a keyword-sign pack for users who already sign," not the mapping layer.
- HaGRID's separate annotations hold MediaPipe landmarks for static hand shapes (like, dislike, stop, ok, palm, fist, point, mute, call). If the annotations can be downloaded without the images (size not verified), it is a cheaper path to a generic hand-shape detector than WLASL, but those labels are shape names, not meanings [S32].

### 3.5 Dialogue-act corpora [S20-S27]

**SwDA.** Extends Switchboard-1 Release 2 with utterance-level DA tags: 221,616 utterances, over 200 tags, collapsed to 44 in the coders' manual [S21]. (The Stolcke et al. paper uses 42 and 1,155 conversations / 205,000 utterances [S22]; sources disagree on 42 vs 43 vs 44 tags, which does not matter for Tink.) Counts for the tags that matter, from the full corpus table [S21]:

| Tag | Code | Count |
|---|---|---|
| Acknowledge (backchannel) | `b` | 38,298 |
| Agree/Accept | `aa` | 11,133 |
| Yes-No-Question | `qy` | 4,727 |
| Yes answers | `ny` | 3,034 |
| Wh-Question | `qw` | 1,979 |
| No answers | `nn` | 1,377 |
| Action-directive (request) | `ad` | 746 |
| Repeat-phrase | `b^m` | 688 |
| Open-Question | `qo` | 656 |
| Signal-non-understanding | `br` | 298 |

Other answer tags exist (affirmative non-yes `na`, negative non-no `ng`, other answers `no`, dispreferred `arp_nd`), plus Reject `ar` and Maybe/Accept-part `aap_am`, backchannel in question form `bh`, response acknowledgement `bk`, downplayer `bd` [S21].

Fit to Tink's 8 intents [J]: Yes to `ny`/`na`/`aa`; No to `nn`/`ng`/`ar`; Repeat to `br` (partner-side signal-non-understanding) and `b^m`; Help to `ad` (as a request) but only roughly; Attention and Show have no counterpart; Want has none. The Wh- and Yes-No-Question tags are the partner-turn side, which is exactly what `meaning.py` already uses for context. Caveats: (a) Switchboard is phone chat between strangers on assigned topics, and the 38k backchannels dominate any unconditional rate, so condition on the previous act and drop backchannels from reply targets; (b) many yes/no answers in speech are not literally "yes" ("Yeah", "uh-huh" are tagged by function), which is the same function-over-form rule AMI states [S26]; (c) Tink's users give short replies, so SwDA's long turns are not a natural-length match, but the *act transition* is what you take, not the words.

License is the open question. The annotation tarball page states no terms [S20]; the Potts GitHub repo is GPL-2.0 for its code and does not state terms for the corpus [S21b]; the Potts page footer is CC BY-NC-SA 3.0 for the page content [S21]; a Hugging Face mirror card was reported as CC BY-NC-SA 3.0 and an aggregator pointed to LDC97S62 [from search, unverified]. For a non-commercial hackathon demo that is fine; for anything shipped, check LDC terms.

**ICSI MRDA.** 75 meetings, ~72 hours, 180,000+ dialog act tags, SWBD-DAMSL-derived with one general tag and optional special tags per act [S24]. The Edinburgh site offers the ICSI core annotations (19 MB; transcripts + DA coding) under CC BY 4.0 with no registration noted [S25]. I could not read the ICSI labeling guide (connection refused), so I do not know the general-tag inventory or exactly how yes/no answers are coded; confirm that before writing a mapping. Meetings are task talk among colleagues, which is farther from Tink's conversations than Switchboard.

**AMI.** 15 DA types: BACKCHANNEL, STALL, FRAGMENT, INFORM, ELICIT-INFORM, SUGGEST, OFFER, ELICIT-OFFER-OR-SUGGESTION, ASSESS, COMMENT-ABOUT-UNDERSTANDING, ELICIT-ASSESSMENT, ELICIT-COMMENT-ABOUT-UNDERSTANDING, BE-POSITIVE, BE-NEGATIVE, OTHER [S26]. There is no yes-answer / no-answer tag; the manual codes by function ("no" can be agreement after a negative-form question) and, notably, says nods count as dialogue acts but only acts with a transcript trace are annotated [S26]. So AMI gives no handle on nonverbal replies either. Skip.

**DailyDialog** (already in the pipeline) has 4 act labels. The commissive label (accepting or rejecting requests and suggestions) is the only one near yes/no [S27]. Keep it for reply phrases and use SwDA for act transitions.

### 3.6 L2 conversation corpora [S16-S18]

- **Trinity Lancaster Corpus.** 4.2 M words from 2,000+ L2 speakers in GESE exams, with two to four tasks of spoken interaction with an L1 examiner [S16]. Access is the password-protected TLC Hub; whether text can be exported in bulk and under what license is not stated on the page I read. No act labels found.
- **ICNALE Spoken Dialogue.** 30-40 minute interviews with picture description and role play, 425 participants, ~1.6 M words, but the terms bar redistribution and analysis of interviewer speech [S17]. Pairing partner turns with replies is exactly an analysis of interviewer speech, so this is blocked by its own terms. The monologue module (4,400 one-minute samples) has no partner turn.
- **NICT JLE.** 1,281 OPI interviews, ~1.2 M words, CC BY-SA 3.0 direct 7 MB download, 30+ basic tags covering interview structure, speaker turns and utterance phenomena (fillers, repetitions, self-corrections) [S18]. It has interviewee level ratings and disfluency tags but no communicative act labels. Use only if you want realistic L2 disfluent replies; interview structure makes it a poor stand-in for casual conversation.
- None of the three is annotated for intent, and all three are test or interview settings where the examiner controls the topic.

## 4. Recommendation: top 3 to add now, in order

1. **SwDA for conversation-layer act priors.** Download `swb1_dialogact_annot.tar.gz` from Jurafsky's page (or Potts's `swda.zip`), keep the collapsed tags, build a transition table from partner act (`qy`, `qw`, `qo`, `ad`, statements) to reply act (`ny`, `nn`, `aa`, `ar`, `br`, `b^m`, answers, requests), drop backchannel-only turns from reply targets, and use it to weight Yes, No, Repeat and Help given the partner's turn type. Cost: about 2 to 3 hours to parse and validate. Why first: it is the only source found that directly supplies the acts Tink's replies are made of, with usable counts (3,034 yes, 1,377 no answers; 4,727 yes-no questions), open to download today. Treat the license as non-commercial until checked. [J]
2. **WLASL core-gloss subset (plus ASL Citizen if its zip is manageable) as an optional keyword-sign pack and as evaluation clips.** Core words are present (yes, no, help, more, drink, eat, bathroom, finish, stop, again, want, please). Download the clips for about 10 target glosses only; extract MediaPipe landmarks; train or evaluate a small sequence classifier. Cost: 3 to 5 hours, mostly clip acquisition and the risk of dead links. Label this as ASL-based and aimed at users who already sign; it does not replace per-user enrollment. Do not start this unless the demo needs sign input; otherwise the gain is mostly a talking point. [J]
3. **ICSI MRDA annotations (CC BY 4.0) as the second DA corpus.** It is a 19 MB download, has a clean license, and lets you check whether SwDA's act-transition rates hold in a different domain. Before writing a mapping, read the ICSI labeling guide to see how yes/no answers are tagged (not verified here). Cost: 1.5 to 2 hours after step 1 exists, since the parser and mapping can be reused. If the license of SwDA text turns out to be a problem, MRDA is the fallback. [J]

**Do not spend the weekend on:** AphasiaBank, DementiaBank, Speech Accessibility Project, UASpeech, ANAWC or Euphonia (all blocked by applications or sponsorship, with no dataset worth the wait for a hackathon). If Tink continues past the hackathon, start the Speech Accessibility Project proposal and, through a faculty advisor, the AphasiaBank request in parallel, because both take days to weeks. TORGO, ICNALE and NICT JLE are open or nearly so but give nothing the product needs.

## Could not verify

- **ASL Citizen core-word coverage** (YES, NO, MORE, STOP, HELP, EAT, DRINK, BATHROOM, ALL DONE): no gloss list was read, and the zip size is not stated.
- **WLASL license text** (C-UDA 1.0 PDF not read), the **current count of downloadable videos** (the 12k figure is from a secondhand mirror), and signer counts beyond the metadata (119 is my count of `signer_id` values).
- **SwDA license for the transcript text.** No terms on the Jurafsky page; Potts's GitHub says GPL-2.0 for the repo; footer CC BY-NC-SA 3.0 applies to the page; the LDC pointer and the Hugging Face license came from search summaries. The tag-count discrepancy (42/43/44) is unexplained in my sources.
- **MRDA general-tag inventory, yes/no-answer coding and mapping to SWBD-DAMSL:** the ICSI tech report (TR-04-002) could not be fetched. The "5 general / 11 / 52" figures I started with were not confirmed and are not used.
- **AMI size in meetings and the coding of license for each annotation layer**; one source says download needs registration, the Edinburgh annotation page I read does not mention it.
- **AphasiaBank size** (participants, hours), what the English transcripts include for gestures outside lab studies, and which AphasiaBank/DementiaBank corpora carry gesture tiers. The "under review" notice on TalkBank pages could mean access changes.
- **ASDBank corpus list**: the corpus index was not read, so whether any corpus has minimally verbal participants or gesture/function coding is unknown.
- **ANAWC access.** **Trinity Lancaster** bulk export and data license. **ICNALE** audio availability and license name. **UASpeech** turnaround and exact vocabulary composition.
- **Speech Accessibility Project** exact license name, hours per etiology, and whether the 1,090 h / 1,130 participants figure is current (it comes from a search summary of the project page).
- **HaGRID annotations-only download size** and the exact license terms (PDF not read).
- **CNVVE-like head nod/shake datasets** and datasets of gestures with communicative intent in clinical populations beyond the ones above: searches found nothing, which is a fact about my queries, not proof none exists.
- **Makaton / key-word-sign datasets**: none found by one search; the 2023 recognizer paper's data availability was not checked.

## Sources

- [S1] AphasiaBank home (access rules, "under review" notice): https://talkbank.org/aphasia/ ; [S1b] gesture resources (Ferré ELAN files, City Gesture Checklist): https://talkbank.org/aphasia/gesture/
- [S2] TalkBank Ground Rules and access levels; DementiaBank: https://talkbank.org/share/rules.html ; https://talkbank.org/0share/access.html ; https://talkbank.org/dementia/
- [S3] NEURAL lab co-speech gesture coding manual (AphasiaBank protocol): https://talkbank.org/aphasia/access/English/Protocol/0docs/NEURAL-2-Gesture.docx
- [S4] ASDBank home and access levels: https://talkbank.org/asd/ ; https://talkbank.org/asd/access.html
- [S5] CHILDES Rowe corpus: https://talkbank.org/childes/access/Eng-NA/Rowe.html
- [S6] Rowe Lab gesture coding manual (2017): https://talkbank.org/childes/access/Eng-NA/0docs/Rowe-Gesture.docx
- [S7] INCA-A / `%spa` speech-act codes (search summary; speech-act code list): https://talkbank.org/0info/resources/rely/codes.doc ; https://talkbank.org/childes/access/Eng-NA/NewEngland.html
- [S8] ANAWC: https://research.polyu.edu.hk/en/publications/an-introduction-to-the-anawc-the-aac-and-non-aac-workplace-corpus/ ; https://research.uniroma1.it/node/30714
- [S9] Vertanen, crowdsourced AAC-like communications (comm2): https://www.keithv.com/pub/comm2/vertanen-comm2.pdf
- [S10] TORGO (Toronto): https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html ; [S11] LDC2012S02: https://catalog.ldc.upenn.edu/LDC2012S02
- [S12] UASpeech (Illinois): https://speechtechnology.web.illinois.edu/uaspeech/
- [S13] Speech Accessibility Project, research page and FAQ: https://speechaccessibilityproject.beckman.illinois.edu/conduct-research-through-the-project ; [S14] https://speechaccessibilityproject.beckman.illinois.edu/conduct-research-through-the-project/faq-on-using-our-data ; registry: https://clinicaltrials.gov/study/NCT05889260
- [S15] Project Euphonia: https://sites.research.google/euphonia/about/faq ; Frontiers 2025: https://www.frontiersin.org/journals/language-sciences/articles/10.3389/flang.2025.1569448/pdf
- [S16] Trinity Lancaster Corpus: https://cass.lancs.ac.uk/trinity-lancaster-corpus/
- [S17] ICNALE: http://language.sakura.ne.jp/icnale/
- [S18] NICT JLE: https://alaginrc.nict.go.jp/nict_jle/index_E.html
- [S19] WLASL repository (C-UDA, download steps): https://github.com/dxli94/WLASL ; [S19b] ASLLRP annotations for WLASL and mirror notes: https://dai.cs.rutgers.edu/dai/s/aboutwlasl
- [S20] SwDA annotations (Jurafsky): https://web.stanford.edu/~jurafsky/ws97/
- [S21] SwDA (Potts; tag table and counts): https://compprag.christopherpotts.net/swda.html ; [S21b] https://github.com/cgpotts/swda
- [S22] Stolcke et al., Dialogue Act Modeling for Automatic Tagging and Recognition of Conversational Speech: https://arxiv.org/pdf/cs/0006023
- [S23] AMI corpus: https://groups.inf.ed.ac.uk/ami/corpus/ ; annotation page https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml
- [S24] Shriberg et al., ICSI MRDA corpus (SIGDIAL 2004): https://www.sri.com/publication/the-icsi-meeting-recorder-dialog-act-mrda-corpus/ ; labeling guide (not fetched): https://www1.icsi.berkeley.edu/ftp/pub/techreports/2004/tr-04-002.pdf
- [S25] ICSI download on the AMI site: https://groups.inf.ed.ac.uk/ami/icsi/download
- [S26] AMI dialogue act guidelines 1.0: https://groups.inf.ed.ac.uk/ami/corpus/Guidelines/dialogue_acts_manual_1.0.pdf ; [S26b] AMI download: https://groups.inf.ed.ac.uk/ami/download
- [S27] DailyDialog (Li et al. 2017): https://alphaxiv.org/abs/1710.03957 ; card https://huggingface.co/datasets/ConvLab/dailydialog/blob/main/README.md
- [S28] ASL Citizen paper: https://arxiv.org/abs/2304.05934 ; [S29] datasheet: https://www.microsoft.com/en-us/research/project/asl-citizen/datasheet/ ; [S30] project page: https://www.microsoft.com/en-us/research/project/asl-citizen/
- [S31] How2Sign: https://how2sign.github.io/
- [S32] HaGRID: https://github.com/hukenovs/hagrid
- [S33] Makaton recognition paper (summary only): https://paperguide.ai/papers/83a9fb6d-2d9f-4c97-a299-eb0d40623d03-makaton-sign-language-recognition
- Local file read: `docs/RESEARCH-mapping.md`. WLASL metadata counted from https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json (annotation list, no video).
