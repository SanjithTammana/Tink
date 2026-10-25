# Modeling Tink's final layer on published work (2026-10-24)

Read-only literature and code survey. Nothing here was run on Tink's data. Tags: **[E]** read in a source listed under Sources,
**[S]** secondary summary only (search snippet or third-party page, not the primary text), **[J]** my judgment or extrapolation.
Where a number or license could not be confirmed it says so, and it is repeated under "Could not verify".

Why this survey: the two LLM attempts in `llm.py` (compose from frame; select-and-adapt) used the LLM as a **replacement generator**
and lost to retrieval on held-out dialogues (hit@3 0.42 vs 0.48, `EVALUATION.md` 4b). The published systems below mostly use the
LLM differently: as a **proposal generator plus scorer** inside a noisy-channel or n-best rescoring loop, where the sensor-side
model keeps a vote and real candidates are never discarded. [J]

## 1. Systems table

| System | Year | What the LLM layer does | Inputs | Outputs | Metric / result | Code (license) |
|---|---|---|---|---|---|---|
| SpeakFaster, Cai et al. (Nat. Commun.) [E] | 2024 | Fine-tuned LaMDA 64B (decoder-only) expands abbreviations (KeywordAE) and fills masked words (FillMask). Two separately fine-tuned models, about 1.8M synthesized {context, abbreviation, phrase} triplets from four public dialogue sets. | Previous dialogue turns (separated by curly braces) + abbreviation (initials, optionally spelled keywords) | Top 5 phrases. 128 (or 256 when more than 5 initials+keywords) temperature samples at T=1.0, deduplicated, ranked (ranking rule not stated; frequency is my reading [J]) | Exact match in top 5: 72.6% (v1) / 76.3% (v2) for initials only. Simulated keystroke saving 0.640-0.657 vs Gboard 0.482. Eye-gaze ALS users 29-60% faster. Latency about 843 ms (AE) | `webui` only: TeamGleason/SpeakFaster (MIT). Model weights and server not released |
| KWickChat, Shen, Yang, Dudley, Kristensson (IUI) [S] | 2022 | Fine-tuned GPT-2 generates a sentence from a bag of keywords + dialogue history + persona. Generates M=50 candidates, **re-ranks them with an unmodified GPT-2** (exp of cross-entropy as score), shows top N. | Keywords, multi-turn history, persona tags | N suggested sentences | Re-ranking raised average keystroke saving 15.9% relative at 3 suggestions; negligible gain above 8 suggestions. About 71% keystroke saving reported with word prediction at WER threshold 0.65. Human judges: median 4/5 for best sentence | No repository found |
| KWickChat 2 (reported in Promptor) [S] | 2023 | Adds punctuation as input to the keyword model | Keywords + 4 punctuation marks | Sentences | BLEU +79.3% over no punctuation at bag size 3 | None found |
| Valencia et al. (CHI) "The less I type, the better" [E via earlier note] | 2023 | LLM phrase suggestions (extend a reply, answer, ask for help) in a study with 12 AAC users | Short user input + context | Phrases | Qualitative: users want time savings but their own style | None found |
| COMPA, Valencia et al. (CHI) [S] | 2024 | LLM writes starter phrases for a transcript turn the user marks, given a user-chosen intent | Live transcript turn + intent (reply, opinion, question, yes/no) | Starter phrases | Qualitative only (not read) | None found |
| AACessTalk, Choi et al. (CHI Best Paper) [S] | 2025 | LLM proposes topic/action cards; core cards are fixed lists | Conversation context | Cards | 2-week study, 11 dyads, high engagement | None found |
| AACKGDemo (HF space, willwade) [E] | 2025 | Prompts a hosted LLM (default gemini-1.5-flash) via the `llm` CLI | Persona, listener, topic, time, mood, last 2 conversations, partner message | 1 utterance; 3 by varying temperature +/-0.05 | None | Space file, no license stated; no logprobs requested |
| FreeSpeech (Merkie) [S] | n/d | GPT-3 for phrase and word conjugation in an open AAC app (Svelte) | Typed words | Sentences | None | github.com/Merkie/freespeech, **no license** per GitHub API; pushed 2026-10-20 |
| MozoLM (Google) [E] | 2022+ | Not an LLM layer: LM serving library to combine several LMs behind one interface for AAC text entry | Text prefix | Next-symbol probabilities | n/a | google-research/mozolm (Apache-2.0, C++) |
| Gaines and Vertanen, "Adapting LLMs for character-based AAC" (Findings of EMNLP) [S] | 2025 | Algorithm that turns a subword LLM into character predictions; domain adaptation on sentences scored for AAC usefulness | Text prefix | Character distribution | Better than classification layer, byte-level LLM, n-gram | OSF link on project page (contents not checked) |
| Willett et al. speech BCI (Nature) [E partly] | 2023 | n-gram over phoneme probabilities, n-best list, then **OPT-6.7B rescoring** (repo) | RNN phoneme logits | One sentence | 23.8% WER, 125k vocab, 62 wpm (paper). Repo page: 3-gram 18.8% vs 5-gram+OPT 13.7% WER on the competition test partition | fwillett/speechBCI (no license detected by GitHub API) |
| Card et al. (NEJM) [E partly] | 2024 | Phonemes to words with an open language model, then two further open-source LMs to a sentence (details in Supplement S3, not read) | Phoneme probabilities | Sentence | 9.8% WER at session 2, 2.5% by session 15 (125k vocab) | Neuroprosthetics-Lab/nejm-brain-to-text (no license detected by GitHub API; subfolder has its own LICENSE, unread) |
| Gloss2Text, Fayyazsanavi et al. (EMNLP Findings) [E] | 2024 | Fine-tunes NLLB-200 (final 3.3B) with LoRA (r=16, alpha=32) + semantic label smoothing + paraphrase/back-translation augmentation | Gloss sequence | German sentence, beam 5 | PHOENIX-2014T about 28 BLEU-4 (dev and test) | pooyafayyaz/Gloss2Text (no license) |
| Gong et al., "LLMs are good sign language translators" (CVPR) [S] | 2024 | Frozen off-the-shelf LLM does sign-to-text from a sign encoder's output | Sign video features | Text | Not read | Not checked |
| AulSign (ECAI) [S] | 2025 | Prompts an LLM with retrieved examples to map between text and canonical sign descriptions | Text or sign descriptions | Sign descriptions or text | Low-resource focus; numbers not read | gabrix00/AulSign (license unchecked) |
| LanGaze [S] | n/d | Combines gaze-trajectory decoding with LLM probabilities for gaze typing | Gaze trajectory | Text | +6 points in an ablation (as summarized) | Not found; page returned 403 |

## 2. Family A: AAC with LLM utterance generation

**SpeakFaster** is the strongest published template for the structure "context + partial signal -> LLM -> short list -> user picks". [E]
- Prompt: earlier turns, then the abbreviation; the target is the full phrase. Training inserts spaces between abbreviation letters
  because it tokenizes more consistently. Context matters: in my earlier note one conversation turn more than doubled expansion accuracy.
- Decoding is **temperature sampling at T=1.0**, not beam search. Many samples (128 or 256) are drawn, normalized (lowercase,
  whitespace, final punctuation), and the top 5 are shown. Samples that repeat are evidence for a candidate. [E for sampling; J for
  "frequency = probability estimate"]
- Real-world keystroke savings (0.32-0.36) fell far below simulation because of typos and misselection, and reviewing options costs
  seconds. The transferable lesson for Tink is that the metric must include review cost.
- Only the UI (`webui`, MIT) is public. There is no released model, prompt file or server, so "port SpeakFaster" means reusing the
  UI patterns and the sampling protocol, not code for the LLM layer. [E: repo listing and arXiv text]

**KWickChat** is the closest published structure to what Tink needs and its idea is simple enough to reimplement. [S]
It does **generate-many-then-rerank**: M=50 candidates from a conditioned generator, scored by an *unmodified* GPT-2 (perplexity),
top N shown. The re-ranker gives a 15.9% relative keystroke saving at N=3 and almost nothing above N=8, i.e. re-ranking matters most
when the display is small, which is exactly Tink's three-slot display. Details come from the Promptor paper's description; I could not
read the original IUI PDF (binary fetch) and no code exists to my search.

**Valencia 2023 / COMPA / AACessTalk.** All three keep a human gate (user marks context or intent; core vocabulary not LLM-written;
at most 3 suggestions). None released code that I could find, and none is a scorer, so they inform UX, not algorithm. COMPA's
"user chooses the intent, LLM writes the phrase" is the closest to Tink's frame-to-sentence step. [S]

**Open-source AAC projects with an LLM.** I found only FreeSpeech (GPT-3, no license), the AACKGDemo space (Gemini through the `llm`
CLI, no license), an AMD-NPU demo "AACwithLLM" on Hackster (Llama-2-7B, license unclear), and MozoLM (Apache-2.0, LM serving). None is a
rescoring layer; none returns logprobs. Cboard, AsTeRICS and OpenAAC searches returned nothing LLM-related (absence in one search
engine, not proof). [S]

## 3. Family B: LM rescoring over a probabilistic decoder (structurally Tink's problem)

Speech BCI is the best match: a noisy neural decoder gives a distribution over symbols, a cheap LM builds an n-best list, and a big LM
rescoring step picks the sentence. Replace "phoneme logits" with "intent and referent posterior" and the shape is the same. [J]

**Concrete algorithm from the public code** (`language_model/language-model-standalone.py`, read in full for the scoring path) [E]:
```
totalScore(h) = acoustic_scale * acousticScore(h)
              + (1 - alpha) * ngramScore(h)
              + alpha       * llmScore(h)
best = argmax_h totalScore(h)           # softmax over totalScore gives a confidence
llmScore(h) = sum_{t=1..n-1} log p(token_t | token_<t)  -  n_tokens * length_penalty
```
- n-best size 100 (`--nbest`); `--alpha` 0.5 and `--length_penalty` 0.0 are the argparse defaults; `--acoustic_scale` 0.3 and
  `--blank_penalty` 9.0 are the argparse defaults, while the README example commands use 0.325 and 90, and a code comment lists
  0.8 / 7 / 0.5 for the OPT setup. The values differ by configuration, so I did not treat any one set as canonical.
- The LLM score is a **sum** of token log-probs (first token unscored, no BOS added by this file), batch-forwarded; padding is excluded
  using the attention mask. A failed batch retries in 5 chunks, then falls back to zero LLM scores. Empty hypotheses are dropped.
  Only complete hypotheses are rescored, never partial ones.
- If a context string exists, it is prepended to each hypothesis and its tokens are scored too; since the context is the same for
  every hypothesis it only shifts all totals equally, so it is harmless for argmax but matters if you want calibrated confidence. [J]
- Reported effect: 3-gram 18.8% WER vs 5-gram + OPT 13.7% on the competition test partition (repo page). The Nature paper's own
  language-model ablation and the NEJM supplement (S3, Fig. S8) were not readable here.
- Hardware cost reported by the repo: OPT-6.7B needs about 12.4 GB GPU memory; the 5-gram LM about 300 GB RAM. Tink does not need
  either; the lesson is the *combination rule*, not the models.

**Noisy-channel framing in text entry.** Dwell-free eye typing (Kristensson and Vertanen) and noisy-channel eye-typing recognizers
argue that the LM carries most of the inference when the signal is ambiguous, and prune the candidate set to items within k
edits of the input. [S] "Noisy neural language modeling for typing prediction in BCI communication" (SLPAT 2019) found that n-gram
and RNN LMs trained on clean text degrade when the typing history contains errors, a warning for Tink's history (confirmed
sentences are clean, but unconfirmed misreads must not enter the context). [S]

## 4. Family C: Sign language and fingerspelling

- **Gloss to text with an LLM.** Gloss2Text fine-tunes NLLB-3.3B with LoRA (beam 5) and reaches about 28 BLEU-4 on PHOENIX-2014T. [E]
  This is *trained translation* on thousands of gloss-sentence pairs, not prompting. Tink's enrolled signs (5 signs, one user) have no
  such corpus, so the transferable part is only "the LLM sees an ordered, closed vocabulary of labels and writes a sentence". [J]
- **Frozen-LLM sign translation** (Gong et al. CVPR 2024) and AulSign (retrieved examples in the prompt) point the same way: the
  LLM is handed discrete sign tokens plus retrieved examples. I did not read either. [S]
- **Fingerspelling with LMs.** The 2019 "fingerspelling recognition in the wild" paper reports a small gain (about 1%) from a
  language model. [S] The Kaggle ASL Fingerspelling winner (Christof Henkel) is described as a landmark-sequence encoder-decoder with
  a two-stage out-of-fold training scheme; I found no confirmation of an LM correction step in any top solution, so I do **not**
  claim one exists. [S]
- For Tink the useful, cheap analogue is word-level: the fingerspelled string is a noisy observation of a word, and a lexicon with a
  unigram or conversation-conditioned prior corrects it (edit distance as channel model, as in the eye-typing work above). This is
  my design choice, not a published Tink-like result. [J]

## 5. Family D: Gestures, gaze or partial signals to LLM sentences

I found **no** peer-reviewed system that maps camera gestures or pointing plus conversation context to LLM-written AAC sentences,
consistent with `RESEARCH-interpretation.md`. Adjacent items: an AAC-journal 2025 paper on AI interpreting gestures through
body-worn sensors for people with motor impairments (no LLM sentence stage visible; full text returned 403) [S]; a 2025 webcam
eye-sign system (deep learning plus text prediction, not an LLM) [S]; SemanticScanpath (gaze plus speech with LLMs for robots, not
AAC) [S]; LanGaze (gaze decoding combined with LLM probabilities; see Section 3). A June 2026 position paper, "It's Complicated:
on the design and evaluation of AI-powered AAC interfaces" (Frisch, Wade, Gaines, Kinsella, Peters, Broderick, Vertanen), argues that
current metrics miss users' varied needs; it is relevant to how we report hit@3, but I read only its abstract. [E: abstract]

## 6. Getting P_LM(sentence | conversation): what each backend exposes

| Backend | Can it score an *arbitrary given* sentence? | Notes |
|---|---|---|
| Local transformers / vLLM / llama.cpp | Yes (forward pass over prompt+candidate; vLLM exposes `prompt_logprobs` and OpenAI-style `echo`+`logprobs` on completions) [S for vLLM] | Best option. One batched forward of about 60 tokens per candidate. A 0.5-1.5B model is enough for fluency and context fit [J] |
| Ollama native API | Returns logprobs for *generated* tokens (`logprobs`, `top_logprobs` 0-20 per a community API reference) [S]; no echo found | Use the multiple-choice trick below, not echo. `:cloud` models may not expose it; unverified |
| OpenAI-compatible chat endpoints | Generated tokens only; legacy `echo` is on completions endpoints [S] | Same trick |
| Gemini API | `response_logprobs`+`logprobs` existed for 2.0-flash; forum reports say 2.5 and Vertex 3.x reject them [S] | Do not depend on it |
| Anthropic Claude API | Reported not to expose logprobs [S] (no official page retrieved) | Use sampling frequency or a separate local scorer |

Two ways to get a usable P_LM without per-sentence echo:
1. **Listwise scoring through the label token.** Show the model the numbered candidate list and the conversation, ask for the number,
   and read `top_logprobs` of the first generated token. Softmax over the candidate labels is P_LM(candidate | context). Average over 2-3
   random orderings to cancel position bias. This is the idea behind logprob-based classification in the OpenAI cookbook; applying it to
   candidate ranking is my adaptation. [J]
2. **Sampling frequency** (SpeakFaster): sample N=64-128 completions at T=1.0 conditioned on the frame, dedupe, and use counts as a
   Monte Carlo estimate of P(sentence | context, frame). Works with any API including Claude, costs N tokens. Candidates not hit
   by any sample get a floor value. [E for sampling; J for the estimator]

## 7. Recommended architecture for Tink's final layer

**Idea: n-best rescoring, with the LLM as one proposer and one scorer, and the real-reply pool as the backbone.** [J, modeled on
Willett/Card rescoring and KWickChat generate-then-rerank]

### 7.1 Candidate pool (high recall first)
```
C = retrieval top-K1 (K1 ~ 30, conversation model, real replies)
  U templates and phrase DB for the top intents
  U LLM proposals: N=20-50 samples at T~1.0 conditioned on the frame (SpeakFaster/KWickChat style)
  U referent-substituted variants of the above (insert the pointed or signed noun)
dedupe by normalized text; cap |C| ~ 100 (Willett nbest=100; KWickChat M=50)
```
First measurement, before any weights: **oracle hit@3 and hit@K of the pool** on the existing 163 pairs. If pool oracle recall is not well
above 0.48, rescoring cannot help and the work belongs in the pool; if it is, scoring is the lever.

### 7.2 Scoring
```
score(s) = l1 * log P_sig(s)
         + l2 * log P_LM(s | history, partner turn)       # PMI, or length-normalized, or - n_tok * l_len
         + l3 * log P_ret(s)                              # retrieval/conversation-model likelihood
         + l4 * log P_style(s)                            # optional: similarity to user's confirmed phrases

P_sig(s) = sum_{i,r} P(i, r | signals) * P(i, r | s) / P(i, r | ctx)
```
- `P(i,r | signals)` is the existing Bayesian posterior. `P(i,r | s)` is the rule labeler (`conversation.py` labels) plus a "mentions
  referent r" check; soft, with a floor (for example 0.05) so a mislabel cannot veto a sentence.
- The division by `P(i,r | ctx)` removes the conversation prior from the signal term, because that prior is already inside `P_LM` and
  `P_ret`; leaving it in double counts it (the same trap noted for recursive filtering in `RESEARCH-interpretation.md`). [J]
- Use PMI instead of raw `log P_LM`: `log P_LM(s | ctx) - log P_LM(s)` (or a small constant weight on the unconditioned term), otherwise
  short generic sentences ("Yes.", "OK.") win on length alone. The Willett code handles length only through `length_penalty` (default
  0.0) because its hypotheses all share one acoustic length; Tink's do not. [J]
- Selection: top 3 by score with a diversity step (MMR already exists in `retrieval.py`), always keep a deterministic Yes/No slot when
  the partner asked a yes/no question (AACessTalk's fixed core vocabulary). [J]

### 7.3 Weights
- Published practice: Willett/Card interpolate with a single `alpha=0.5` between n-gram and LLM and a separate `acoustic_scale`
  (0.3-0.8 across configurations in the same file); there is no learned model, just a small grid on dev data. [E]
- Plan for Tink: freeze `l1=1`, grid `l2, l3 in {0, 0.25, 0.5, 1}`, `l4 in {0, 0.25}`, then refine by coordinate ascent. Optimize
  hit@3 (and mean best cosine) on a **dev split, report on a separate test split**. 163 pairs is small: with p=0.45 the standard
  error is about 0.04, so a 0.06 difference is within noise unless the comparison is paired. Generate more simulated episodes from the
  96,956 real pairs for tuning and keep the 163 for confirmation. [J]
- Once `data/confirmations.jsonl` exists, replace the grid with a listwise logistic regression on the same feature vector
  (`log P_sig`, `log P_LM`, `log P_ret`, length). The weights are then learned from the user's own choices. [J]

### 7.4 What to port or reuse
1. **Rescoring combination and batched sequence log-prob** from `nejm-brain-to-text/language_model/language-model-standalone.py`
   (`gpt2_lm_decode`, `rescore_with_gpt2`): about 40 lines of logic. GitHub reports no license for the repository, so
   **reimplement from the formula in Section 3, do not copy the file**; with transformers (Apache-2.0) the scorer is a causal-LM
   forward pass and a gather of next-token log-probs.
2. **KWickChat's pipeline** (generate M=50, rerank with an unmodified LM, show N): no code exists; the algorithm is a few lines and is
   the direct justification for Section 7.1-7.2.
3. **SpeakFaster's sampling protocol and UI** (`webui`, MIT): reuse UI patterns for a 3-5 option panel and the "sample many at T=1,
   normalize, dedupe, rank" procedure. License allows direct reuse of the web UI code.
Optional: MozoLM (Apache-2.0) if we ever want a serving layer that combines a small n-gram LM with a neural LM.

### 7.5 Backend choice
- Primary (scoring needs logprobs): a **local small model** (0.5-1.5B class) through llama.cpp, vLLM or transformers, in its own
  process, event-triggered. Scoring 100 candidates is one batched pass; it is cheaper than generating. At present `ollama list` shows
  only `:cloud` tags, so a local model must be pulled first. [E for the list in the earlier note; J for the rest]
- LLM proposals can stay on the hosted model (Claude or Gemini). They do not need logprobs; they need a diverse sample.
- If no local scorer is available: use the label-token trick on any logprob-exposing endpoint, or fall back to sampling frequency.
  For Claude and recent Gemini models, assume **no logprobs** and use the fallback.
- Keep the existing 2-4 s timeout and silent fallback to retrieval-only; rescoring with `l2=0` reduces exactly to the current pipeline,
  which makes ablation clean.

### 7.6 Why this should beat the earlier attempts (hypotheses, not results) [J]
- The earlier LLM runs removed retrieved real replies from the final list or rewrote them; here they stay in the pool and compete.
- The LLM contributes a *conversation-conditioned prior* where it has real information (the partner's words), rather than
  re-describing the frame the Bayesian layer already holds.
- The metric (best cosine to the real reply) rewards candidates that are close to actual phrasing; LLM samples generically fluent
  but unlike real replies will rarely win unless context supports them, which is the desired behavior.

## Sources

- SpeakFaster, arXiv: https://arxiv.org/abs/2312.01532 and https://arxiv.org/html/2312.01532v1 (Nat. Commun. 2024: https://www.nature.com/articles/s41467-024-53873-3)
- SpeakFaster repo (MIT): https://github.com/TeamGleason/SpeakFaster
- Context-aware abbreviation expansion (NAACL 2022): https://arxiv.org/abs/2205.03767
- KWickChat (IUI 2022): https://www.pokristensson.com/pubs/ShenEtAlIUI2022.pdf (DOI https://doi.org/10.1145/3490099.3511145); described in Promptor: https://arxiv.org/pdf/2310.08101
- Gaines and Vertanen, character-based AAC LLM adaptation: https://arxiv.org/pdf/2501.10582 and https://keithv.com/pub/aacllm/
- "It's Complicated: on the design and evaluation of AI-powered AAC interfaces": https://arxiv.org/abs/2606.24854
- Willett et al. 2023 code (speechBCI): https://github.com/fwillett/speechBCI ; paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9882398
- Card et al. 2024 (NEJM): https://pmc.ncbi.nlm.nih.gov/articles/PMC11030484/ and https://pmc.ncbi.nlm.nih.gov/articles/PMC11328962/
- Brain-to-text code with OPT rescoring: https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text ; scoring file: https://raw.githubusercontent.com/Neuroprosthetics-Lab/nejm-brain-to-text/main/language_model/language-model-standalone.py
- Gloss2Text: https://arxiv.org/html/2407.01394v2 ; code https://github.com/pooyafayyaz/Gloss2Text
- AulSign: https://github.com/gabrix00/AulSign ; Pseudo-gloss generation: https://arxiv.org/abs/2505.15438
- LLMs are good sign language translators (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/papers/Gong_LLMs_are_Good_Sign_Language_Translators_CVPR_2024_paper.pdf
- Fingerspelling in the wild: https://arxiv.org/pdf/1810.11438 ; Kaggle winner summary (unofficial): https://deepwiki.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution
- Dwell-free eye typing: https://www.keithv.com/pub/dwellfree/kristensson-dwell-free.pdf ; BCI noisy LM: https://preview.aclanthology.org/menus/W19-1707/
- MozoLM: https://github.com/google-research/mozolm ; FreeSpeech: https://github.com/Merkie/freespeech ; AACKGDemo: https://huggingface.co/spaces/willwade/AACKGDemo
- AACwithLLM write-up: https://www.hackster.io/mjcrowe99/fast-aac-on-device-with-llms-and-modern-text-to-speech-algs-3318a7
- AACessTalk (press): https://www.alphagalileo.org/en-gb/Item-Display/ItemId/258661
- Gesture interpretation for AAC (AAC journal 2025): https://www.tandfonline.com/doi/pdf/10.1080/07434618.2025.2495905
- Logprobs availability: Gemini forum threads https://discuss.ai.google.dev/t/logprobs-is-not-enabled-for-gemini-models/107989 and https://discuss.ai.google.dev/t/were-logprobs-disabled-for-gemini-3-3-1-in-vertex-api/132426 ; vLLM completions protocol https://docs.vllm.ai/en/v0.16.0/api/vllm/entrypoints/openai/completion/protocol/ ; OpenAI logprobs cookbook https://developers.openai.com/cookbook/examples/using_logprobs ; llama.cpp server `post_sampling_probs` commit mirror (search result, not the official README)

## Could not verify

- KWickChat primary text: the PDF fetch returned binary; all KWickChat details (M=50, GPT-2 perplexity rerank, +15.9%, 71% keystroke saving) come from
  Promptor's description and a search summary. No KWickChat code found.
- Willett 2023 language-model details (n-gram order in the paper's main pipeline, transformer rescoring ablation) and Card 2024 Supplement S3 (which LMs, how combined) were not read.
  The repo's rescoring code was read; the 18.8% vs 13.7% WER figures are from the repo page, not the paper.
- Licenses: GitHub's API returned 404 for a license on `fwillett/speechBCI` and `Neuroprosthetics-Lab/nejm-brain-to-text` (I read that as "none detected", not "none exists");
  the `language_model/` subfolder has its own LICENSE that I did not read. Gloss2Text and FreeSpeech have no detected license. AulSign, Gong et al. and the Hackster project were not checked.
- SpeakFaster: no released model or prompts; the final ranking rule among sampled phrases is not stated in the text I read.
- Valencia CHI 2023 and COMPA: no code found and only secondary descriptions available; COMPA details are not from the paper.
- Gemini/Claude logprob status: Gemini rests on forum reports, Anthropic on third-party statements. Test the exact model id before relying on either. Ollama `logprobs` rests on a community API reference; the release that added it was not found, and `:cloud` model support is unknown.
- llama.cpp server prompt-logprob/echo support and llama-cpp-python scoring behavior: not confirmed.
- Kaggle ASL fingerspelling top solutions' use of LM correction: not found; I claim none.
- LanGaze details, the gesture-AAC paper's methods, and the Gaines/Vertanen OSF contents: not read.
- Whether any 2024-2026 paper maps camera gestures plus conversation context to LLM AAC sentences: not found in my searches; absence in search is not proof.
