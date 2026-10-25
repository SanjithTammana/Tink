# LLM composer: which model, local or API (2026-10-24)

Task: a "composer" that takes a small frame (intent distribution over ~8 intents, a referent, the partner's last utterance,
2-3 turns of history, up to ~10 retrieved candidate sentences) and returns exactly 3 first-person sentences of at most 15 words
as JSON, within about 1 s, a few calls per minute. Context: `EVALUATION.md` section 6 and `RESEARCH-interpretation.md` section A.

Evidence labels: **[M]** measured and cited, **[V]** vendor or model-card claim, **[X]** extrapolated by me from cited numbers,
**[J]** my judgment. Everything is as of 2026-10-24; many sources are third-party aggregators, noted where it matters.

## 0. Bottom line

| Slot | Pick | Why (one line) |
|---|---|---|
| Primary local | `qwen3.5:4b` via Ollama (thinking off) | Apache-2.0, 3.3 GB, IFEval 89.8 on the model card [V]; leaves headroom beside the vision stack. |
| Fallback local | `gemma4:e2b` | Apache-2.0, measured 60 tok/s decode on a 16 GB M4 [M]; the only option likely to hit about 1.3 s on-device. |
| Primary API | `claude-haiku-5-5` (Anthropic) | $0.10 / $0.50 per MTok, "fastest" in the lineup, GA JSON-schema structured outputs, paid terms. |
| Free API fallback | Groq `openai/gpt-oss-20b` (free key) | Strict `json_schema`, OpenAI-compatible URL, vendor-listed 1000 tok/s; free tier is for non-sensitive demo text only. |

Expected local latency for the whole call (800 input tokens, about 70 output tokens) on the M5 Air 16 GB: about 2 s for
`qwen3.5:4b`, about 1.3-1.5 s for `gemma4:e2b` [X, not measured on this hardware]. A strict "under 1 s" is not realistic
locally except by cutting output tokens (section 2). The API route is the only way to reliably get under about 1 s,
and it fails when venue wifi fails; keep the retrieval-only fallback from `EVALUATION.md`.

## 1. Local candidates (fit 16 GB with headroom)

Headroom rule of thumb [J]: macOS plus the vision stack (MediaPipe, YOLO, Whisper, voc2vec) leaves roughly 8-9 GB; keep the
LLM under about 5 GB resident including KV cache. That excludes gpt-oss-20b (12 GB measured [S4]) and gemma4:26b/31b.

| Model | Params (total / active) | License | Q4 size | Instruction following / structure evidence | Mac speed | Ollama / MLX tag |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 4B dense | Apache-2.0 (from the 9B card; 4B card not opened) [S1] | Ollama 3.3-4.0 GB; MLX-4bit about 2.9 GB [S2][S3] | IFEval 89.8, IFBench 59.2, MultiChallenge 49.0 (model card, vendor-run) [S1] | No measured number found. Extrapolated about 45-55 tok/s decode on M5 [X] | `qwen3.5:4b`, `qwen3.5:4b-mlx` (4.0 GB) [S2]; HF `mlx-community/Qwen3.5-4B-MLX-4bit` [S3] |
| Qwen3.5-9B | 9B dense (10B tensors) | Apache-2.0 [S1] | Ollama 6.6-7.6 GB; MLX-4bit about 4.7 GB [S2][S3] | IFEval 91.5, IFBench 64.5, MultiChallenge 54.5 [S1] | Estimate only: about 26 tok/s for a 16 GB M5 Air (aggregator model, not a run) [S5] | `qwen3.5:9b` (also `latest`), `qwen3.5:9b-mlx` (8.9 GB) [S2] |
| Qwen3.5-2B / 0.8B | 2B / 0.8B dense | not checked | 2.7-3.1 GB / 1.2-1.3 GB [S2] | No IFEval found for these sizes | not found | `qwen3.5:2b`, `qwen3.5:0.8b` |
| Gemma 4 E4B | 4.5B effective (8B with embeddings) | Apache-2.0 [S6] | Ollama 6.6-9.5 GB (includes vision/audio towers); 4.6 GB peak in MLX [S7] | No IFEval/LMArena on the card. MMLU-Pro 69.4, Tau2 42.2 [S8]. Native `system` role and function calling [S8] | **Measured**: 33-34 tok/s decode, about 560-600 tok/s prefill, 16 GB M4 (10-core GPU), oMLX 0.3.3, one community run [S7] | `gemma4:e4b` (= `gemma4`), `gemma4:e4b-mlx` [S8] |
| Gemma 4 E2B | 2.3B effective (5.1B with embeddings) | Apache-2.0 [S6] | Ollama 4.6-7.5 GB; 3.0 GB peak in MLX [S9] | MMLU-Pro 60.0, Tau2 24.5 [S8]; no IFEval | **Measured**: about 60 tok/s decode at 4k context, 16 GB M4, oMLX [S9] | `gemma4:e2b`, `gemma4:e2b-mlx` [S8] |
| Gemma 4 12B | not stated on the Ollama page | not checked | Ollama 7.7-8.0 GB [S8] | IFEval listed "coming soon" on a comparison site [S10] | Estimate only: about 17 tok/s (aggregator) [S5] | `gemma4:12b`, `gemma4:12b-mlx` [S8] |
| Gemma 4 26B-A4B (MoE) | 25.2B / 3.8B active | Apache-2.0 [S6] | 16-19 GB | MMLU-Pro 82.6 [S8] | Too large for this Mac alongside vision | `gemma4:26b` |
| gpt-oss-20b | 20B MoE | Apache-2.0 (my knowledge, not re-checked) | 12.1 GB (MXFP4, measured memory) [S4] | Always reasons (low/medium/high) [J]; strict `json_schema` supported on Groq [S11] | M5 vs M4 decode x1.24 [S4]; no absolute figure for a 16 GB Air | `gpt-oss:20b` |
| Others seen in the Ollama list, not evaluated | | | | | | `ministral-3` (3b/8b/14b), `granite4.1` (3b/8b), `lfm2.5` (8b), `nemotron-3.5-bolt` (e4b), `phi4-mini`, `llama3.2` (1b/3b), `smollm2` [S12] |

Notes:
- Qwen3.5 thinks by default. The card says to set `enable_thinking: false`; `/nothink` is not supported [S1]. In Ollama use `think:false` (native API) or
  `reasoning_effort: "none"` (OpenAI endpoint) [S13]. Recommended non-thinking sampling on the card: temperature 0.7, top_p 0.8, top_k 20,
  presence_penalty 1.5 [S1]. For a deterministic JSON composer use a lower temperature (0.2-0.4) and test [J].
- Gemma 4 thinking is opt-in (a `<|think|>` token in the system prompt), so it is off by default [S8]. Card sampling: temperature 1.0, top_p 0.95, top_k 64 [S8].
- Llama 3.2, Phi-4-mini, SmolLM2 are over a year old in the Ollama listing [S12]; I did not gather current numbers for them or for SmolLM3, and I would not choose them over the 2026 families.
- **No faithfulness or hallucination benchmark was found for any of these small models,** and no chat-arena ranking for the small variants. Quality for "no invented facts" has to be measured with our own harness.
- Structured-output research on small models [S14]: hard schema constraints guarantee validity but can cut correctness in sub-2B models (a calendar tool-call task fell from 91.5% to 48.0% executable accuracy in one study; models were Qwen2.5-0.5B/1.5B and SmolLM2-1.7B, not these). Their advice: measure prompt-only JSON first, track "valid but wrong" separately. [J] Use a schema, but make the schema trivial (see section 5).
- Quantity: a 4B model composing three fluent sentences that stay inside a frame is plausible, but I have no benchmark for the exact task. Treat the picks as hypotheses to test on 10-20 scripted frames.

## 2. Speed on Apple Silicon

Measured anchors (none is the M5 Air 16 GB with these exact models):

| Source | Setup | Result |
|---|---|---|
| Apple ML Research, MLX on M5 [S4] | M5 vs M4 MacBook Pro 24 GB, `mlx_lm.generate`, 4096-token prompt, 128 output tokens | Time-to-first-token 3.3-4.1x faster than M4 (Qwen3-8B-4bit 3.97x, Qwen3-14B-4bit 4.06x, gpt-oss-20b 3.33x); decode only 1.19-1.27x faster. Bandwidth 120 GB/s (M4) to 153 GB/s (M5). Absolute tok/s not given. |
| Rattner, fanless M5 Air 32 GB [S15] | mlx-lm 0.31.3, 10-minute runs | Qwen3-30B-A3B-4bit decode: 57 tok/s cold, 44 sustained (battery) or 39 (mains). Qwen3-14B-4bit prefill: 355 cold, 274 sustained (battery). Prompt-processing numbers are for 4,096-token prompts. |
| oMLX community run [S7][S9] | 16 GB M4 (10-core GPU), macOS 26.3.1, oMLX 0.3.3 | Gemma 4 E4B 4-bit: 34.2 tok/s decode, 562 tok/s prefill at 1k context. Gemma 4 E2B: 60.3 tok/s decode at 4k. |
| ModelFit aggregator [S5] | model-based estimates for a 16 GB M5 Air | Qwen3.5-9B Q4_K_M about 26 tok/s; Gemma 4 12B about 17 tok/s. Marked by the site as estimates, not runs. |

**Latency arithmetic for one call (800 prompt tokens + 70 output tokens)** [X, scaling the M4 run by Apple's M5 ratios: prefill x3.5, decode x1.2]:

| Model | Prefill | Decode | Total |
|---|---|---|---|
| `gemma4:e2b` | about 0.3-0.4 s | 70 / ~72 tok/s = about 1.0 s | about 1.3-1.4 s |
| `gemma4:e4b` | about 0.4 s | 70 / ~40 tok/s = about 1.75 s | about 2.2 s |
| `qwen3.5:4b` | about 0.3-0.5 s | 70 / ~45-55 tok/s = about 1.3-1.6 s | about 1.7-2.1 s |
| `qwen3.5:9b` | about 0.7 s | 70 / ~25 tok/s = about 2.8 s | about 3.5 s |

Findings:
- **Is 60-90 tokens in under 1-1.5 s realistic locally?** Only with a ~2B model, or by shrinking the output. 3 sentences of at most 15 words is roughly 55-65 tokens plus JSON
  syntax, so an array of three plain strings (`{"s":["...","...","..."]}`) is about 70 tokens; field names like `sentence_1` add more. [J]
- **Tricks that help:** (1) stream and render sentence 1 as soon as it closes, so perceived latency is prefill + about 20 tokens (about 0.7-1.0 s for E4B-class) [J];
  (2) keep the prompt short and the system prompt byte-identical so the KV prefix can be reused [J]; (3) warm the model at startup and keep it resident (`OLLAMA_KEEP_ALIVE=-1`) [J];
  (4) disable thinking.
- **MLX vs llama.cpp/Ollama:** Ollama announced an MLX backend for Apple Silicon on 2026-03-30 (0.19 preview; one benchmark, Qwen3.5-35B-A3B decode 58 to 112 tok/s vs llama.cpp) [S16].
  A secondary source says MLX became the default Apple Silicon path in v0.30 (May 2026) [S16b], and the MLX gain is largest for MoE (up to about 3x), about 1.4-1.6x for dense models. I could not confirm this from Ollama's own release notes.
  Only safetensors/`-mlx` tags use MLX; GGUF tags still go through llama.cpp [S16b]. The Ollama library shows `-mlx` tags for these models [S2][S8]. Latest Ollama is v0.40.11 (2026-10-24) [S17].
- **Thermal throttling (documented for the fanless M5 Air):** sustained throughput falls to 69-77% of cold within 30-60 s of continuous load and settles by about 150 s; first-token time rose 30-41% under sustained load; mains was 7-12% slower than battery in the sustained state [S15]. The composer is bursty (a few calls per minute), so single calls mostly see cold speeds, but the always-on vision stack will keep the chip warm [J]. Budget for the sustained numbers (multiply the table above by about 1.3) when planning a stage demo.

## 3. API options

Prices are USD per million tokens, standard tier. Latency column is vendor-listed or "not measured".

| Provider | Model (ID) | Free tier / limits | Price in / out | Latency | JSON schema | Data-use terms |
|---|---|---|---|---|---|---|
| Anthropic | `claude-haiku-5-5` | none (paid) | $0.10 / $0.50 (prompts up to 100k) [S18] | "Fastest" in the lineup; no ms figure published; not measured here [S19] | Yes, GA: `output_config.format = {type: "json_schema", schema}`; `additionalProperties:false` required; no `maxLength`; first use of a schema adds compile latency, cached 24 h [S20] | Paid API terms: not re-read this session (unverified) |
| Anthropic | `claude-haiku-4-5-20251001` | none | $1 / $5 [S18] | not measured | Yes [S20] | as above |
| OpenAI | `gpt-6-luna`, `gpt-5.4-nano`, `gpt-5-nano` (listed) | none found | $0.10 / $0.50; $0.20 / $1.25; $0.05 / $0.40 [S21] | not measured | not checked this session | not checked |
| Google | `gemini-3.1-flash-lite`, `gemini-2.5-flash-lite` | Free tier exists for all Flash/Flash-Lite models; exact RPM/RPD only visible in AI Studio [S22][S23] | $0.25 / $1.50; $0.10 / $0.40 (output includes thinking tokens) [S22] | not measured | not checked | **Free tier: inputs and outputs used to improve products. Paid tier: not used.** [S22] |
| Groq | `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` (preview) | Limits table on the docs page: gpt-oss-20b/120b and qwen3.8-27b all 30 RPM, 1K RPD, 8K TPM, 200K TPD. The fetch said the table is the Developer plan base and did not show a separate Free table, so free numbers are unconfirmed [S24] | gpt-oss-20b $0.075 / $0.30; gpt-oss-120b $0.15 / $0.60; qwen3.8-27b $0.80 / $4.00 [S25] | Vendor "speed": 1000 tok/s (20b), 500 (120b), 450 (qwen3.8) [S25]. Not measured end to end | Strict `json_schema` on gpt-oss-20b/120b, qwen3.8-27b; other models JSON-object mode only [S11] | Not checked |
| Cerebras | `gpt-oss-120b`, `qwen-3.8-27b` | Free trial only: 5 RPM, 1M tokens/day, $5 credit after adding a verified payment method, expires in 30 days; no permanent free tier [S26] | not checked | not measured | not checked | not checked |
| OpenRouter | `:free` models | 20 RPM; 50/day under 10 credits purchased, 1,000/day with 10+ credits [S27]. The free list today is unusual models (for example `xingchen-agi/xing4.0-29b-a4b:free`, `recursive/flint-v1.0:free`) [S28] | $0 | not measured | not shown | Logging/training note not retrieved |
| Ollama cloud | `gemma4:31b-cloud`, `glm-5.3-flash:cloud`, ... | usage-based, not stated on the cloud docs page [S29] | n/a | not measured | **Not supported on cloud** [S30] | "Not used to train models"; retention period not stated [S29] |
| Mistral | `mistral-small` family | Third parties say 1 RPS / 500K TPM / 1B tokens per month, phone verification [S31] | not checked | not measured | not checked | Several third-party pages say the free tier needs opt-in to training; not confirmed from Mistral [S31] |
| GitHub Models | n/a | One source says closed to new users on 2026-06-16; not confirmed from GitHub [S32] | n/a | n/a | n/a | n/a |
| Cloudflare Workers AI | open models | 10,000 Neurons/day shared across models; overage $0.011 per 1,000 Neurons on the Paid plan [S33] | per-model Neurons | not measured | JSON mode not found | Training policy not found |

**Weekend-demo cost** (500 calls x 800 input + 80 output tokens = 400K input + 40K output tokens) [X from the prices above, assuming no hidden thinking tokens]:

| Model | Cost |
|---|---|
| Claude Haiku 5.5 | $0.04 + $0.02 = **$0.06** |
| Claude Haiku 4.5 | $0.40 + $0.20 = $0.60 |
| Groq gpt-oss-20b | $0.03 + $0.012 = $0.04 |
| Groq gpt-oss-120b | $0.06 + $0.024 = $0.08 |
| Gemini 3.1 Flash-Lite | $0.10 + $0.06 = $0.16 |
| `gpt-6-luna` / `gpt-5.4-nano` | $0.06 / $0.13 |

Adaptive thinking on Haiku 5.5 (default effort `medium` [S19]) and reasoning on gpt-oss can add hidden output tokens and seconds; set effort low or disable thinking where the API allows
(check `thinking.types.disabled.supported` for the model in the Models API [S19]). Even at 10x the tokens, every row stays under a few dollars [J].

**The six Ollama `:cloud` models on this machine** (what they are; none is a speed pick):

| Tag | What it is |
|---|---|
| `deepseek-v4.1-flash` | DeepSeek-AI MoE, about 8B active in prefill and 16B in decode, 1M context, thinking, vision (page lists 763B vs 552B total, inconsistent) [S34] |
| `glm-5.3-flash` | Z.ai, about 321B total / 18B active, MIT, 1M context, thinking, vision [S35] |
| `gemma4` (`gemma4:cloud`) | Google Gemma 4, cloud variant is the 31B (256K context) [S8] |
| `glm-5.2`, `minimax-m2.7`, `minimax-m3` | Listed as cloud on the Ollama library (4, 7 and 4 months old); I did not open their pages. `minimax-m2.7` also appears on Groq as a preview model [S12][S25] |

All are large reasoning-capable models behind Ollama's servers: slower than Haiku-class or Groq for a 70-token answer, no schema enforcement on Ollama cloud, and the same wifi dependence as any API [J].

## 4. Recommendation

1. **Primary local: `qwen3.5:4b` (thinking off).** Apache-2.0, 3.3 GB, IFEval 89.8 [S1], smallest dense model with a published, strong instruction-following score; leaves memory for the vision stack. Expect about 2 s per call [X]. Caveat: its card numbers come from Qwen, and the card defaults to thinking, so check that the 89.8 applies to non-thinking use [J].
2. **Fallback local: `gemma4:e2b`.** Apache-2.0, measured 60 tok/s on a 16 GB M4 [S9]; the likeliest local model to finish in about 1.3-1.4 s on an M5 Air [X]. Lower quality (Tau2 24.5 vs 42.2 for E4B [S8]); pair it with the id-selection schema so it cannot invent content. If quality is short and latency is acceptable, `gemma4:e4b` (33 tok/s measured on M4 [S7]) is the next step up; run the A/B on `simulate.py dialogue`. Do not use 9B/12B on the live path (about 3+ s, plus about 7-8 GB of the 16 GB).
3. **Primary API: `claude-haiku-5-5`.** Cheapest Anthropic tier ($0.10 / $0.50), described as the fastest model "for high-volume, latency-sensitive tasks such as classification, extraction, and routing" [S19], GA JSON-schema outputs [S20]. Constrain with an `enum` of candidate ids; set low effort. No measured latency; time it from the venue network before the demo.
4. **Free API fallback: Groq `openai/gpt-oss-20b`.** OpenAI-compatible, strict `json_schema` [S11], vendor 1000 tok/s [S25]. Use a free key only for scripted, non-personal test frames: Groq's free-tier data terms were not verified, and the Gemini free tier explicitly trains on inputs [S22], so do not send real partner speech to either. At 8K TPM / 200K TPD the limits allow about 9 calls/min and about 220 calls/day (900 tokens each) [X from S24]; the weekend 500-call test needs the paid Developer plan (about $0.04) or a split.

Privacy [J]: send text only; for a real user session prefer the local model (nothing leaves the laptop) or a paid API; keep a visible LLM on/off toggle and the retrieval-only fallback.

## 5. Serving setup on the Mac, one client for local and API

Use the OpenAI-compatible chat API everywhere and switch by base URL, key and model name:

| Target | Base URL | Key | Model |
|---|---|---|---|
| Ollama local | `http://localhost:11434/v1/` [S13] | any string, ignored (`"ollama"`) [S13] | `qwen3.5:4b`, `gemma4:e2b` |
| Groq | `https://api.groq.com/openai/v1` (from my knowledge, not fetched) | Groq key | `openai/gpt-oss-20b` |
| Ollama cloud | `https://ollama.com/v1` [S13] | `OLLAMA_API_KEY` | e.g. `gemma4:31b` (API name differs from the `:cloud` CLI tag) [S29] |
| LM Studio local (optional) | `http://localhost:1234/v1` (from my knowledge, not fetched) | any | name shown in LM Studio |
| `mlx_lm.server` (optional) | `http://localhost:8080/v1` (default port, from my knowledge, not fetched) | any | HF repo id, e.g. `mlx-community/Qwen3.5-4B-MLX-4bit` |
| Anthropic | not OpenAI-compatible by default; use the Anthropic SDK with `output_config.format` [S20] (an OpenAI-SDK compatibility layer exists but I did not verify it for Haiku 5.5) | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |

Recommended: **Ollama** (already installed, v0.40.x, MLX backend on Apple Silicon [S16][S17], `-mlx` tags available). Only consider `mlx_lm.server` or LM Studio if the Ollama measurement misses budget.
- Pull the MLX build where it exists, then compare it with the GGUF tag on your own timing script: `qwen3.5:4b-mlx` vs `qwen3.5:4b`; `gemma4:e2b-mlx` vs `gemma4:e2b` [S2][S8]. (Not pulled in this research.)
- Structured output: Ollama's native `/api/chat` takes `format: <JSON schema>` [S30]; the OpenAI-compat page lists `response_format` but does not explicitly say `json_schema` [S13], so test it, or call the native endpoint for the local path. Local only; the cloud does not support it [S30]. Also put the schema text in the prompt and set temperature low [S30].
- Disable thinking: `reasoning_effort: "none"` on the OpenAI endpoint, or `think:false` natively [S13]. Set a small `num_ctx` (2048) to limit KV memory [S13 shows `num_ctx` via Modelfile].
- Keep it resident and warm: `OLLAMA_KEEP_ALIVE=-1`, one warm-up call at app start [J].
- Schema design [J]: `{"picks":[{"id":<enum of candidate ids>}], "free":["..."]}` for selection, or `{"s":["...","...","..."]}` for composition. Validate length and word limit in code, and fall back to the retrieval candidates on timeout (about 2 s hosted, about 3 s local) or invalid JSON.
- Note that Anthropic's schema subset has no `maxLength`/`minItems>1` [S20]: enforce "15 words" in the prompt and in post-validation, not in the schema.

## Could not verify

- Any measured tokens/sec for Qwen3.5-4B/2B/9B, Gemma 4 12B, or any model on the **M5 Air 16 GB** itself; every M5 number here is scaled from an M4 run with Apple's ratios [X]. The only published M5 Air run is on the 32 GB model with Qwen3 [S15].
- IFEval for Gemma 4 (any size), Qwen3.5-2B/0.8B, and whether Qwen3.5's IFEval figures are thinking or non-thinking mode.
- Chat-arena or writing-quality rankings and hallucination/faithfulness numbers for all small models.
- Licence text for Qwen3.5 sizes other than 9B; the Ollama pages do not state licences.
- Exact free-tier limits for Groq (Free vs Developer), Gemini, and Mistral; training/data terms for Groq, Mistral, OpenRouter free models, Cloudflare, and Anthropic's paid API in this session.
- OpenAI mini/nano structured-output support and latency; end-to-end latency for Haiku 5.5, Groq and Gemini from venue wifi.
- Whether Ollama's OpenAI endpoint enforces `json_schema` (docs list `response_format` only), and whether `thinking: disabled` is accepted by Haiku 5.5.
- GitHub Models status (single unconfirmed source), the Ollama v0.30 "MLX default" date (secondary source), and details of `glm-5.2`, `minimax-m2.7`, `minimax-m3`.
- Default ports and URLs for Groq, LM Studio and `mlx_lm.server` are from memory.

## Sources

- [S1] Qwen3.5-9B model card (IFEval, license, thinking off): https://huggingface.co/Qwen/Qwen3.5-9B
- [S2] Ollama Qwen3.5 tags and sizes: https://ollama.com/library/qwen3.5
- [S3] MLX 4-bit conversions: https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit , https://huggingface.co/Micklavin/Qwen3.5-9B-4bit
- [S4] Apple ML Research, MLX on M5 (TTFT/decode ratios, memory): https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- [S5] ModelFit estimates (not measured): https://modelfit.io/blog/best-llm-macbook-air-m5-16gb/
- [S6] Gemma 4 E4B model card (Apache-2.0): https://huggingface.co/google/gemma-4-E4B-it
- [S7] oMLX run, Gemma 4 E4B on M4 16 GB: https://omlx.ai/benchmarks/6kn71ty3
- [S8] Ollama Gemma 4 tags, parameters, benchmark table, sampling: https://ollama.com/library/gemma4
- [S9] oMLX run, Gemma 4 E2B on M4 16 GB (as summarized by search): https://omlx.ai/benchmarks/lk2w8231
- [S10] BenchLM Gemma 4 12B comparison (IFEval "coming soon"): https://benchlm.ai/md/compare/gemma-4-12b-vs-qwen3-5-122b-a10b.md
- [S11] Groq structured outputs: https://console.groq.com/docs/structured-outputs
- [S12] Ollama model library (tags, ages, cloud list): https://ollama.com/library
- [S13] Ollama OpenAI compatibility: https://docs.ollama.com/api/openai-compatibility
- [S14] Small-model structured output studies: https://arxiv.org/abs/2605.26128 , https://arxiv.org/pdf/2605.02363 , https://arxiv.org/pdf/2604.25359
- [S15] Fanless M5 Air sustained benchmark: https://zachrattner.com/projects/ai-mac-cluster/fanless-benchmarking
- [S16] Ollama MLX preview announcement: https://registry.ollama.ai/blog/mlx ; [S16b] secondary: https://dev.to/jovan_chan_9500711396d4e6/ollama-v030-on-apple-silicon-what-the-stable-mlx-release-actually-changed-from-the-preview-25h7 , https://levelup.gitconnected.com/ollama-mlx-nearly-doubles-llm-speed-on-your-mac-58-to-112-tok-s-and-your-old-models-get-none-of-fd1a9ec531fb
- [S17] Ollama releases (v0.40.11): https://github.com/ollama/ollama/releases
- [S18] Anthropic pricing: https://platform.claude.com/docs/en/about-claude/pricing
- [S19] Anthropic models overview (IDs, latency, effort): https://platform.claude.com/docs/en/about-claude/models/overview
- [S20] Anthropic structured outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- [S21] OpenAI pricing: https://developers.openai.com/api/docs/pricing
- [S22] Gemini API pricing and free-tier data use: https://ai.google.dev/gemini-api/docs/pricing
- [S23] Gemini rate limits (no numbers on page): https://ai.google.dev/gemini-api/docs/rate-limits
- [S24] Groq rate limits: https://console.groq.com/docs/rate-limits
- [S25] Groq models, speed and prices: https://console.groq.com/docs/models
- [S26] Cerebras rate limits: https://inference-docs.cerebras.ai/support/rate-limits
- [S27] OpenRouter limits: https://openrouter.ai/docs/api-reference/limits
- [S28] OpenRouter free models: https://openrouter.ai/models?max_price=0
- [S29] Ollama cloud docs: https://docs.ollama.com/cloud
- [S30] Ollama structured outputs: https://docs.ollama.com/capabilities/structured-outputs
- [S31] Mistral free-tier third-party listings: https://mintlify.com/cheahjs/free-llm-api-resources/providers/free/mistral-plateforme , https://free-llm.com/provider/mistral-ai
- [S32] GitHub Models status (single source): https://klymentiev.com/blog/github-models-shutdown
- [S33] Cloudflare Workers AI pricing: https://developers.cloudflare.com/workers-ai/platform/pricing/
- [S34] Ollama deepseek-v4.1-flash: https://ollama.com/library/deepseek-v4.1-flash
- [S35] Ollama glm-5.3-flash: https://ollama.com/library/glm-5.3-flash
