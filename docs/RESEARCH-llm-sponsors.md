# LLM providers with a sponsor-prize angle: Gemini, Meta, xAI (2026-10-24)

Scope: extends `RESEARCH-llm.md` (local models, Claude Haiku 5.5, Groq; not repeated). Task is unchanged: a small JSON frame in, exactly
3 short first-person sentences out as JSON, about 1-2 s, a few calls per minute (`llm.py`, `config/llm.json`).

Evidence labels: **[V]** read on the provider's own page this session, **[S]** secondary source (blog, aggregator, news), **[I]** my inference.
Nothing here was run: no key was created, no API was called, no latency was measured by me.

## 0. Bottom line

- **Prize reality first.** HackTX 2026 track and prize rules are not published online (hacktx.com is a JS shell showing only the title; the Devpost page returns 403; the opening ceremony is Sat 10:30). Nothing below can be tied to a HackTX-specific rule. [V]
- **Gemini is the one clear MLH LLM prize** in the earlier event notes (MLH's HackTX list included "Gemini API"). MLH's own category text is one sentence ("push the boundaries of what's possible with AI using Google Gemini"), prize "MLH Swag Kits", with a "Build with Gemini" link and **no published eligibility or judging rules**. The generic MLH prizes page labels the category "Q1 2025 - Q1 2026", which has already lapsed on paper; whether HackTX 2026 still carries it must be confirmed on the HackTX MLH page / Devpost after the ceremony. [V]
- **Meta and xAI have no LLM-API prize I could find.** Meta has a workshop (Sat 4-5 pm, GDC 2.216) and the Instagram teaser shows Meta Ray-Ban prizes on one unnamed track; nothing says it is an API-use prize. xAI is not visibly a HackTX sponsor; the "grokbot" the owner mentioned is almost certainly MLH's "Build Your Hackathon A-Team with SpaceXSI" category (section 3), which is about dev tooling, not about putting Grok in your product. [V for the MLH text; I for the HackTX link]
- **Recommended wiring:** add `gemini-3.5-flash-lite` (paid-tier key) as an optional profile to qualify for the Gemini prize honestly, keep local Ollama as the privacy default and Claude Haiku 5.5 as the primary API. Add xAI and Meta as profiles only if time allows; neither earns a prize that I can verify.
- **Biggest technical risk for Gemini:** Artificial Analysis shows about 6-7 s time-to-first-token for `gemini-3.5-flash-lite` on a 10k-token workload [S]. Our prompt is about 800 tokens, so it should be far lower, but nobody has measured that. Time it from the venue network before relying on it (`timeout_s` is 6.0 in `llm.json`).

## 1. Comparison table

| Provider | Model IDs | Free tier / credits | Price (USD per MTok in / out) | Latency | JSON schema | OpenAI-compatible base URL | Data-use terms | Prize relevance |
|---|---|---|---|---|---|---|---|---|
| **Google Gemini API** | `gemini-3.5-flash-lite` (stable, "fastest" 3.5 model; new projects are told to use it or 3.8 Flash), `gemini-3.1-flash-lite`, `gemini-3.8-flash`, `gemini-3.6-flash`; older `gemini-2.5-flash-lite` ($0.10 / $0.40) exists but is "limited to past users" [V] | Free tier exists for every Flash/Flash-Lite model; numbers only visible per project in AI Studio, not on the docs page [V]. Third-party blogs disagree widely (for example 5-15 RPM, 100-1,500 RPD) [S]; treat as unknown. Tier 1 = link a billing account, upgrade is instant [V] | 3.5 Flash-Lite $0.30 / $2.50; 3.1 Flash-Lite $0.25 / $1.50; 3.8 Flash $0.75 / $3.75 until 2026-12-31, then $1.50 / $7.50; output includes thinking tokens [V] | Vendor: about 350 output tok/s. Artificial Analysis (AI Studio, 10k-token input): TTFT 6-10 s across snapshots, 412-490 tok/s [S]. Not measured at our 800-token size | Yes: native `response_json_schema`; supports `enum`, `minItems`/`maxItems`, `required`, `additionalProperties`; `minLength`/`maxLength` are stripped, not enforced [V] | `https://generativelanguage.googleapis.com/v1beta/openai/` [V]. Docs show structured output only via the OpenAI SDK `parse()` with a Pydantic model, so raw `response_format: json_schema` through `llm.py` is likely but untested [V/I] | **Free tier: content and responses are used to improve Google's products and humans may read them; Google says do not submit sensitive, confidential or personal information. Paid tier: not used.** In the EEA, Switzerland and UK the paid terms apply to everything. 18+ only [V] | MLH "Best Use of Gemini API" (see section 2) |
| **Meta Model API** (successor to Llama API) | `muse-spark-1.3` (standard), `muse-spark-1.3-contributor`, `muse-spark-1.2`, `muse-spark-1.1`, `muse-wonder-1.0` (preview) [V]. Open-weight Muse Spark Open is not served here [V] | Reports of $20 free credits per new account at the 2026-07 launch, US-only preview, waitlist for some [S]; one tutorial instead shows a card prompt. Unconfirmed today | Standard $1.25 / $4.25; contributor $0.10 / $0.20 [S, one tutorial; matches launch coverage for standard] | No ms figure. Artificial Analysis cited about 182 tok/s direct [S] | Yes, `response_format: {type: json_schema, json_schema: {name, schema}}`, constrained decoding always; `strict` optional; depth 10, no recursion [V] | `https://api.meta.ai/v1` (Chat Completions, Responses and Anthropic-style Messages) [V] | **Standard tier: no training on your data. Contributor tier (the default in Muse Code): content "may be used for product improvement".** Zero retention only via sales [S] | No Meta prize found. Old "Llama API" preview and `api.llama.com` retired 2026-07-06 [S: promptfoo] |
| **xAI Grok API** (company now styled SpaceXAI; `x.ai/api` is titled "SpaceXSI API") | `grok-4.20-0309-non-reasoning`, `grok-4.3`, `grok-build-0.1` on the models page; `grok-4.7` / `4.6` / `4.5` exist per the rate-limit page [V]. No "mini" or "fast" tier on the models page; "Grok 4.1 Fast" at $0.20 / $0.50 appears only in secondary sources and may be legacy [S] | Unverified. Older $25 sign-up credit gone or unclear; a "$150 per month if you opt in to data sharing" scheme appears only in blogs [S]. xAI's own enterprise FAQ says it may give credits in exchange for permission to train on business data [V] | `grok-4.20-0309-non-reasoning` and `grok-4.3`: $1.25 / $2.50 (1M context); `grok-build-0.1`: $1.00 / $2.00; `grok-4.7`: $2.00 / $6.00 [V models page; S for 4.7] | No data found | Yes, `response_format` json_schema through the OpenAI SDK; example uses `grok-4.7`; no per-model support list [V] | `https://api.x.ai/v1` [V] | API: not trained on business inputs/outputs by default per enterprise FAQ; credits-for-training exception exists [V, via search snippet of the FAQ]. Consumer-product terms not checked | None found (see section 3) |
| Snowflake Cortex (MLH "Best Use of Snowflake API") | many hosted LLMs behind one account | Trial account needed; not checked | not checked | not checked | not checked | not checked | not checked | MLH category window "Q3 2025 - Q1 2026", likely lapsed; the HackTX list included it [V] |
| Vultr Serverless Inference (MLH "Best Use of Vultr") | `gpt-oss-120b`, `kimi-k2-instruct`, DeepSeek-R1 distills, Qwen2.5-Coder and others [S] | Plan from $10 per month for 50M tokens [S]; no hackathon credit found | $0.0002 per 1K tokens overage [S] | not checked | not checked | `https://api.vultrinference.com/v1`, key env `VULTR_SERVERLESS_INFERENCE_API_KEY` [S] | not checked | MLH category window "Q4 2025" (probably lapsed) [V] |

On-device counterpart: Gemma 4 is already profiled in `RESEARCH-llm.md` (`gemma4:e2b`, `gemma4:e4b`, Apache-2.0, run through Ollama). It uses Google's open model, **not** the Gemini API, so I would not claim it satisfies "Best Use of Gemini API"; ask the MLH mentor if you want to try. [I]

## 2. Gemini detail

**Prize.** MLH lists "Best Use of Gemini API" (prize: MLH Swag Kits; also a related "Best Use of Gen AI" category sharing the same asset and window) on its events prizes page. The only description is the single sentence quoted above; no per-event rules, README requirement, key type or model restriction is published. An older MHacks guide said judges look at how the project showcases Gemini, with extra credit for audio/video modalities [S, a past event, not a rule]. A Google forum reply says a key from AI Studio or Vertex both work [S]. MLH's challenge pages say submissions open only during the event [S]. So: "any real use of the Gemini API, from an AI Studio key, shown in the submission" is the safe reading, and it is an inference, not a published rule. Also note that "Gemini Computer Use" was a separate MLH category earlier in 2026 [S].

**Models for this task.**
- `gemini-3.5-flash-lite`: Google's page calls it the fastest, most cost-effective 3.5 model. Thinking is on at level `minimal` by default; levels `minimal/low/medium/high` [V]. Weekend cost: 400K input + 40K output tokens is about $0.12 + $0.10 = **about $0.22** [I from V prices]; hidden thinking tokens at `minimal` should be small but are billed as output.
- `gemini-3.1-flash-lite`: stable, $0.25 / $1.50, no thinking-level notes; backup profile.
- `gemini-3.8-flash`: smarter, thinking cannot go below `low`, $0.75 / $3.75 (price doubles 2027-01-01). Too slow and pricey for this use [I].
- `gemini-2.5-flash-lite` is the cheapest ($0.10 / $0.40, thinking off by default) but the models page says access is limited to past users; a new key may be refused [V]. Do not plan around it.
- Through the OpenAI-compatible layer, `reasoning_effort: "none"` works **only for 2.5 models**; reasoning cannot be turned off for Gemini 3 models [V]. Since 3.5 Flash-Lite already defaults to `minimal`, no `extra` field is needed. `reasoning_effort` and `thinking_level`/`thinking_budget` cannot be sent together [V].

**Free-tier terms, stated plainly.** The unpaid Gemini API and AI Studio are "Unpaid Services": Google uses submitted content and generated responses (prompts, system instructions, cached content, files) to provide, improve and develop its products and ML technology, and human reviewers may read and annotate them after Google disconnects them from the account and key. The terms say not to submit sensitive, confidential or personal information to the unpaid services. Paid services are not used to improve products. Users in the EEA, Switzerland and UK get the paid terms everywhere, and the API requires users to be 18+. [V] For an assistive app that sends a communication partner's words, the consequence is: **free-tier key only with scripted/demo frames; real partner speech only on a billing-linked key**, and even then prefer local. Billing needs a card, which is the owner's call, not something to automate.

**Python SDK shape (`pip install -U google-genai`, env `GEMINI_API_KEY`).** The current official docs lead with the Interactions API; this form was read on the docs page:

```python
from google import genai
client = genai.Client()                       # reads GEMINI_API_KEY
interaction = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input=json.dumps(frame),
    generation_config={"thinking_level": "minimal"},
    response_format={"type": "text", "mime_type": "application/json", "schema": SCHEMA},
)
options = json.loads(interaction.output_text)["options"]
```

The older `generate_content` form (still shipped in the SDK per a third-party mirror, not re-verified on the official page) is
`client.models.generate_content(model=..., contents=..., config=types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=SCHEMA, thinking_config=types.ThinkingConfig(thinking_level="minimal"), system_instruction=...))`.
Neither combination (schema + thinking level) was shown together on an official page; the first is a close adaptation of two official snippets (`gemini-3.8-flash` examples) with the model changed. Test before the demo. [V/I]

Schema notes: `minItems`/`maxItems` are enforced (so `llm.py`'s SCHEMA with `minItems: 3, maxItems: 3` fits); `maxLength` is stripped, so keep the 15-word cap in the prompt and in `clean_options` (already done). [V]

## 3. xAI / "grokbot"

- xAI now brands itself SpaceXAI (SpaceX acquired Cursor; Grok Bot is a SpaceXAI agent product that logs into apps, remembers tasks and coordinates a team of bots). Sources: news coverage and SpaceXAI event pages [S].
- The MLH prizes page lists **"Build Your Hackathon A-Team with SpaceXSI"**: assemble a team using Cursor and a Grok Bot to speed up development, with a raffle entry; prize is a mechanical keyboard; no date window stated. That is almost certainly the "grokbot" mention. It rewards using the Grok Bot as a development helper, not calling the Grok API from your app, and it is a raffle, not a judged prize. [V text; I for the match to the owner's remark]
- No evidence that SpaceXAI/xAI sponsors HackTX 2026: it is absent from `hacktx26-sponsors.md` (site logos, Discord, handbook) and from the search results. A SpaceXAI x Cursor "Grok hackathon" exists in Austin, hosted by Cursor Austin, not by MLH or Freetail [S]. If the owner heard that xAI is a sponsor, confirm at the opening ceremony. [I]
- Technically `grok-4.20-0309-non-reasoning` at $1.25 / $2.50 is the only plausible latency candidate, and it is 5-12x the price of Haiku 5.5 or Flash-Lite for no known quality or speed advantage on this task. Tier 0 limits (37 RPS, 10M TPM for the 4.3/4.20 family) are not a constraint [V]. Free credits unverified; the credits-for-data-sharing route would send partner speech to xAI for training, which is the opposite of the privacy default. Skip unless a mentor hands out credits.

## 4. Meta detail

- The "Llama API" (llama.developer.meta.com) is gone: promptfoo's docs say the public preview and `api.llama.com` were retired 2026-07-06 and point to the new Meta Model API at `api.meta.ai`; the old dev URL now redirects to `dev.meta.ai`. [S + V redirect]
- Current offering is Muse Spark / Muse Wonder (Meta Superintelligence Labs models), not Llama, via `https://api.meta.ai/v1`. Muse Spark 1.1 opened as a paid public preview in July 2026 (US developers at launch, waitlist for some, $20 free credits, $1.25 / $4.25) [S]. Whether the account is open to a Texas student today without a waitlist is not verified.
- **Use only `muse-spark-1.3`, never `-contributor`.** The cheap contributor tier ($0.10 / $0.20) trades data for price; a tutorial reports it is the default in Muse Code. [S]
- If Meta's HackTX track is "use Meta AI", the match is weak for Tink and the Muse API is not obviously what they will ask for; the open-weight route (Llama via Ollama, or Muse Spark Open self-hosted) is the on-device story. I could not confirm any "Best use of Llama/Muse" prize at HackTX or at other 2026 Meta-sponsored hackathons. [I]

## 5. Other LLM angles in the event files

- **MLH list at HackTX** (from `event-facts-2026.md`, MLH page seen 10-22): Gemini API, Solana, Tiger Data, Vultr, DEV writing challenge, Snowflake API, .Tech. Only Gemini, Snowflake API and Vultr touch LLMs.
- **Snowflake API:** "access leading LLMs through one account". Using it to run the composer would need a Snowflake trial account (owner's decision; I did not create one) and a Cortex REST call. Tink has no data-warehouse story, so this would be a contrived use; its listed window is Q3 2025 - Q1 2026. [V list; I fit]
- **Vultr:** hosting plus Serverless Inference (OpenAI-compatible). A "best use of Vultr" would reward deploying on Vultr, not merely using their LLM. Window Q4 2025 (likely lapsed). Low fit. [I]
- **Annapurna Labs (AWS):** Trainium/Inferentia inference or Bedrock would be the LLM angle; no track published, no history. I did not research Bedrock's current terms. Low fit for a laptop-and-webcam app. [I]
- **Cerebras:** covered in `RESEARCH-llm.md` (trial limits, no permanent free tier); no HackTX link found.
- **Non-LLM MLH categories that fit Tink better than any LLM prize** (from the same MLH page, windows as printed): **ElevenLabs** (Q4 2025 - Q4 2026; natural-sounding speech; Tink's output could be voiced), **Backboard** (persistent memory for stateless model APIs; Tink's user-phrase history), **Presage** (human-sensing layer for physical and emotional state; a stretch). These are not LLM providers but each could be a stronger and more honest prize angle than forcing an LLM sponsor. [V descriptions; I fit]

## 6. Recommendation

**(a) Best technical fit.** Unchanged from `RESEARCH-llm.md`: local `qwen3.5:4b` / `gemma4:e2b`, API `claude-haiku-5-5`. Among this document's providers, `gemini-3.5-flash-lite` is the only credible peer on cost and structured output; Meta and xAI cost 5-12x more per token with no demonstrated benefit.

**(b) Prize eligibility.** Wire Gemini and be able to show it working (a profile in `config/llm.json`, a line in the Devpost description, a short A/B result from `simulate.py dialogue --llm`). That is the cheapest honest way to be eligible if HackTX carries the MLH Gemini category. Do not add xAI or Meta for prize reasons: no qualifying prize exists that I can find.

**(c) Privacy-safe default.** Keep `active` on a local profile (Ollama) for any session with real partner speech. Gemini only (1) on a billing-linked key or (2) with scripted frames. Never the free tier with real speech.

**Profiles to add to `config/llm.json`** (do not change `active`; env var names are my choice except where noted):

```json
"gemini_flash_lite": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-3.5-flash-lite",
                      "api_key_env": "GEMINI_API_KEY", "json_schema": true,
                      "note": "MLH Gemini prize profile; $0.30/$2.50 per MTok; thinking defaults to minimal; free tier trains on prompts, use a billing-linked key or scripted frames only"},
"gemini_3_1_flash_lite": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-3.1-flash-lite",
                          "api_key_env": "GEMINI_API_KEY", "json_schema": true, "note": "stable backup, $0.25/$1.50"},
"xai_grok_non_reasoning": {"base_url": "https://api.x.ai/v1", "model": "grok-4.20-0309-non-reasoning",
                           "api_key_env": "XAI_API_KEY", "json_schema": true,
                           "note": "optional; $1.25/$2.50; credits-for-training offers send data to xAI, do not accept"},
"meta_muse_spark": {"base_url": "https://api.meta.ai/v1", "model": "muse-spark-1.3",
                    "api_key_env": "META_API_KEY", "json_schema": true,
                    "note": "optional; standard tier only (never -contributor); US preview, access unverified"}
```

`llm.py` already appends `/chat/completions` to the base URL and sends `Authorization: Bearer <key>`, which the Gemini compatibility layer accepts per its docs [V for the layer, I for the header]. If the Gemini OpenAI-compat layer rejects the raw `json_schema` body, set `"json_schema": false` (falls back to `json_object`, and the prompt already asks for JSON) or add a `call_gemini` branch with `google-genai` as above.

**Setup steps (owner actions; I created nothing).**
1. Get a key at `https://aistudio.google.com/api-keys` (AI Studio creates a project and key for new users) [V].
2. For real-speech use, link a billing account in AI Studio to move the project to Tier 1 (instant); otherwise keep the key free-tier and demo-only [V].
3. PowerShell: `$env:GEMINI_API_KEY = "<key>"` in the session; do not put it in `llm.json`.
4. `python llm.py demo` with `profile=gemini_flash_lite`; record wall time from the venue network over about 20 calls. Adopt only if p95 is under the 6 s timeout, preferably under 2 s.
5. Optional: `pip install -U google-genai` for the native-SDK path.
6. At the opening ceremony and on the HackTX Devpost, check (i) whether "Best Use of Gemini API" is offered and its wording, (ii) whether Meta, xAI/SpaceXAI or Google have tracks, and ask the MLH table whether Gemma-on-device counts. Choose the Devpost track slots (max 3) after that.

## Could not verify

- Any HackTX 2026 track, prize rule or sponsor-specific API requirement (not published; hacktx.com and Devpost not readable by my tools).
- That MLH's Gemini prize is active at HackTX 2026 (generic MLH page says Q1 2025 - Q1 2026; the earlier event-facts file says the HackTX MLH list included it) and what qualifies beyond "use Gemini".
- Gemini free-tier RPM/RPD for any current model (docs defer to AI Studio; blogs conflict).
- Real latency for any of these models at our 800-token size; Artificial Analysis numbers are for a 10k-token workload and differ across snapshots.
- Whether the Gemini OpenAI-compatible layer accepts a raw `json_schema` `response_format` and a `minimal` reasoning effort for 3.x models (docs show only `parse()` and say `none` is 2.5-only); whether schema + `thinking_level` combine in `google-genai` as written above.
- Meta Model API current access (waitlist, US-only, free credits) and its official data terms and rate limits; the Muse Spark 1.3 prices come from one tutorial.
- xAI free credits for students or hackathons, any HackTX sponsorship by xAI/SpaceXAI, the exact meaning of "SpaceXSI" (the name appears on MLH's page and on `x.ai/api`'s title; I did not find an explanation), whether `grok-4.20-0309-non-reasoning` supports structured outputs (docs say all examples use `grok-4.7`), and any latency data.
- Snowflake Cortex and Vultr details beyond the base URL and plans above; AWS Bedrock/Annapurna terms (not researched).
- Whether the owner's "xAI is a sponsor" remark reflects anything published.

## Sources

- MLH prizes page (categories, windows): https://www.mlh.com/events/prizes
- MLH HackGT 13 prizes (Gemini text, no extra rules): https://www.mlh.com/events/hackgt-13/prizes
- MLH challenge page (submissions only during event, per search summary): https://www.mlh.com/events/code-api/challenges/019d43c3-bed0-523b-a04f-80bd5fcc8276
- Gemini pricing and free-tier data use: https://ai.google.dev/gemini-api/docs/pricing
- Gemini models page: https://ai.google.dev/gemini-api/docs/models
- Gemini OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai
- Gemini structured output: https://ai.google.dev/gemini-api/docs/structured-output
- Gemini thinking levels: https://ai.google.dev/gemini-api/docs/thinking
- Gemini rate limits: https://ai.google.dev/gemini-api/docs/rate-limits
- Gemini quickstart (SDK, key): https://ai.google.dev/gemini-api/docs/quickstart
- Gemini API additional terms (unpaid services): https://ai.google.dev/gemini-api/terms
- Artificial Analysis, Gemini 3.5 Flash-Lite: https://artificialanalysis.ai/models/gemini-3-5-flash-lite/providers
- Meta Model API overview (redirected from llama.developer.meta.com): https://dev.meta.ai/docs/overview.md
- Meta structured output: https://dev.meta.ai/docs/structured-output.md
- Muse Spark 1.3 tutorial (pricing, tiers, data): https://www.datacamp.com/tutorial/muse-spark-1-3-tutorial
- Llama API retirement note: https://www.promptfoo.dev/docs/providers/llamaApi/
- Muse Spark launch coverage: https://www.thestar.com.my/tech/tech-news/2026/07/09/meta-debuts-muse-spark-11-with-preview-open-to-developers , https://aiweekly.co/alerts/meta-prices-muse-spark-11-api-at-125425-per-m-tokens
- xAI models and pricing: https://docs.x.ai/docs/models
- xAI structured outputs: https://docs.x.ai/docs/guides/structured-outputs
- xAI rate limits: https://docs.x.ai/docs/key-information/consumption-and-rate-limits
- xAI enterprise FAQ (training and credits): https://x.ai/legal/faq-enterprise
- xAI API page (title only): https://x.ai/api
- Grok Bot / SpaceXAI coverage: https://www.thestar.com.my/tech/tech-news/2026/08/12/spacexai-unveils-grok-bot-to-work-like-a-team-of-ai-agents , https://luma.com/cursor-tp8z , https://www.princeton.edu/events/2026/spacexai-x-hackprinceton-grok-bot-build-night
- xAI credits (blogs, conflicting): https://agentdeals.dev/vendor/xai , https://www.getaiperks.com/en/blogs/22-xai-grok-free-credits
- Vultr Serverless Inference docs: https://docs.vultr.com/how-to-use-vultr-serverless-inference-in-python , https://docs.vultr.com/products/serverless/inference/faq
- Local: `docs/RESEARCH-llm.md`, `docs/EVALUATION.md`, `llm.py`, `config/llm.json`, `ClaudeOS/projects/hacktx/reference/event-facts-2026.md`, `hacktx26-sponsors.md`
