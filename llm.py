"""Optional LLM composer: turns Tink's frame into 3 short sentences in the user's voice.

The frame is what Tink already knows: the intent distribution, the referent (pointed at, signed or mentioned),
the partner's last words, a few turns of history and the deterministic candidates. The model may only
rephrase or specify within that frame; it must not add facts the user never signalled, and every option is
still confirmed by the user. Any OpenAI-compatible endpoint works (Ollama, LM Studio, mlx_lm.server, Groq,
OpenRouter, Cerebras, Gemini's compatibility endpoint), so local vs API is a base URL in config/llm.json.
Calls run in a background thread with a timeout; on timeout or bad output Tink keeps its own candidates.

Run: python llm.py demo        (one frame through the configured endpoint)
"""
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'config' / 'llm.json'
SYSTEM = (
    "You write what an AAC user wants to say, in their voice: first person, plain, at most 15 words per "
    "sentence. You get a frame of evidence about their intent. Rules: use only what the frame contains (the "
    "intent, the referent, the partner's words, the history); never add new facts, names, quantities or "
    "preferences; if the partner asked a yes/no question and the intent is Yes or No, answer that question. "
    "Return JSON only: {\"options\": [{\"text\": \"...\", \"intent\": \"...\"}, ...]} with exactly 3 options, "
    "most likely first, each different in meaning or wording."
)
SCHEMA = {'type': 'object', 'properties': {'options': {'type': 'array', 'minItems': 3, 'maxItems': 3, 'items': {
    'type': 'object', 'properties': {'text': {'type': 'string'}, 'intent': {'type': 'string'}},
    'required': ['text', 'intent']}}}, 'required': ['options']}


def load_config(path=CONFIG):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    profile = cfg['profiles'][cfg['active']]
    return {**profile, 'timeout_s': cfg.get('timeout_s', 4.0), 'name': cfg['active']}


def frame_for(meaning, belief, partner_text, history, k_candidates=6):
    """The compact frame sent to the model (no images, no raw signals, no personal data beyond the chat)."""
    refs = [(r, round(p, 2)) for r, p in (meaning.get('referents') or {}).items() if r != '(none)'][:2]
    return {
        'intents': {i: round(p, 2) for i, p in list((meaning.get('intents') or belief).items())[:4]},
        'referents': dict(refs),
        'partner_said': partner_text or None,
        'question_type': (meaning.get('context') or {}).get('type'),
        'history': [f"{'partner' if w == 'partner' else 'me'}: {t}" for w, t in history[-4:]],
        'candidates': [c['text'] for c in (meaning.get('candidates') or [])[:k_candidates]],
        'mixed_signals': (meaning.get('clarify') or {}).get('detail'),
    }


def clean_options(opts):
    clean = []
    for o in opts:
        t = ' '.join(str(o.get('text', '')).split())
        if t and len(t.split()) <= 20 and t not in [c['text'] for c in clean]:
            clean.append({'text': t, 'intent': str(o.get('intent', ''))})
    if not clean:
        raise ValueError('no usable options')
    return clean[:3]


def call_anthropic(cfg, frame):
    """Claude through the official SDK with native structured output (output_config.format). Thinking is
    disabled for latency (allowed on Claude Haiku 5.5 at its default effort); no temperature (rejected)."""
    import anthropic
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['options'], 'properties': {
        'options': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                                               'required': ['text', 'intent'],
                                               'properties': {'text': {'type': 'string'}, 'intent': {'type': 'string'}}}}}}
    client = anthropic.Anthropic(timeout=cfg['timeout_s'], max_retries=0)
    response = client.messages.create(
        model=cfg['model'], max_tokens=400, system=SYSTEM, thinking={'type': 'disabled'},
        messages=[{'role': 'user', 'content': json.dumps(frame, ensure_ascii=False)}],
        output_config={'format': {'type': 'json_schema', 'schema': schema}})
    if response.stop_reason == 'refusal':
        raise ValueError('refused')
    text = next(b.text for b in response.content if b.type == 'text')
    return clean_options(json.loads(text)['options'])


def call(cfg, frame):
    """One completion; returns [{'text', 'intent'}] or raises."""
    if cfg.get('provider') == 'anthropic':
        return call_anthropic(cfg, frame)
    body = {'model': cfg['model'], 'temperature': 0.3, 'max_tokens': 220,
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': json.dumps(frame, ensure_ascii=False)}]}
    if cfg.get('json_schema', True):
        body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'options', 'schema': SCHEMA}}
    else:
        body['response_format'] = {'type': 'json_object'}
    body.update(cfg.get('extra') or {})        # e.g. {"reasoning_effort": "none"} to turn thinking off
    headers = {'Content-Type': 'application/json'}
    key = os.environ.get(cfg.get('api_key_env') or '', '')
    if key:
        headers['Authorization'] = f'Bearer {key}'
    req = urllib.request.Request(cfg['base_url'].rstrip('/') + '/chat/completions',
                                 data=json.dumps(body).encode('utf-8'), headers=headers)
    with urllib.request.urlopen(req, timeout=cfg['timeout_s']) as r:
        out = json.loads(r.read().decode('utf-8'))
    text = out['choices'][0]['message']['content'].strip()
    if text.startswith('```'):
        text = text.strip('`').split('\n', 1)[-1]
    return clean_options(json.loads(text[text.find('{'):text.rfind('}') + 1])['options'])


class LLMComposer:
    """Non-blocking: request(frame) starts a call unless one is running; latest() returns the newest options
    with the frame they answer, plus status and latency."""

    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        self.lock = threading.Lock()
        self.busy = False
        self.result = {'status': 'idle', 'options': [], 'key': None, 'latency_s': None}

    def request(self, frame, key):
        with self.lock:
            if self.busy or self.result.get('key') == key:
                return
            self.busy = True
        threading.Thread(target=self._run, args=(frame, key), daemon=True).start()

    def _run(self, frame, key):
        t0 = time.monotonic()
        try:
            opts = call(self.cfg, frame)
            res = {'status': 'ok', 'options': opts, 'key': key, 'latency_s': round(time.monotonic() - t0, 2)}
        except Exception as e:
            res = {'status': f'error: {str(e)[:60]}', 'options': [], 'key': key,
                   'latency_s': round(time.monotonic() - t0, 2)}
        with self.lock:
            self.result, self.busy = res, False

    def latest(self):
        with self.lock:
            return dict(self.result)


def demo():
    cfg = load_config()
    frame = {'intents': {'Yes / I agree': 0.78, 'No / I disagree': 0.07, 'I want something': 0.05},
             'referents': {'water': 0.6}, 'partner_said': 'Do you want some water?', 'question_type': 'yesno',
             'history': ['partner: Do you want some water?'],
             'candidates': ['Yes, I want some water.', 'Yes please. Thank you.', 'No, I don\'t want some water.'],
             'mixed_signals': None}
    print(f"profile {cfg['name']}: {cfg['model']} at {cfg['base_url']}")
    t0 = time.monotonic()
    try:
        opts = call(cfg, frame)
        print(f'{time.monotonic() - t0:.2f} s')
        for o in opts:
            print(f"  [{o['intent']}] {o['text']}")
    except Exception as e:
        print(f'failed after {time.monotonic() - t0:.2f} s: {e}')


if __name__ == '__main__':
    demo()
