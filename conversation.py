"""Conversation model: what people say back, given what the partner just said, learned from dialogue corpora
by embedding.

Every (partner turn -> reply) pair in the corpora is embedded (MiniLM) on the partner side and its reply is
labelled with one of Tink's intents, or 'other' when the reply is something no intent expresses (most
statements). For a new partner utterance q, the k nearest partner turns give
  P(intent | q) = (sum_i w_i [label_i = intent] + alpha / K) / (sum_i w_i + alpha),  w_i = exp((cos_i - cos_max) / tau)
a data-driven context prior (replacing hand rules such as "a yes/no question boosts Yes and No"), and the
replies themselves as sentence candidates. The labels come from transparent phrase rules (REPLY_RULES) on the
reply's opening words; anything else is 'other'.

Corpora (downloaded by `python conversation.py fetch`, kept out of git):
- DailyDialog (Li et al. 2017), 13,118 everyday dialogues, CC BY-NC-SA 4.0, via ConvLab on Hugging Face.
- Vertanen (2017) crowdsourced AAC-style dialogues, 1,419 x 6 turns; no license stated, so local use only.

Run: python conversation.py fetch | build | query "Do you want some water?" | eval
"""
import argparse
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'dialogue'
INDEX = ROOT / 'models' / 'conversation_index.npz'
OTHER = 'other'
SOURCES = {
    'dailydialog_convlab.zip': 'https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip',
    'turk-dialogues.txt': 'https://keithv.com/data/turk-dialogues.txt',
}
# reply opening -> intent. Checked in order; first match wins. Lower-cased, punctuation stripped.
REPLY_RULES = [
    ('Can you repeat that?', r"^(pardon|sorry\?|what did you say|say that again|can you repeat|could you repeat|"
                             r"i didn'?t (catch|hear|understand)|come again|what\?$)"),
    ('No / I disagree', r"^(no|nope|nah|not really|not at all|no thanks|i don'?t think so|i'?m afraid not|"
                        r"never|i disagree|i don'?t agree|of course not|not yet)\b"),
    ('Yes / I agree', r"^(yes|yeah|yep|yup|sure|of course|ok|okay|certainly|definitely|absolutely|i agree|"
                      r"sounds good|that'?s right|right|exactly|i'?d love to|alright|all right|why not|great)\b"),
    ('I need help', r"^(help|can you help|could you help|please help|i need (some |your )?help|"
                    r"would you help|i need you to)\b"),
    ('I need a break', r"^((i'?m|i am) (so |very |really |a bit |a little )?(tired|exhausted|sleepy|worn out)|i need (a |to )?(break|rest|sleep)|"
                       r"can we (stop|take a break)|let me rest)\b"),
    ('I want to show you something', r"^(look at (this|that)|look!|check (this|that) out|see (this|that)|"
                                     r"have a look|i want to show you)\b"),
    ('Please pay attention', r"^(excuse me|listen|hey|look at me|wait|hold on|attention)\b"),
    ('I want something', r"^(i want|i'?d like|i would like|can i have|could i have|may i have|i need|"
                         r"give me|get me|bring me|can you get|could you get|could you bring|can you bring|"
                         r"can i get|i'?ll have|please (bring|get|give|pass))\b"),
]


def label_reply(text):
    t = re.sub(r"[^a-z' ?!]", ' ', (text or '').lower()).strip()
    t = ' '.join(t.split())
    for intent, pattern in REPLY_RULES:
        if re.search(pattern, t):
            return intent
    return OTHER


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = DATA / name
        if not path.exists():
            print('Downloading', url)
            urllib.request.urlretrieve(url, path)
    print('data in', DATA)


def load_pairs():
    """[(partner turn, reply, corpus, split)] from both corpora."""
    pairs = []
    z = zipfile.ZipFile(DATA / 'dailydialog_convlab.zip')
    dialogues = json.loads(z.read('data/dialogues.json'))
    for d in dialogues:
        turns = [t['utterance'].strip() for t in d['turns']]
        split = d['data_split']
        pairs += [(a, b, 'dailydialog', split) for a, b in zip(turns, turns[1:])]
    lines = (DATA / 'turk-dialogues.txt').read_text(encoding='utf-8').splitlines()[1:]
    for line in lines:
        cols = line.split('\t')
        split = 'test' if cols[0] == 'test' else ('validation' if cols[0] in ('dev', 'validation') else 'train')
        turns = [c.strip() for c in cols[2:] if c.strip()]
        pairs += [(a, b, 'aac-dialogues', split) for a, b in zip(turns, turns[1:])]
    return pairs


def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')


def build():
    pairs = load_pairs()
    model = embedder()
    ctx = [p[0] for p in pairs]
    print(f'embedding {len(ctx)} partner turns (one-off, a few minutes on CPU)')
    vecs = model.encode(ctx, batch_size=256, normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False).astype(np.float16)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INDEX, ctx=np.array(ctx), reply=np.array([p[1] for p in pairs]),
                        label=np.array([label_reply(p[1]) for p in pairs]),
                        corpus=np.array([p[2] for p in pairs]), split=np.array([p[3] for p in pairs]), vecs=vecs)
    labels = [label_reply(p[1]) for p in pairs]
    print('saved', INDEX, '| reply labels:', {l: labels.count(l) for l in sorted(set(labels))})


class ConversationModel:
    def __init__(self, intents, path=INDEX, splits=('train', 'validation'), k=40, tau=0.05, alpha=1.0,
                 min_cos=0.45, model=None):
        z = np.load(path, allow_pickle=False)
        keep = np.isin(z['split'], list(splits))
        self.ctx, self.reply, self.label = z['ctx'][keep], z['reply'][keep], z['label'][keep]
        self.vecs = z['vecs'][keep].astype(np.float32)
        self.intents = [i for i in intents]
        self.k, self.tau, self.alpha, self.min_cos = k, tau, alpha, min_cos
        self.model = model or embedder()

    def query(self, text, k=None):
        """{'prior': {intent or 'other': p}, 'replies': [(reply, label, weight, partner turn)], 'match': cos}
        or None when nothing in the corpora is close enough (max cos < min_cos)."""
        if not text or not text.strip():
            return None
        q = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.vecs @ q
        k = k or self.k
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        if sims[top[0]] < self.min_cos:
            return None
        w = np.exp((sims[top] - sims[top[0]]) / self.tau)
        classes = self.intents + [OTHER]
        mass = {c: self.alpha / len(classes) for c in classes}
        for i, wi in zip(top, w):
            mass[str(self.label[i])] = mass.get(str(self.label[i]), 0.0) + float(wi)
        z = sum(mass.values())
        prior = dict(sorted(((c, m / z) for c, m in mass.items()), key=lambda x: -x[1]))
        replies, seen = [], set()
        for i, wi in zip(top, w):
            r = str(self.reply[i])
            if r.lower() not in seen and len(r.split()) <= 20:
                seen.add(r.lower())
                replies.append((r, str(self.label[i]), float(wi / w.sum()), str(self.ctx[i])))
        return {'prior': prior, 'replies': replies, 'match': float(sims[top[0]])}


def evaluate(n=600, seed=0):
    """Held-out test pairs: does P(intent | partner turn) predict the reply's label, and can the system produce
    something close to the actual reply (best cosine of the candidates the model offers)?"""
    from retrieval import intents_from_config
    intents = [i for i in intents_from_config() if i != '(no message)']
    cm = ConversationModel(intents)
    z = np.load(INDEX, allow_pickle=False)
    test = np.flatnonzero(z['split'] == 'test')
    rng = np.random.default_rng(seed)
    test = rng.choice(test, size=min(n, len(test)), replace=False)
    classes = intents + [OTHER]
    rule = lambda t: ('Yes / I agree' if t.split()[0].lower() in
                      ('do', 'does', 'did', 'is', 'are', 'can', 'could', 'would', 'will', 'should', 'have')
                      else OTHER) if t.split() else OTHER
    base = {c: float(np.mean(z['label'][z['split'] != 'test'] == c)) for c in classes}
    ll = {'uniform': [], 'base rates': [], 'conversation model': []}
    acc = {'base rates': [], 'yes/no-question rule': [], 'conversation model': []}
    best_cos, best_cos_label = [], []
    for i in test:
        ctx, reply, truth = str(z['ctx'][i]), str(z['reply'][i]), str(z['label'][i])
        out = cm.query(ctx)
        prior = out['prior'] if out else {c: 1 / len(classes) for c in classes}
        ll['uniform'].append(-np.log(1 / len(classes)))
        ll['base rates'].append(-np.log(max(base[truth], 1e-9)))
        ll['conversation model'].append(-np.log(max(prior.get(truth, 1e-9), 1e-9)))
        acc['base rates'].append(max(base, key=base.get) == truth)
        acc['yes/no-question rule'].append(rule(ctx) == truth)
        acc['conversation model'].append(next(iter(prior)) == truth)
        if out:
            cands = [r for r, _, _, _ in out['replies'][:20]]
            sims = cm.model.encode([reply] + cands, normalize_embeddings=True, convert_to_numpy=True)
            best_cos.append(float((sims[1:] @ sims[0]).max()))
        else:
            best_cos.append(0.0)
        best_cos_label.append(truth)
    print(f'{len(test)} held-out (partner turn -> reply) pairs\n')
    print('Predicting the reply type from the partner turn:')
    for name, v in ll.items():
        a = acc.get(name)
        print(f"  {name:22s} log-loss {np.mean(v):.2f}" + (f"   top-1 acc {np.mean(a):.2f}" if a else ''))
    print(f"  {'yes/no-question rule':22s}                top-1 acc {np.mean(acc['yes/no-question rule']):.2f}")
    print(f"\nReply types in held-out data: " + ', '.join(
        f'{c} {np.mean(np.array(best_cos_label) == c) * 100:.0f}%' for c in classes if np.any(np.array(best_cos_label) == c)))
    bc = np.array(best_cos)
    print('\nCan a top-20 retrieved reply stand in for the real one?')
    for th in (0.8, 0.7, 0.6, 0.5):
        print(f'  best cosine >= {th}: {np.mean(bc >= th) * 100:5.1f}% of pairs')
    for c in classes:
        m = np.array(best_cos_label) == c
        if m.sum() >= 5:
            print(f'    {c:30s} n={m.sum():3d}  share >= 0.7: {np.mean(bc[m] >= 0.7) * 100:5.1f}%')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('cmd', choices=['fetch', 'build', 'query', 'eval'])
    p.add_argument('text', nargs='?')
    a = p.parse_args()
    if a.cmd == 'fetch':
        fetch()
    elif a.cmd == 'build':
        build()
    elif a.cmd == 'eval':
        evaluate()
    else:
        from retrieval import intents_from_config
        cm = ConversationModel([i for i in intents_from_config() if i != '(no message)'])
        out = cm.query(a.text)
        if not out:
            print('no close context')
            return
        print(f"closest partner turn cos {out['match']:.2f}")
        print('P(reply type | partner):', {c: round(p, 2) for c, p in list(out['prior'].items())[:5]})
        for r, l, w, c in out['replies'][:8]:
            print(f'  {w:.2f}  [{l}] {r}    <- {c!r}')


if __name__ == '__main__':
    main()
