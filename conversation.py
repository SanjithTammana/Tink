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

Run: python conversation.py fetch | build | soften | fit | query "Do you want some water?" | eval
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


SOFT = ROOT / 'models' / 'conversation_soft.npz'
PARAMS = ROOT / 'models' / 'conversation_params.json'
RULE_WEIGHT = 0.5    # soft reply label = RULE_WEIGHT * rule one-hot + (1 - RULE_WEIGHT) * act-model distribution


def soften(intents):
    """Soft class distribution for every corpus reply: half the rule label (it knows Tink-only intents such as
    'I need a break'), half the SwDA act model's P(class | reply). Saved beside the index."""
    from acts import ActModel
    am = ActModel()
    z = np.load(INDEX, allow_pickle=False)
    classes = list(intents) + [OTHER]
    replies = [str(r) for r in z['reply']]
    print(f'soft-labelling {len(replies)} replies with the act model (one-off, a few minutes on CPU)')
    act_soft = am.reply_class(replies)                     # (n, am.classes)
    soft = np.zeros((len(replies), len(classes)), np.float32)
    for j, c in enumerate(am.classes):
        soft[:, classes.index(c)] += (1 - RULE_WEIGHT) * act_soft[:, j]
    for i, label in enumerate(z['label']):
        soft[i, classes.index(str(label))] += RULE_WEIGHT
    np.savez_compressed(SOFT, soft=soft.astype(np.float16), classes=np.array(classes))
    print('saved', SOFT, '| mean soft mass:', {c: round(float(v), 3) for c, v in zip(classes, soft.mean(0))})


class ConversationModel:
    POOL = 300
    """P(class | partner text) = w * P_knn + (1 - w) * P_act, where
    P_knn = (alpha / K + sum_i w_i soft_i) / (alpha + sum_i w_i), w_i = exp((cos_i - cos_max) / tau) over the k
    nearest partner turns (soft_i: the neighbour reply's soft label, or its rule one-hot without soft labels);
    P_act = the SwDA act-transition estimate (acts.ActModel.context_class). tau and w are fitted on held-out
    data by cross-entropy (`python conversation.py fit`) and read from models/conversation_params.json."""

    def __init__(self, intents, path=INDEX, splits=('train', 'validation'), k=40, tau=0.05, alpha=1.0,
                 min_cos=0.45, model=None, w_knn=1.0, use_soft=True, beta=0.0):
        import json
        z = np.load(path, allow_pickle=False)
        keep = np.isin(z['split'], list(splits))
        self.ctx, self.reply, self.label = z['ctx'][keep], z['reply'][keep], z['label'][keep]
        self.vecs = z['vecs'][keep].astype(np.float32)
        self.intents = [i for i in intents]
        self.classes = self.intents + [OTHER]
        if PARAMS.exists():
            params = json.loads(PARAMS.read_text(encoding='utf-8'))
            tau, w_knn, beta = params.get('tau', tau), params.get('w_knn', w_knn), params.get('beta', beta)
        self.k, self.tau, self.alpha, self.min_cos, self.w_knn, self.beta = k, tau, alpha, min_cos, w_knn, beta
        self.soft = None
        if use_soft and SOFT.exists():
            s = np.load(SOFT, allow_pickle=False)
            if list(s['classes']) == self.classes:
                self.soft = s['soft'][keep].astype(np.float32)
        if self.soft is None:
            self.soft = np.eye(len(self.classes), dtype=np.float32)[[self.classes.index(str(l)) for l in self.label]]
        from acts import MODEL, ActModel
        self.acts = ActModel() if MODEL.exists() else None
        # dialogue-act distribution of every partner turn, straight from the stored embeddings (no re-encoding)
        self.ctx_acts = self.acts.act_probs(vecs=self.vecs).astype(np.float32) if self.acts else None
        self.model = model or embedder()
        if self.acts:
            self.acts.encoder = self.model

    def act_prior(self, vec):
        p = self.acts.context_class(vecs=vec[None])[0]
        out = np.zeros(len(self.classes))
        for j, c in enumerate(self.acts.classes):
            out[self.classes.index(c)] += p[j]
        return out

    def kernel(self, q, sims, qa=None, tau=None, beta=None, k=None):
        """Indices and weights of the k neighbours: w_i = exp((cos_i - cos_max) / tau) * BC(q, i) ** beta, over a
        pool of the POOL most similar partner turns. BC is the Bhattacharyya coefficient between the two turns'
        dialogue-act distributions (1 = same kind of turn), so a yes/no question mostly draws on replies to
        other yes/no questions rather than on topically similar requests."""
        tau, beta, k = tau or self.tau, self.beta if beta is None else beta, k or self.k
        pool = np.argpartition(-sims, self.POOL)[:self.POOL]
        logw = (sims[pool] - sims[pool].max()) / tau
        if beta and self.ctx_acts is not None:
            qa = self.acts.act_probs(vecs=q[None])[0] if qa is None else qa
            bc = np.sqrt(self.ctx_acts[pool] * qa).sum(axis=1)
            logw = logw + beta * np.log(np.clip(bc, 1e-6, 1))
        order = np.argsort(-logw)[:k]
        top, logw = pool[order], logw[order]
        return top, np.exp(logw - logw.max())

    def query(self, text, k=None):
        """{'prior': {intent or 'other': p}, 'replies': [(reply, label, weight, partner turn)], 'match': cos}
        or None when nothing in the corpora is close enough (max cos < min_cos)."""
        if not text or not text.strip():
            return None
        q = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.vecs @ q
        if sims.max() < self.min_cos:
            return None
        top, w = self.kernel(q, sims, k=k)
        p = (self.alpha / len(self.classes) + w @ self.soft[top]) / (self.alpha + w.sum())
        if self.acts and self.w_knn < 1.0:
            p = self.w_knn * p + (1 - self.w_knn) * self.act_prior(q)
        prior = dict(sorted(zip(self.classes, p.tolist()), key=lambda x: -x[1]))
        replies, seen = [], set()
        for i, wi in zip(top, w):
            r = str(self.reply[i])
            if r.lower() not in seen and len(r.split()) <= 20:
                seen.add(r.lower())
                replies.append((r, str(self.label[i]), float(wi / w.sum()), str(self.ctx[i])))
        return {'prior': prior, 'replies': replies, 'match': float(sims.max())}


def fit(intents, n=800, seed=0):
    """Choose tau and w_knn by cross-entropy on validation pairs (neighbours from train only). The target is
    the true reply's soft label, so a sharper or flatter curve is rewarded only if it predicts better."""
    import json
    from acts import ActModel
    if PARAMS.exists():
        PARAMS.unlink()      # fit from the defaults, not from a previous fit
    cm = ConversationModel(intents, splits=('train',), w_knn=0.5)
    z = np.load(INDEX, allow_pickle=False)
    s = np.load(SOFT, allow_pickle=False)['soft'].astype(np.float32) if SOFT.exists() else None
    val = np.flatnonzero(z['split'] == 'validation')
    val = np.random.default_rng(seed).choice(val, size=min(n, len(val)), replace=False)
    Q = cm.model.encode([str(z['ctx'][i]) for i in val], batch_size=256, normalize_embeddings=True,
                        convert_to_numpy=True)
    target = s[val] if s is not None else np.eye(len(cm.classes))[[cm.classes.index(str(z['label'][i])) for i in val]]
    act = np.array([cm.act_prior(q) for q in Q]) if cm.acts else None
    QA = cm.acts.act_probs(vecs=Q) if cm.acts else [None] * len(Q)
    results = []
    for tau in (0.2, 0.5, 1.0, 2.0):
        for beta in ((0.0, 1.0, 2.0, 4.0) if cm.acts else (0.0,)):
            knn = []
            for q, qa in zip(Q, QA):
                top, w = cm.kernel(q, cm.vecs @ q, qa=qa, tau=tau, beta=beta)
                knn.append((cm.alpha / len(cm.classes) + w @ cm.soft[top]) / (cm.alpha + w.sum()))
            knn = np.array(knn)
            for w_knn in ((0.5, 0.75, 1.0) if act is not None else (1.0,)):
                p = w_knn * knn + (1 - w_knn) * (act if act is not None else 0)
                ce = -np.mean(np.sum(target * np.log(np.clip(p, 1e-9, 1)), axis=1))
                results.append((ce, tau, beta, w_knn))
    results.sort()
    base = -np.mean(np.sum(target * np.log(np.clip(target.mean(0), 1e-9, 1)), axis=1))
    print(f'{len(val)} validation pairs | base-rate cross-entropy {base:.3f}')
    for ce, tau, beta, w_knn in results[:8]:
        print(f'  tau={tau:<4} beta={beta:<4} w_knn={w_knn:<5} cross-entropy {ce:.3f}')
    for ce, tau, beta, w_knn in results:
        if beta == 0.0 and w_knn == 1.0 and tau == results[0][1]:
            print(f'  (same tau, meaning-only kernel, no act mixture: {ce:.3f})')
    ce, tau, beta, w_knn = results[0]
    PARAMS.write_text(json.dumps({'tau': tau, 'beta': beta, 'w_knn': w_knn, 'validation_cross_entropy': round(ce, 4),
                                  'base_rate_cross_entropy': round(float(base), 4)}, indent=2), encoding='utf-8')
    print('saved', PARAMS)


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
    p.add_argument('cmd', choices=['fetch', 'build', 'soften', 'fit', 'query', 'eval'])
    p.add_argument('text', nargs='?')
    a = p.parse_args()
    if a.cmd == 'fetch':
        fetch()
    elif a.cmd == 'build':
        build()
    elif a.cmd == 'eval':
        evaluate()
    elif a.cmd in ('soften', 'fit'):
        from retrieval import intents_from_config
        intents = [i for i in intents_from_config() if i != '(no message)']
        soften(intents) if a.cmd == 'soften' else fit(intents)
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
