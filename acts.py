"""Dialogue acts as probabilities (Switchboard Dialog Act corpus), for the conversation layer.

Three pieces, all distributions rather than hard labels:
- P(act | text): logistic regression on MiniLM embeddings, trained on SwDA's human-tagged utterances.
- T[a -> b] = P(reply act b | partner act a): counts of consecutive turns by different speakers in SwDA,
  Dirichlet-smoothed (pseudo-count ALPHA_T spread by the reply-act base rates).
- M[b -> c] = P(Tink class c | act b): a small hand-written soft map (ACT_TO_INTENT) from SwDA acts to Tink's
  intents and 'other'; acts not listed are 'other'.
Together: P(c | partner text) = sum_a P(a | text) sum_b T[a, b] M[b, c]  (the act-transition estimate), and a
reply's soft class P(c | reply) = sum_b P(b | reply) M[b, c] (used to soft-label the dialogue corpora).

SwDA: Jurafsky, Shriberg & Biasca (1997); Potts' CSV release (github.com/cgpotts/swda), CC BY-NC-SA 3.0.
Downloaded to data/dialogue/ by `python acts.py fetch`, not committed.

Run: python acts.py fetch | train | eval | demo
"""
import argparse
import csv
import io
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'dialogue'
SWDA = DATA / 'swda.zip'
MODEL = ROOT / 'models' / 'acts.joblib'
URL = 'https://github.com/cgpotts/swda/raw/master/swda.zip'
OTHER = 'other'
ALPHA_T = 20.0
MIN_COUNT = 150          # acts rarer than this in training are folded into 'x_rare'
# SwDA act -> Tink class. Probabilities; the remainder goes to 'other'. Names from the SWBD-DAMSL manual.
ACT_TO_INTENT = {
    'ny': {'Yes / I agree': 0.95},                  # yes answer
    'na': {'Yes / I agree': 0.8},                   # affirmative non-yes answer
    'aa': {'Yes / I agree': 0.85},                  # agree / accept
    'aap_am': {'Yes / I agree': 0.6},               # maybe / accept-part
    'bk': {'Yes / I agree': 0.4},                   # response acknowledgement
    'b': {'Yes / I agree': 0.3},                    # backchannel (uh-huh)
    'nn': {'No / I disagree': 0.95},                # no answer
    'ng': {'No / I disagree': 0.8},                 # negative non-no answer
    'ar': {'No / I disagree': 0.85},                # reject
    'arp_nd': {'No / I disagree': 0.6},             # dispreferred answer
    'br': {'Can you repeat that?': 0.9},            # signal non-understanding
    'ad': {'I want something': 0.4, 'I need help': 0.2, 'Please pay attention': 0.1},   # action directive
    'qy': {}, 'qw': {}, 'qo': {}, 'qh': {}, 'qrr': {},          # questions: 'other' (asking, not answering)
}


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    if not SWDA.exists():
        print('Downloading', URL)
        urllib.request.urlretrieve(URL, SWDA)
    print('data in', DATA)


def clean(text):
    """Strip SwDA transcription markup ({F uh, }, [ ... + ... ], <noise>, slashes, dashes)."""
    t = re.sub(r'<<?[^>]*>>?', ' ', text)
    t = re.sub(r'\{[A-Z] ', ' ', t).replace('}', ' ')
    t = re.sub(r'[\[\]+/#()]', ' ', t).replace(' -', ' ').replace('--', ' ')
    return ' '.join(t.split())


def base_tag(tag):
    tag = tag.strip()
    if tag in ('qy^d', 'qw^d', 'b^m'):
        return tag
    t = tag.split('^')[0].split('(')[0].strip() or tag
    return {'fo_o_fw_"_by_bc': 'fo_o_fw_by_bc', 'sd(^q)': 'sd'}.get(t, t)


def load_conversations():
    """[(conversation_no, [(caller, act, text)])] with '+' continuations merged into the utterance they
    continue (same caller), and abandoned/uninterpretable fragments ('%', 'x') kept as their own acts."""
    z = zipfile.ZipFile(SWDA)
    convs = []
    for name in sorted(n for n in z.namelist() if n.endswith('.utt.csv') and not n.startswith('__MACOSX')):
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding='utf-8')))
        turns = []
        last = {}
        for r in rows:
            caller, tag, text = r['caller'], r['act_tag'], clean(r['text'])
            if tag == '+' and caller in last:
                i = last[caller]
                turns[i] = (turns[i][0], turns[i][1], (turns[i][2] + ' ' + text).strip())
                continue
            turns.append((caller, base_tag(tag), text))
            last[caller] = len(turns) - 1
        convs.append((rows[0]['conversation_no'], [t for t in turns if t[2]]))
    return convs


def split_of(conv_no):
    """Deterministic 80/10/10 split by conversation."""
    h = int(conv_no) % 10
    return 'test' if h == 0 else ('validation' if h == 1 else 'train')


class ActModel:
    def __init__(self, path=MODEL):
        import joblib
        data = joblib.load(path)
        self.clf, self.acts, self.T = data['clf'], data['acts'], data['T']
        self.classes = data['classes']
        self.M = np.array([[self.act_to_class(a).get(c, 0.0) for c in self.classes] for a in self.acts])
        self.encoder = None

    def act_to_class(self, act):
        row = dict(ACT_TO_INTENT.get(act, {}))
        row[OTHER] = row.get(OTHER, 0.0) + max(0.0, 1 - sum(row.values()))
        return row

    def embed(self, texts):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        return self.encoder.encode(list(texts), batch_size=256, normalize_embeddings=True, convert_to_numpy=True)

    def act_probs(self, texts=None, vecs=None):
        vecs = self.embed(texts) if vecs is None else vecs
        return self.clf.predict_proba(vecs)               # (n, acts), columns in self.acts order

    def reply_class(self, texts=None, vecs=None):
        """P(Tink class | reply text) for each reply: (n, classes)."""
        return self.act_probs(texts, vecs) @ self.M

    def context_class(self, texts=None, vecs=None):
        """Act-transition estimate P(Tink class of the reply | partner text): (n, classes)."""
        return self.act_probs(texts, vecs) @ self.T @ self.M


def train(cap=5000):
    import joblib
    from sklearn.linear_model import LogisticRegression
    convs = load_conversations()
    utts = [(split_of(no), act, text) for no, turns in convs for _, act, text in turns]
    counts = Counter(a for s, a, _ in utts if s == 'train')
    keep = {a for a, n in counts.items() if n >= MIN_COUNT}
    norm = lambda a: a if a in keep else 'x_rare'
    rng = np.random.default_rng(0)
    by_act = defaultdict(list)
    for i, (s, a, t) in enumerate(utts):
        if s == 'train':
            by_act[norm(a)].append(i)
    idx = []
    for a, ids in by_act.items():             # cap frequent acts so statements/backchannels don't swamp it
        idx += list(rng.choice(ids, size=min(cap, len(ids)), replace=False))
    texts = [utts[i][2] for i in idx]
    labels = [norm(utts[i][1]) for i in idx]
    print(f'{len(convs)} conversations, {len(utts)} utterances; training on {len(idx)} '
          f'(capped at {cap} per act), {len(set(labels))} acts')
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    X = enc.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
    clf = LogisticRegression(max_iter=2000, C=2.0).fit(X, labels)
    # capped sampling distorts the class prior; restore the true training base rates (prior correction)
    true_prior = np.array([sum(counts[a] for a in counts if norm(a) == c) for c in clf.classes_], float)
    true_prior /= true_prior.sum()
    sample_prior = np.array([labels.count(c) for c in clf.classes_], float) / len(labels)
    clf.intercept_ += np.log(true_prior) - np.log(sample_prior)
    acts = [str(a) for a in clf.classes_]
    ai = {a: i for i, a in enumerate(acts)}
    T = np.zeros((len(acts), len(acts)))
    for no, turns in convs:
        if split_of(no) != 'train':
            continue
        for (c1, a1, _), (c2, a2, _) in zip(turns, turns[1:]):
            if c1 != c2:
                T[ai[norm(a1)], ai[norm(a2)]] += 1
    base = T.sum(axis=0) / T.sum()
    T = (T + ALPHA_T * base) / (T.sum(axis=1, keepdims=True) + ALPHA_T)
    classes = sorted({c for row in ACT_TO_INTENT.values() for c in row} | {OTHER})
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'clf': clf, 'acts': acts, 'T': T, 'classes': classes}, MODEL)
    print('saved', MODEL)


def held_out_pairs(split='test'):
    """(partner text, partner act, reply text, reply act) for consecutive turns by different speakers."""
    out = []
    for no, turns in load_conversations():
        if split_of(no) == split:
            out += [(t1, a1, t2, a2) for (c1, a1, t1), (c2, a2, t2) in zip(turns, turns[1:]) if c1 != c2]
    return out


def evaluate(n=3000):
    """Cross-entropy of predicted P(class of reply | partner text) against the class distribution implied by
    the reply's human-annotated act (M row), on held-out SwDA conversations. Lower is better."""
    am = ActModel()
    pairs = held_out_pairs('test')
    rng = np.random.default_rng(0)
    pairs = [pairs[i] for i in rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)]
    target = np.array([[am.act_to_class(a).get(c, 0.0) for c in am.classes] for _, _, _, a in pairs])
    vecs = am.embed([p for p, _, _, _ in pairs])
    pa = am.act_probs(vecs=vecs)
    acc = np.mean([am.acts[i] == (a if a in am.acts else 'x_rare') for i, (_, a, _, _) in zip(pa.argmax(1), pairs)])
    preds = {
        'uniform': np.full(target.shape, 1 / len(am.classes)),
        'base rates': np.tile(target.mean(0), (len(pairs), 1)),
        'act transitions (soft)': pa @ am.T @ am.M,
    }
    print(f'{len(pairs)} held-out SwDA turn pairs | act classifier top-1 on partner turns: {acc:.2f} '
          f'({len(am.acts)} acts)')
    print('Cross-entropy H(target, predicted) over Tink classes (lower is better):')
    for name, p in preds.items():
        ce = -np.mean(np.sum(target * np.log(np.clip(p, 1e-9, 1)), axis=1))
        print(f'  {name:26s} {ce:.3f}')
    yes_no = np.array([a in ('ny', 'nn', 'na', 'ng') for _, _, _, a in pairs])
    if yes_no.any():
        p = preds['act transitions (soft)']
        cls = am.classes
        yn = p[:, cls.index('Yes / I agree')] + p[:, cls.index('No / I disagree')]
        print(f"  P(Yes)+P(No) predicted before a real yes/no answer: {yn[yes_no].mean():.2f} "
              f"vs before other replies: {yn[~yes_no].mean():.2f}")


def demo():
    am = ActModel()
    for text in ['Do you want some water?', 'Are you tired?', 'What would you like for dinner?',
                 'Can you pass me the remote?', 'I went to the store yesterday.', 'Sorry, what did you say?']:
        p = am.context_class([text])[0]
        top = sorted(zip(am.classes, p), key=lambda x: -x[1])[:4]
        acts = sorted(zip(am.acts, am.act_probs([text])[0]), key=lambda x: -x[1])[:2]
        print(f'{text!r:42s} partner act {acts[0][0]} {acts[0][1]:.2f} -> reply: ' +
              ', '.join(f'{c.split(" /")[0]} {v:.2f}' for c, v in top))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('cmd', choices=['fetch', 'train', 'eval', 'demo'])
    a = p.parse_args()
    {'fetch': fetch, 'train': train, 'eval': evaluate, 'demo': demo}[a.cmd]()
