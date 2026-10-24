"""Semantic retrieval and phrase ranking for Tink.

The Bayesian posterior is over intents (the 'phrases' list in config/likelihoods.json). Every phrase in
config/phrases.json is embedded once with a small CPU sentence model. Each phrase gets an intent profile
P(intent | phrase) = softmax(cos(phrase, intent) / tau), and its score is the chance that it and the
posterior agree on the intent: score = sum_k P(k | evidence) * P(k | phrase). A classical ML ranker
(logistic regression) replaces that formula once confirmed selections have been logged.

Run: python retrieval.py demo
     python retrieval.py train-ranker
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch  # keep before anything that imports tensorflow (c10.dll load error on this machine)
import numpy as np

ROOT = Path(__file__).resolve().parent
PHRASES = ROOT / 'config' / 'phrases.json'
INTENTS = ROOT / 'config' / 'likelihoods.json'
LOG = ROOT / 'data' / 'confirmations.jsonl'
RANKER = ROOT / 'models' / 'ranker.joblib'
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


class PhraseDatabase:
    """Phrase texts (optionally tagged with an intent) and their unit-norm embeddings."""

    def __init__(self, path=PHRASES, model_id=EMBED_MODEL):
        self.path = Path(path)
        self.model_id = model_id
        self.model = None
        self.items = json.loads(self.path.read_text(encoding='utf-8'))['phrases']
        self.texts = [p['text'] for p in self.items]
        self.vectors = None

    def embed(self, texts):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_id, device='cpu')
        return self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)

    def load(self):
        if self.vectors is None:
            self.vectors = self.embed(self.texts)
        return self

    def add(self, text, intent=None, save=True):
        if text in self.texts:
            return
        self.items.append({'text': text, 'intent': intent} if intent else {'text': text})
        self.texts.append(text)
        if self.vectors is not None:
            self.vectors = np.vstack([self.vectors, self.embed([text])])
        if save:
            self.path.write_text(json.dumps({'phrases': self.items}, indent=2), encoding='utf-8')


class PhraseRanker:
    """Posterior over intents -> ranked phrases. Formula by default, logistic regression once trained."""

    def __init__(self, db, intents, tau=0.05, embed_weight=0.0, ranker_path=RANKER):
        self.db = db.load()
        self.intents = list(intents)
        self.tau = tau
        sims = self.db.vectors @ self.db.embed(self.intents).T          # (phrases, intents)
        z = sims / tau
        z -= z.max(axis=1, keepdims=True)
        profile = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        # Short-label similarity is partly lexical ("I don't agree" ~ "I agree"), so a tagged phrase's
        # profile is its tag (embed_weight=0 keeps phrase order = intent order on flat posteriors);
        # untagged phrases rely on embeddings alone. Embeddings still drive MMR diversity.
        for i, item in enumerate(self.db.items):
            if item.get('intent') in self.intents:
                tag = np.eye(len(self.intents))[self.intents.index(item['intent'])]
                profile[i] = (1 - embed_weight) * tag + embed_weight * profile[i]
        self.profile = profile                                            # P(intent | phrase)
        self.model = None
        if Path(ranker_path).exists():
            import joblib
            self.model = joblib.load(ranker_path)

    @staticmethod
    def features(score, post, profile_row):
        p = np.maximum(post, 1e-12)
        return [score, post.max(), float(-(p * np.log(p)).sum()), profile_row.max(),
                float(profile_row @ post / max(profile_row.max(), 1e-12))]

    def rank(self, posterior, k=3, diversity=0.3):
        """posterior {intent: p} -> top-k [{'text', 'intent', 'score', 'match'}].

        Picks are made by maximal marginal relevance so the top-k are not near-copies of one phrase:
        next = argmax (1 - diversity) * score - diversity * max cosine to phrases already picked.
        """
        post = np.array([posterior.get(i, 0.0) for i in self.intents])
        match = self.profile @ post
        score = match
        if self.model is not None:
            X = [self.features(m, post, row) for m, row in zip(match, self.profile)]
            score = self.model.predict_proba(X)[:, 1]
        order = [int(np.argmax(score))]
        while len(order) < min(k, len(score)):
            redundancy = (self.db.vectors @ self.db.vectors[order].T).max(axis=1)
            mmr = (1 - diversity) * score - diversity * redundancy
            mmr[order] = -np.inf
            order.append(int(np.argmax(mmr)))
        return [{'text': self.db.texts[i], 'intent': self.intents[int(self.profile[i].argmax())],
                 'score': round(float(score[i]), 4), 'match': round(float(match[i]), 4)}
                for i in order]

    def log(self, posterior, shown, chosen, path=LOG):
        """Record what was shown and what the user confirmed (None = rejected) for train_ranker."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps({'timestamp': time.time(), 'posterior': posterior,
                                 'shown': [s['text'] for s in shown], 'chosen': chosen}) + '\n')


def train_ranker(ranker, path=LOG, out=RANKER):
    """Logistic regression on logged confirmations: each shown phrase is a row, label = it was chosen."""
    from sklearn.linear_model import LogisticRegression
    X, y = [], []
    index = {t: i for i, t in enumerate(ranker.db.texts)}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        rec = json.loads(line)
        post = np.array([rec['posterior'].get(i, 0.0) for i in ranker.intents])
        match = ranker.profile @ post
        for text in rec['shown']:
            if text in index:
                i = index[text]
                X.append(ranker.features(match[i], post, ranker.profile[i]))
                y.append(int(text == rec['chosen']))
    if len(set(y)) < 2:
        raise ValueError(f'need both chosen and not-chosen rows; have {len(y)} rows')
    import joblib
    model = LogisticRegression(class_weight='balanced', max_iter=1000).fit(X, y)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f'trained on {len(y)} rows ({sum(y)} chosen) -> {out}')


def import_corpus(db, intents, files, source, min_sim=0.6, margin=0.05, dry_run=False, max_words=15):
    """Tag corpus sentences with an intent by nearest hand-tagged seed phrase; keep confident ones.

    Seeds are the phrases already tagged without a 'source' (the hand-written set). A sentence joins
    intent k when its best cosine to k's seeds is >= min_sim and beats every other intent by margin.
    """
    seeds = [(p['text'], p['intent']) for p in db.items if p.get('intent') in intents and 'source' not in p]
    seed_vecs = db.embed([t for t, _ in seeds])
    norm = lambda s: ' '.join(re.sub(r"[^a-z' ]", ' ', s.lower()).split())
    seen = {norm(t) for t in db.texts}
    lines = []
    for f in files:
        for line in Path(f).read_text(encoding='utf-8').splitlines():
            line = ' '.join(line.split())
            key = norm(line)
            # skip empty, long, duplicate (case/punctuation-insensitive), and all-lowercase or all-caps forms
            if key and len(line.split()) <= max_words and key not in seen and line[0].isupper() \
                    and not line.isupper():
                seen.add(key)
                lines.append(line)
    vecs = db.embed(lines)
    sims = vecs @ seed_vecs.T
    best = np.full((len(lines), len(intents)), -1.0)
    for j, (_, intent) in enumerate(seeds):
        k = intents.index(intent)
        best[:, k] = np.maximum(best[:, k], sims[:, j])
    order = np.sort(best, axis=1)
    keep = (order[:, -1] >= min_sim) & (order[:, -1] - order[:, -2] >= margin)
    picked = {}
    for i in np.flatnonzero(keep):
        picked.setdefault(intents[int(best[i].argmax())], []).append((lines[i], float(order[i, -1])))
    print(f'{len(lines)} candidate sentences, {int(keep.sum())} matched (min_sim {min_sim}, margin {margin})')
    for intent in intents:
        rows = sorted(picked.get(intent, []), key=lambda r: -r[1])
        print(f'  {len(rows):4d}  {intent}: ' + ' | '.join(t for t, _ in rows[:6]))
        if dry_run and rows:
            print('        weakest: ' + ' | '.join(f'{t} ({s:.2f})' for t, s in rows[-6:]))
    if not dry_run:
        for intent, rows in picked.items():
            for text, sim in rows:
                db.items.append({'text': text, 'intent': intent, 'source': source,
                                 'match': round(sim, 3)})
                db.texts.append(text)
        db.path.write_text(json.dumps({'note': NOTE, 'phrases': db.items}, indent=1), encoding='utf-8')
        db.vectors = None
    return picked


NOTE = ("Phrase database, editable. 'intent' tags the intent (hand-written seeds have no 'source'). "
        "Imported rows carry 'source' and 'match' (cosine to the nearest seed). Imported AAC messages: "
        "Vertanen & Kristensson (2011), The Imagination of Crowds, EMNLP; CC BY 4.0; aactext.org/imagine.")


def intents_from_config(path=INTENTS):
    return json.loads(Path(path).read_text(encoding='utf-8'))['phrases']


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('demo', help='rank phrases for a few hand-made posteriors')
    sub.add_parser('train-ranker', help='fit the classical ML ranker on data/confirmations.jsonl')
    p = sub.add_parser('import-corpus', help='tag and add corpus sentences to the phrase database')
    p.add_argument('files', nargs='+')
    p.add_argument('--source', required=True)
    p.add_argument('--min-sim', type=float, default=0.6)
    p.add_argument('--margin', type=float, default=0.05)
    p.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    intents = intents_from_config()
    if args.cmd == 'import-corpus':
        import_corpus(PhraseDatabase(), intents, args.files, args.source, args.min_sim, args.margin,
                      args.dry_run)
        return
    ranker = PhraseRanker(PhraseDatabase(), intents)
    if args.cmd == 'train-ranker':
        train_ranker(ranker)
        return
    uniform = {i: 1 / len(intents) for i in intents}
    cases = {'uniform': uniform,
             'mostly yes (0.84)': {**{i: 0.16 / 7 for i in intents}, intents[0]: 0.84},
             'repeat 0.41 / yes 0.27': {**{i: 0.32 / 6 for i in intents}, 'Can you repeat that?': 0.41,
                                        intents[0]: 0.27}}
    for name, post in cases.items():
        print(f'\n== {name}')
        for r in ranker.rank(post, k=5):
            print(f"  {r['score']:.3f}  {r['text']}  [{r['intent']}]")


if __name__ == '__main__':
    main()
