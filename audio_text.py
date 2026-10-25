"""Audio, text and Bayesian fusion for Tink.

faster-whisper transcription, voc2vec vocalization embeddings with a light
classifier on top, and a Bayesian fusion of all evidence over a small, editable
phrase list. Tables live in config/likelihoods.json.

Run: python audio_text.py train --data data/vocalizations --out models/voc_clf.joblib
     python audio_text.py classify clip.wav
     python audio_text.py fuse-demo
"""
import argparse
import copy
import csv
import json
import re
import threading
import time
from collections import Counter
from pathlib import Path

import torch
import numpy as np

from transcribe import Transcriber

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
CONFIG = ROOT / 'config' / 'likelihoods.json'
VOC_MODEL = 'alkiskoudounas/voc2vec-hubert-ls-pt'
SR = 16000
SILENCE_RMS = 0.008
AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.ogg'}


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


class VocalizationEncoder:
    """Frozen voc2vec (HuBERT) encoder: 16 kHz mono waveform -> 768-d mean-pooled embedding."""

    def __init__(self, model_id=VOC_MODEL):
        self.model_id = model_id
        self.model = None

    def load(self):
        if self.model is None:
            from transformers import AutoFeatureExtractor, AutoModel
            self.extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id).eval().to('cpu')
        return self

    def embed(self, wave):
        self.load()
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        if len(wave) < SR // 2:
            wave = np.pad(wave, (0, SR // 2 - len(wave)))
        inputs = self.extractor(wave, sampling_rate=SR, return_tensors='pt')
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        return hidden.mean(dim=1)[0].numpy().astype(np.float32)


def load_audio(path):
    import librosa
    wave, _ = librosa.load(str(path), sr=SR, mono=True)
    return wave.astype(np.float32)


def collect(sources):
    """Dataset dirs (<dir>/<label>/*.wav) and/or manifest CSVs (path,label) -> [(Path, label)]."""
    found = {}
    def add(f, label):
        key = str(f.resolve())
        if key in found and found[key][1] != label:
            print(f'warning: {f} listed with two labels, keeping {found[key][1]!r}')
        found.setdefault(key, (f, label))
    for src in map(Path, sources):
        if src.is_dir():
            for d in sorted(p for p in src.iterdir() if p.is_dir()):
                for f in sorted(d.rglob('*')):
                    if f.suffix.lower() in AUDIO_EXTS:
                        add(f, d.name)
        elif src.suffix.lower() == '.csv':
            with open(src, newline='', encoding='utf-8-sig') as fh:
                for row in csv.DictReader(fh):
                    f = Path(row['path'].strip())
                    add(f if f.is_absolute() else src.parent / f, row['label'].strip())
        else:
            raise ValueError(f'not a dataset dir or manifest csv: {src}')
    return list(found.values())


class VocalizationClassifier:
    def __init__(self, encoder=None, cache_dir=MODELS / 'cache'):
        self.encoder = encoder or VocalizationEncoder()
        self.cache_path = Path(cache_dir) / 'voc2vec_embeddings.npz'
        self.pipe = None

    @property
    def labels(self):
        return [str(c) for c in self.pipe.classes_] if self.pipe is not None else []

    @staticmethod
    def _make():
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight='balanced', max_iter=5000))

    def _read_cache(self):
        if not self.cache_path.exists():
            return {}
        with np.load(self.cache_path, allow_pickle=False) as z:
            return dict(zip(z['keys'].tolist(), z['emb']))

    def _write_cache(self, cache):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_name('tmp_' + self.cache_path.name)
        np.savez_compressed(tmp, keys=np.array(list(cache)), emb=np.stack(list(cache.values())))
        tmp.replace(self.cache_path)

    def embed_items(self, items):
        """Embeddings for [(path, label)], embedding only files not already cached (path + mtime)."""
        cache = self._read_cache()
        X, y, fresh = [], [], 0
        for i, (f, label) in enumerate(items, 1):
            try:
                key = f'{self.encoder.model_id}|{f.resolve()}|{f.stat().st_mtime_ns}'
                if key not in cache:
                    cache[key] = self.encoder.embed(load_audio(f))
                    fresh += 1
                    if fresh % 10 == 0:
                        print(f'  embedded {fresh} new files ({i}/{len(items)})', flush=True)
                    if fresh % 100 == 0:
                        self._write_cache(cache)
                X.append(cache[key])
                y.append(label)
            except Exception as e:
                print(f'skipping {f}: {e}')
        if fresh:
            self._write_cache(cache)
        print(f'{len(X)} clips ({fresh} newly embedded, {len(X) - fresh} cached)')
        return np.array(X), np.array(y)

    def train(self, sources):
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        X, y = self.embed_items(collect(sources))
        counts = Counter(y.tolist())
        print('classes:', dict(sorted(counts.items())))
        if len(counts) < 2:
            raise ValueError('need at least 2 classes to train')
        uar = None
        if min(counts.values()) < 2:
            print('skipping cross-validation: a class has fewer than 2 samples')
        else:
            k = min(10, min(counts.values()))
            pred = cross_val_predict(self._make(), X, y,
                                     cv=StratifiedKFold(k, shuffle=True, random_state=0))
            uar = float(balanced_accuracy_score(y, pred))
            print(f'{k}-fold stratified CV, UAR (balanced accuracy): {uar:.3f}')
        self.pipe = self._make().fit(X, y)
        return {'uar': uar, 'n': len(y), 'classes': dict(counts)}

    def save(self, path):
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'pipe': self.pipe, 'model_id': self.encoder.model_id}, path)

    def load(self, path):
        import joblib
        data = joblib.load(path)
        self.pipe = data['pipe']
        self.encoder.model_id = data.get('model_id', self.encoder.model_id)
        return self

    def predict_proba(self, emb):
        p = self.pipe.predict_proba(np.asarray(emb).reshape(1, -1))[0]
        return {str(c): float(v) for c, v in zip(self.pipe.classes_, p)}


class AudioPipeline:
    """Background mic loop: Whisper transcript plus voc2vec vocalization class per 4 s chunk."""

    def __init__(self, enabled, whisper_size='base.en',
                 voc_classifier_path='models/voc_clf.joblib', use_voc2vec=True):
        self.enabled = enabled
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.transcriber = Transcriber(whisper_size)
        self.encoder = None
        self.clf = None
        self.voc_error = None
        path = resolve(voc_classifier_path)
        if use_voc2vec and path.exists():
            try:
                self.encoder = VocalizationEncoder()
                self.clf = VocalizationClassifier(self.encoder).load(path)
            except Exception as e:
                self.voc_error = str(e)
        voc = self._voc('disabled' if not enabled else self._idle_status())
        self.latest = {'transcript': '', 'timestamp': None,
                       'status': 'disabled' if not enabled else 'starting', 'rms': 0.0,
                       'vocalization': voc}
        if enabled:
            import sounddevice as sd
            self.sd = sd
            self.transcriber.load()
            if self.clf is not None:
                try:
                    self.encoder.load()
                except Exception as e:
                    self.clf, self.voc_error = None, str(e)
                    self.latest['vocalization'] = self._voc('error: ' + str(e))

    def _idle_status(self):
        if self.voc_error:
            return 'error: ' + self.voc_error
        return 'untrained' if self.clf is None else 'silence'

    @staticmethod
    def _voc(status, label=None, probs=None):
        return {'status': status, 'label': label, 'probs': probs or {}}

    def start(self):
        if not self.enabled:
            return
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def process(self, mono):
        """One chunk (16 kHz mono float32) -> latest dict. Whisper and voc2vec fail independently."""
        mono = np.asarray(mono, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
        if rms < SILENCE_RMS:
            transcript, status = '', 'silence/quiet'
            voc = self._voc(self._idle_status())
        else:
            try:
                transcript = self.transcriber.transcribe(mono)
                status = 'transcribed' if transcript else 'no speech detected'
            except Exception as e:
                transcript, status = '', 'audio error: ' + str(e)
            if self.clf is None:
                voc = self._voc(self._idle_status())
            else:
                try:
                    probs = self.clf.predict_proba(self.encoder.embed(mono))
                    voc = self._voc('ok', max(probs, key=probs.get), probs)
                except Exception as e:
                    voc = self._voc('error: ' + str(e))
        return {'transcript': transcript, 'timestamp': time.time(), 'status': status,
                'rms': round(rms, 5), 'vocalization': voc}

    def loop(self):
        while not self.stop_event.is_set():
            try:
                samples = self.sd.rec(SR * 4, samplerate=SR, channels=1, dtype='float32')
                self.sd.wait()
                if self.stop_event.is_set():
                    break
                result = self.process(samples[:, 0])
                with self.lock:
                    self.latest = result
            except Exception as e:
                with self.lock:
                    self.latest = {'transcript': '', 'timestamp': time.time(),
                                   'status': 'audio error: ' + str(e), 'rms': 0.0,
                                   'vocalization': self._voc('error: ' + str(e))}
                time.sleep(1)

    def get(self):
        with self.lock:
            return copy.deepcopy(self.latest)

    def close(self):
        self.stop_event.set()
        if self.enabled:
            self.sd.stop()
        if self.thread:
            self.thread.join(timeout=1)


class BayesianFusion:
    """Video sets the initial distribution over phrases; audio updates it. No decision is made here.

    P(phrase | video)        ~ prior(phrase) * prod over video evidence terms
    P(phrase | video, audio) ~ P(phrase | video) * prod over audio evidence terms
    A term is sum_o P(o | phrase) q(o) (soft evidence): q is a detector's own probability over its
    labels, or a hard observation (q = 1 on one keyword). A source that is missing, stale, silent or
    untrained adds no term. A label with no table entry, and the mass a detector leaves unassigned
    (an implicit 'other'), get the same likelihood for every phrase (1/K for a K-label table), so
    they never move the distribution. Sources: vocalization, transcript (audio); mp_gesture,
    envision (video).
    """
    FRESH_S = 7.0

    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.phrases = list(self.config['phrases'])
        self.config.setdefault('sources', {})
        self._votes = {}

    @classmethod
    def from_json(cls, path=CONFIG):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def save_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.config, indent=2), encoding='utf-8')

    def prior(self):
        prior = self.config.get('prior', 'uniform')
        if isinstance(prior, dict):
            w = np.array([max(float(prior.get(p, 0.0)), 0.0) for p in self.phrases])
            if w.sum() > 0:
                return w / w.sum()
        return np.full(len(self.phrases), 1.0 / len(self.phrases))

    def _labels(self, source):
        src = self.config['sources'].get(source, {})
        labels = list(src.get('labels', []))
        for row in src.get('table', {}).values():
            labels += [o for o in row if o not in labels]
        return labels

    def term(self, source, q):
        """log sum_o P(o | phrase) q(o) for every phrase, or None if the source has no table."""
        labels = self._labels(source)
        if not labels:
            return None
        flat = 1.0 / len(labels)
        total = sum(q.values())
        if total > 1:
            q = {o: v / total for o, v in q.items()}
        other = max(0.0, 1.0 - sum(q.values()))
        table = self.config['sources'][source].get('table', {})
        out = []
        for phrase in self.phrases:
            row = table.get(phrase, {})
            s = sum(v * row.get(o, flat) for o, v in q.items()) + other * flat
            out.append(np.log(max(s, 1e-12)))
        return np.array(out)

    def keywords(self, audio):
        if audio.get('timestamp') is None or time.time() - audio['timestamp'] >= self.FRESH_S:
            return None
        known = set(self._labels('transcript'))
        found = []
        for word in re.findall(r"[a-z']+", audio.get('transcript', '').lower()):
            if word in known and word not in found:
                found.append(word)
        return found

    def _combine(self, logp, source, qs, detail, evidence):
        terms = [t for t in (self.term(source, q) for q in qs) if t is not None]
        evidence[source] = {'used': bool(terms),
                            'informative': any(np.ptp(t) > 1e-9 for t in terms), **detail}
        # weight < 1 tempers a source whose evidence overlaps another's (same frames, same body)
        weight = float(self.config['sources'].get(source, {}).get('weight', 1.0))
        if terms:
            self._votes[source] = weight * sum(terms)
        return logp + weight * (sum(terms) if terms else 0.0)

    def votes(self):
        """Each source's own distribution over intents (its likelihood alone, uniform prior)."""
        out = {}
        for source, ll in self._votes.items():
            if np.ptp(ll) < 1e-9:
                continue
            p = np.exp(ll - ll.max())
            out[source] = dict(zip(self.phrases, (p / p.sum()).tolist()))
        return out

    def conflicts(self, votes, ratio=2.0):
        """Pairs of sources that each back a different intent and each reject the other's pick.

        Source s backs intent a when P_s(a) >= ratio / N (well above uniform). s rejects b when P_s(b) <=
        P_s(a) / ratio. Severity = Jensen-Shannon divergence between the two votes (0..ln 2)."""
        n = len(self.phrases)
        backed = {s: max(v, key=v.get) for s, v in votes.items()}
        backed = {s: a for s, a in backed.items() if votes[s][a] >= ratio / n and a != '(no message)'}
        out, names = [], sorted(backed)
        for i, s1 in enumerate(names):
            for s2 in names[i + 1:]:
                a, b = backed[s1], backed[s2]
                if a != b and votes[s1][b] <= votes[s1][a] / ratio and votes[s2][a] <= votes[s2][b] / ratio:
                    p, q = np.array(list(votes[s1].values())), np.array(list(votes[s2].values()))
                    m = (p + q) / 2
                    js = 0.5 * float(np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
                    out.append({'sources': (s1, s2), 'intents': (a, b), 'severity': round(js, 3)})
        return sorted(out, key=lambda c: -c['severity'])

    def _dist(self, logp):
        m = logp.max()
        p = np.exp(logp - (m + np.log(np.exp(logp - m).sum())))
        return dict(sorted(zip(self.phrases, p.tolist()), key=lambda x: -x[1]))

    def _start(self, start):
        if start is None:
            return np.log(self.prior())
        return np.log(np.maximum([start.get(p, 0.0) for p in self.phrases], 1e-12))

    def _audio(self, audio, start=None):
        audio, evidence = audio or {}, {}
        logp = self._start(start)
        voc = audio.get('vocalization') or {}
        ok = voc.get('status') == 'ok' and voc.get('probs')
        logp = self._combine(logp, 'vocalization', [voc['probs']] if ok else [],
                             {'probs': voc['probs']} if ok else {'status': voc.get('status', 'missing')},
                             evidence)
        words = self.keywords(audio)
        logp = self._combine(logp, 'transcript', [{w: 1.0} for w in words or []],
                             {'keywords': words} if words is not None
                             else {'status': 'no fresh transcript'}, evidence)
        return self._dist(logp), evidence

    def _video(self, visual, start=None):
        visual, evidence = visual or {}, {}
        logp = self._start(start)
        hands = []
        for g in visual.get('gestures') or []:
            alts = g.get('alternatives') or [{'label': g['label'], 'score': g['score']}]
            hands.append({a['label']: float(a['score']) for a in alts})
        logp = self._combine(logp, 'mp_gesture', hands, {'hands': hands}, evidence)
        env = visual.get('envision') or {}
        ok = env.get('status') == 'ok' and env.get('probs')
        logp = self._combine(logp, 'envision', [env['probs']] if ok else [],
                             {'probs': env['probs']} if ok else {'status': env.get('status', 'missing')},
                             evidence)
        head = visual.get('head') or {}
        ok = head.get('status') == 'ok' and head.get('q')
        logp = self._combine(logp, 'head', [head['q']] if ok else [],
                             {'q': head['q']} if ok else {'status': head.get('status', 'missing')}, evidence)
        user = next((p for p in visual.get('people') or [] if p['id'] == visual.get('target_id')), None)
        sig = (user or {}).get('signals')
        arm = None
        if sig:
            wave, hold = sig['wave'], sig['hold'] * (1 - sig['wave'])
            arm = {'wave': wave, 'hold': hold, 'none': max(0.0, 1 - wave - hold)}
        logp = self._combine(logp, 'arm', [arm] if arm else [],
                             {'q': arm} if arm else {'status': 'no tracked user'}, evidence)
        point = None
        if user is not None:
            s = float((user.get('pointing') or {}).get('score', 0.0))
            point = {'point': s, 'none': 1 - s}
        logp = self._combine(logp, 'point', [point] if point else [],
                             {'q': point} if point else {'status': 'no tracked user'}, evidence)
        sign = visual.get('sign') or {}
        ok = sign.get('status') == 'ok' and sign.get('probs') and 'sign' in self.config['sources']
        logp = self._combine(logp, 'sign', [sign['probs']] if ok else [],
                             {'probs': sign['probs']} if ok else {'status': sign.get('status', 'missing')}, evidence)
        return self._dist(logp), evidence

    def add_sign_source(self, meanings, enrolled, no_sign='(no sign)'):
        """Likelihood table for enrolled keyword signs: P(sign | intent). Each sign weighs `peak` for its mapped
        intent against 1 for every other intent; 'no sign' weighs 3 for every intent and 8 for '(no message)'.
        Rows are normalized. peak comes from meanings['strength'] (0..1 -> 4..20), so a weaker setting makes a
        recognised sign count for less."""
        peak = 4 + 16 * float(meanings.get('strength', 0.7))
        labels = list(enrolled) + [no_sign]
        table = {}
        for intent in self.phrases:
            row = {s: (peak if meanings['signs'].get(s, {}).get('intent') == intent else 1.0) for s in enrolled}
            row[no_sign] = 8.0 if intent == '(no message)' else 3.0
            z = sum(row.values())
            table[intent] = {k: v / z for k, v in row.items()}
        self.config['sources']['sign'] = {'labels': labels, 'table': table, 'weight': 1.0,
                                          'note': 'built from config/signs.json and the enrolled signs'}

    def make_filter(self):
        f = self.config.get('filter', {})
        return BeliefFilter(self.phrases, self.prior(), f.get('half_life_s', 2.0), f.get('evidence_rate', 1.5))

    def visual_distribution(self, visual):
        """Initial distribution {phrase: P(phrase | video)}, sorted, starting from the prior."""
        return self._video(visual)[0]

    def update(self, initial, audio):
        """Bayesian update of an initial {phrase: p} with audio evidence; sorted."""
        return self._audio(audio, initial)[0]

    def infer(self, visual, audio):
        self._votes = {}
        # Video gives the initial distribution, audio updates it (Sanjith's diagram). The posterior is the
        # same in either order; only which distribution is shown as 'initial' depends on it.
        initial, ev_video = self._video(visual)
        post, ev_audio = self._audio(audio, initial)
        votes = self.votes()
        return {'initial_distribution': initial, 'posterior': post, 'evidence': {**ev_video, **ev_audio},
                'votes': votes, 'conflicts': self.conflicts(votes)}


class BeliefFilter:
    """Temporal Bayesian filter over intents: belief builds up while a signal is held and drifts back to the
    prior after it stops, at any frame rate.

    Each step, with dt seconds since the last one:
      predict: b <- s * b + (1 - s) * prior,  s = 0.5 ** (dt / half_life_s)
      update:  log b <- log b + evidence_rate * dt * (log P(intent | evidence now) - log prior), normalized
    Scaling the evidence by dt tempers it by elapsed time: one second of a held signal adds the same evidence
    at 5 or 30 fps, and an unchanged snapshot repeated by a fast display loop is not counted once per frame.
    """

    def __init__(self, phrases, prior, half_life_s=2.0, evidence_rate=1.5):
        self.phrases = list(phrases)
        self.prior = np.asarray(prior, dtype=float)
        self.half_life_s, self.rate = float(half_life_s), float(evidence_rate)
        self.reset()

    def reset(self):
        self.belief = self.prior.copy()
        self.t = None

    def step(self, posterior, t):
        dt = 0.0 if self.t is None else min(max(t - self.t, 0.0), 1.0)   # cap gaps (pauses, stalls) at 1 s
        self.t = t
        s = 0.5 ** (dt / self.half_life_s)
        b = s * self.belief + (1 - s) * self.prior
        loglik = np.log(np.maximum([posterior.get(p, 0.0) for p in self.phrases], 1e-12)) - np.log(self.prior)
        logb = np.log(np.maximum(b, 1e-12)) + self.rate * dt * loglik
        logb -= logb.max()
        self.belief = np.exp(logb) / np.exp(logb).sum()
        return dict(sorted(zip(self.phrases, self.belief.tolist()), key=lambda x: -x[1]))


def cmd_train(args):
    clf = VocalizationClassifier()
    clf.train(args.data)
    clf.save(args.out)
    print('saved', args.out, '| labels:', clf.labels)


def cmd_classify(args):
    clf = VocalizationClassifier().load(args.clf)
    probs = clf.predict_proba(clf.encoder.embed(load_audio(args.wav)))
    for label, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f'{p*100:5.1f}%  {label}')


def cmd_fuse_demo(args):
    fusion = BayesianFusion.from_json(args.config)
    now = time.time()
    def hand(label, score, rest=()):
        return {'label': label, 'score': score,
                'alternatives': [{'label': label, 'score': score}] + [
                    {'label': l, 'score': s} for l, s in rest]}
    say = lambda text, age=0: {'transcript': text, 'timestamp': now - age}
    cases = {
        'no evidence': ({}, {}),
        'video only: Thumb_Up 0.9': ({'gestures': [hand('Thumb_Up', 0.9, [('None', 0.05)])]}, {}),
        'video only: Thumb_Up 0.55 (weak)': ({'gestures': [hand('Thumb_Up', 0.55, [('None', 0.3), ('Open_Palm', 0.1)])]}, {}),
        'audio only: "yes"': ({}, say('Yes.')),
        'audio "yes" + Thumb_Up 0.9': ({'gestures': [hand('Thumb_Up', 0.9)]}, say('yes')),
        'audio "say that again" + Thumb_Up 0.9 (conflict)': ({'gestures': [hand('Thumb_Up', 0.9)]},
                                                             say('Can you say that again?')),
        'audio "I want that" + Pointing_Up 0.8': ({'gestures': [hand('Pointing_Up', 0.8)]}, say('I want that')),
        'two hands: Open_Palm + Pointing_Up': ({'gestures': [hand('Open_Palm', 0.8), hand('Pointing_Up', 0.8)]}, {}),
        'stale transcript (30 s old) + Thumb_Down 0.9': ({'gestures': [hand('Thumb_Down', 0.9)]}, say('help', 30)),
        'envision + vocalization (empty/neutral tables)': (
            {'envision': {'status': 'ok', 'probs': {'Gesture': 0.8, 'NoGesture': 0.2}}},
            {'vocalization': {'status': 'ok', 'probs': {'a': 0.7, 'b': 0.3}}}),
    }
    show = lambda dist: '; '.join(f'{p:.3f} {n}' for n, p in list(dist.items())[:3])
    for name, (visual, audio) in cases.items():
        out = fusion.infer(visual, audio)
        print(f'\n== {name}')
        print('  video     :', show(out['initial_distribution']), f"(sum {sum(out['initial_distribution'].values()):.6f})")
        print('  + audio   :', show(out['posterior']), f"(sum {sum(out['posterior'].values()):.6f})")
        print('  used      :', {s: e['informative'] for s, e in out['evidence'].items() if e['used']})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('train', help='train the vocalization classifier')
    p.add_argument('--data', nargs='+', required=True, help='dataset dirs and/or manifest CSVs')
    p.add_argument('--out', default=str(MODELS / 'voc_clf.joblib'))
    p.set_defaults(func=cmd_train)
    p = sub.add_parser('classify', help='classify one audio file')
    p.add_argument('wav')
    p.add_argument('--clf', default=str(MODELS / 'voc_clf.joblib'))
    p.set_defaults(func=cmd_classify)
    p = sub.add_parser('fuse-demo', help='print posteriors for hand-made inputs')
    p.add_argument('--config', default=str(CONFIG))
    p.set_defaults(func=cmd_fuse_demo)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
