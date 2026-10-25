"""Audio-layer bench: each step of the nonverbal-audio branch measured on its own (docs/LAYER-audio.md).

Data: CNVVE (data/vocalizations/cnvve/raw/<class>/*.wav, 48 speakers, 6 classes) and negatives made here
(Windows SAPI partner speech + noise). Embeddings are cached in data/vocalizations/cache/ (git-ignored).

Run: python tools/bench_audio.py negatives     # TTS partner sentences -> data/vocalizations/synth_neg/
     python tools/bench_audio.py classify      # step 3+4: held-out-speaker UAR per embedding / layer
     python tools/bench_audio.py fewshot       # step 4: k-shot personalization on unseen speakers
     python tools/bench_audio.py recanvo       # steps 4+5 on ReCANVo (8 minimally-speaking people)
     python tools/bench_audio.py gate          # step 3: partner speech vs vocalization (Silero VAD)
     python tools/bench_audio.py stream        # whole layer on 4 s mic chunks: baseline vs segmenter
     python tools/bench_audio.py cpu           # ms per component on this machine
"""
import argparse
import glob
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
VOC = ROOT / 'data' / 'vocalizations'
CNVVE = VOC / 'cnvve' / 'raw'
NEG = VOC / 'synth_neg'
CACHE = VOC / 'cache'
SR = 16000
VOC_MODEL = 'alkiskoudounas/voc2vec-hubert-ls-pt'
RNG = np.random.default_rng(0)

PARTNER = [
    'Do you need help?', 'Do you want some water?', 'Are you tired?', 'Should we take a break?',
    'What do you want to show me?', 'Can you say that again?', 'Is this the one you wanted?',
    'I will be right back.', 'Look over here.', 'Do you want to go outside?', 'Are you hungry?',
    'Yes, I see it.', 'No, not that one.', 'Okay, let me get it for you.', 'How are you feeling today?',
    'Tell me when you are ready.', 'Do you want the blue cup or the red cup?', 'That is great.',
    'Hold on a second, I am on the phone.', 'Mm, I think so.', 'We are going to the doctor at three.',
    'Can you hear me?', 'Hello, good morning!', 'Do you want to watch something?', 'Let me fix your chair.',
    'Is it too cold in here?', 'Did you like the food?', 'I do not understand, show me again.',
    'Point to the one you want.', 'We can finish this later.',
]


def ensure_ignored(d):
    d.mkdir(parents=True, exist_ok=True)
    if not (d / '.gitignore').exists():
        (d / '.gitignore').write_text('*\n')


def load(path, sr=SR):
    import librosa
    return librosa.load(str(path), sr=sr, mono=True)[0].astype(np.float32)


def cnvve_items():
    """[(path, label, speaker)]; speaker id is the second field of the CNVVE file name."""
    out = []
    for f in sorted(glob.glob(str(CNVVE / '*' / '*.wav'))):
        p = Path(f)
        out.append((p, p.parent.name, p.name.split('_')[1]))
    if not out:
        raise SystemExit(f'no CNVVE clips under {CNVVE}')
    return out


# ---------- negatives ----------

def cmd_negatives(args):
    import pyttsx3
    ensure_ignored(NEG)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    n = 0
    for vi, v in enumerate(voices):
        for rate in (150, 190):
            engine.setProperty('voice', v.id)
            engine.setProperty('rate', rate)
            for si, s in enumerate(PARTNER):
                engine.save_to_file(s, str(NEG / f'speech_v{vi}_r{rate}_{si:02d}.wav'))
                n += 1
    engine.runAndWait()
    print(f'{n} partner sentences -> {NEG}')


def neg_speech():
    """[(wave, group)]; group = sentence index, so train/test never share a sentence."""
    out = []
    for f in sorted(NEG.glob('speech_*.wav')):
        out.append((load(f), int(f.stem.split('_')[-1])))
    if not out:
        raise SystemExit('run: python tools/bench_audio.py negatives')
    return out


def pink(n, rng):
    x = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.sqrt(np.arange(n // 2 + 1) + 1.0), n)
    return (x / (np.std(x) + 1e-9)).astype(np.float32)


def background(n, rng, kind):
    """Room background at mic level: 'quiet' (-55 dBFS pink), 'noisy' (-35 dBFS pink + 120 Hz hum)."""
    if kind == 'quiet':
        return pink(n, rng) * 10 ** (-55 / 20)
    t = np.arange(n) / SR
    hum = 0.3 * np.sin(2 * np.pi * 120 * t + rng.uniform(0, 6.3))
    return ((pink(n, rng) + hum) * 10 ** (-35 / 20)).astype(np.float32)


# ---------- embeddings ----------

class Voc2Vec:
    name = 'voc2vec'

    def __init__(self, layers=None, quantize=False):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        self.torch = torch
        self.fx = AutoFeatureExtractor.from_pretrained(VOC_MODEL)
        self.model = AutoModel.from_pretrained(VOC_MODEL).eval()
        if layers is not None:   # run only the first `layers` transformer blocks
            self.model.encoder.layers = self.model.encoder.layers[:layers]
        if quantize:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

    def all_layers(self, wave):
        """[13, 768] mean-pooled hidden states (0 = CNN features after projection, 12 = last layer)."""
        wave = np.asarray(wave, np.float32)
        if len(wave) < SR // 2:
            wave = np.pad(wave, (0, SR // 2 - len(wave)))
        inp = self.fx(wave, sampling_rate=SR, return_tensors='pt')
        with self.torch.no_grad():
            hs = self.model(**inp, output_hidden_states=True).hidden_states
        return np.stack([h[0].mean(0).numpy() for h in hs]).astype(np.float32)


class Clap:
    name = 'clap'

    def __init__(self):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.proc = ClapProcessor.from_pretrained('laion/clap-htsat-unfused')
        self.model = ClapModel.from_pretrained('laion/clap-htsat-unfused').eval()

    def embed(self, wave):
        import librosa
        w48 = librosa.resample(np.asarray(wave, np.float32), orig_sr=SR, target_sr=48000)
        inp = self.proc(audio=w48, sampling_rate=48000, return_tensors='pt')
        with self.torch.no_grad():
            return self.model.get_audio_features(**inp)[0].numpy().astype(np.float32)


def mfcc_stats(wave):
    """MFCC + delta mean/std (160-d): the near-free baseline embedding."""
    import librosa
    wave = np.asarray(wave, np.float32)
    if len(wave) < 2048:
        wave = np.pad(wave, (0, 2048 - len(wave)))
    m = librosa.feature.mfcc(y=wave, sr=SR, n_mfcc=40)
    d = librosa.feature.delta(m, width=9 if m.shape[1] >= 9 else 3, mode='nearest')
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]).astype(np.float32)


def cached(name, keys, fn):
    """Embeddings for keys (strings) via fn(i) -> array, cached in CACHE/<name>.npz."""
    ensure_ignored(CACHE)
    path = CACHE / f'{name}.npz'
    have = {}
    if path.exists():
        with np.load(path) as z:
            have = dict(zip(z['keys'].tolist(), z['emb']))
    out, fresh = [], 0
    for i, k in enumerate(keys):
        if k not in have:
            have[k] = fn(i).astype(np.float16)
            fresh += 1
            if fresh % 100 == 0:
                print(f'  {name}: {fresh} new', flush=True)
        out.append(have[k].astype(np.float32))
    if fresh:
        np.savez(path, keys=np.array(list(have)), emb=np.stack(list(have.values())))
    return np.stack(out)


def clip_features(items, which):
    keys = [str(p.relative_to(VOC)) for p, _, _ in items]
    waves = {}
    def wave(i):
        if i not in waves:
            waves.clear()
            waves[i] = load(items[i][0])
        return waves[i]
    if which == 'voc2vec':
        enc = Voc2Vec()
        return cached('voc2vec_layers', keys, lambda i: enc.all_layers(wave(i)))
    if which == 'clap':
        enc = Clap()
        return cached('clap', keys, lambda i: enc.embed(wave(i)))
    if which == 'mfcc':
        return cached('mfcc', keys, lambda i: mfcc_stats(wave(i)))
    raise ValueError(which)


# ---------- classifiers ----------

def make_lr(C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=5000))


def speaker_cv(X, y, groups, model=make_lr, folds=5):
    """Held-out-speaker predictions (GroupKFold): no speaker is in both train and test."""
    from sklearn.model_selection import GroupKFold
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(folds).split(X, y, groups):
        pred[te] = model().fit(X[tr], y[tr]).predict(X[te])
    return pred


def uar(y, pred):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, pred.astype(str)))


def cmd_classify(args):
    items = cnvve_items()
    y = np.array([l for _, l, _ in items])
    g = np.array([s for _, _, s in items])
    print(f'CNVVE: {len(items)} clips, {len(set(g))} speakers, classes {dict(zip(*np.unique(y, return_counts=True)))}')
    print('UAR, 5-fold held-out speakers, logistic regression (chance 0.167)')
    V = clip_features(items, 'voc2vec')
    for layer in range(V.shape[1]):
        print(f'  voc2vec layer {layer:2d} mean-pool : {uar(y, speaker_cv(V[:, layer], y, g)):.3f}')
    print(f'  voc2vec layers 4-8 avg  : {uar(y, speaker_cv(V[:, 4:9].mean(1), y, g)):.3f}')
    M = clip_features(items, 'mfcc')
    print(f'  MFCC+delta stats        : {uar(y, speaker_cv(M, y, g)):.3f}')
    if args.clap:
        C = clip_features(items, 'clap')
        print(f'  CLAP htsat-unfused      : {uar(y, speaker_cv(C, y, g)):.3f}')


def cmd_fewshot(args):
    """Unseen speaker gives k takes per class; test on that speaker's other takes."""
    from sklearn.model_selection import GroupKFold
    items = cnvve_items()
    y = np.array([l for _, l, _ in items])
    g = np.array([s for _, _, s in items])
    V = clip_features(items, 'voc2vec')[:, args.layer]
    classes = sorted(set(y))
    res = defaultdict(list)
    for tr, te in GroupKFold(5).split(V, y, g):
        pop = make_lr().fit(V[tr], y[tr])
        for spk in sorted(set(g[te])):
            idx = te[g[te] == spk]
            for k in (1, 2):
                rng = np.random.default_rng(hash(spk) % 2**32)
                sup, qry = [], []
                for c in classes:
                    ic = rng.permutation(idx[y[idx] == c])
                    if len(ic) <= k:
                        break
                    sup += list(ic[:k]); qry += list(ic[k:])
                else:
                    sup, qry = np.array(sup), np.array(qry)
                    res[f'population only        k={k}'].append(np.mean(pop.predict(V[qry]) == y[qry]))
                    # prototypes from the user's own takes (cosine to class mean)
                    Z = V - V[tr].mean(0)
                    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
                    P = np.stack([Z[sup][y[sup] == c].mean(0) for c in classes])
                    pp = np.array(classes)[np.argmax(Z[qry] @ P.T, 1)]
                    res[f'user prototypes only   k={k}'].append(np.mean(pp == y[qry]))
                    # population + user takes, user weighted (last-layer personalization, Lea & Huang)
                    w = np.r_[np.ones(len(tr)), np.full(len(sup), 20.0)]
                    m = make_lr()
                    m.fit(np.r_[V[tr], V[sup]], np.r_[y[tr], y[sup]], logisticregression__sample_weight=w)
                    res[f'population + user x20  k={k}'].append(np.mean(m.predict(V[qry]) == y[qry]))
    print(f'Few-shot on unseen speakers, voc2vec layer {args.layer} (per-speaker accuracy, mean over speakers)')
    for name, v in sorted(res.items(), key=lambda x: x[0][-3:] + x[0]):
        print(f'  {name}: {np.mean(v):.3f}  (n={len(v)} speakers)')


# ---------- segmentation ----------

def segments(wave, frame=0.02, on_db=9.0, off_db=5.0, min_dur=0.12, merge_gap=0.25, pad=0.08, floor=None):
    """Energy segmentation with hysteresis over an adaptive noise floor: 20 ms frames, start when energy is
    on_db above the floor, end when it drops below off_db; merge gaps < merge_gap, drop events < min_dur."""
    hop = int(frame * SR)
    n = len(wave) // hop
    if n == 0:
        return []
    e = 10 * np.log10(np.mean(wave[:n * hop].reshape(n, hop) ** 2, 1) + 1e-10)
    fl = np.percentile(e, 20) if floor is None else floor
    act, on = np.zeros(n, bool), False
    for i, v in enumerate(e):
        on = v > fl + (off_db if on else on_db)
        act[i] = on
    segs, i = [], 0
    while i < n:
        if act[i]:
            j = i
            while j < n and act[j]:
                j += 1
            segs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for s in segs:
        if merged and (s[0] - merged[-1][1]) * frame < merge_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    p = int(pad / frame)
    return [(max(0, a - p) * hop, min(n, b + p) * hop) for a, b in merged if (b - a) * frame >= min_dur]


# ---------- whole layer on 4 s chunks ----------

def build_chunks(items, speech, rng, n_pos, n_neg, chunk_s=4.0, prefix_s=1.0):
    """Mic-like chunks. Positives: one CNVVE clip at a random offset over a background, half with partner
    speech elsewhere in the chunk. Negatives: background only, or partner speech. Each chunk comes with
    prefix_s of the same background before it, so a streaming segmenter has a noise floor."""
    L, P = int(chunk_s * SR), int(prefix_s * SR)
    out = []
    pos_idx = rng.choice(len(items), n_pos, replace=False)
    for k, i in enumerate(pos_idx):
        kind = 'quiet' if k % 2 == 0 else 'noisy'
        bg = background(P + L, rng, kind)
        clip = load(items[i][0])[:L - SR // 4]
        off = int(rng.integers(0, L - len(clip)))
        bg[P + off:P + off + len(clip)] += clip
        with_speech = k % 4 >= 2
        if with_speech:   # partner speech before or after the vocalization, not overlapping
            s = speech[int(rng.integers(len(speech)))][0] * 0.7
            room = (off, 0) if off > L - off - len(clip) else (L - off - len(clip), off + len(clip))
            s = s[:max(0, room[0] - SR // 10)]
            if len(s):
                start = room[1] + (0 if room[1] == 0 else SR // 20)
                bg[P + start:P + start + len(s)] += s
        out.append({'wave': bg, 'P': P, 'label': items[i][1], 'item': int(i), 'span': (off, off + len(clip)),
                    'cond': kind + ('+speech' if with_speech else '')})
    for k in range(n_neg):
        kind = ('quiet', 'noisy', 'speech', 'speech')[k % 4]
        bg = background(P + L, rng, 'quiet' if kind == 'speech' else kind)
        if kind == 'speech':
            s = speech[int(rng.integers(len(speech)))][0][:L] * 0.7
            o = int(rng.integers(0, L - len(s) + 1))
            bg[P + o:P + o + len(s)] += s
        out.append({'wave': bg, 'P': P, 'label': None, 'cond': kind})
    return out


def cmd_stream(args):
    """Baseline (today's AudioPipeline.process) vs the segmenter pipeline, same chunks, same speakers held out."""
    from sklearn.model_selection import GroupShuffleSplit
    items = cnvve_items()
    y = np.array([l for _, l, _ in items])
    g = np.array([s for _, _, s in items])
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(y, y, g))
    speech = neg_speech()
    sp_tr = [s for s in speech if s[1] % 3 != 0]
    sp_te = [s for s in speech if s[1] % 3 == 0]
    V = clip_features(items, 'voc2vec')
    enc = Voc2Vec()
    rng = np.random.default_rng(1)
    chunks = build_chunks([items[i] for i in te], sp_te, rng, args.pos, args.neg)
    print(f'{len(chunks)} chunks of 4 s ({args.pos} with a vocalization from {len(set(g[te]))} held-out speakers)')

    # baseline: RMS over 4 s, voc2vec last layer mean-pooled over the whole chunk, 6-class LR (no 'none')
    base = make_lr().fit(V[tr, 12], y[tr])
    # candidate: segmenter -> embed segment only (layer L) -> LR with a 'none' class trained on negatives
    L = args.layer
    negX = []
    for w, _ in sp_tr:
        for a, b in segments(np.r_[background(SR, rng, 'quiet'), w + background(len(w), rng, 'quiet')]):
            negX.append(enc.all_layers(np.r_[background(SR, rng, 'quiet'), w][a:b])[L])
    for _ in range(40):   # noise bursts the segmenter may fire on
        w = background(2 * SR, rng, 'noisy') * rng.uniform(1, 4)
        negX.append(enc.all_layers(w[:int(rng.uniform(0.2, 1.5) * SR)])[L])
    negX = np.stack(negX)
    cand = make_lr().fit(np.r_[V[tr, L], negX], np.r_[y[tr], ['none'] * len(negX)])
    print(f'candidate none-class: {len(negX)} speech/noise segments')

    rows = defaultdict(lambda: defaultdict(int))
    t_base, t_cand = [], []
    for c in chunks:
        w = c['wave'][c['P']:]
        cond = c['cond']
        # baseline
        t0 = time.perf_counter()
        rms = float(np.sqrt(np.mean(w ** 2)))
        if rms >= 0.008:
            p = base.predict_proba(enc.all_layers(w)[12:13])[0]
            lab, conf = base.classes_[p.argmax()], p.max()
            emitted = [(lab, conf)]
        else:
            emitted = []
        t_base.append(time.perf_counter() - t0)
        score(rows['baseline'], c, emitted, cond)
        # candidate (the prefix gives the floor a running streaming detector would already have)
        t0 = time.perf_counter()
        full = c['wave']
        hop = int(0.02 * SR)
        pe = 10 * np.log10(np.mean(full[:c['P'] // hop * hop].reshape(-1, hop) ** 2, 1) + 1e-10)
        segs = [(a - c['P'], b - c['P']) for a, b in segments(full, floor=np.percentile(pe, 50))
                if b > c['P']]
        emitted = []
        for a, b in segs:
            p = cand.predict_proba(enc.all_layers(w[max(0, a):b])[L:L + 1])[0]
            k = p.argmax()
            if cand.classes_[k] != 'none' and p[k] >= args.thr:
                emitted.append((cand.classes_[k], p[k]))
        t_cand.append(time.perf_counter() - t0)
        score(rows['segment+none'], c, emitted, cond)
        seg = rows['segmenter alone']   # step 2 on its own: did a segment cover the planted sound?
        key = ('pos ' if c['label'] else 'neg ') + cond
        seg[key + '|n'] += 1
        if c['label']:
            s0, s1 = c['span']
            seg[key + '|hit'] += any(min(b, s1) - max(a, s0) >= 0.3 * min(b - a, s1 - s0) for a, b in segs)
            seg[key + '|emit'] += bool(segs)
        else:
            seg[key + '|fp'] += len(segs)

    for name, r in rows.items():
        print(f'\n== {name}')
        for cond in sorted({k.split('|')[0] for k in r}):
            n = r[cond + '|n']
            if cond.startswith('pos'):
                print(f'  {cond:22s} n={n:3d}  emitted {r[cond + "|emit"] / n:.2f}  correct top-1 {r[cond + "|hit"] / n:.2f}'
                      f'  wrong-label events/chunk {r[cond + "|wrong"] / n:.2f}')
            else:
                fp = r[cond + '|fp'] / n
                print(f'  {cond:22s} n={n:3d}  false events/chunk {fp:.2f}  (= {fp * 900:.0f} per hour of this audio)')
    print(f'\nmean ms per 4 s chunk: baseline {1000 * np.mean(t_base):.0f}, candidate {1000 * np.mean(t_cand):.0f}')


def score(r, c, emitted, cond):
    if c['label'] is None:
        key = 'neg ' + cond
        r[key + '|n'] += 1
        r[key + '|fp'] += len(emitted)
    else:
        key = 'pos ' + cond
        r[key + '|n'] += 1
        r[key + '|emit'] += bool(emitted)
        r[key + '|hit'] += any(l == c['label'] for l, _ in emitted)
        r[key + '|wrong'] += sum(l != c['label'] for l, _ in emitted)


# ---------- ReCANVo: real minimally-speaking communicators ----------

RECANVO = VOC / 'recanvo' / 'raw'


def recanvo_items(cap=80, min_n=15):
    """[(path, label, participant)], at most `cap` clips per (participant, label), labels with >= min_n clips."""
    import csv
    rows = list(csv.DictReader(open(RECANVO / 'dataset_file_directory.csv', encoding='utf-8-sig')))
    by = defaultdict(list)
    for r in rows:
        f = RECANVO / r['Filename']
        if f.exists():
            by[(r['Participant'], r['Label'])].append(f)
    rng = np.random.default_rng(0)
    out = []
    for (p, lab), fs in sorted(by.items()):
        if len(fs) >= min_n:
            out += [(f, lab, p) for f in rng.permutation(fs)[:cap]]
    return out


def cmd_recanvo(args):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    items = recanvo_items(args.cap)
    keys = [str(p.relative_to(VOC)) for p, _, _ in items]
    enc = Voc2Vec()
    V = cached('recanvo_voc2vec_layers', keys, lambda i: enc.all_layers(load(items[i][0])[:4 * SR]))
    M = cached('recanvo_mfcc', keys, lambda i: mfcc_stats(load(items[i][0])[:4 * SR]))
    y = np.array([l for _, l, _ in items])
    g = np.array([p for _, _, p in items])
    print(f'ReCANVo: {len(items)} clips (<= {args.cap} per participant x label, labels with >= 15 clips)')
    feats = {'voc2vec L12 (today)': V[:, 12], 'voc2vec L5': V[:, 5], 'voc2vec L7': V[:, 7], 'MFCC stats': M}
    print('\nA. Personal model: 5-fold CV inside one participant (UAR; chance = 1 / #labels)')
    print('   participant  #labels  ' + '  '.join(f'{k:>19s}' for k in feats))
    agg = defaultdict(list)
    for p in sorted(set(g)):
        i = np.where(g == p)[0]
        labs = sorted(set(y[i]))
        if len(labs) < 2:
            continue
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        r = {k: uar(y[i], cross_val_predict(make_lr(), X[i], y[i], cv=cv)) for k, X in feats.items()}
        for k, v in r.items():
            agg[k].append(v)
        print(f'   {p:11s}  {len(labs):7d}  ' + '  '.join(f'{v:19.3f}' for v in r.values()))
    print('   mean                  ' + '  '.join(f'{np.mean(v):19.3f}' for v in agg.values()))
    shared = ['delighted', 'frustrated', 'selftalk', 'request']
    m = np.isin(y, shared)
    print(f'\nB. Population model, leave-one-participant-out, shared labels {shared} (UAR, chance 0.25)')
    for k, X in feats.items():
        res = []
        for p in sorted(set(g[m])):
            tr, te = m & (g != p), m & (g == p)
            if len(set(y[te])) == len(shared):
                res.append(uar(y[te], make_lr().fit(X[tr], y[tr]).predict(X[te])))
        print(f'   {k:19s}: {np.mean(res):.3f}  (per participant {np.round(res, 2).tolist()})')


# ---------- step 3: speech gate ----------

def silero_speech(wave):
    """Silero VAD v6 (bundled with faster-whisper): per-32 ms speech probability -> (mean, max)."""
    from faster_whisper.vad import get_vad_model
    w = np.asarray(wave, np.float32)
    w = np.pad(w, (0, (-len(w)) % 512))
    p = get_vad_model()(w).reshape(-1)
    return float(p.mean()), float(p.max())


def cmd_gate(args):
    """Can partner speech be told apart from a user's vocalization before any embedding? AUROC per score."""
    from sklearn.metrics import roc_auc_score
    items = cnvve_items()
    rng = np.random.default_rng(2)
    voc = [silero_speech(load(items[i][0])) for i in rng.choice(len(items), 300, replace=False)]
    by_class = defaultdict(list)
    for i in rng.choice(len(items), 300, replace=False):
        by_class[items[i][1]].append(silero_speech(load(items[i][0]))[0])
    sp = [silero_speech(w) for w, _ in neg_speech()]
    y = np.r_[np.zeros(len(voc)), np.ones(len(sp))]
    for j, name in enumerate(('mean', 'max')):
        s = np.r_[[v[j] for v in voc], [v[j] for v in sp]]
        print(f'Silero {name} speech prob: AUROC speech-vs-vocalization {roc_auc_score(y, s):.3f}')
    print('mean speech prob by vocalization class:', {c: round(float(np.mean(v)), 2) for c, v in sorted(by_class.items())},
          '| partner speech:', round(float(np.mean([v[0] for v in sp])), 2))


# ---------- CPU ----------

def cmd_cpu(args):
    import torch
    print(f'torch {torch.__version__}, threads {torch.get_num_threads()}')
    wave1 = (np.random.default_rng(0).standard_normal(SR) * 0.05).astype(np.float32)
    wave4 = np.tile(wave1, 4)
    def bench(fn, w, n=10):
        fn(w)
        t0 = time.perf_counter()
        for _ in range(n):
            fn(w)
        return 1000 * (time.perf_counter() - t0) / n
    print(f'  energy segmenter, 4 s          : {bench(segments, wave4, 50):7.2f} ms')
    print(f'  MFCC stats, 1 s                : {bench(mfcc_stats, wave1):7.2f} ms')
    print(f'  Silero VAD, 1 s                : {bench(silero_speech, wave1, 50):7.2f} ms')
    for layers, q in ((None, False), (None, True), (args.layer, False), (args.layer, True)):
        enc = Voc2Vec(layers=layers, quantize=q)
        tag = f'voc2vec {"all 12" if layers is None else f"first {layers}"} layers{" int8" if q else ""}'
        print(f'  {tag:31s}: 1 s {bench(enc.all_layers, wave1):6.1f} ms | 4 s {bench(enc.all_layers, wave4, 5):6.1f} ms')
    if args.clap:
        enc = Clap()
        print(f'  CLAP htsat, 1 s                : {bench(enc.embed, wave1, 5):7.1f} ms')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('negatives').set_defaults(func=cmd_negatives)
    p = sub.add_parser('classify')
    p.add_argument('--clap', action='store_true')
    p.set_defaults(func=cmd_classify)
    p = sub.add_parser('fewshot')
    p.add_argument('--layer', type=int, default=12)
    p.set_defaults(func=cmd_fewshot)
    p = sub.add_parser('stream')
    p.add_argument('--layer', type=int, default=12)
    p.add_argument('--pos', type=int, default=160)
    p.add_argument('--neg', type=int, default=120)
    p.add_argument('--thr', type=float, default=0.5)
    p.set_defaults(func=cmd_stream)
    sub.add_parser('gate').set_defaults(func=cmd_gate)
    p = sub.add_parser('recanvo')
    p.add_argument('--cap', type=int, default=80)
    p.set_defaults(func=cmd_recanvo)
    p = sub.add_parser('cpu')
    p.add_argument('--layer', type=int, default=6)
    p.add_argument('--clap', action='store_true')
    p.set_defaults(func=cmd_cpu)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
