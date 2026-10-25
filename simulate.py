"""Simulate Tink end to end (without cameras) and score it, plus a coverage test against real AAC messages.

simulate: a generative user model picks a true frame (intent, referent) and a partner utterance, then produces
noisy detector outputs for 2 s at 10 Hz (nod/shake, gestures, hold/wave, pointing with angular error, missed
detections, false alarms). The real decoder runs on them: BayesianFusion -> BeliefFilter -> MeaningComposer.
Scored: intent and referent accuracy, and whether the true frame is the top suggestion (hit@1) or in the top 3.
The user model's probabilities are set separately from the decoder's mapping (config/mapping_provisional.json),
so the decoder does not grade itself; both are assumptions, and every number here is simulated, not measured.

coverage: how much of the Vertanen & Kristensson AAC corpus the non-LLM generator can say: the best cosine
(MiniLM) between each real message and anything the templates + phrase database can produce.

Run: python simulate.py simulate [--n 150] [--seed 0]
     python simulate.py coverage
"""
import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from audio_text import BayesianFusion
from meaning import NONE, MeaningComposer
from retrieval import PhraseDatabase, PhraseRanker

ROOT = Path(__file__).resolve().parent
W, H = 1280, 720
ORIGIN = np.array([640.0, 420.0])          # user's pointing wrist
OBJECTS = ['bottle', 'cup', 'tv', 'book', 'cell phone', 'remote', 'laptop', 'bowl', 'apple', 'chair']
NOISE = {   # per level: score noise, miss prob, false-alarm rate, pointing angle error (deg)
    'low': dict(sd=0.05, miss=0.05, fa=0.02, angle=5),
    'medium': dict(sd=0.12, miss=0.15, fa=0.06, angle=12),
    'high': dict(sd=0.2, miss=0.3, fa=0.12, angle=20),
}
# User model: what a user produces when they mean each intent (probability per 2 s episode). Kept apart from
# the decoder's mapping on purpose.
PRODUCE = {
    'Yes / I agree': {'nod': 0.85, 'Thumb_Up': 0.3},
    'No / I disagree': {'shake': 0.85, 'Thumb_Down': 0.25},
    'I want something': {'point': 0.85, 'Open_Palm': 0.1},
    'I want to show you something': {'point': 0.85, 'Pointing_Up': 0.25},
    'I need help': {'hold': 0.7, 'point': 0.4},
    'Please pay attention': {'wave': 0.75, 'hold': 0.2},
    'I need a break': {'Closed_Fist': 0.6},
    'Can you repeat that?': {},
    '(no message)': {},          # idle: the user is not communicating
}


def load_decoder():
    fusion = BayesianFusion.from_json()
    cfg = copy.deepcopy(fusion.config)
    prov = json.loads((ROOT / 'config' / 'mapping_provisional.json').read_text(encoding='utf-8'))
    for src, spec in prov['sources'].items():
        cfg['sources'][src].update({'labels': spec['labels'], 'table': spec['table']})
    fusion = BayesianFusion(cfg)
    ranker = PhraseRanker(PhraseDatabase(), fusion.phrases)
    return fusion, ranker


def place_objects(rng, names):
    objs = []
    for i, name in enumerate(names):
        ang = rng.uniform(-170, -10) if rng.random() < 0.8 else rng.uniform(10, 170)   # mostly above the wrist
        dist = rng.uniform(200, 450)
        c = ORIGIN + dist * np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
        c = np.clip(c, [40, 40], [W - 40, H - 40])
        objs.append({'label': name, 'bbox': [int(c[0] - 35), int(c[1] - 35), int(c[0] + 35), int(c[1] + 35)],
                     'score': 0.8, 'centre': c})
    # keep objects at least 25 degrees apart as seen from the wrist, else pointing is ambiguous by design
    return objs


def scenario(rng):
    """True frame, partner utterance and scene."""
    names = rng.sample(OBJECTS, rng.randint(2, 4))
    intent = rng.choice(list(PRODUCE))
    ref, partner, expect = NONE, '', None
    from meaning import Lexicon
    lex = Lexicon()
    if intent in ('Yes / I agree', 'No / I disagree'):
        if rng.random() < 0.7:
            noun = rng.choice(['water', 'juice', 'coffee', 'a blanket', 'the TV on', 'your medicine'])
            partner = rng.choice([f'Do you want {noun}?', 'Are you tired?', 'Would you like to go outside?',
                                  'Is it too cold in here?', f'Do you need {noun}?'])
    elif intent in ('I want something', 'I want to show you something') or (intent == 'I need help' and rng.random() < 0.5):
        ref = lex.name(names[0])
        if intent == 'I want something':
            r = rng.random()
            if r < 0.25:
                partner = 'What do you want?'
            elif r < 0.5 and len(names) >= 2:
                partner = f'The {lex.name(names[0])} or the {lex.name(names[1])}?'
    return {'intent': intent, 'referent': ref, 'partner': partner, 'objects': names, 'target': names[0]}


def episode(rng, sc, noise, fusion, composer, use_context=True, steps=20, dt=0.1):
    n = NOISE[noise]
    objs = place_objects(rng, sc['objects'])
    target = objs[0]['centre']
    produced = {s for s, p in PRODUCE[sc['intent']].items() if rng.random() < p}
    detected = {s for s in produced if rng.random() > n['miss']}
    bf = fusion.make_filter()
    composer.referents.reset()
    aim = target - ORIGIN
    aim_ang = math.atan2(aim[1], aim[0]) + math.radians(rng.gauss(0, n['angle']))
    partner = sc['partner'] if use_context else ''
    out = None
    for i in range(steps):
        t = i * dt
        on = 0.3 <= t                  # the user starts signalling after 0.3 s
        noisy = lambda base: float(np.clip(base + rng.gauss(0, n['sd']), 0, 1))
        fa = lambda: rng.random() < n['fa']
        nod = noisy(0.8) if on and 'nod' in detected else (noisy(0.4) if fa() else 0.0)
        shake = noisy(0.8) if on and 'shake' in detected else (noisy(0.4) if fa() else 0.0)
        nod, shake = nod * (1 - shake / 2), shake * (1 - nod / 2)
        head = {'status': 'ok', 'q': {'nod': nod, 'shake': shake, 'still': max(0.0, 1 - nod - shake)}}
        hold = noisy(0.75) if on and 'hold' in detected else (noisy(0.3) if fa() else 0.0)
        wave = noisy(0.8) if on and 'wave' in detected else (noisy(0.3) if fa() else 0.0)
        pscore = noisy(0.8) if on and 'point' in detected else (noisy(0.35) if fa() else 0.0)
        pang = aim_ang + math.radians(rng.gauss(0, 3)) if 'point' in detected else rng.uniform(-math.pi, math.pi)
        gestures = []
        for g in ('Thumb_Up', 'Thumb_Down', 'Open_Palm', 'Pointing_Up', 'Closed_Fist'):
            if on and g in detected:
                s = noisy(0.75)
                gestures.append({'label': g, 'score': s, 'alternatives': [{'label': g, 'score': s},
                                                                           {'label': 'None', 'score': 1 - s}]})
        person = {'id': 1, 'signals': {'hold': hold, 'wave': wave},
                  'pointing': {'score': pscore, 'origin': ORIGIN.tolist(), 'dir': [math.cos(pang), math.sin(pang)]}}
        visual = {'head': head, 'people': [person], 'target_id': 1, 'gestures': gestures,
                  'envision': {'status': 'disabled'}}
        audio = {'transcript': partner, 'timestamp': None}   # partner speech is context, not user keywords
        post = fusion.infer(visual, audio)['posterior']
        belief = bf.step(post, t)
        scene = {'objects': [{k: v for k, v in o.items() if k != 'centre'} for o in objs
                             if rng.random() > n['miss'] / 3], 'pointing': person['pointing']}
        out = composer.compose(belief, scene, partner, t)
    return out


def score(sc, out):
    top_intent = next(iter(out['intents']))
    top_ref = next(iter(out['referents']))
    if sc['intent'] == '(no message)':       # idle: right if Tink also says nothing is being communicated
        ok = top_intent == sc['intent']
        return {'intent': ok, 'referent': None, 'hit1': ok, 'hit3': ok}
    if out.get('idle'):                       # Tink thinks nothing is being said: a miss for any real message
        return {'intent': False, 'referent': False if sc['referent'] != NONE else None, 'hit1': False, 'hit3': False}
    truth = (sc['intent'], sc['referent'])
    frames = [(c['intent'], c['referent']) for c in out['candidates']]
    # a yes/no answer is right if it answers with the right polarity; "The X, please." to a choice is Want+X
    hit = [f == truth or (truth[0] in ('Yes / I agree', 'No / I disagree') and f[0] == truth[0]) for f in frames]
    return {'intent': top_intent == sc['intent'],
            'referent': (top_ref == sc['referent']) if sc['referent'] != NONE else None,
            'hit1': bool(hit[:1] and hit[0]), 'hit3': any(hit)}


def cmd_simulate(args):
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    fusion, ranker = load_decoder()
    composer = MeaningComposer(ranker)
    scenarios = [scenario(rng) for _ in range(args.n)]
    rows = []
    for noise in NOISE:
        for ctx in (True, False):
            agg, by_intent = defaultdict(list), defaultdict(lambda: defaultdict(list))
            for sc in scenarios:
                s = score(sc, episode(random.Random(rng.random()), sc, noise, fusion, composer, ctx))
                for k, v in s.items():
                    if v is not None:
                        agg[k].append(v)
                        by_intent[sc['intent']][k].append(v)
            rows.append((noise, ctx, {k: np.mean(v) for k, v in agg.items()}, by_intent))
    print(f'{args.n} scenarios per condition, 2 s episodes at 10 Hz (simulated)\n')
    print('| noise | partner context | intent acc | referent acc | hit@1 | hit@3 |')
    print('|---|---|---|---|---|---|')
    for noise, ctx, m, _ in rows:
        print(f"| {noise} | {'on' if ctx else 'off'} | {m['intent']:.2f} | {m['referent']:.2f} | "
              f"{m['hit1']:.2f} | {m['hit3']:.2f} |")
    noise, ctx, _, by_intent = rows[2]   # medium noise, context on
    print(f'\nPer intent (noise={noise}, context on): hit@1 / hit@3 / n')
    for intent, m in sorted(by_intent.items()):
        print(f"  {intent:30s} {np.mean(m['hit1']):.2f} / {np.mean(m['hit3']):.2f} / {len(m['hit1'])}")


def cmd_dialogue(args):
    """Real held-out conversations: the partner turn is real; the simulated user signals what the real reply
    meant (its label); a hit means a top-3 suggestion says what the person actually said (cosine >= 0.7 to the
    real reply, or the right polarity for a yes/no answer). Replies labelled 'other' cannot be gestured, so the
    user gives no signal there and only the suggestions themselves can help."""
    from conversation import INDEX, ConversationModel
    fusion, ranker = load_decoder()
    intents = [i for i in fusion.phrases if i != '(no message)']
    conv = ConversationModel(intents)
    plain, with_conv = MeaningComposer(ranker), MeaningComposer(ranker, conversation=conv)
    z = np.load(INDEX, allow_pickle=False)
    test = np.flatnonzero(z['split'] == 'test')
    rng = random.Random(args.seed)
    by_label = defaultdict(list)
    for i in test:
        by_label[str(z['label'][i])].append(int(i))
    picked = []                                   # stratified: up to n / labels per reply type
    per = max(1, args.n // len(by_label))
    for label, idx in by_label.items():
        picked += rng.sample(idx, min(per, len(idx)))
    res = defaultdict(lambda: defaultdict(list))
    names = ('templates + phrase DB', '+ conversation model')
    for i in picked:
        partner, reply, label = str(z['ctx'][i]), str(z['reply'][i]), str(z['label'][i])
        sc = {'intent': label if label in PRODUCE else '(no message)', 'referent': NONE, 'partner': partner,
              'objects': rng.sample(OBJECTS, 3)}
        sc['target'] = sc['objects'][0]
        for name, comp in zip(names, (plain, with_conv)):
            out = episode(random.Random(i), sc, args.noise, fusion, comp, True)
            cands = out['candidates']
            vec = conv.model.encode([reply] + [c['text'] for c in cands], normalize_embeddings=True,
                                    convert_to_numpy=True)
            cos = vec[1:] @ vec[0] if cands else np.array([0.0])
            yn = label in ('Yes / I agree', 'No / I disagree')
            hit3 = bool((yn and any(c['intent'] == label for c in cands[:3])) or (cos[:3] >= 0.7).any())
            hit1 = bool((yn and cands and cands[0]['intent'] == label) or (cos[:1] >= 0.7).any())
            for key in (name, name + '|' + label):
                res[key]['hit1'].append(hit1)
                res[key]['hit3'].append(hit3)
                res[key]['best'].append(float(cos.max()))
    print(f'{len(picked)} held-out real (partner turn -> reply) pairs, stratified by reply type, '
          f'noise={args.noise} (signals simulated, conversations real)\n')
    print('| suggestions from | hit@1 | hit@3 | mean best cosine |')
    print('|---|---|---|---|')
    for name in names:
        r = res[name]
        print(f"| {name} | {np.mean(r['hit1']):.2f} | {np.mean(r['hit3']):.2f} | {np.mean(r['best']):.2f} |")
    print('\nhit@3 by real reply type (templates + DB -> + conversation model):')
    for label in sorted(by_label):
        a, b = res[names[0] + '|' + label]['hit3'], res[names[1] + '|' + label]['hit3']
        if a:
            print(f'  {label:30s} n={len(a):3d}  {np.mean(a):.2f} -> {np.mean(b):.2f}')


def cmd_coverage(args):
    from meaning import Lexicon
    lex = Lexicon()
    db = PhraseDatabase()
    refs = sorted(set(lex.display.values()) | lex.nouns)
    gen = set(db.texts)
    for tpls in MeaningComposer.TEMPLATES.values():
        for tpl in tpls:
            gen.update(tpl.format(r=r) for r in refs)
    gen.update(f'The {r}, please.' for r in refs)
    gen = sorted(gen)
    corpus = []
    for name in ('sent_train_aac.txt', 'sent_dev_aac.txt', 'sent_test_aac.txt'):
        corpus += (ROOT / 'data' / 'aac_corpus' / 'vertanen_kristensson_2011' / name).read_text(encoding='utf-8').splitlines()
    corpus = sorted({c.strip() for c in corpus if c.strip()})
    random.Random(0).shuffle(corpus)
    corpus = corpus[:args.n]
    print(f'{len(gen)} sentences the non-LLM generator can produce; {len(corpus)} real AAC messages sampled')
    G = db.embed(gen)
    C = db.embed(corpus)
    best = (C @ G.T).max(axis=1)
    arg = (C @ G.T).argmax(axis=1)
    for th in (0.9, 0.8, 0.7, 0.6):
        print(f'  cosine >= {th}: {np.mean(best >= th) * 100:5.1f}% of real messages')
    order = np.argsort(best)
    print('\nWell covered (real -> generated):')
    for i in order[::-1][:6]:
        print(f'  {best[i]:.2f}  {corpus[i]!r} -> {gen[arg[i]]!r}')
    print('\nNot covered:')
    for i in order[:8]:
        print(f'  {best[i]:.2f}  {corpus[i]!r} -> {gen[arg[i]]!r}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('simulate')
    s.add_argument('--n', type=int, default=150)
    s.add_argument('--seed', type=int, default=0)
    c = sub.add_parser('coverage')
    c.add_argument('--n', type=int, default=1500)
    d = sub.add_parser('dialogue')
    d.add_argument('--n', type=int, default=270)
    d.add_argument('--seed', type=int, default=0)
    d.add_argument('--noise', choices=list(NOISE), default='medium')
    args = p.parse_args()
    {'simulate': cmd_simulate, 'coverage': cmd_coverage, 'dialogue': cmd_dialogue}[args.cmd](args)
