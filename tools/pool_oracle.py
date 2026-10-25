"""Oracle recall of the candidate pool: is a sentence close to the real reply anywhere in the pool?"""
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))   # repo root
import numpy as np
import simulate as S
from conversation import INDEX, ConversationModel
from meaning import MeaningComposer, NONE

fusion, ranker = S.load_decoder()
intents = [i for i in fusion.phrases if i != '(no message)']
conv = ConversationModel(intents)
comp = MeaningComposer(ranker, conversation=conv, conversation_replies=40)
z = np.load(INDEX, allow_pickle=False)
test = np.flatnonzero(z['split'] == 'test')
rng = random.Random(0)
by_label = defaultdict(list)
for i in test:
    by_label[str(z['label'][i])].append(int(i))
picked = []
for label, idx in by_label.items():
    picked += rng.sample(idx, min(30, len(idx)))
res = defaultdict(lambda: defaultdict(list))
for i in picked:
    partner, reply, label = str(z['ctx'][i]), str(z['reply'][i]), str(z['label'][i])
    sc = {'intent': label if label in S.PRODUCE else '(no message)', 'referent': NONE, 'partner': partner,
          'objects': rng.sample(S.OBJECTS, 3)}
    sc['target'] = sc['objects'][0]
    out = S.episode(random.Random(i), sc, 'medium', fusion, comp, True)
    ranked = [c['text'] for c in comp.compose(out['intents'] if False else {k: v for k, v in out['intents'].items() if k != 'other'} or {}, {}, partner, 99, k=60)['candidates']] if False else None
    pool_ret = [r for r, _, _, _ in (out.get('conversation') or {}).get('replies', [])]
    pool = list(dict.fromkeys([c['text'] for c in out['candidates']] + pool_ret))
    vec = conv.model.encode([reply] + pool, normalize_embeddings=True, convert_to_numpy=True)
    cos = vec[1:] @ vec[0]
    order_cos = cos  # pool order: composer top-3 first, then retrieved replies by kernel weight
    for K in (3, 10, 20, 50):
        hit = bool((cos[:K] >= 0.7).any())
        res['all'][K].append(hit)
        res[label][K].append(hit)
    res['all']['size'].append(len(pool))
print(f"{len(picked)} pairs; mean pool size {np.mean(res['all']['size']):.0f}")
print('hit@K over the pool in its current order (cosine >= 0.7):')
for K in (3, 10, 20, 50):
    print(f'  K={K:<3} {np.mean(res["all"][K]):.2f}')
print('oracle (anywhere in pool) by type:')
for label in sorted(by_label):
    if res[label][50]:
        print(f'  {label:30s} n={len(res[label][50]):3d}  @3 {np.mean(res[label][3]):.2f}  @20 {np.mean(res[label][20]):.2f}  @50 {np.mean(res[label][50]):.2f}')
