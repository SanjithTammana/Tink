"""Meaning layer: belief over intents + scene + conversation -> precise sentences, without an LLM.

A frame is (intent, referent). The intent distribution is the BeliefFilter's belief. The referent is what the
user means: an object they point at (scene objects from the detector) or something the partner just mentioned.
The partner's last utterance also reshapes the intents: a yes/no question makes Yes / No answers likely, and
"water or juice?" makes those two the referent candidates. A template composer turns the top frames into
sentences (answers echo the question: "Do you want water?" -> "Yes, I want water."); retrieval over the phrase
database covers frames without a referent.

Run: python meaning.py demo
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LEXICON = ROOT / 'config' / 'referents.json'
NONE = '(none)'
NO_MESSAGE = '(no message)'
AUX = {'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'would', 'will', 'should', 'shall',
       'may', 'have', 'has', 'am'}
WH = {'what', 'which', 'where', 'who', 'when', 'why', 'how'}


def words(text):
    return re.findall(r"[a-z']+", (text or '').lower())


class Lexicon:
    """Referent words: display names for detector classes and the nouns we listen for in partner speech."""

    def __init__(self, path=LEXICON):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        self.display = data['display']            # detector class -> what to call it in a sentence
        self.synonyms = data['synonyms']          # spoken word -> canonical referent
        self.affords = {k: set(v) for k, v in data.get('affords', {}).items()}   # object -> what it can hold
        self.nouns = set(data['nouns']) | set(self.display.values())
        self.max_len = max(len(w.split()) for w in list(self.synonyms) + list(self.nouns))

    def name(self, label):
        return self.display.get(label, label)

    def mentions(self, text):
        """Referents mentioned in text, in order, longest match first ('ice cream' before 'cream')."""
        ws, found, i = words(text), [], 0
        while i < len(ws):
            for n in range(min(self.max_len, len(ws) - i), 0, -1):
                phrase = ' '.join(ws[i:i + n])
                ref = self.synonyms.get(phrase) or (phrase if phrase in self.nouns else None)
                if ref is None and n == 1 and phrase.endswith('s') and phrase[:-1] in self.nouns:
                    ref = phrase[:-1]
                if ref:
                    if ref not in found:
                        found.append(ref)
                    i += n
                    break
            else:
                i += 1
        return found


class PartnerContext:
    """What the partner's last utterance asks: yes/no question, choice ('A or B'), wh-question, or statement."""

    SWAP = {'you': 'I', 'your': 'my', 'yours': 'mine', 'yourself': 'myself', 'me': 'you', 'my': 'your',
            'mine': 'yours', 'i': 'you', 'myself': 'yourself'}

    def __init__(self, lexicon):
        self.lexicon = lexicon

    def parse(self, text):
        ws = words(text)
        if not ws:
            return {'type': None, 'mentions': [], 'text': ''}
        mentions = self.lexicon.mentions(text)
        if ' or ' in f' {" ".join(ws)} ' and len(mentions) >= 2:
            kind = 'choice'
        elif ws[0] in AUX:
            kind = 'yesno'
        elif ws[0] in WH:
            kind = 'wh'
        else:
            kind = 'statement'
        return {'type': kind, 'mentions': mentions, 'text': text.strip(), 'words': ws}

    def answer(self, ctx, positive):
        """'Do you want some water?' -> 'I want some water' / "I don't want some water"; None if no echo."""
        ws = ctx.get('words') or []
        if ctx.get('type') != 'yesno' or len(ws) < 3 or ws[1] != 'you':
            return None
        aux, rest = ws[0], [self.SWAP.get(w, w) for w in ws[2:]]
        rest = ' '.join(rest)
        if aux in ('do', 'does', 'did'):
            verb = '' if aux == 'do' else aux + ' '
            neg = {"do": "don't", "does": "doesn't", "did": "didn't"}[aux]
            return f'I {verb}{rest}'.replace('  ', ' ') if positive else f'I {neg} {rest}'
        if aux in ('are', 'were'):
            return f"I'm {rest}" if positive else f"I'm not {rest}"
        if aux in ('can', 'could', 'would', 'will', 'should', 'have'):
            return f'I {aux} {rest}' if positive else f"I {aux} not {rest}".replace('can not', "can't")
        return None


class ReferentFilter:
    """Belief over referents that builds while the user points and drifts back to the context prior.

    Same rule as BeliefFilter: s = 0.5 ** (dt / half_life_s); b <- s * b + (1 - s) * prior; then
    log b += rate * dt * log likelihood. Candidates change over time (objects appear, the partner names
    things), so the state is a dict and new candidates start at their prior.
    """

    def __init__(self, half_life_s=2.0, evidence_rate=1.5):
        self.half_life_s, self.rate = half_life_s, evidence_rate
        self.reset()

    def reset(self):
        self.belief, self.t = {}, None

    def step(self, prior, loglik, t):
        dt = 0.0 if self.t is None else min(max(t - self.t, 0.0), 1.0)
        self.t = t
        s = 0.5 ** (dt / self.half_life_s)
        logb = {}
        for r, p in prior.items():
            b = s * self.belief.get(r, p) + (1 - s) * p
            logb[r] = math.log(max(b, 1e-12)) + self.rate * dt * loglik.get(r, 0.0)
        m = max(logb.values())
        z = sum(math.exp(v - m) for v in logb.values())
        self.belief = {r: math.exp(v - m) / z for r, v in logb.items()}
        return dict(sorted(self.belief.items(), key=lambda x: -x[1]))


SOURCE_NAMES = {'head': 'head', 'mp_gesture': 'hand', 'arm': 'arm', 'point': 'pointing', 'envision': 'Envision',
                'vocalization': 'voice', 'transcript': 'speech'}


class ClarifyTracker:
    """Decides when to stop guessing and ask. Two triggers, both smoothed over time so one noisy frame does not
    flip the screen:
    - conflict: two confident sources back different intents (BayesianFusion.conflicts); each intent pair keeps
      an exponentially decaying score s <- s * 0.5 ** (dt / half_life_s) + (1 - 0.5 ** (dt / half_life_s)) * [seen
      this frame]; raised when s >= on_level;
    - split: the two leading intents are both >= split_min and within split_gap of each other.
    """

    def __init__(self, half_life_s=1.0, on_level=0.5, split_min=0.25, split_gap=0.08):
        self.half_life_s, self.on_level = half_life_s, on_level
        self.split_min, self.split_gap = split_min, split_gap
        self.reset()

    def reset(self):
        self.score, self.detail, self.t = {}, {}, None

    def step(self, conflicts, intents, t):
        dt = 0.0 if self.t is None else min(max(t - self.t, 0.0), 1.0)
        self.t = t
        a = 0.5 ** (dt / self.half_life_s)
        seen = {}
        for c in conflicts:
            key = tuple(sorted(c['intents']))
            seen.setdefault(key, c)
        for key in set(self.score) | set(seen):
            self.score[key] = self.score.get(key, 0.0) * a + (1 - a) * (1.0 if key in seen else 0.0)
            if key in seen:
                self.detail[key] = seen[key]
        live = [k for k, s in self.score.items() if s >= self.on_level]
        if live:
            key = max(live, key=lambda k: self.score[k])
            c = self.detail[key]
            (s1, s2), (i1, i2) = c['sources'], c['intents']
            said = f"{SOURCE_NAMES.get(s1, s1)} says {i1.split(' /')[0]}, {SOURCE_NAMES.get(s2, s2)} says {i2.split(' /')[0]}"
            return {'kind': 'conflict', 'options': [i1, i2], 'detail': said,
                    'prompt': f"Mixed signals ({said}). Partner: ask a yes/no question to check."}
        top = [(i, p) for i, p in intents.items() if i not in (NO_MESSAGE, 'other')][:2]
        if len(top) == 2 and top[1][1] >= self.split_min and top[0][1] - top[1][1] <= self.split_gap:
            return {'kind': 'split', 'options': [top[0][0], top[1][0]],
                    'detail': f'{top[0][0]} or {top[1][0]}',
                    'prompt': f"Unsure: {top[0][0].split(' /')[0]} or {top[1][0].split(' /')[0]}? Partner: ask to check."}
        return None


class MeaningComposer:
    """(belief over intents, scene, partner utterance) -> ranked sentences with their frames."""

    NEEDS_REFERENT = {'I want something', 'I want to show you something', 'I need help'}
    TEMPLATES = {
        'I want something': ['Can I have the {r}, please?', 'I want the {r}.', 'Could you get me the {r}?'],
        'I want to show you something': ['Look at the {r}.', 'I want to show you the {r}.'],
        'I need help': ['Can you help me with the {r}?', 'I need help with the {r}.'],
        'No / I disagree': ['No, not the {r}.'],
    }
    POINT_KAPPA = 8.0       # von Mises-style concentration on the angle between pointing ray and object
    NONE_PRIOR = 0.3        # referent mass left for "no particular thing"
    MENTION_SHARE = 0.5     # share of the rest for things the partner just mentioned
    CONTEXT_BOOST = 3.0     # likelihood ratio a question type gives the intents that answer it

    def __init__(self, ranker, lexicon=None, half_life_s=2.0, evidence_rate=1.5, conversation=None,
                 conversation_replies=6):
        self.ranker = ranker
        self.lexicon = lexicon or Lexicon()
        self.context = PartnerContext(self.lexicon)
        self.referents = ReferentFilter(half_life_s, evidence_rate)
        self.clarify = ClarifyTracker()
        self.conversation = conversation          # conversation.ConversationModel, or None for the rule fallback
        self.conversation_replies = conversation_replies
        self._conv_cache = (None, None)

    def conversation_context(self, text):
        if not self.conversation or not text:
            return None
        if self._conv_cache[0] != text:           # the same transcript repeats every frame
            self._conv_cache = (text, self.conversation.query(text))
        return self._conv_cache[1]

    @staticmethod
    def data_context(belief, conv_prior):
        """P(c | cues, context) ∝ L(c) * P(c | context) over intents and 'other'.
        L(c) = belief(c) / prior(c) is the cue evidence (uniform prior), L('other') = 1: gestures cannot
        signal a free-form statement, so cues neither favour nor rule it out. '(no message)' is left out."""
        n = len(belief)
        post = {}
        for c, pc in conv_prior.items():
            lik = belief.get(c, 1.0 / n) * n if c in belief else 1.0
            post[c] = lik * pc
        z = sum(post.values()) or 1.0
        return dict(sorted(((c, p / z) for c, p in post.items()), key=lambda x: -x[1]))

    # ---- context -> intents
    def intent_context(self, belief, ctx):
        """Reweight the belief by what the partner's question invites; returns a normalized dict."""
        boost = {}
        kind = ctx.get('type')
        if kind == 'yesno':
            boost = {'Yes / I agree': self.CONTEXT_BOOST, 'No / I disagree': self.CONTEXT_BOOST}
        elif kind == 'choice':
            boost = {'I want something': self.CONTEXT_BOOST}
        elif kind == 'wh' and {'want', 'like', 'need'} & set(ctx.get('words', [])):
            boost = {'I want something': self.CONTEXT_BOOST}
        w = {i: p * boost.get(i, 1.0) for i, p in belief.items()}
        z = sum(w.values()) or 1.0
        return dict(sorted(((i, p / z) for i, p in w.items()), key=lambda x: -x[1]))

    # ---- referents
    def referent_prior(self, objects, ctx):
        mentioned = list(ctx.get('mentions') or [])
        seen = [self.lexicon.name(o['label']) for o in objects]
        seen = [s for s in dict.fromkeys(seen) if s not in mentioned]
        prior = {NONE: self.NONE_PRIOR}
        rest = 1 - self.NONE_PRIOR
        share = self.MENTION_SHARE if mentioned and seen else (1.0 if mentioned else 0.0)
        for r in mentioned:
            prior[r] = rest * share / len(mentioned)
        for r in seen:
            prior[r] = prior.get(r, 0.0) + rest * (1 - share) / len(seen)
        if len(prior) == 1:
            prior[NONE] = 1.0
        return prior

    def pointing_loglik(self, pointing, objects, candidates):
        """log P(pointing | referent) up to a constant, for every referent candidate.

        k(object) = exp(kappa * (cos angle - 1)) between the pointing ray and the object's centre. A candidate
        gets the best k of the objects it can be: the object itself, or one that affords it (pointing at a
        bottle can mean 'water'); a candidate no visible object stands for gets k = 0. Mixed with 'not really
        pointing' by the pointing score p: log(p * k + 1 - p). No pointing gives no evidence."""
        p = (pointing or {}).get('score', 0.0)
        if p <= 0 or not objects:
            return {}
        origin, d = np.array(pointing['origin']), np.array(pointing['dir'])
        k_obj = {}
        for o in objects:
            x1, y1, x2, y2 = o['bbox']
            v = np.array([(x1 + x2) / 2, (y1 + y2) / 2]) - origin
            n = float(np.hypot(*v))
            cos = 1.0 if n < 1 else float(v @ d) / n
            name = self.lexicon.name(o['label'])
            k_obj[name] = max(k_obj.get(name, 0.0), math.exp(self.POINT_KAPPA * (cos - 1)))
        out = {}
        for r in candidates:
            if r == NONE:
                continue
            k = max([k for name, k in k_obj.items() if name == r or r in self.lexicon.affords.get(name, ())],
                    default=0.0)
            out[r] = math.log(p * k + (1 - p) + 1e-9)
        out[NONE] = math.log(1 - p + 1e-6)
        return out

    # ---- composition
    def compose(self, belief, scene, partner_text, t, k=3, conflicts=None):
        """Returns {'intents', 'referents', 'context', 'candidates': [{'text', 'intent', 'referent', 'score'}]}."""
        ctx = self.context.parse(partner_text)
        conv = self.conversation_context(partner_text)
        intents = self.data_context(belief, conv['prior']) if conv else self.intent_context(belief, ctx)
        clarify = self.clarify.step(conflicts or [], intents, t)
        objects = scene.get('objects') or []
        prior = self.referent_prior(objects, ctx)
        # a recognised sign that names something (WATER, EAT, BATHROOM) adds that referent, weighted by the
        # sign's probability; the prior is renormalized after
        for r, w in (scene.get('referent_hints') or {}).items():
            prior[r] = prior.get(r, 0.0) + (1 - self.NONE_PRIOR) * w
        z = sum(prior.values())
        prior = {r: v / z for r, v in prior.items()}
        pointing = scene.get('pointing') or {}
        refs = self.referents.step(prior, self.pointing_loglik(pointing, objects, prior), t)
        pointed = pointing.get('score', 0.0) >= 0.3
        cands = {}

        def add(text, intent, ref, score):
            if text and score > cands.get(text, {}).get('score', -1):
                cands[text] = {'text': text, 'intent': intent, 'referent': ref, 'score': round(score, 4)}

        p_none = refs.get(NONE, 1.0)
        if conv:   # real replies from similar conversations: P(label | cues, context) x neighbour weight
            by_label = {}
            for text, label, w, _ in conv['replies']:
                by_label.setdefault(label, []).append((text, w))
            for label, rows in by_label.items():
                total = sum(w for _, w in rows)
                for text, w in rows[:self.conversation_replies]:
                    add(text, label, NONE, intents.get(label, 0.0) * w / total)
        for intent, pi in [(i, p) for i, p in intents.items() if i not in (NO_MESSAGE, 'other')][:4]:
            if intent in ('Yes / I agree', 'No / I disagree') and ctx.get('type') == 'yesno':
                echo = self.context.answer(ctx, intent.startswith('Yes'))
                word = 'Yes' if intent.startswith('Yes') else 'No'
                add(f'{word}, {echo}.' if echo else f'{word}.', intent, NONE, pi)
            if intent.startswith('Yes') and ctx.get('type') == 'choice':
                for r, pr in refs.items():
                    if r != NONE:
                        add(f'The {r}, please.', 'I want something', r, pi * pr)
            for r, pr in refs.items():
                if r == NONE:
                    continue
                if ctx.get('type') == 'yesno' and intent in ('Yes / I agree', 'No / I disagree') and not pointed:
                    continue   # an answer to a yes/no question is about the question unless the user points
                for j, tpl in enumerate(self.TEMPLATES.get(intent, [])):
                    add(tpl.format(r=r), intent, r, pi * pr * (1 - 0.05 * j))
            # frames without a particular referent: the phrase database
            for c in self.ranker.rank({intent: 1.0}, k=2):
                if c['intent'] == intent:
                    weight = p_none if intent in self.NEEDS_REFERENT else 1.0
                    add(c['text'], intent, NONE, pi * weight * 0.9)
        ranked = sorted(cands.values(), key=lambda c: -c['score'])
        top, used = [], set()
        if clarify:                            # mixed signals: put the best sentence for each reading first
            for option in clarify['options']:
                best = next((c for c in ranked if c['intent'] == option), None)
                if best:
                    top.append(best)
                    used.add((best['intent'], best['referent']))
        for c in ranked:                       # diversity: one sentence per (intent, referent) frame first
            if (c['intent'], c['referent']) not in used and c not in top:
                top.append(c)
                used.add((c['intent'], c['referent']))
            if len(top) == k:
                break
        for c in ranked:
            if len(top) == k:
                break
            if c not in top:
                top.append(c)
        return {'intents': intents, 'referents': refs, 'context': ctx, 'candidates': top,
                'idle': next(iter(belief)) == NO_MESSAGE, 'conversation': conv, 'clarify': clarify}


def demo():
    from retrieval import PhraseDatabase, PhraseRanker, intents_from_config
    intents = intents_from_config()
    composer = MeaningComposer(PhraseRanker(PhraseDatabase(), intents))
    flat = {i: 1 / len(intents) for i in intents}
    peak = lambda name, p: {i: (p if i == name else (1 - p) / (len(intents) - 1)) for i in intents}
    objects = [{'label': 'bottle', 'bbox': [400, 200, 440, 300], 'score': 0.8},
               {'label': 'tv', 'bbox': [20, 50, 200, 160], 'score': 0.9}]
    point_bottle = {'score': 0.9, 'origin': [300, 250], 'dir': [1.0, 0.0]}
    cases = [
        ('pointing at the bottle, intent Want 0.5', peak('I want something', 0.5), point_bottle, ''),
        ('partner: "Do you want some water?", nod -> Yes 0.6', peak('Yes / I agree', 0.6), {}, 'Do you want some water?'),
        ('partner: "Are you tired?", shake -> No 0.6', peak('No / I disagree', 0.6), {}, 'Are you tired?'),
        ('partner: "Water or juice?", flat belief, pointing at bottle', flat, point_bottle, 'Water or juice?'),
        ('Help 0.5, pointing at the TV', peak('I need help', 0.5), {'score': 0.8, 'origin': [300, 250], 'dir': [-0.85, -0.53]}, ''),
        ('Break 0.6, no scene, no context', peak('I need a break', 0.6), {}, ''),
    ]
    for name, belief, pointing, partner in cases:
        composer.referents.reset()
        out = None
        for i in range(15):        # 1.5 s of the same evidence at 10 Hz
            out = composer.compose(belief, {'objects': objects, 'pointing': pointing}, partner, i / 10)
        print(f'\n== {name}')
        print('  context:', out['context']['type'], out['context']['mentions'],
              '| referents:', {r: round(p, 2) for r, p in list(out['referents'].items())[:3]})
        for c in out['candidates']:
            print(f"  {c['score']:.3f}  {c['text']}   [{c['intent']} / {c['referent']}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('cmd', choices=['demo'])
    parser.parse_args()
    demo()
