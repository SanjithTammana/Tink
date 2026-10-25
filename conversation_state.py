"""Bounded, revision-aware conversation evidence; suggestions are never user speech."""
from copy import deepcopy
import math
import re


def _number(value, default=None):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def _normalized(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


class ConversationState:
    def __init__(self, max_turns=32, max_age_s=300.0):
        self.max_turns = max(1, int(max_turns))
        self.max_age_s = max(1.0, float(max_age_s))
        self.revision = 0
        self._turns = {}
        self._confirmed = []
        self._rejected = []
        self._tts = []
        self._next_id = 0

    def add_turn(self, turn):
        text = str(turn.get('text', '')).strip()[:2000]
        if not text or turn.get('source') in {'tts', 'suggestion', 'llm'}:
            return False
        start = _number(turn.get('start_s'))
        end = _number(turn.get('end_s'), start)
        if start is not None and end is not None and end < start:
            return False
        for spoken in self._tts:
            if (start is not None and end is not None and
                    start <= spoken['end_s'] + 0.5 and end >= spoken['start_s'] - 0.5 and
                    _normalized(text) == _normalized(spoken['text'])):
                return False
        identity = turn.get('turn_id') or turn.get('id')
        if identity is None:
            identity = 'asr:' + str(start) + ':' + str(end) + ':' + str(turn.get('speaker_id'))
        identity = str(identity)[:200]
        revision = max(0, int(turn.get('revision', 0)))
        old = self._turns.get(identity)
        if old and (revision < old['revision'] or (old['final'] and not turn.get('final', True))):
            return False
        item = {'turn_id': identity, 'text': text, 'start_s': start, 'end_s': end,
                'speaker_id': turn.get('speaker_id') or 'unknown',
                'speaker_role': turn.get('speaker_role') or 'unknown',
                'source': turn.get('source') or 'asr', 'final': bool(turn.get('final', True)),
                'revision': revision, 'overlap': bool(turn.get('overlap', False)),
                'overlap_known': bool(turn.get('overlap_known', False)),
                'speaker_label': turn.get('speaker_label'),
                'speaker_scope': turn.get('speaker_scope', 'unknown'),
                'speaker_uncertain': bool(turn.get('speaker_uncertain', True)),
                'backend': turn.get('backend'),
                'timestamp_quality': turn.get('timestamp_quality', 'unknown'),
                'language': turn.get('language')}
        if old == item:
            return False
        self._turns[identity] = item
        while len(self._turns) > self.max_turns:
            del self._turns[next(iter(self._turns))]
        self.revision += 1
        return True

    def confirm_user(self, text, event_ids, at_s):
        text = str(text).strip()[:2000]
        at_s = _number(at_s)
        if not text or at_s is None:
            raise ValueError('A confirmed message needs text and a finite timestamp.')
        self._next_id += 1
        item = {'turn_id': f'confirmed:{self._next_id}', 'text': text, 'at_s': at_s,
                'event_ids': list(dict.fromkeys(str(v) for v in event_ids))[:64],
                'speaker_role': 'communicator', 'source': 'user_confirmation', 'final': True}
        self._confirmed.append(item)
        self._confirmed = self._confirmed[-self.max_turns:]
        self.revision += 1
        return deepcopy(item)

    def reject_suggestions(self, option_ids, at_s):
        timestamp = _number(at_s)
        if timestamp is None:
            raise ValueError('A rejection needs a finite timestamp.')
        self._rejected.append({'option_ids': list(dict.fromkeys(str(v) for v in option_ids))[:32],
                               'at_s': timestamp})
        self._rejected = self._rejected[-self.max_turns:]
        self.revision += 1

    def record_tts(self, text, start_s, end_s=None):
        start_s = _number(start_s)
        end_s = _number(end_s, start_s)
        if start_s is None or end_s < start_s:
            raise ValueError('TTS timing must be finite and ordered.')
        # Caller can update the end time after playback; absent end gets a conservative echo window.
        if end_s == start_s:
            end_s = start_s + max(1.0, len(str(text).split()) / 2.0)
        self._tts.append({'text': str(text)[:2000], 'start_s': start_s, 'end_s': end_s})
        self._tts = self._tts[-self.max_turns:]

    def snapshot(self, now_s):
        now_s = _number(now_s)
        if now_s is None:
            raise ValueError('Snapshot timing must be finite.')
        def fresh(at):
            return at is not None and 0 <= now_s - at <= self.max_age_s
        turns = [deepcopy(t) for t in self._turns.values() if fresh(t['end_s'])]
        turns.sort(key=lambda t: (t['start_s'] if t['start_s'] is not None else -1, t['turn_id']))
        for turn in turns:
            turn['recency_weight'] = max(0.0, 1.0 - (now_s - turn['end_s']) / self.max_age_s)
        finalized = [t for t in turns if t['final']]
        questions = [t for t in finalized if t['speaker_role'] == 'partner' and
                     not t['overlap'] and t['text'].rstrip().endswith('?')]
        confirmed = [deepcopy(t) for t in self._confirmed if fresh(t['at_s'])]
        pending = questions[-1] if questions else None
        if pending and any(t['at_s'] >= pending['end_s'] for t in confirmed):
            pending = None
        # These are literal utterance mentions, never an extracted fact database.
        mentions = [{'text': t['text'], 'turn_id': t['turn_id'], 'speaker_id': t['speaker_id'],
                     'status': 'mentioned_in_transcript'} for t in finalized]
        return {'revision': self.revision, 'turns': turns, 'finalized_turns': finalized,
                'confirmed_user_turns': confirmed, 'pending_question': pending,
                'mentions': mentions, 'rejections': [deepcopy(r) for r in self._rejected if fresh(r['at_s'])],
                'unknown_speakers': any(t['speaker_id'] == 'unknown' or t['speaker_role'] == 'unknown' for t in turns),
                'overlap_present': any(t['overlap'] for t in turns),
                'source': 'local_bounded_history'}
