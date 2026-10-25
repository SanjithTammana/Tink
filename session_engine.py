"""Sensor-independent evidence, confirmation and suggestion state for the test lab."""
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import re
import uuid

from conversation_state import ConversationState
from motif_learning import MotifLearner


class SessionEngine:
    def __init__(self, session_id, learner=None, speaker=None, root=None):
        self.session_id, self.speaker = session_id, speaker
        self.learner = learner if learner is not None else MotifLearner()
        self.context = ConversationState()
        self.events = deque(maxlen=128)
        self.evidence_revision = 0
        self.person_id, self.owned = None, False
        self.clock_valid = True
        self.spelled, self.options, self.option_packet = '', [], None
        self.rejected, self.last_discovery = [], {}
        self.rejected_wording = deque(maxlen=32)
        self.status = 'Discovering patterns; meanings require confirmation'
        self.phrases = ['Yes.', 'No.', 'I need help.', 'I want a break.']
        if root:
            path = Path(root) / 'config/phrases.json'
            if path.is_file():
                data = json.loads(path.read_text(encoding='utf-8'))
                self.phrases = list(dict.fromkeys(self.phrases + [p['text'] for p in data.get('phrases', [])
                    if isinstance(p.get('text'), str) and 0 < len(p['text']) <= 160]))

    def set_role(self, person_id, owned):
        if self.person_id is not None and person_id != self.person_id:
            self.events.clear()
            self.options, self.option_packet, self.spelled = [], None, ''
            self.context = ConversationState()
            self.evidence_revision += 1
        self.person_id, self.owned = person_id, bool(owned)

    def observe(self, event):
        event = deepcopy(event)
        event['context_id'] = self.context.revision
        result = self.learner.observe(event)
        self.events.append(event)
        self.evidence_revision += 1
        self.last_discovery = result
        # Only measured overlap on the shared clock: no nearest-neighbor semantic pairing.
        if event['modality'] in {'audio', 'video'}:
            opposite = 'video' if event['modality'] == 'audio' else 'audio'
            for old in reversed(list(self.events)[:-1]):
                if old['modality'] != opposite:
                    continue
                audio, video = (event, old) if event['modality'] == 'audio' else (old, event)
                pair = self.learner.pair(audio, video)
                if pair.get('pair_status') == 'paired':
                    paired = pair['event']
                    if not any(e['event_id'] == paired['event_id'] for e in self.events):
                        self.events.append(paired)
                    break
        return result

    def version(self):
        return (self.context.revision, self.evidence_revision, self.person_id, self.owned,
                self.clock_valid, self.spelled)

    def packet(self, now_s):
        active = [e for e in self.events if e['person_id'] == self.person_id and
                  e['ownership_valid'] and e['clock_valid'] and 0 <= now_s - e['t_end'] <= 3]
        ids = [e['event_id'] for e in active] if self.owned and self.clock_valid else []
        belief = self.learner.posterior(ids, now_s)
        history = self.context.snapshot(now_s)
        tokens = set(re.findall(r'\w+', ' '.join(t['text'] for t in history['finalized_turns'][-3:]).lower()))
        learned = [i['label'] for i in belief.get('items', []) if i['label'] != '(unknown)']
        # Lexical retrieval only; this ordering carries no posterior confidence.
        ranked = sorted(self.phrases, key=lambda text: -len(tokens & set(re.findall(r'\w+', text.lower()))))
        candidates = list(dict.fromkeys(learned + ranked[:12] + ([self.spelled] if self.spelled else [])))[:16]
        rejected_text = {r['text'] for r in self.rejected_wording if set(r['event_ids']) & set(ids)}
        candidates = [text for text in candidates if text not in rejected_text or text == self.spelled]
        return {'session_id': self.session_id, 'context_revision': self.context.revision,
                'event_ids': ids, 'user_role': {'person_id': self.person_id, 'ownership_valid': self.owned},
                'belief': belief, 'quality': {'clock_valid': self.clock_valid,
                    'scores_calibrated': False, 'audio_speaker_default': 'unknown'},
                'conversation': history, 'candidate_pool': candidates, 'spelled': self.spelled,
                'rejected_option_ids': self.rejected[-32:]}

    def present(self, result, packet, version, now_s=None):
        fresh = (now_s is None or all(any(e['event_id'] == identity and 0 <= now_s - e['t_end'] <= 3
                                        for e in self.events) for identity in packet['event_ids']))
        if version != self.version() or not fresh:
            self.status = 'Discarded stale suggestions; R refreshes'
            return False
        self.options = [dict(o, option_id=uuid.uuid4().hex) for o in result['options']]
        self.option_packet = deepcopy(packet)
        self.status = 'Possibilities awaiting your confirmation; R refreshes'
        return True

    def suggest_local(self, now_s):
        packet = self.packet(now_s)
        if self.spelled:
            options = [{'text': self.spelled, 'kind': 'message', 'evidence_ids': []}]
        elif packet['event_ids']:
            options = [{'text': text, 'kind': 'message', 'evidence_ids': packet['event_ids']}
                       for text in packet['candidate_pool'][:4]]
        else:
            options = [{'text': 'Please show or spell what you want to say.', 'kind': 'clarify', 'evidence_ids': []}]
        self.present({'options': options}, packet, self.version())

    def confirm(self, index, now_s):
        if not 0 <= index < len(self.options):
            return None
        option = self.options[index]
        if option['kind'] != 'message':
            self.status = 'Clarification prompt: provide a gesture, spelling or typed message'
            return None
        packet = self.option_packet
        if not self.owned or packet['user_role']['person_id'] != self.person_id:
            self.status = 'Recover/confirm the communicator before selecting a message'
            return None
        return self.confirm_text(option['text'], option.get('evidence_ids', []), now_s)

    def confirm_text(self, text, event_ids, now_s):
        text = str(text).strip()[:500]
        if not text:
            return None
        outcome = self.learner.confirm(event_ids, text, now_s) if event_ids else {'status': 'text_only_no_training'}
        self.context.confirm_user(text, outcome.get('evidence_ids', []), now_s)
        if self.speaker:
            self.context.record_tts(text, now_s)
            self.speaker.say(text)
        self.options, self.option_packet, self.spelled = [], None, ''
        self.status = 'Confirmed: ' + text + ' | ' + outcome['status']
        return outcome

    def reject(self, now_s):
        ids = [o['option_id'] for o in self.options]
        for option in self.options:
            self.rejected_wording.append({'text': option['text'],
                'event_ids': (self.option_packet or {}).get('event_ids', [])})
        self.context.reject_suggestions(ids, now_s)
        self.learner.reject(ids, event_ids=(self.option_packet or {}).get('event_ids', []), at_s=now_s)
        self.rejected.extend(ids)
        self.options, self.option_packet = [], None
        self.status = 'Rejected wording; no intent-negative label trained'

    def undo(self):
        undone = self.learner.undo()
        if undone:
            self.evidence_revision += 1
            self.options, self.option_packet = [], None
        self.status = 'Undid latest learned association (spoken audio cannot be recalled)' if undone else 'No association to undo'
        return undone
