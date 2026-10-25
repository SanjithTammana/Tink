"""Discover personal signal motifs, then ground them only through confirmations.

Cosine scores describe a feature match, not an intent probability. The optional
count-based posterior is an explicitly uncalibrated model over confirmed text.
No microphone, camera, model loading, cloud service or speech is owned here.
"""
from collections import Counter, OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading
import uuid

import numpy as np


SCHEMA = 1
ALGORITHM = 'spherical-prototypes-confirmed-counts-v1'
BAD_FLAGS = ('tts', 'overlap', 'tracking_lost', 'source_lost')
UNKNOWN = '(unknown)'


def _name(value, field, maximum=160):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{field} must be a nonempty string of at most {maximum} characters')
    return value


def _time(value, field):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{field} must be finite and nonnegative') from error
    if not math.isfinite(value) or not 0 <= value <= 1e12:
        raise ValueError(f'{field} must be finite and nonnegative')
    return value


def _ids(values, field, maximum=512):
    if not isinstance(values, (list, tuple)) or len(values) > maximum:
        raise ValueError(f'{field} must be a bounded list of IDs')
    return list(dict.fromkeys(_name(value, field) for value in values))


def _unit(values):
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not 2 <= len(vector) <= 4096 or not np.isfinite(vector).all():
        raise ValueError('features must be a finite one-dimensional vector of 2-4096 values')
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-10:
        raise ValueError('zero or numerically unbounded features cannot form a motif')
    return vector / norm


def _record(event):
    if not isinstance(event, dict):
        raise ValueError('event must be an object')
    identity = _name(event.get('event_id'), 'event_id')
    session = _name(event.get('session_id'), 'session_id', 128)
    modality = event.get('modality')
    if modality not in ('audio', 'video', 'paired'):
        raise ValueError('modality must be audio, video or paired')
    start, end = _time(event.get('t_start'), 't_start'), _time(event.get('t_end'), 't_end')
    if end < start:
        raise ValueError('event end precedes start')
    person = event.get('person_id')
    if person is not None:
        person = _name(person, 'person_id', 128)
    quality = event.get('quality') or {}
    if not isinstance(quality, dict):
        raise ValueError('quality must be an object')
    quality = {key: quality.get(key) is True for key in (*BAD_FLAGS, 'confirmed_by_user')}
    features = _unit(event.get('features')).tolist()
    feature_schema = _name(event.get('feature_schema', 'unspecified-v1'), 'feature_schema', 256)
    sources = _ids(event.get('source_event_ids') or [], 'source_event_id', 2)
    if len(sources) > 2:
        raise ValueError('a paired observation may contain at most two source events')
    for source in sources:
        _name(source, 'source_event_id')
        if source == identity:
            raise ValueError('an event cannot be its own source')
    if modality == 'paired' and len(sources) != 2:
        raise ValueError('paired events require two provenance IDs')
    group = event.get('group_id')
    if group is not None:
        group = _name(group, 'group_id')
    negative = quality['confirmed_by_user'] and (
        event.get('negative') is True or event.get('is_communicative') is False
        or event.get('confirmed_negative') is True)
    value = {'event_id': identity, 'session_id': session, 'modality': modality,
             't_start': start, 't_end': end, 'person_id': person,
             'ownership_valid': event.get('ownership_valid') is True and person is not None,
             'clock_valid': event.get('clock_valid') is True,
             'complete': event.get('complete', True) is True,
             'features': features, 'feature_schema': feature_schema,
             'source_event_ids': sources, 'group_id': group,
             'quality': quality, 'confirmed_negative': bool(negative)}
    if event.get('context_id') is not None:
        context = event['context_id']
        if not isinstance(context, int) or isinstance(context, bool) or context < 0:
            raise ValueError('context_id must be a nonnegative integer')
        value['context_id'] = context
    canonical = {**value, 'features': [round(x, 8) for x in features]}
    value['fingerprint'] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':'),
                                                    allow_nan=False).encode('utf-8')).hexdigest()
    return value


def _blocked(event):
    if not event['clock_valid']:
        return 'invalid_clock'
    if not event['complete']:
        return 'incomplete'
    if any(event['quality'][flag] for flag in BAD_FLAGS):
        return 'ambiguous_source'
    if event['confirmed_negative']:
        return 'confirmed_negative'
    return None


def _space(event):
    # Ownership is kept separate: unknown-room audio is never silently promoted
    # into the communicator's personal codebook.
    owner = event['person_id'] if event['ownership_valid'] else None
    return json.dumps([event['modality'], owner, event['feature_schema'], len(event['features'])],
                      separators=(',', ':'))


class MotifLearner:
    """Bounded online discovery with explicit, reversible human grounding.

    ``path`` is optional. Save locations inside Git must be under an ignored
    .scratch* or models directory; repository-root ignore rules are never changed.
    Persist after confirmation/Undo/rejection or call save() explicitly on close.
    """
    def __init__(self, path=None, min_observations=3, similarity_threshold=.85,
                 max_events=512, max_prototypes=64, max_confirmations=512,
                 max_sessions=32, active_ttl_s=3.):
        if not isinstance(min_observations, int) or min_observations < 2:
            raise ValueError('min_observations must be at least two')
        if not math.isfinite(similarity_threshold) or not 0 < similarity_threshold <= 1:
            raise ValueError('similarity threshold must be in (0,1]')
        if any(not isinstance(n, int) or n < 2 for n in
               (max_events, max_prototypes, max_confirmations, max_sessions)):
            raise ValueError('capacity limits must be integers of at least two')
        if not math.isfinite(active_ttl_s) or active_ttl_s <= 0:
            raise ValueError('active TTL must be positive')
        self.path = Path(path) if path is not None else None
        self.min_observations, self.similarity_threshold = min_observations, float(similarity_threshold)
        self.max_events, self.max_prototypes = max_events, max_prototypes
        self.max_confirmations, self.max_sessions = max_confirmations, max_sessions
        self.active_ttl_s = float(active_ttl_s)
        self.events, self.prototypes = OrderedDict(), OrderedDict()
        self.feature_spaces, self.retired_before = {}, {}
        self.confirmations, self.rejections = [], []
        self.counter = 0
        self._lock = threading.RLock()
        if self.path is not None and self.path.exists():
            self._load()

    def _settings(self):
        return {'min_observations': self.min_observations,
                'similarity_threshold': self.similarity_threshold,
                'max_events': self.max_events, 'max_prototypes': self.max_prototypes,
                'max_confirmations': self.max_confirmations, 'max_sessions': self.max_sessions,
                'active_ttl_s': self.active_ttl_s}

    def _remember(self, event, result):
        event['discovery'] = result
        self.events[event['event_id']] = event
        while len(self.events) > self.max_events:
            _, expired = self.events.popitem(last=False)
            session = expired['session_id']
            self.retired_before[session] = max(self.retired_before[session], expired['t_end'])
        return deepcopy(result)

    def observe(self, event):
        with self._lock:
            event = _record(event)
            identity, session = event['event_id'], event['session_id']
            if identity in self.events:
                if self.events[identity]['fingerprint'] != event['fingerprint']:
                    raise ValueError('event_id was reused for different observations')
                return {**deepcopy(self.events[identity]['discovery']), 'duplicate': True}
            result = {'event_id': identity, 'motif_id': None, 'status': 'unknown',
                      'memberships': [], 'score_kind': 'cosine_similarity', 'calibrated': False}
            if session not in self.retired_before:
                if len(self.retired_before) >= self.max_sessions:
                    return {**result, 'status': 'session_capacity_reached'}
                self.retired_before[session] = -1.
            if event['t_end'] <= self.retired_before[session]:
                return {**result, 'status': 'expired_event'}
            reason = _blocked(event)
            if reason:
                return self._remember(event, {**result, 'status': reason})
            if event['modality'] == 'paired':
                sources = [self.events.get(identity) for identity in event['source_event_ids']]
                if (any(source is None for source in sources)
                        or {source['modality'] for source in sources} != {'audio', 'video'}
                        or any(_blocked(source) or not source['ownership_valid']
                               or source['person_id'] != event['person_id']
                               or source['session_id'] != event['session_id'] for source in sources)):
                    return self._remember(event, {**result, 'status': 'invalid_pair_provenance'})
                sources.sort(key=lambda source: source['modality'] != 'audio')
                expected = _unit(sources[0]['features'] + sources[1]['features'])
                if (len(expected) != len(event['features']) or not np.allclose(expected, event['features'], atol=1e-8)
                        or event['t_start'] != min(source['t_start'] for source in sources)
                        or event['t_end'] != max(source['t_end'] for source in sources)
                        or max(source['t_start'] for source in sources) - min(source['t_end'] for source in sources) > .25):
                    return self._remember(event, {**result, 'status': 'invalid_pair_provenance'})
            signature = [len(event['features']), event['feature_schema']]
            modality = event['modality']
            if modality in self.feature_spaces and self.feature_spaces[modality] != signature:
                raise ValueError('feature dimension/schema changed; use a separate learner profile')
            self.feature_spaces[modality] = signature
            space, vector = _space(event), np.asarray(event['features'])
            ranked = []
            for proto in self.prototypes.values():
                if proto['space'] == space:
                    center = _unit(proto['sum'])
                    ranked.append((proto['id'], float(np.clip(center @ vector, -1., 1.))))
            ranked.sort(key=lambda item: item[1], reverse=True)
            result['memberships'] = [{'id': mid, 'score': score} for mid, score in ranked[:5]]
            if ranked and ranked[0][1] >= self.similarity_threshold:
                proto = self.prototypes[ranked[0][0]]
                proto['sum'] = (np.asarray(proto['sum']) + vector).tolist()
                proto['count'] += 1
            elif len(self.prototypes) < self.max_prototypes:
                self.counter += 1
                mid = f'm{self.counter:06d}'
                proto = {'id': mid, 'space': space, 'modality': modality,
                         'person_id': event['person_id'] if event['ownership_valid'] else None,
                         'sum': vector.tolist(), 'count': 1}
                self.prototypes[mid] = proto
                result['memberships'] = [{'id': mid, 'score': 1.}]
            else:
                return self._remember(event, {**result, 'status': 'prototype_capacity_unknown'})
            result.update(motif_id=proto['id'], observations=proto['count'],
                          status='matched' if proto['count'] >= self.min_observations else 'provisional',
                          ownership_valid=event['ownership_valid'])
            return self._remember(event, result)

    def pair(self, audio_event, visual_event, max_lag_s=None):
        """Observe one joint candidate, not two independent votes.

        Require >=25% overlap of the shorter interval by default. A caller may
        explicitly allow a nonoverlapping gap of at most 0.25 seconds. This is a
        temporal/ownership candidate, not proof of communication or intent.
        """
        with self._lock:
            a, v = _record(audio_event), _record(visual_event)
            reject = lambda reason: {'status': 'rejected', 'pair_status': 'rejected', 'reason': reason,
                                     'event_id': None, 'motif_id': None, 'calibrated': False}
            if a['modality'] != 'audio' or v['modality'] != 'video':
                return reject('need_audio_then_video')
            if a['session_id'] != v['session_id']:
                return reject('different_sessions')
            if a['person_id'] != v['person_id'] or not a['ownership_valid'] or not v['ownership_valid']:
                return reject('ownership_unknown_or_different')
            if _blocked(a) or _blocked(v):
                return reject('invalid_clock_or_source')
            overlap = max(0., min(a['t_end'], v['t_end']) - max(a['t_start'], v['t_start']))
            shorter = min(a['t_end'] - a['t_start'], v['t_end'] - v['t_start'])
            enough = shorter > 0 and overlap / shorter >= .25
            gap = max(0., max(a['t_start'], v['t_start']) - min(a['t_end'], v['t_end']))
            if max_lag_s is not None:
                max_lag_s = _time(max_lag_s, 'max_lag_s')
                if max_lag_s > .25:
                    raise ValueError('explicit pairing lag may not exceed 0.25 seconds')
            if not enough and not (overlap == 0 and max_lag_s is not None and gap <= max_lag_s):
                return reject('not_temporally_aligned')
            source_ids = [a['event_id'], v['event_id']]
            paired_id = 'pair-' + hashlib.sha256(json.dumps([a['session_id'], *source_ids]).encode()).hexdigest()[:24]
            if paired_id in self.events:
                return {**deepcopy(self.events[paired_id]['discovery']), 'pair_status': 'paired',
                        'duplicate': True, 'event': self._public_event(self.events[paired_id])}
            for existing in self.events.values():
                if existing['modality'] == 'paired' and set(source_ids) & set(existing['source_event_ids']):
                    return reject('source_already_paired')
            for source in (audio_event, visual_event):
                observed = self.observe(source)
                if observed['status'] in ('expired_event', 'session_capacity_reached'):
                    return reject('source_outside_history')
            signature = 'joint:' + hashlib.sha256(json.dumps([
                a['feature_schema'], len(a['features']), v['feature_schema'], len(v['features'])]).encode()).hexdigest()[:20]
            joint = {'event_id': paired_id, 'session_id': a['session_id'], 'modality': 'paired',
                     't_start': min(a['t_start'], v['t_start']), 't_end': max(a['t_end'], v['t_end']),
                     'person_id': a['person_id'], 'ownership_valid': True, 'clock_valid': True,
                     'complete': True, 'features': np.r_[a['features'], v['features']].tolist(),
                     'feature_schema': signature, 'source_event_ids': source_ids,
                     'group_id': paired_id, 'quality': {}}
            result = self.observe(joint)
            return {**result, 'pair_status': 'paired', 'event': deepcopy(joint)}

    @staticmethod
    def _public_event(record):
        return {k: deepcopy(v) for k, v in record.items() if k not in ('fingerprint', 'discovery')}

    def _eligible(self, ids, at_s, ttl=None):
        records = []
        for identity in _ids(ids, 'event_id', self.max_events):
            if identity not in self.events:
                continue
            event = self.events[identity]
            if (_blocked(event) or not event['ownership_valid'] or event['t_end'] > at_s + .001
                    or (ttl is not None and at_s - event['t_end'] > ttl)):
                continue
            if event['discovery']['motif_id'] is not None:
                records.append(event)
        return records

    def _effective(self, records):
        """Keep joint evidence, and conservatively avoid correlated frame votes."""
        ordered = sorted(records, key=lambda r: (r['modality'] != 'paired',
                         -self.prototypes[r['discovery']['motif_id']]['count']))
        selected, covered = [], set()
        for event in ordered:
            if event['event_id'] in covered or set(event['source_event_ids']) & covered:
                continue
            duplicate_group = any(
                (event.get('group_id') is not None and event['group_id'] == old.get('group_id'))
                or (event['session_id'] == old['session_id'] and event['person_id'] == old['person_id']
                    and min(event['t_end'], old['t_end']) > max(event['t_start'], old['t_start']))
                for old in selected)
            if duplicate_group:
                continue
            selected.append(event)
            covered.update([event['event_id'], *event['source_event_ids']])
        return selected

    def confirm(self, event_ids, text, at_s):
        with self._lock:
            text = ' '.join(_name(text, 'confirmed text', 500).split())
            if text == UNKNOWN:
                raise ValueError('(unknown) is reserved for abstention')
            at_s = _time(at_s, 'at_s')
            records = self._effective(self._eligible(event_ids, at_s, ttl=120.))
            if not records:
                return {'status': 'no_eligible_evidence', 'evidence_ids': [], 'label': text}
            if len({(e['session_id'], e['person_id']) for e in records}) != 1:
                return {'status': 'mixed_sources_rejected', 'evidence_ids': [], 'label': text}
            all_sources = set()
            for e in records:
                all_sources.update([e['event_id'], *e['source_event_ids']])
            if any(all_sources & set(c['source_event_ids']) for c in self.confirmations):
                return {'status': 'already_confirmed', 'evidence_ids': sorted(all_sources), 'label': text}
            start, end = min(e['t_start'] for e in records), max(e['t_end'] for e in records)
            if any(c['session_id'] == records[0]['session_id'] and c['person_id'] == records[0]['person_id']
                   and min(end, c['t_end']) > max(start, c['t_start']) for c in self.confirmations):
                return {'status': 'correlated_confirmation_rejected', 'evidence_ids': sorted(all_sources), 'label': text}
            confirmation = {'id': uuid.uuid4().hex, 'text': text, 'at_s': at_s,
                            'session_id': records[0]['session_id'], 'person_id': records[0]['person_id'],
                            't_start': start, 't_end': end,
                            'event_ids': [e['event_id'] for e in records], 'source_event_ids': sorted(all_sources),
                            'motif_ids': list(dict.fromkeys(e['discovery']['motif_id'] for e in records))}
            self.confirmations.append(confirmation)
            self.confirmations = self.confirmations[-self.max_confirmations:]
            self._autosave()
            return {'status': 'confirmed', 'confirmation_id': confirmation['id'], 'label': text,
                    'evidence_ids': confirmation['event_ids'], 'motif_ids': confirmation['motif_ids']}

    def reject(self, option_ids, event_ids=None, at_s=None):
        with self._lock:
            at_s = _time(at_s, 'at_s')
            options = _ids(option_ids, 'option_id', 32)
            entry = {'option_ids': options, 'event_ids': _ids(event_ids or [], 'event_id', 32), 'at_s': at_s}
            self.rejections.append(entry)
            self.rejections = self.rejections[-128:]
            self._autosave()
            return {'status': 'rejection_recorded', **deepcopy(entry), 'trained_negative': False}

    def undo(self):
        with self._lock:
            if not self.confirmations:
                return False
            self.confirmations.pop()
            self._autosave()
            return True

    def posterior(self, event_ids, now_s):
        with self._lock:
            now_s = _time(now_s, 'now_s')
            records = self._effective(self._eligible(event_ids, now_s, ttl=self.active_ttl_s))
            evidence_ids = [e['event_id'] for e in records]
            ungrounded = {'kind': 'ungrounded_motifs', 'calibrated': False,
                          'items': [{'label': UNKNOWN, 'probability': 1., 'support': 0, 'evidence_ids': evidence_ids}],
                          'method': 'anonymous motifs are not communication meanings',
                          'motifs': [deepcopy(e['discovery']) for e in records]}
            if not records:
                return {**ungrounded, 'reason': 'missing_stale_or_unowned_evidence'}
            if len({(e['session_id'], e['person_id']) for e in records}) != 1:
                return {**ungrounded, 'reason': 'mixed_sources'}
            person = records[0]['person_id']
            records = [e for e in records if self.prototypes[e['discovery']['motif_id']]['count'] >= self.min_observations]
            if not records:
                return {**ungrounded, 'reason': 'provisional_motifs'}
            counts, support = {}, Counter()
            for confirmation in self.confirmations:
                if confirmation['person_id'] != person:
                    continue
                label = confirmation['text']
                support[label] += 1
                counts.setdefault(label, Counter()).update(confirmation['motif_ids'])
            current = [e['discovery']['motif_id'] for e in records]
            labels = [label for label in sorted(counts) if support[label] >= 2
                      and any(counts[label][mid] >= 2 for mid in current)]
            if not labels:
                return {**ungrounded, 'reason': 'need_two_independent_confirmations'}
            labels = sorted(label for label in counts if support[label] >= 2)
            # One shared prior, then one likelihood per effective independent group.
            # The explicit unknown model uses a uniform motif likelihood; its prior
            # is the average supported message count, avoiding a vanishing open set.
            weights = {label: float(support[label] + 1) for label in labels}
            weights[UNKNOWN] = float(np.mean(list(weights.values())))
            logp = {label: math.log(value) for label, value in weights.items()}
            alpha = .5
            for event in records:
                mid = event['discovery']['motif_id']
                space = self.prototypes[mid]['space']
                vocabulary = {m for m, proto in self.prototypes.items() if proto['space'] == space}
                width = len(vocabulary) + 1  # reserve a never-seen motif outcome
                for label in labels:
                    denominator = sum(counts[label][m] for m in vocabulary) + alpha * width
                    logp[label] += math.log((counts[label][mid] + alpha) / denominator)
                logp[UNKNOWN] += math.log(1. / width)
            maximum = max(logp.values())
            values = {label: math.exp(value - maximum) for label, value in logp.items()}
            total = sum(values.values())
            items = [{'label': label, 'probability': value / total,
                      'support': int(support[label]) if label != UNKNOWN else 0,
                      'evidence_ids': [e['event_id'] for e in records]} for label, value in values.items()]
            items.sort(key=lambda item: item['probability'], reverse=True)
            return {'kind': 'intent_posterior', 'calibrated': False, 'items': items,
                    'method': 'Dirichlet-smoothed confirmed motif counts; shared prior once; explicit unknown'}

    def _autosave(self):
        if self.path is not None:
            self.save()

    def save(self, path=None):
        with self._lock:
            destination = Path(path) if path is not None else self.path
            if destination is None:
                raise ValueError('supply a private persistence path')
            if destination.suffix != '.json' or destination.is_symlink():
                raise ValueError('motif profile must be a regular .json file')
            absolute = destination.absolute()
            if destination.parent.resolve() != destination.parent.absolute():
                raise ValueError('profile directories may not redirect through symlinks or junctions')
            repository = next((p for p in absolute.parents if (p / '.git').exists()), None)
            if repository is not None:
                first = absolute.relative_to(repository).parts[0]
                if not (first.startswith('.scratch') or first == 'models'):
                    raise ValueError('store a motif profile under ignored .scratch* or models, not tracked source')
            destination.parent.mkdir(parents=True, exist_ok=True)
            ignore = destination.parent / '.gitignore'
            if not ignore.exists():
                ignore.write_text('*\n', encoding='utf-8')
            else:
                rules = [line.strip() for line in ignore.read_text(encoding='utf-8').splitlines()]
                if '*' not in rules or any(line.startswith('!') for line in rules):
                    raise ValueError('private profile directory needs an unconditional * ignore rule')
            data = {'schema': SCHEMA, 'algorithm': ALGORITHM, 'settings': self._settings(),
                    'counter': self.counter, 'feature_spaces': self.feature_spaces,
                    'retired_before': self.retired_before, 'prototypes': list(self.prototypes.values()),
                    'events': list(self.events.values()), 'confirmations': self.confirmations,
                    'rejections': self.rejections}
            encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            if len(encoded.encode('utf-8')) > 32 * 1024 * 1024:
                raise ValueError('motif profile exceeds the 32 MB persistence limit')
            pending = destination.with_name(destination.name + '.pending-' + uuid.uuid4().hex)
            try:
                pending.write_text(encoded, encoding='utf-8')
                pending.replace(destination)
            finally:
                if pending.exists():
                    pending.unlink()

    def _load(self):
        if self.path.is_symlink() or self.path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError('invalid motif profile file')
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if (data.get('schema') != SCHEMA or data.get('algorithm') != ALGORITHM
                or data.get('settings') != self._settings()):
            raise ValueError('motif profile schema/algorithm/settings differ; use a separate profile')
        if (len(data['events']) > self.max_events or len(data['prototypes']) > self.max_prototypes
                or len(data['confirmations']) > self.max_confirmations or len(data['retired_before']) > self.max_sessions
                or len(data['rejections']) > 128):
            raise ValueError('motif profile exceeds its declared bounds')
        feature_spaces = data['feature_spaces']
        if not isinstance(feature_spaces, dict) or any(k not in ('audio', 'video', 'paired') for k in feature_spaces):
            raise ValueError('invalid feature-space metadata')
        for dimension, name in feature_spaces.values():
            if not isinstance(dimension, int) or not 2 <= dimension <= 4096:
                raise ValueError('invalid feature-space dimension')
            _name(name, 'feature_schema', 256)
        prototypes = OrderedDict()
        for prototype in data['prototypes']:
            mid = _name(prototype['id'], 'motif_id')
            if (not mid.startswith('m') or not mid[1:].isdigit() or mid in prototypes
                    or not isinstance(prototype['count'], int) or not 1 <= prototype['count'] <= 1e9):
                raise ValueError('invalid motif prototype')
            vector = _unit(prototype['sum'])
            signature = feature_spaces.get(prototype['modality'])
            if signature is None or len(vector) != signature[0]:
                raise ValueError('prototype dimension disagrees with metadata')
            if json.loads(prototype['space']) != [prototype['modality'], prototype['person_id'], signature[1], signature[0]]:
                raise ValueError('prototype feature schema/owner differs from metadata')
            prototypes[mid] = prototype
        events = OrderedDict()
        for saved in data['events']:
            restored = _record(saved)
            mid = saved['discovery']['motif_id']
            if restored['event_id'] in events or (mid is not None and mid not in prototypes):
                raise ValueError('invalid motif event provenance')
            restored['discovery'] = saved['discovery']
            events[restored['event_id']] = restored
        for session, retired in data['retired_before'].items():
            _name(session, 'session_id', 128)
            if not isinstance(retired, (int, float)) or not math.isfinite(retired) or retired < -1:
                raise ValueError('invalid retired-event horizon')
        for confirmation in data['confirmations']:
            _name(confirmation['text'], 'confirmed text', 500)
            if confirmation['text'] == UNKNOWN:
                raise ValueError('reserved confirmation text')
            _name(confirmation['person_id'], 'person_id', 128)
            _name(confirmation['session_id'], 'session_id', 128)
            _name(confirmation['id'], 'confirmation_id')
            _time(confirmation['at_s'], 'confirmation time')
            if _time(confirmation['t_end'], 'end') < _time(confirmation['t_start'], 'start'):
                raise ValueError('invalid confirmation interval')
            _ids(confirmation['event_ids'], 'event_id', self.max_events)
            _ids(confirmation['source_event_ids'], 'source_event_id', self.max_events * 3)
            if not confirmation['motif_ids'] or any(mid not in prototypes for mid in confirmation['motif_ids']):
                raise ValueError('confirmation refers to an unknown motif')
            if len(confirmation['source_event_ids']) > self.max_events * 3:
                raise ValueError('unbounded confirmation provenance')
        for rejection in data['rejections']:
            _ids(rejection['option_ids'], 'option_id', 32)
            _ids(rejection['event_ids'], 'event_id', 32)
            _time(rejection['at_s'], 'rejection time')
        if (not isinstance(data['counter'], int) or data['counter'] <
                max((int(mid[1:]) for mid in prototypes), default=0)):
            raise ValueError('invalid motif counter')
        self.counter, self.feature_spaces, self.retired_before = data['counter'], feature_spaces, data['retired_before']
        self.prototypes, self.events = prototypes, events
        self.confirmations, self.rejections = data['confirmations'], data['rejections']
