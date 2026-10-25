"""Standalone nonverbal audio events. No video, fusion, transcription or TTS.

One microphone cannot establish who made a sound. All events have unknown person
attribution. Cosine template scores are not calibrated intent probabilities.
"""
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import re
import threading
import time
import uuid

import numpy as np

SR = 16000
MODEL = 'alkiskoudounas/voc2vec-hubert-ls-pt'
LAYER = 5
NEGATIVES = ('speech', 'other')


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,31}', value):
        raise ValueError('use a name beginning with a letter, then up to 31 letters/numbers/_/-')
    return value


def rms_db(wave):
    wave = np.asarray(wave, np.float32)
    return float(10 * np.log10(np.mean(wave * wave) + 1e-12)) if len(wave) else -120.


@dataclass
class AudioSegment:
    start: int
    end: int
    wave: np.ndarray
    complete: bool = True
    reason: str = 'release'


class EnergySegmenter:
    """20ms sample-clock segmentation with noise-relative hysteresis.

    Initial calibration requires quiet room audio. A max-length event is emitted
    incomplete once, then blocked until rest: a long sound is not counted again.
    Gaps discard partial events rather than joining unrelated audio.
    """
    def __init__(self, noise_db=None, calibration_s=1., on_db=9., off_db=5.,
                 min_s=.12, release_s=.26, pre_s=.08, max_s=4., sr=SR):
        if sr != SR or not 0 <= off_db < on_db or min_s <= 0 or max_s <= min_s:
            raise ValueError('invalid sample rate or segmentation thresholds')
        if noise_db is not None and not math.isfinite(noise_db):
            raise ValueError('noise floor must be finite')
        self.sr, self.hop = sr, int(.02 * sr)
        self.noise_db = noise_db
        self.calibration_frames = max(1, math.ceil(calibration_s / .02))
        self.calibration = []
        self.on_db, self.off_db = on_db, off_db
        self.min_frames = math.ceil(min_s / .02)
        self.release_frames = max(1, math.ceil(release_s / .02))
        self.pre_frames = max(0, math.ceil(pre_s / .02))
        self.max_frames = math.ceil(max_s / .02)
        self.expected = self.pending_start = None
        self.pending = np.empty(0, np.float32)
        self.pre = deque(maxlen=self.pre_frames)
        self.active = None
        self.blocked = False
        self.quiet_frames = 0
        self.gaps = self.discarded_segments = 0
        self.level_db = -120.

    @property
    def state(self):
        if self.noise_db is None:
            return 'CALIBRATING: keep quiet'
        return 'WAIT FOR REST' if self.blocked else ('SOUND ACTIVE' if self.active else 'LISTENING')

    def reset_gap(self):
        if self.active:
            self.discarded_segments += 1
        self.active, self.blocked, self.quiet_frames = None, False, 0
        self.pending = np.empty(0, np.float32)
        self.pre.clear()
        self.gaps += 1

    def feed(self, start, wave, discontinuity=False):
        wave = np.asarray(wave, np.float32).reshape(-1)
        if start < 0 or not np.isfinite(wave).all():
            raise ValueError('audio must be finite and its sample index nonnegative')
        if discontinuity or (self.expected is not None and start != self.expected):
            self.reset_gap()
        if not len(self.pending):
            self.pending_start = start
        self.expected = start + len(wave)
        self.pending = np.concatenate([self.pending, wave])
        output = []
        while len(self.pending) >= self.hop:
            frame, self.pending = self.pending[:self.hop], self.pending[self.hop:]
            index = self.pending_start
            self.pending_start += self.hop
            result = self._frame(index, frame)
            if result is not None:
                output.append(result)
        return output

    def _frame(self, index, frame):
        level = self.level_db = rms_db(frame)
        if self.noise_db is None:
            self.calibration.append(level)
            if len(self.calibration) >= self.calibration_frames:
                self.noise_db = float(np.median(self.calibration))
                self.calibration.clear()
            self.pre.append((index, frame.copy()))
            return None
        # Avoid a digital-zero floor making any numerical noise a sound.
        floor = max(-80., self.noise_db)
        above = level >= floor + (self.off_db if self.active else self.on_db)
        if self.blocked:
            self.quiet_frames = self.quiet_frames + 1 if not above else 0
            if self.quiet_frames >= self.release_frames:
                self.blocked, self.quiet_frames = False, 0
            self.pre.append((index, frame.copy()))
            return None
        if self.active is None:
            if not above:
                self.noise_db = .995 * self.noise_db + .005 * level
                self.pre.append((index, frame.copy()))
                return None
            frames = list(self.pre) + [(index, frame.copy())]
            self.active = {'frames': frames, 'active_n': 1, 'last': index + self.hop,
                           'last_position': len(frames), 'quiet': 0}
            return None
        current = self.active
        current['frames'].append((index, frame.copy()))
        if above:
            current['active_n'] += 1
            current['last'], current['last_position'], current['quiet'] = index + self.hop, len(current['frames']), 0
        else:
            current['quiet'] += 1
        if len(current['frames']) >= self.max_frames:
            result = self._finish(False, 'max_duration')
            self.blocked = True
            return result
        if current['quiet'] >= self.release_frames:
            return self._finish(True, 'release')
        return None

    def _finish(self, complete, reason):
        current, self.active = self.active, None
        self.pre.clear()
        if current is None or current['active_n'] < self.min_frames:
            return None
        # Retain at most preroll-sized trailing padding, not the entire release wait.
        frames = current['frames'][:current['last_position'] + self.pre_frames]
        wave = np.concatenate([v for _, v in frames])
        return AudioSegment(frames[0][0], frames[0][0] + len(wave), wave, complete, reason)

    def finish(self):
        return self._finish(False, 'eof')


class SampleCapture:
    """Nonblocking sounddevice callback bridge with explicit losses and sample clock."""
    def __init__(self, max_blocks=128):
        self.blocks = queue.Queue(maxsize=max_blocks)
        self.sample_count = self.dropped_blocks = self.dropped_samples = self.overflows = 0
        self.lock = threading.Lock()

    @property
    def clock_valid(self):
        # Device loss has unknown duration. Never call this compressed delivered-
        # sample clock valid capture time after a PortAudio overflow/status flag.
        return self.overflows == 0

    def callback(self, data, frames, time_info, status):
        # Device timestamps are not assumed comparable to this application's clock.
        wave = np.asarray(data, np.float32).reshape(frames, -1).mean(axis=1).copy()
        with self.lock:
            start = self.sample_count
            self.sample_count += frames
            if status:
                self.overflows += 1
        item = start, wave, bool(status)
        try:
            self.blocks.put_nowait(item)
        except queue.Full:
            try:
                old = self.blocks.get_nowait()
                self.dropped_blocks += 1
                self.dropped_samples += len(old[1])
            except queue.Empty:
                pass
            try:
                self.blocks.put_nowait(item)
            except queue.Full:
                self.dropped_blocks += 1
                self.dropped_samples += frames

    def drain(self):
        result = []
        while True:
            try:
                result.append(self.blocks.get_nowait())
            except queue.Empty:
                return result


def annotate_capture_timing(event, delivered_samples, clock_valid):
    """Keep device-side loss from becoming apparently fresh, valid evidence."""
    result = dict(event, capture_clock_valid=bool(clock_valid))
    if clock_valid:
        result['age_s_at_publish'] = max(0., delivered_samples / SR - result['t_end'])
    else:
        result.update(sample_t_start=result['t_start'], sample_t_end=result['t_end'],
                      t_start=None, t_end=None, age_s_at_publish=None,
                      accepted=False, label='unknown', reason='device_overflow_clock_unknown')
    return result


class Voc2VecEncoder:
    """Exact layer-5 hidden-state mean pooling; load cached files only."""
    def __init__(self):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        from huggingface_hub import try_to_load_from_cache
        self.torch = torch
        torch.set_num_threads(2)
        try:
            config = try_to_load_from_cache(MODEL, 'config.json')
            if not isinstance(config, str):
                raise OSError('cached model config is missing')
            snapshot = Path(config).parent
            self.revision = snapshot.name
            self.extractor = AutoFeatureExtractor.from_pretrained(str(snapshot), local_files_only=True)
            self.model = AutoModel.from_pretrained(str(snapshot), local_files_only=True).eval()
        except (OSError, RuntimeError) as error:
            raise RuntimeError('Cached voc2vec model is unavailable. No download was attempted; use the existing Tink environment/cache.') from error

    def __call__(self, wave):
        wave = np.asarray(wave, np.float32).reshape(-1)
        if len(wave) < SR // 2:
            wave = np.pad(wave, (0, SR // 2 - len(wave)))
        inputs = self.extractor(wave, sampling_rate=SR, return_tensors='pt')
        with self.torch.no_grad():
            states = self.model(**inputs, output_hidden_states=True).hidden_states
        result = states[LAYER][0].mean(0).cpu().numpy().astype(np.float32)
        if result.shape != (768,) or not np.isfinite(result).all():
            raise RuntimeError('unexpected voc2vec embedding shape/values')
        return result


class TemplateClassifier:
    def __init__(self, vectors, labels, min_score=.75, min_margin=.05):
        if not -1 <= min_score <= 1 or not 0 <= min_margin <= 2:
            raise ValueError('cosine threshold must be in [-1,1] and margin in [0,2]')
        vectors = np.asarray(vectors, np.float32)
        if vectors.ndim != 2 or len(vectors) != len(labels) or not len(vectors) or not np.isfinite(vectors).all():
            raise ValueError('need finite embedding rows with matching labels')
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms < 1e-8):
            raise ValueError('zero embedding is invalid')
        self.vectors = vectors / norms[:, None]
        self.labels = np.asarray([safe_name(str(label)) for label in labels])
        self.min_score, self.min_margin = float(min_score), float(min_margin)

    def predict(self, vector):
        vector = np.asarray(vector, np.float32)
        if vector.shape != (self.vectors.shape[1],) or not np.isfinite(vector).all():
            raise ValueError('embedding dimension/values do not match profile')
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return {'label': 'unknown', 'accepted': False, 'reason': 'invalid_embedding', 'scores': {}}
        sims = self.vectors @ (vector / norm)
        scores = {label: float(sims[self.labels == label].max()) for label in sorted(set(self.labels))}
        ranked = sorted(scores, key=scores.get, reverse=True)
        best = ranked[0]
        margin = scores[best] - (scores[ranked[1]] if len(ranked) > 1 else 1.)
        reason = ('negative_example' if best in NEGATIVES else 'low_similarity' if scores[best] < self.min_score
                  else 'ambiguous' if margin < self.min_margin else 'template_match')
        accepted = reason == 'template_match'
        label = best if accepted or best in NEGATIVES else 'unknown'
        return {'label': label, 'accepted': accepted, 'reason': reason, 'scores': scores,
                'best_template_label': best, 'margin': float(margin), 'calibrated': False}


def save_profile(path, vectors, labels, noise_db, min_score=.75, min_margin=.05, revision=None):
    if not math.isfinite(noise_db):
        raise ValueError('noise floor must be finite')
    if not isinstance(revision, str) or not revision:
        raise ValueError('profile requires the resolved model revision')
    classifier = TemplateClassifier(vectors, labels, min_score, min_margin)
    counts = {label: int(np.sum(classifier.labels == label)) for label in set(classifier.labels)}
    if any(counts.get(label, 0) < 3 for label in NEGATIVES) or not (set(counts) - set(NEGATIVES)):
        raise ValueError('profile needs user classes and at least three speech and three other examples')
    if any(n < 3 for n in counts.values()):
        raise ValueError('every enrolled class needs at least three examples')
    meta = {'schema': 2, 'model': MODEL, 'model_revision': revision, 'layer': LAYER, 'sr': SR,
            'embedding_dim': classifier.vectors.shape[1], 'noise_db': float(noise_db),
            'min_score': float(min_score), 'min_margin': float(min_margin), 'calibrated': False}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / '.gitignore').write_text('*\n', encoding='utf-8')
    pending = path.with_suffix('.pending.npz')
    np.savez_compressed(pending, vectors=classifier.vectors, labels=classifier.labels, metadata=json.dumps(meta))
    pending.replace(path)


def load_profile(path, expected_dim=768, revision=None):
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved['metadata'].item()))
        vectors, labels = saved['vectors'].copy(), saved['labels'].copy()
    if any(meta.get(k) != v for k, v in {'schema': 2, 'model': MODEL, 'layer': LAYER, 'sr': SR,
                                        'embedding_dim': expected_dim}.items()):
        raise ValueError('profile model/layer/sample rate/embedding schema is incompatible; reenroll')
    if not meta.get('model_revision') or (revision is not None and meta['model_revision'] != revision):
        raise ValueError('profile model revision differs from the cached encoder; reenroll')
    if vectors.shape[1:] != (expected_dim,):
        raise ValueError('profile vector shape disagrees with metadata')
    classifier = TemplateClassifier(vectors, labels, meta['min_score'], meta['min_margin'])
    counts = {name: int(np.sum(classifier.labels == name)) for name in set(classifier.labels)}
    if (any(counts.get(name, 0) < 3 for name in NEGATIVES)
            or not set(counts) - set(NEGATIVES) or any(n < 3 for n in counts.values())):
        raise ValueError('profile needs at least three examples per user/speech/other class')
    return classifier, meta


class EventProcessor:
    def __init__(self, encoder, classifier=None):
        self.encoder, self.classifier = encoder, classifier
        self.session, self.counter = uuid.uuid4().hex, 0

    def process(self, segment):
        begin = time.perf_counter()
        vector = self.encoder(segment.wave)
        result = (self.classifier.predict(vector) if self.classifier else
                  {'label': 'unknown', 'accepted': False, 'reason': 'unenrolled', 'scores': {}})
        if not segment.complete:
            result.update(accepted=False, reason=segment.reason, label='unknown')
        self.counter += 1
        return {'type': 'audio_event', 'event_id': f'{self.session}:{self.counter}',
                'source': 'vocalization', 'person_id': None, 'attribution': 'unknown_single_microphone',
                't_start': segment.start / SR, 't_end': segment.end / SR,
                'complete': segment.complete, 'calibrated': False,
                'embedding_dim': len(vector), 'embedding_ms': 1000 * (time.perf_counter() - begin),
                **result}
