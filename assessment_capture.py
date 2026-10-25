"""Bounded recording with common timestamps; no camera, microphone or network ownership."""
from copy import deepcopy
from collections import OrderedDict
import json
import math
from pathlib import Path
import threading
import uuid

import cv2
import numpy as np

from audio_layer import SR
from audio_layer import SampleCapture


class RecordedAudioCapture(SampleCapture):
    """Per-block ADC/host timestamps preserve raw sound after an earlier device gap.

    Legacy labs retain their fail-closed sample clock. This recorder independently
    timestamps every block and marks any bad block inside its clip as uncertain.
    """
    def __init__(self, clock, max_blocks=256):
        super().__init__(max_blocks)
        self.clock, self.timings = clock, OrderedDict()
        self.timing_lock = threading.Lock()
        self.timestamp_basis = 'per-block host estimate; sensor latency uncertain'

    def callback(self, data, frames, time_info, status):
        now = self.clock.now()
        timestamp = now - frames / SR
        basis = 'host callback estimate'
        try:
            adc, current = float(time_info.inputBufferAdcTime), float(time_info.currentTime)
            if math.isfinite(adc + current) and adc > 0 and 0 <= current - adc <= 5:
                timestamp = now - (current - adc)
                basis = 'per-block PortAudio ADC offset'
        except (AttributeError, ValueError, TypeError):
            pass
        with self.timing_lock:
            self.timings[self.sample_count] = (timestamp, not bool(status))
            while len(self.timings) > 1024:
                self.timings.popitem(last=False)
        self.timestamp_basis = basis
        super().callback(data, frames, time_info, status)

    def at(self, sample_index):
        with self.timing_lock:
            entry = self.timings.get(sample_index)
            return entry[0] if entry else None

    def block_valid(self, sample_index):
        with self.timing_lock:
            entry = self.timings.get(sample_index)
            return entry[1] if entry else False


def compact_observation(obs):
    value = {k: deepcopy(obs[k]) for k in ['t_capture', 'person_id', 'logical_person_id',
             'ownership_valid', 'target_status', 'target_id', 'asl'] if k in obs}
    value['people'] = [{k: p[k] for k in ['id', 'bbox', 'score'] if k in p} for p in obs.get('people', [])]
    value['channels'] = {}
    for name, channel in obs.get('channels', {}).items():
        item = {k: deepcopy(channel[k]) for k in ['status', 'features', 'model_ms', 'cnn_heads',
                 'cnn_distribution', 'lgbm_gesture', 'calibrating_neutral'] if k in channel}
        if name == 'hands':
            item['hands'] = [{k: deepcopy(h[k]) for k in ['features', 'scores'] if k in h}
                             for h in channel.get('hands', [])]
        value['channels'][name] = item
    return value


class ClipRecorder:
    def __init__(self, folder, session_id, start_s, max_seconds=30., fps=8.):
        if not math.isfinite(start_s) or start_s < 0 or not 0 < max_seconds <= 30 or not 1 <= fps <= 30:
            raise ValueError('invalid capture clock or duration')
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        (self.folder / '.gitignore').write_text('*\n', encoding='utf-8')
        self.session_id, self.clip_id = session_id, 'clip-' + uuid.uuid4().hex
        self.start_s, self.maximum, self.fps = float(start_s), float(max_seconds), float(fps)
        self.lock = threading.RLock()
        self.writer, self.last_frame, self.frames = None, None, 0
        self.audio = np.zeros(round(max_seconds * SR), np.float32)
        self.audio_end, self.audio_gaps, self.clock_valid = 0, [], True
        self.result, self.observations = None, []
        self.frame_times = (self.folder / 'video_times.jsonl').open('w', encoding='utf-8')
        self.audio_times = (self.folder / 'audio_times.jsonl').open('w', encoding='utf-8')

    def add_frame(self, frame, at_s):
        with self.lock:
            if self.result is not None or at_s < self.start_s or at_s > self.start_s + self.maximum:
                return
            if self.writer is None:
                h, w = frame.shape[:2]
                scale = min(640 / w, 480 / h, 1.)
                self.size = (max(2, round(w * scale) // 2 * 2), max(2, round(h * scale) // 2 * 2))
                self.writer = cv2.VideoWriter(str(self.folder / 'video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
                if not self.writer.isOpened():
                    raise RuntimeError('Could not create the local MP4 recording')
            image = cv2.resize(frame, self.size)
            # Uniform timeline: duplicate last observed image across missing frame slots.
            while self.start_s + self.frames / self.fps <= at_s and self.frames < math.ceil(self.maximum * self.fps):
                slot = self.start_s + self.frames / self.fps
                self.writer.write(self.last_frame if self.last_frame is not None and slot < at_s - .5 / self.fps else image)
                self.frame_times.write(json.dumps({'frame': self.frames, 'session_s': slot, 'source_read_s': at_s,
                    'resampled': True}) + '\n')
                self.frames += 1
            self.last_frame = image

    def add_audio(self, wave, at_s, clock_valid):
        with self.lock:
            if self.result is not None:
                return
            if at_s is not None and (at_s + len(wave) / SR <= self.start_s or at_s >= self.start_s + self.maximum):
                return  # an older callback must not invalidate a later recording
            self.clock_valid &= bool(clock_valid)
            if at_s is None:
                self.audio_times.write(json.dumps({'clock_valid': False, 'samples': len(wave)}) + '\n')
                return
            wave = np.asarray(wave, np.float32).reshape(-1)
            if not np.isfinite(wave).all():
                self.clock_valid = False
                return
            start = round((at_s - self.start_s) * SR)
            if start < 0:
                wave, start = wave[-start:], 0
            stop = min(len(self.audio), start + len(wave))
            if stop <= start or start >= len(self.audio):
                return
            if start > self.audio_end:
                self.audio_gaps.append([self.start_s + self.audio_end / SR, self.start_s + start / SR])
            begin = max(start, self.audio_end)
            if stop > begin:
                self.audio[begin:stop] = wave[begin-start:stop-start]
                self.audio_end = stop
            self.audio_times.write(json.dumps({'sample_start': start, 'samples': stop-start,
                'session_s': at_s, 'clock_valid': bool(clock_valid)}) + '\n')

    def add_observation(self, observation):
        with self.lock:
            t = observation['t_capture']
            if self.result is None and self.start_s <= t <= self.start_s + self.maximum and len(self.observations) < 512:
                self.observations.append(compact_observation(observation))

    def finish(self, end_s):
        with self.lock:
            if self.result is not None:
                return deepcopy(self.result)
            duration = max(.001, min(self.maximum, end_s - self.start_s))
            samples = round(duration * SR)
            if self.audio_end < samples:
                self.audio_gaps.append([self.start_s + self.audio_end / SR, self.start_s + duration])
            while self.writer and self.frames < math.ceil(duration * self.fps) and self.last_frame is not None:
                self.writer.write(self.last_frame)
                self.frames += 1
            if self.writer:
                self.writer.release()
            self.frame_times.close()
            self.audio_times.close()
            import soundfile as sf
            audio_path = self.folder / 'audio.wav'
            sf.write(str(audio_path), self.audio[:samples], SR, subtype='PCM_16')
            self.result = {'session_id': self.session_id, 'clip_id': self.clip_id, 'start_s': self.start_s,
                'end_s': self.start_s + duration, 'duration_s': duration, 'clock_valid': self.clock_valid,
                'audio_gaps': self.audio_gaps, 'audio_samples': samples, 'audio_rate': SR,
                'audio_path': str(audio_path), 'video_path': str(self.folder / 'video.mp4') if self.writer else None,
                'video_frames': self.frames, 'video_fps': self.fps,
                'media_timing': 'common host monotonic origin; video resampled to 8 fps; frame duration quantization <= 1/fps; audio gaps padded with silence',
                'video_observations': len(self.observations)}
            (self.folder / 'capture.json').write_text(json.dumps(self.result, indent=2), encoding='utf-8')
            (self.folder / 'features.json').write_text(json.dumps(self.observations, allow_nan=False), encoding='utf-8')
            return deepcopy(self.result)
