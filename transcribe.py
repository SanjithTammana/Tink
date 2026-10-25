"""Speech to text for the conversation partner: faster-whisper on the CPU, or Meta's Muse Voice Transcribe.

Run: python transcribe.py some.wav [--model base.en]
     python transcribe.py some.wav --meta          (needs MODEL_API_KEY from dev.meta.ai)
"""
import argparse
import io
import json
import math
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np

META_ASR_URL = 'https://api.meta.ai/v1/asr/transcribe'
META_ASR_MODEL = 'muse-voice-transcribe-1.0'
META_MAX_RESPONSE_BYTES = 256 * 1024


class MetaTranscriptionError(RuntimeError):
    """Sanitized configuration, input, transport or response failure."""


def meta_settings(root=None, environ=None):
    """Read only literal ASR settings; OS values (including empty ones) take precedence."""
    names = {'MODEL_API_KEY', 'TINK_TRANSCRIPTION_PROVIDER'}
    values = {}
    if root is not None:
        root = Path(root).resolve(strict=True)
        path = root / '.env'
        if path.is_symlink() or (path.exists() and (not path.is_file() or
                path.resolve().parent != root or path.stat().st_size > 16384)):
            raise MetaTranscriptionError('Muse settings require a regular project .env file.')
        try:
            if path.exists():
                for line in path.read_text(encoding='utf-8-sig').splitlines():
                    if '=' not in line or line.lstrip().startswith('#'):
                        continue
                    name, value = (s.strip() for s in line.split('=', 1))
                    if name not in names:
                        continue
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                        value = value[1:-1]
                    values[name] = value
        except (OSError, UnicodeError):
            raise MetaTranscriptionError('Muse settings could not be read.') from None
    source = os.environ if environ is None else environ
    values.update({name: source[name] for name in names if name in source})
    provider = values.get('TINK_TRANSCRIPTION_PROVIDER', 'local')
    if provider not in ('local', 'meta'):
        raise MetaTranscriptionError('TINK_TRANSCRIPTION_PROVIDER must be local or meta.')
    values['TINK_TRANSCRIPTION_PROVIDER'] = provider
    return values


class _NoMetaRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise MetaTranscriptionError('Muse redirected the request; it was refused.')


def _meta_transport(request, timeout, max_bytes):
    # Do not forward bearer credentials to redirects or environment-configured proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoMetaRedirect())
    with opener.open(request, timeout=timeout) as response:
        if response.geturl() != META_ASR_URL:
            raise MetaTranscriptionError('Muse returned an unexpected destination.')
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise MetaTranscriptionError('Muse response exceeded the size limit.')
    return data


class Transcriber:
    def __init__(self, model_size='base.en'):
        self.model_size = model_size
        self.model = None

    def load(self):
        if self.model is None:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.model_size, device='cpu', compute_type='int8')
        return self

    def transcribe(self, wave_data):
        """16 kHz mono float32 array -> text."""
        self.load()
        segments, _ = self.model.transcribe(
            wave_data, language='en', vad_filter=True, beam_size=5,
            condition_on_previous_text=False)
        return ' '.join(s.text.strip() for s in segments).strip()

    def transcribe_turns(self, wave_data, start_s=0, speaker_id=None):
        """Segment timestamps are ASR estimates, not speaker diarization."""
        self.load()
        segments, info = self.model.transcribe(
            wave_data, language='en', vad_filter=True, beam_size=5,
            condition_on_previous_text=False)
        return [{'turn_id': f'whisper:{start_s}:{i}', 'text': s.text.strip(),
                 'start_s': float(start_s) + float(s.start),
                 'end_s': float(start_s) + float(s.end),
                 'speaker_id': speaker_id or 'unknown', 'speaker_role': 'unknown',
                 'source': 'whisper', 'final': True, 'revision': 0,
                 'timestamp_quality': 'asr_segment', 'overlap': False,
                 'language': getattr(info, 'language', None)}
                for i, s in enumerate(segments) if s.text.strip()]


def wav_bytes(wave_data, sr=16000):
    """float32 mono -> 16-bit PCM WAV bytes."""
    pcm = (np.clip(np.asarray(wave_data, np.float32), -1, 1) * 32767).astype('<i2')
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class MetaTranscriber:
    """Opt-in Muse file transcription with recording-local diarization and estimated turn times.

    Legacy captions retain their local fallback. Recorded assessment passes use_fallback=False
    because its existing local transcript is already the fallback; failures never trigger a
    second cloud provider or a new model download here. No request is made by load().
    """

    def __init__(self, fallback_size='base.en', timeout_s=15., api_key_env='MODEL_API_KEY',
                 *, root=None, environ=None, transport=None, use_fallback=True):
        source = os.environ if environ is None else environ
        self.key = (meta_settings(root, source).get('MODEL_API_KEY', '')
                    if api_key_env == 'MODEL_API_KEY' else source.get(api_key_env, ''))
        if not isinstance(self.key, str) or len(self.key) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in self.key):
            raise MetaTranscriptionError('Muse API key has an unsupported format.')
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
            raise MetaTranscriptionError('Muse timeout must be between zero and sixty seconds.')
        self.timeout_s, self.transport = timeout_s, transport or _meta_transport
        self.fallback = Transcriber(fallback_size) if use_fallback else None
        self.last_backend, self.last_error = None, None

    @property
    def configured(self):
        return bool(self.key)

    def load(self):
        if not self.key and self.fallback is not None:
            self.fallback.load()
        return self

    def _call_json(self, samples):
        if not self.key:
            raise MetaTranscriptionError('Add MODEL_API_KEY before enabling Muse transcription.')
        boundary = uuid.uuid4().hex
        settings = json.dumps({'model': META_ASR_MODEL, 'audioEncoding': 'WAV', 'mode': 'DIARIZATION'})
        body = b''.join([
            f'--{boundary}\r\nContent-Disposition: form-data; name="request"\r\n'
            f'Content-Type: application/json\r\n\r\n{settings}\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="chunk.wav"\r\n'
            f'Content-Type: audio/wav\r\n\r\n'.encode(), wav_bytes(samples), b'\r\n',
            f'--{boundary}--\r\n'.encode()])
        request = urllib.request.Request(META_ASR_URL, data=body, method='POST', headers={
            'Authorization': 'Bearer ' + self.key, 'Accept': 'application/json',
            'Content-Type': 'multipart/form-data; boundary=' + boundary})
        try:
            data = self.transport(request, self.timeout_s, META_MAX_RESPONSE_BYTES)
            if not isinstance(data, bytes) or len(data) > META_MAX_RESPONSE_BYTES:
                raise ValueError()
            return json.loads(data)
        except Exception:
            # Protocol failures (including HTTPException), custom transports and decoders
            # may put response bytes or credentials in their exception strings. Never let
            # any of them cross this cloud boundary. Process-control exceptions still pass.
            raise MetaTranscriptionError('Muse transcription failed; use the local transcript or retry explicitly.') from None

    @staticmethod
    def _samples(wave_data, start_s):
        try:
            samples = np.asarray(wave_data, dtype=np.float32)
            if (samples.ndim != 1 or not 1 <= samples.size <= 30 * 16000 or
                    not np.isfinite(samples).all() or isinstance(start_s, bool) or
                    not math.isfinite(float(start_s)) or float(start_s) < 0):
                raise ValueError()
            return samples, float(start_s)
        except (ValueError, TypeError, OverflowError):
            raise MetaTranscriptionError('Muse needs up to thirty seconds of finite 16 kHz mono samples and a valid clock.') from None

    def _turns(self, samples, start_s):
        payload = self._call_json(samples)
        try:
            duration_ms = len(samples) / 16.
            duration = payload['audioDurationMs']
            raw = payload['turns']
            if (type(duration) not in (int, float) or not math.isfinite(duration) or
                    not 0 <= duration <= duration_ms + 50 or not isinstance(raw, list) or len(raw) > 96 or
                    not isinstance(payload.get('transcript'), str) or len(payload['transcript']) > 16000):
                raise ValueError()
            scope, identities, result = uuid.uuid4().hex, set(), []
            for item in raw:
                identity, text = item['turnId'], item['transcript']
                begin, end = item['startMs'], item['endMs']
                label = item.get('speaker') or 'unknown'
                if (type(identity) is not int or identity in identities or not isinstance(text, str) or
                        not text.strip() or len(text) > 2000 or not isinstance(label, str) or
                        not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', label) or
                        type(begin) not in (int, float) or type(end) not in (int, float) or
                        not math.isfinite(begin + end) or not 0 <= begin <= end <= duration_ms + 50):
                    raise ValueError()
                identities.add(identity)
                result.append({'turn_id': f'meta:{scope}:{identity}', 'text': text.strip(),
                    'start_s': start_s + min(begin, duration_ms) / 1000,
                    'end_s': start_s + min(end, duration_ms) / 1000,
                    'speaker_id': f'meta:{scope}:{label}' if label.casefold() != 'unknown' else 'unknown',
                    'speaker_label': label, 'speaker_scope': 'recording', 'speaker_role': 'unknown',
                    'speaker_uncertain': True, 'source': 'cloud_audio', 'backend': 'meta_muse',
                    'final': True, 'revision': 0, 'timestamp_quality': 'asr_turn',
                    'overlap': False, 'overlap_known': False, 'language': None})
            # DIARIZATION should return timed turns for speech. Never manufacture word/turn times.
            if payload['transcript'].strip() and not result:
                raise ValueError()
            for turn in result:
                if any(other is not turn and other['start_s'] < turn['end_s'] and
                       other['end_s'] > turn['start_s'] for other in result):
                    turn['overlap'], turn['overlap_known'] = True, True
            result.sort(key=lambda turn: (turn['start_s'], turn['end_s']))
            return result
        except (KeyError, ValueError, TypeError, AttributeError):
            raise MetaTranscriptionError('Muse returned an unsupported transcript; local captions remain available.') from None

    def transcribe_turns(self, wave_data, start_s=0, speaker_id=None):
        samples, start_s = self._samples(wave_data, start_s)
        self.last_error = None
        try:
            result = self._turns(samples, start_s)
            self.last_backend = 'meta'
            return result
        except MetaTranscriptionError as exc:
            self.last_error = str(exc)
            if self.fallback is None:
                raise
        self.last_backend = 'whisper'
        return self.fallback.transcribe_turns(samples, start_s, speaker_id)

    def call(self, wave_data):
        samples, start_s = self._samples(wave_data, 0)
        return ' '.join(turn['text'] for turn in self._turns(samples, start_s))

    def transcribe(self, wave_data):
        return ' '.join(turn['text'] for turn in self.transcribe_turns(wave_data))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio')
    parser.add_argument('--model', default='base.en')
    parser.add_argument('--meta', action='store_true', help='Muse Voice Transcribe (MODEL_API_KEY)')
    args = parser.parse_args()
    import librosa
    wave_data, _ = librosa.load(args.audio, sr=16000, mono=True)
    t = MetaTranscriber(args.model, root=Path(__file__).resolve().parent) if args.meta else Transcriber(args.model)
    print(t.transcribe(wave_data))
    if args.meta:
        print('(backend:', t.last_backend + ')')


if __name__ == '__main__':
    main()
