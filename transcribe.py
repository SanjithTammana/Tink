"""Speech to text for the conversation partner: faster-whisper on the CPU, or Meta's Muse Voice Transcribe.

Run: python transcribe.py some.wav [--model base.en]
     python transcribe.py some.wav --meta          (needs MODEL_API_KEY from dev.meta.ai)
"""
import argparse
import io
import json
import os
import urllib.request
import uuid
import wave

import numpy as np

META_ASR_URL = 'https://api.meta.ai/v1/asr/transcribe'
META_ASR_MODEL = 'muse-voice-transcribe-1.0'


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
    """Muse Voice Transcribe, file endpoint (POST /v1/asr/transcribe, multipart: a JSON 'request' part and a
    WAV 'audio' part, PUSH_TO_TALK mode, text/plain reply with one turn per line). Each 4 s microphone chunk is
    one request. If the call fails (no key, no network, rate limit), the local Whisper fallback answers, so
    captions keep working. Sends partner audio to Meta: use with consent."""

    def __init__(self, fallback_size='base.en', timeout_s=8.0, api_key_env='MODEL_API_KEY'):
        self.key = os.environ.get(api_key_env, '')
        self.timeout_s = timeout_s
        self.fallback = Transcriber(fallback_size)
        self.last_backend = None

    def load(self):
        if not self.key:
            self.fallback.load()
        return self

    def call(self, wave_data):
        boundary = uuid.uuid4().hex
        settings = json.dumps({'model': META_ASR_MODEL, 'audioEncoding': 'WAV', 'mode': 'PUSH_TO_TALK'})
        body = b''.join([
            f'--{boundary}\r\nContent-Disposition: form-data; name="request"\r\n'
            f'Content-Type: application/json\r\n\r\n{settings}\r\n'.encode(),
            f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="chunk.wav"\r\n'
            f'Content-Type: audio/wav\r\n\r\n'.encode(), wav_bytes(wave_data), b'\r\n',
            f'--{boundary}--\r\n'.encode()])
        req = urllib.request.Request(META_ASR_URL, data=body, headers={
            'Authorization': f'Bearer {self.key}', 'Accept': 'text/plain',
            'Content-Type': f'multipart/form-data; boundary={boundary}'})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            lines = r.read().decode('utf-8').splitlines()
        # text/plain: one turn per line; diarized lines look like 'A:Hello' or 'unknown:...'
        turns = [l.split(':', 1)[1] if ':' in l[:10] and l.split(':', 1)[0].isalnum() else l for l in lines]
        return ' '.join(t.strip() for t in turns if t.strip())

    def transcribe(self, wave_data):
        if self.key:
            try:
                text = self.call(wave_data)
                self.last_backend = 'meta'
                return text
            except Exception as e:
                print('Muse Voice Transcribe failed, using Whisper:', str(e)[:80])
        self.last_backend = 'whisper'
        return self.fallback.transcribe(wave_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio')
    parser.add_argument('--model', default='base.en')
    parser.add_argument('--meta', action='store_true', help='Muse Voice Transcribe (MODEL_API_KEY)')
    args = parser.parse_args()
    import librosa
    wave_data, _ = librosa.load(args.audio, sr=16000, mono=True)
    t = MetaTranscriber(args.model) if args.meta else Transcriber(args.model)
    print(t.transcribe(wave_data))
    if args.meta:
        print('(backend:', t.last_backend + ')')


if __name__ == '__main__':
    main()
