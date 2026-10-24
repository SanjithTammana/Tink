"""faster-whisper transcription (CPU, int8).

Run: python transcribe.py some.wav [--model base.en]
"""
import argparse


class Transcriber:
    def __init__(self, model_size='base.en'):
        self.model_size = model_size
        self.model = None

    def load(self):
        if self.model is None:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.model_size, device='cpu', compute_type='int8')
        return self

    def transcribe(self, wave):
        """16 kHz mono float32 array -> text."""
        self.load()
        segments, _ = self.model.transcribe(
            wave, language='en', vad_filter=True, beam_size=5,
            condition_on_previous_text=False)
        return ' '.join(s.text.strip() for s in segments).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio')
    parser.add_argument('--model', default='base.en')
    args = parser.parse_args()
    import librosa
    wave, _ = librosa.load(args.audio, sr=16000, mono=True)
    print(Transcriber(args.model).transcribe(wave))


if __name__ == '__main__':
    main()
