"""Offline text-to-speech (pyttsx3 / Windows SAPI) on a background thread.

Run: python tts.py "Hello there"
"""
import queue
import sys
import threading


class Speaker:
    def __init__(self, rate=170, enabled=True):
        self.enabled = enabled
        self.rate = rate
        self.queue = queue.Queue()
        self.thread = None
        if enabled:
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def loop(self):
        import pyttsx3  # the SAPI engine must live on the thread that uses it
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        while True:
            text = self.queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print('TTS error:', e)

    def say(self, text):
        if self.enabled and text:
            self.queue.put(text)

    def close(self):
        if self.thread:
            self.queue.put(None)
            self.thread.join(timeout=5)


if __name__ == '__main__':
    s = Speaker()
    s.say(' '.join(sys.argv[1:]) or 'Text to speech is working.')
    s.close()
