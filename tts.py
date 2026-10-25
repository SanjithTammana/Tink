"""Offline text-to-speech (pyttsx3 / Windows SAPI) on a background thread.

Run: python tts.py "Hello there"
"""
import queue
from collections import deque
import sys
import threading
import time


class Speaker:
    def __init__(self, rate=170, enabled=True):
        self.enabled = enabled
        self.rate = rate
        self.queue = queue.Queue()
        self.thread = None
        self.speaking = threading.Event()
        self.last_finished = float('-inf')
        self.intervals = deque(maxlen=128)
        self.interval_lock = threading.Lock()
        self.error = False
        if enabled:
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def loop(self):
        import pyttsx3  # the SAPI engine must live on the thread that uses it
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
        except Exception:
            self.error = True
            self.speaking.clear()
            with self.interval_lock:
                for interval in self.intervals:
                    interval['end'] = time.perf_counter()
            return
        while True:
            item = self.queue.get()
            if item is None:
                break
            text, interval = item
            try:
                self.speaking.set()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                self.error = True
                print('Local TTS playback failed.')
            finally:
                self.last_finished = time.perf_counter()
                with self.interval_lock:
                    interval['end'] = self.last_finished
                self.speaking.clear()

    def say(self, text):
        if self.enabled and not self.error and text:
            interval = {'start': time.perf_counter(), 'end': None}
            with self.interval_lock:
                self.intervals.append(interval)
            self.speaking.set()  # suppress capture from enqueue, including engine startup
            self.queue.put((text, interval))

    def overlaps(self, start_s, end_s, origin=0., guard=.7):
        """Playback ledger, independent of UI polling; timestamps use perf_counter."""
        with self.interval_lock:
            return any(start_s + origin <= (i['end'] + guard if i['end'] is not None else float('inf'))
                       and end_s + origin >= i['start'] for i in self.intervals)

    def close(self):
        if self.thread:
            self.queue.put(None)
            self.thread.join(timeout=5)


if __name__ == '__main__':
    s = Speaker()
    s.say(' '.join(sys.argv[1:]) or 'Text to speech is working.')
    s.close()
