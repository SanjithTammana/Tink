"""One monotonic origin for video, delivered audio and conversation turns."""
import math
import time
import uuid

from audio_layer import SR, SampleCapture


class SessionClock:
    def __init__(self, source=time.perf_counter):
        self.source, self.origin = source, source()
        self.session_id = uuid.uuid4().hex

    def now(self):
        return self.source() - self.origin


class SharedAudioCapture(SampleCapture):
    def __init__(self, clock, max_blocks=128):
        super().__init__(max_blocks)
        self.clock = clock
        self.anchor_s = None
        self.timestamp_basis = 'uninitialized'

    def callback(self, data, frames, time_info, status):
        if self.anchor_s is None:
            now = self.clock.now()
            try:
                adc = float(time_info.inputBufferAdcTime)
                current = float(time_info.currentTime)
                if not math.isfinite(adc + current) or adc <= 0 or current < adc or current - adc > 5:
                    raise ValueError('device clock unavailable')
                self.anchor_s = now - (current - adc)
                self.timestamp_basis = 'portaudio_adc_offset_to_session_monotonic'
            except (AttributeError, TypeError, ValueError):
                self.anchor_s = now - frames / SR
                self.timestamp_basis = 'host_callback_estimate; sensor latency uncertain'
        super().callback(data, frames, time_info, status)

    def at(self, sample_index):
        return self.anchor_s + sample_index / SR if self.clock_valid and self.anchor_s is not None else None
