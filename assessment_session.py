"""Existing OpenCV test surface over the Django-callable clip assessment service."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import threading
import textwrap
import time
import uuid

import cv2
import numpy as np

from assessment_capture import ClipRecorder, RecordedAudioCapture
from assessment_service import ClipAssessmentService
from audio_layer import rms_db
from gemini_client import GeminiError
from tools.audio_layer_lab import input_stream
from tools.video_layer_lab import LiveCapture
from tools.video_overlay import render_tracking
from tts import Speaker


class AudioPump:
    """Continuously drain callback audio independently of video inference/network."""
    def __init__(self, capture, active):
        self.capture, self.active = capture, active
        self.stop, self.thread = threading.Event(), None
        self.level, self.quiet, self.noise = -120., [], None
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self.run, name='assessment-audio-writer', daemon=True)
        self.thread.start()
        return self

    def run(self):
        try:
            while not self.stop.is_set():
                for start, wave, bad in self.capture.drain():
                    self.level = rms_db(wave)
                    if self.noise is None:
                        self.quiet.append(self.level)
                        if len(self.quiet) >= 50:
                            self.noise = float(np.median(self.quiet))
                    clip = self.active()
                    if clip is not None:
                        clip.add_audio(wave, self.capture.at(start), self.capture.block_valid(start) and not bad)
                self.stop.wait(.01)
        except Exception as error:
            self.error = error

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2.)


def draw(frame, obs, status, clip, now, audio_level, result, spelled):
    image = render_tracking(frame, obs, neutral_ready=True, recording=clip is not None)
    out = cv2.copyMakeBorder(image, 0, 390, 0, 0, cv2.BORDER_CONSTANT, value=(23, 28, 35))
    y = image.shape[0] + 26
    def line(text, color=(225, 232, 239), size=.48):
        nonlocal y
        for part in textwrap.wrap(str(text), 145)[:2] or ['']:
            cv2.putText(out, part.encode('ascii', 'replace').decode(), (18, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
            y += 23
    elapsed = max(0., now - clip.start_s) if clip else 0.
    line(('RECORDING %.1f / 30 s' % elapsed if clip else 'TINK / RECORDED ASSESSMENT') + f' | mic {audio_level:.1f} dBFS', (130, 225, 180), .58)
    line(status, (130, 190, 255))
    if result:
        line('Assessment: ' + result['assessment']['summary'])
        context = result['partner_context']
        line('Partner context (hypothesis): ' + (context['pending_question'] or context['summary']))
        for option in result['options']:
            line(f"{option['id']}. [{option['kind']}] {option['text']}", (130, 225, 180), .56)
    else:
        line('The completed clip sends video + audio + measured features + recent conversation to Gemini.')
        line('Three possibilities require confirmation. Quiet room for the first second; then interact naturally.')
    asl = obs.get('asl', {})
    line(f"ASL alphabet {asl.get('best','-')} ready={asl.get('ready',False)} | spelling: {spelled}")
    line('SPACE record | ENTER stop + assess | R retry | 1-3 confirm + speak | P JSON packet | Q quit', size=.43)
    line('Y confirm communicator | N correct person | A append letter | B space | D delete | S speak spelling', size=.43)
    return out


def run(args):
    window = 'Tink / record, assess, choose'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 900)
    blank = np.full((680, 1100, 3), (23, 28, 35), np.uint8)
    cv2.putText(blank, 'Loading existing local video and audio models...', (40, 300), cv2.FONT_HERSHEY_SIMPLEX, .7, (220, 230, 240), 1)
    cv2.imshow(window, blank)
    cv2.waitKey(1)
    with ExitStack() as stack:
        speaker = Speaker(enabled=not args.no_tts)
        stack.callback(speaker.close)
        service = ClipAssessmentService(args.profile, speaker)
        stack.callback(service.close)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='clip-assessment')
        stack.callback(lambda: executor.shutdown(wait=True, cancel_futures=True))
        loading = executor.submit(service.prepare)
        while not loading.done():
            if cv2.waitKey(30) & 255 in (ord('q'), 27):
                return
        loading.result()
        print('Local models loaded; opening camera and microphone.', flush=True)
        video = cv2.VideoCapture(args.camera)
        stack.callback(video.release)
        if not video.isOpened():
            raise RuntimeError('Camera could not be opened')
        active, index, future, completed = [None], -1, None, None
        packet, result = None, None
        def raw_frame(frame, at_s):
            clip = active[0]
            if clip is not None:
                clip.add_frame(frame, at_s)
        live = LiveCapture(video, time_source=service.clock.now, frame_callback=raw_frame).start()
        stack.callback(live.close)
        capture = RecordedAudioCapture(service.clock)
        pump = AudioPump(capture, lambda: active[0]).start()
        stack.callback(pump.close)
        stack.enter_context(input_stream(capture, args.device))
        ready_at = service.clock.now()
        status = 'READY: SPACE starts a recording; ENTER finishes and sends it to Gemini.'
        auto_started, recording_stop = False, None
        print('READY: camera and microphone open. SPACE records; ENTER stops and assesses.', flush=True)
        folder_root = service.home / 'assessments'
        folder_root.mkdir(exist_ok=True)

        def assess(clip, captured, existing=None):
            try:
                current = existing if existing is not None else service.packet(clip, captured, pump.noise)
                try:
                    assessed = service.assess(clip, captured, current)
                    return current, assessed, None
                except GeminiError as error:
                    return current, None, str(error)
            except Exception:
                return existing, None, 'Local analysis failed; recording preserved. Check local model availability.'

        try:
            while True:
                item = live.read_after(index)
                if item is None:
                    break
                index, at_s, frame, _ = item
                now = service.clock.now()
                if future is None:
                    obs = service.observe(frame, at_s)
                    if active[0] is not None:
                        active[0].add_observation(obs)
                else:
                    # Avoid parallel inference/mutation while the finalized clip is assessed.
                    obs = {'t_capture': at_s, 'people': [], 'channels': {}, 'session_mode': 'integrated',
                           'asl': {}, 'target_status': 'assessing saved clip', 'person_id': None}
                if args.auto_record_seconds and not auto_started and now - ready_at >= 5:
                    key = 32
                    auto_started = True
                else:
                    key = -1
                if active[0] is not None and (now >= active[0].start_s + 30 or (recording_stop and now >= recording_stop)):
                    key = 13
                if future is not None and future.done():
                    packet, result, error = future.result()
                    future = None
                    if error:
                        status = error + ' | R retries the same saved clip; no automatic retry.'
                        print('ASSESSMENT ERROR:', error, flush=True)
                    else:
                        status = 'Three possibilities ready. 1-3 confirms and speaks; SPACE records another turn.'
                        print('ASSESSMENT READY:', completed[0].folder / 'assessment.json', flush=True)
                cv2.imshow(window, draw(frame, obs, status, active[0], now, pump.level, result, service.engine.spelled))
                pressed = cv2.waitKey(1) & 255
                key = pressed if pressed != 255 else key
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    key = ord('q')
                if pump.error:
                    raise RuntimeError('Audio capture writer failed; recording is incomplete')
                if key in (ord('q'), 27):
                    break
                if key == 32 and active[0] is None and future is None:
                    if speaker.overlaps(now, now, service.clock.origin):
                        status = 'Wait for speech playback and its echo guard to finish.'
                        continue
                    active[0] = ClipRecorder(folder_root / uuid.uuid4().hex, service.clock.session_id, now)
                    packet, result, completed = None, None, None
                    recording_stop = now + args.auto_record_seconds if args.auto_record_seconds and auto_started else None
                    status = 'Recording audio + video. ENTER stops and sends the clip to Gemini.'
                    print('RECORDING:', active[0].folder, flush=True)
                elif key in (10, 13) and active[0] is not None:
                    clip, active[0] = active[0], None
                    captured = clip.finish(service.clock.now())
                    captured['audio_queue_dropped_blocks'] = capture.dropped_blocks
                    captured['audio_timestamp_basis'] = capture.timestamp_basis
                    completed = clip, captured
                    future = executor.submit(assess, clip, captured)
                    status = 'Analyzing recorded signals, conversation and partner context; waiting for Gemini...'
                    print('CAPTURE SAVED:', clip.folder / 'capture.json', flush=True)
                elif key == ord('r') and completed and active[0] is None and future is None:
                    future = executor.submit(assess, *completed, packet)
                    status = 'Explicitly retrying the saved recording...'
                elif key in (ord('1'), ord('2'), ord('3')) and result and active[0] is None and future is None:
                    selected = result['options'][key - ord('1')]
                    service.confirm(selected, packet, now)
                    (completed[0].folder / 'confirmed.json').write_text(json.dumps({'option': selected,
                        'confirmed_at_s': now}, indent=2), encoding='utf-8')
                    result = None
                    status = 'Confirmed wording; speaking locally. SPACE records the next turn.'
                elif key == ord('p') and completed:
                    path = completed[0].folder / ('assessment.json' if result else 'assessment-input.json')
                    if path.is_file():
                        os.startfile(str(path))
                elif future is None and key == ord('y'):
                    service.visual.target.confirm_role()
                elif future is None and key == ord('n'):
                    service.visual.target.select_next(obs['people'])
                elif future is None and key == ord('a') and obs.get('ownership_valid') and obs.get('asl', {}).get('ready'):
                    service.engine.spelled += obs['asl']['best']
                elif future is None and key == ord('b'):
                    service.engine.spelled += ' '
                elif future is None and key == ord('d'):
                    service.engine.spelled = service.engine.spelled[:-1]
                elif key == ord('s') and active[0] is None and future is None and service.engine.spelled.strip():
                    service.engine.confirm_text(service.engine.spelled, [], now)
        finally:
            if active[0] is not None:
                clip, active[0] = active[0], None
                clip.finish(service.clock.now())
                print('Recording saved without a cloud request:', clip.folder, flush=True)
            cv2.destroyAllWindows()
