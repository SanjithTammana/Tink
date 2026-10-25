#!/usr/bin/env python
"""The Tink engine: the teammates' backend, served to the Django site over local HTTP.

Django starts this file in its own process (see tink/engine.py); you normally never run it by hand. It imports the
backend folder (Tink-codex-recorded-gemini-interpretations) without changing it:

  Normal page -> ClipAssessmentService: live camera + microphone, record a clip, send it with the measured
                 evidence to Gemini, get three possibilities back, confirm one.
  ASL page    -> asl.LetterClassifier / LetterStabilizer on one tracked hand: a ranked list of possible letters.

Why a separate process: camera, MediaPipe, torch and PortAudio are native code. If one crashes, the website
survives and shows what happened. Sessions run on the MAIN thread (MediaPipe crashes on macOS when started from a
background thread); the HTTP server runs on a daemon thread.

Routes (all JSON unless noted):  GET /api/state  /api/trace  /api/stream (MJPEG)
  POST /api/session/start {mode, sid}  /api/session/stop {sid}  /api/camera {on}
  POST /api/normal/{record,stop,retry,confirm,reject,role,person}   POST /api/asl/{add,space,delete,clear}
"""
import argparse
import faulthandler
import ipaddress
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

LOCAL_HOSTS = {'127.0.0.1', 'localhost', '[::1]', '::1'}
MAX_CLIP_S = 30.


def valid_sid(sid):
    return isinstance(sid, str) and 0 < len(sid) <= 200 and sid == sid.strip() and all(ord(c) >= 32 for c in sid)


def loopback_host(host):
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def cleanup(action):
    try:
        action()
    except Exception as exc:
        log('Resource cleanup failed:', type(exc).__name__)


def log(*parts):
    print(*parts, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Camera and frames
# ---------------------------------------------------------------------------------------------------------------
class SwitchCamera:
    """cv2.VideoCapture with an on/off switch. Off releases the device (the camera light goes out) and hands the
    pipeline black frames, so timing, audio and the rest of the session keep running."""

    def __init__(self, index, mirror=False):
        import cv2
        import numpy as np
        self.cv2, self.np, self.index, self.mirror = cv2, np, index, mirror
        self.cap, self.on, self.error = None, True, None
        self.shape = (480, 640, 3)
        self.lock = threading.Lock()
        if not self._open():
            raise RuntimeError('The camera could not be opened. Close other apps using it and allow camera access '
                               'for your terminal (macOS: System Settings > Privacy & Security > Camera).')

    def _open(self):
        cv2 = self.cv2
        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            return False
        self.cap = cap
        return True

    def set_on(self, on):
        with self.lock:
            if on and self.cap is None:
                if self._open():
                    self.on, self.error = True, None
                else:
                    self.on, self.error = False, 'The camera could not be reopened.'
            elif not on and self.cap is not None:
                self.cap.release()
                self.cap = None
                self.on, self.error = False, None
            return self.on

    def read(self):
        with self.lock:
            cap = self.cap
            # Native read and release must never run on the same handle concurrently.
            if cap is not None:
                ok, frame = cap.read()
            else:
                ok, frame = True, None
        if cap is None:
            time.sleep(1 / 30)
            return True, self.np.zeros(self.shape, self.np.uint8)
        if not ok:
            return False, None
        if self.mirror:
            frame = self.cv2.flip(frame, 1)
        self.shape = frame.shape
        return True, frame

    def get(self, prop):
        if prop == self.cv2.CAP_PROP_FPS:
            return 30.
        with self.lock:
            return self.cap.get(prop) if self.cap is not None else 0.

    def release(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None


class FrameBox:
    """The latest JPEG, for the MJPEG stream."""

    def __init__(self):
        self.cond, self.jpeg, self.count, self.closed = threading.Condition(), None, 0, False

    def publish(self, image):
        import cv2
        ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            with self.cond:
                self.jpeg, self.count = buf.tobytes(), self.count + 1
                self.cond.notify_all()

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()

    def wait(self, after):
        with self.cond:
            self.cond.wait_for(lambda: self.closed or self.count > after, timeout=2.)
            return (self.jpeg, self.count) if self.count > after else (None, after)


# ---------------------------------------------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------------------------------------------
class Session:
    mode = ''

    def __init__(self, opts):
        self.opts = opts
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.done = threading.Event()
        self.frames = FrameBox()
        self.phase, self.error, self.status = 'loading', None, 'Starting…'
        self.public = {}
        self.camera = None

    def execute(self):
        """Runs on the main thread until stopped."""
        try:
            if self.stop.is_set():
                self.phase = 'stopped'
                return
            self.run()
            self.phase = 'stopped'
        except Exception as exc:                      # shown on the page; the full traceback goes to the engine log
            traceback.print_exc()
            self.phase, self.error = 'error', f'{type(exc).__name__}: {exc}'
        finally:
            self.frames.close()
            self.done.set()

    def run(self):
        raise NotImplementedError

    def request_stop(self):
        self.stop.set()

    def state(self):
        with self.lock:
            base = {'mode': self.mode, 'phase': self.phase, 'error': self.error, 'status': self.status}
            return {**base, **self.public}

    def publish(self, **fields):
        with self.lock:
            self.public = fields

    def trace(self):
        return {'mode': self.mode, 'state': self.state()}

    def command(self, name, data):
        return {'error': 'unknown action'}, 404


def face_box(frame, obs, recording):
    """The plain webcam image with one soft box around the tracked face (or the person, if no face is found)."""
    import cv2
    image = frame.copy()
    h, w = image.shape[:2]
    box = None
    try:
        points = [(p[0], p[1]) for p in (obs.get('channels', {}).get('face', {}).get('landmarks_image') or [])
                  if 0 <= p[0] <= 1 and 0 <= p[1] <= 1]
        if len(points) > 50:
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            padx, pady = (max(xs) - min(xs)) * .22, (max(ys) - min(ys)) * .28
            box = (min(xs) - padx, min(ys) - pady, max(xs) + padx, max(ys) + pady * .6)
            box = tuple(int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(box))
        else:
            chosen = next((q for q in obs.get('people', []) if q.get('id') == obs.get('person_id')), None)
            if chosen:
                sx, sy = obs.get('image_size', [w, h])
                x1, y1, x2, y2 = chosen['bbox']
                box = (int(x1 * w / sx), int(y1 * h / sy), int(x2 * w / sx), int(y2 * h / sy))
    except (TypeError, ValueError, KeyError, IndexError):
        box = None
    if box:
        color = (119, 199, 232) if not recording else (110, 110, 255)         # gold; red while recording (BGR)
        cv2.rectangle(image, (max(0, box[0]), max(0, box[1])), (min(w - 1, box[2]), min(h - 1, box[3])), color, 2, cv2.LINE_AA)
    return image


class AudioPump:
    """Drains the microphone callback into the active clip, independently of video inference and the network."""

    def __init__(self, capture, active):
        from audio_layer import rms_db
        import numpy as np
        self.rms_db, self.np = rms_db, np
        self.capture, self.active = capture, active
        self.stop, self.thread = threading.Event(), None
        self.level, self.quiet, self.noise, self.error = -120., [], None, None

    def start(self):
        self.thread = threading.Thread(target=self._run, name='tink-audio-writer', daemon=True)
        self.thread.start()
        return self

    def _run(self):
        try:
            while not self.stop.is_set():
                for start, wave, bad in self.capture.drain():
                    self.level = self.rms_db(wave)
                    if self.noise is None:
                        self.quiet.append(self.level)
                        if len(self.quiet) >= 50:
                            self.noise = float(self.np.median(self.quiet))
                    clip = self.active()
                    if clip is not None:
                        clip.add_audio(wave, self.capture.at(start), self.capture.block_valid(start) and not bad)
                self.stop.wait(.01)
        except Exception as exc:
            self.error = exc

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2.)


class NormalSession(Session):
    """Record a clip with camera + microphone, assess it with Gemini, choose one of three possibilities."""
    mode = 'normal'

    def __init__(self, opts):
        super().__init__(opts)
        self.commands = queue.Queue()
        self.active = None                       # the ClipRecorder while recording
        self.completed = None                    # (clip, capture) of the last finished clip
        self.packet = None
        self.result = None
        self.job = None
        self.job_error = None
        self.said = []
        self.events = []
        self.cam_error = None
        self.result_id = None
        self.playback_until = 0.

    def note(self, text):
        self.status = text
        self.events.append({'at': time.strftime('%H:%M:%S'), 'text': text})
        del self.events[:-60]
        log('NORMAL:', text)

    def call(self, name, data, timeout=8.):
        """Run a command on the session thread (it owns the models) and wait for its answer."""
        if self.phase != 'running' or self.stop.is_set():
            return {'error': 'Tink is still starting.' if self.phase == 'loading' else 'The session is not running.'}, 409
        future = Future()
        self.commands.put((name, data, future))
        try:
            return future.result(timeout)
        except FutureTimeout:
            future.cancel()  # A queued command reported as failed must not execute later.
            return {'error': 'The engine took too long to respond.'}, 504

    command = call

    def run(self):
        with ExitStack() as resources:
            self._run_session(resources)

    def _run_session(self, resources):
        from assessment_capture import ClipRecorder, RecordedAudioCapture
        from assessment_service import ClipAssessmentService
        from gemini_client import GeminiError
        from tools.video_layer_lab import LiveCapture

        self.note('Loading the video and audio models (first run can take a minute)…')
        service = ClipAssessmentService('web', None)       # no local speaker: the browser speaks
        resources.callback(cleanup, service.close)
        service.prepare()
        if self.stop.is_set():
            return
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tink-assess')
        # This runs before service.close: even a running native/cloud job still owns service state.
        resources.callback(executor.shutdown, wait=True, cancel_futures=True)
        camera = self.camera = SwitchCamera(self.opts.camera)
        resources.callback(cleanup, camera.release)
        active_ref = lambda: self.active                    # noqa: E731
        capture = RecordedAudioCapture(service.clock)
        pump, stream, audio_error = None, None, None
        if self.opts.audio:
            audio_resources = resources.enter_context(ExitStack())
            try:
                from tools.audio_layer_lab import input_stream
                stream = input_stream(capture, self.opts.device)
                audio_resources.callback(cleanup, stream.close)
                stream.start()
                audio_resources.callback(cleanup, stream.stop)
                pump = AudioPump(capture, active_ref).start()
                audio_resources.callback(cleanup, pump.close)
            except Exception as exc:
                audio_error = f'Microphone unavailable: {exc}'
                audio_resources.close()
                stream, pump = None, None
                log('AUDIO:', audio_error)

        def raw_frame(frame, at_s):
            clip = self.active
            if clip is not None:
                clip.add_frame(frame, at_s)

        live = LiveCapture(camera, time_source=service.clock.now, frame_callback=raw_frame)
        resources.callback(cleanup, live.close)
        live.start()
        folder_root = service.home / 'assessments'
        folder_root.mkdir(exist_ok=True)

        def assess(clip, captured, existing):
            try:
                current = existing if existing is not None else service.packet(clip, captured, pump.noise if pump else None)
                try:
                    return current, service.assess(clip, captured, current), None
                except GeminiError as exc:
                    return current, None, str(exc)
            except Exception as exc:
                traceback.print_exc()
                return existing, None, f'Local analysis failed; the recording was kept. ({type(exc).__name__}: {exc})'

        obs, index = {'people': [], 'channels': {}}, -1
        self.note('Ready. Press Record, then Stop to have Tink assess the clip.')
        try:
            while not self.stop.is_set():
                item = live.read_after(index)
                if item is None:
                    raise RuntimeError('The camera stopped delivering frames.')
                index, at_s, frame, _ = item
                now = service.clock.now()
                if self.job is None:
                    obs = service.observe(frame, at_s)
                    if self.active is not None:
                        self.active.add_observation(obs)
                else:                                    # avoid parallel inference while the saved clip is assessed
                    obs = {'t_capture': at_s, 'people': [], 'channels': {}, 'session_mode': 'integrated', 'asl': {},
                           'target_status': 'assessing the saved clip', 'person_id': None}
                if self.active is not None and now >= self.active.start_s + MAX_CLIP_S:
                    self._finish_recording(service, capture, executor, assess)
                self._drain_commands(service, capture, executor, assess, ClipRecorder, folder_root, camera, obs)
                if self.job is not None and self.job.done():
                    self.packet, self.result, error = self.job.result()
                    self.job = None
                    if error:
                        self.result_id = None
                        self.job_error = error
                        self.note(error + ' You can retry the same recording.')
                    else:
                        self.result_id = uuid.uuid4().hex
                        self.job_error = None
                        self.note('Three possibilities are ready. Choose one, or record another turn.')
                if pump is not None and pump.error is not None:
                    raise RuntimeError('Audio capture failed; the recording would be incomplete.')
                self.frames.publish(face_box(frame, obs, self.active is not None))
                self._publish(service, obs, now, pump, audio_error, camera)
                self.phase = 'running'          # only once there is something to show
        finally:
            if self.active is not None:
                clip, self.active = self.active, None
                try:
                    clip.finish(service.clock.now())
                except Exception:
                    pass
            # Resolve callers waiting on commands that can no longer be applied.
            while True:
                try:
                    _, _, pending = self.commands.get_nowait()
                except queue.Empty:
                    break
                if pending.set_running_or_notify_cancel():
                    pending.set_result(({'error': 'The session has stopped.'}, 409))

    # -- commands (always on the session thread) -------------------------------------------------------------------
    def _finish_recording(self, service, capture, executor, assess):
        clip, self.active = self.active, None
        captured = clip.finish(service.clock.now())
        captured['audio_queue_dropped_blocks'] = capture.dropped_blocks
        captured['audio_timestamp_basis'] = capture.timestamp_basis
        self.completed = (clip, captured)
        self.packet = self.result = None
        self.result_id = None
        self.job = executor.submit(assess, clip, captured, None)
        self.note('Clip saved. Analysing the recorded signals and asking Gemini (this takes a few seconds)…')

    def _drain_commands(self, service, capture, executor, assess, ClipRecorder, folder_root, camera, obs):
        while True:
            try:
                name, data, future = self.commands.get_nowait()
            except queue.Empty:
                return
            if not future.set_running_or_notify_cancel():
                continue
            if self.stop.is_set():
                future.set_result(({'error': 'The session has stopped.'}, 409))
                continue
            try:
                future.set_result(self._handle(name, data, service, capture, executor, assess, ClipRecorder, folder_root, camera, obs))
            except Exception as exc:
                traceback.print_exc()
                future.set_result(({'error': f'{type(exc).__name__}: {exc}'}, 500))

    def _handle(self, name, data, service, capture, executor, assess, ClipRecorder, folder_root, camera, obs):
        now = service.clock.now()
        busy = self.job is not None
        if name == 'playback':
            if type(data.get('active')) is not bool:
                return {'error': 'active must be a boolean.'}, 400
            if data['active'] and self.active is not None:
                return {'error': 'Stop recording before speech playback.'}, 409
            self.playback_until = now + (15. if data['active'] else .7)
            return {'ok': True, 'active': data['active']}, 200
        if name == 'camera':
            on = bool(data.get('on'))
            camera.set_on(on)
            self.cam_error = camera.error
            self.note('Camera turned on.' if camera.on else 'Camera turned off: Tink now sees only black frames.')
            return {'ok': True, 'on': camera.on}, 200
        if name == 'record':
            if self.active is not None or busy:
                return {'error': 'Already recording or assessing.'}, 409
            if now < self.playback_until:
                return {'error': 'Wait for speech playback and its echo guard to finish.'}, 409
            self.active = ClipRecorder(folder_root / uuid.uuid4().hex, service.clock.session_id, now)
            self.result = self.packet = self.job_error = None
            self.result_id = None
            self.note('Recording camera and microphone. Press Stop when you are done.')
            return {'ok': True}, 200
        if name == 'stop':
            if self.active is None:
                return {'error': 'Not recording.'}, 409
            self._finish_recording(service, capture, executor, assess)
            return {'ok': True}, 200
        if name == 'retry':
            if not self.completed or self.active is not None or busy:
                return {'error': 'Nothing to retry.'}, 409
            self.job = executor.submit(assess, *self.completed, self.packet)
            self.result = None
            self.result_id = None
            self.note('Retrying the saved recording…')
            return {'ok': True}, 200
        if name == 'confirm':
            options = (self.result or {}).get('options') or []
            index = data.get('index')
            if type(index) is not int or not 0 <= index < len(options):
                return {'error': 'No such option.'}, 400
            selected = options[index]
            if self.active is not None or busy:
                return {'error': 'Wait for the current step to finish.'}, 409
            clip_id = (self.packet or {}).get('clip_id')
            if (not self.result_id or not clip_id or data.get('clip_id') != clip_id
                    or data.get('option_id') != selected['id'] or data.get('result_id') != self.result_id):
                return {'error': 'That choice belongs to an older result. Refresh the current options.'}, 409
            receipt = {'ok': True, 'text': selected['text'], 'kind': selected['kind'],
                       'clip_id': clip_id, 'option_id': selected['id'], 'result_id': self.result_id}
            service.confirm(selected, self.packet, now)
            if self.completed:
                (self.completed[0].folder / 'confirmed.json').write_text(
                    json.dumps({'option': selected, 'confirmed_at_s': now, 'receipt': receipt}, indent=2), encoding='utf-8')
            self.said.append({'text': selected['text'], 'kind': selected['kind']})
            self.result = None
            self.result_id = None
            self.note('Confirmed: ' + selected['text'])
            return receipt, 200
        if name == 'reject':
            self.result = None
            self.result_id = None
            self.note('None of those fit. Record another turn to try again.')
            return {'ok': True}, 200
        if name == 'role' and not busy:
            service.visual.target.confirm_role()
            self.note('Confirmed who the communicator is.')
            return {'ok': True}, 200
        if name == 'person' and not busy:
            service.visual.target.select_next(obs.get('people', []))
            self.note('Switched to the next person in view.')
            return {'ok': True}, 200
        return {'error': 'That is not available right now.'}, 409

    # -- what the page sees ----------------------------------------------------------------------------------------
    def _publish(self, service, obs, now, pump, audio_error, camera):
        result = self.result
        view = None
        if result:
            context = result.get('partner_context', {})
            view = {
                'clip_id': (self.packet or {}).get('clip_id'),
                'summary': result['assessment']['summary'],
                'observations': [o['text'] for o in result['assessment'].get('observations', [])][:6],
                'limitations': result['assessment'].get('limitations', [])[:4],
                'partner': {'summary': context.get('summary', ''), 'topic': context.get('topic', ''),
                            'pending_question': context.get('pending_question', ''),
                            'uncertainties': context.get('uncertainties', [])[:4]},
                'transcript': [{'text': t['text'], 'role': t.get('role_hypothesis')} for t in result.get('transcript', [])][:8],
                'options': [{'id': o['id'], 'kind': o['kind'], 'text': o['text'], 'reason': o['reason']} for o in result['options']],
            }
        clip = self.active
        people = [{'id': p['id'], 'selected': p['id'] == obs.get('person_id')} for p in obs.get('people', [])]
        capture = self.completed[1] if self.completed else None
        self.publish(
            camera={'on': camera.on, 'wanted': camera.on, 'error': camera.error or self.cam_error},
            audio={'enabled': bool(self.opts.audio), 'ok': pump is not None, 'level_db': round(pump.level, 1) if pump else None,
                   'error': audio_error},
            recording={'on': clip is not None, 'elapsed': round(max(0., now - clip.start_s), 1) if clip else 0., 'max': MAX_CLIP_S},
            busy=self.job is not None, people=people, target=obs.get('target_status'),
            result_id=self.result_id, playback={'active': now < self.playback_until, 'until_s': self.playback_until},
            owned=bool(obs.get('ownership_valid')), warnings=list(service.warnings), result=view,
            error_detail=self.job_error, can_retry=bool(self.completed and self.job is None and self.active is None and view is None),
            said=self.said[-8:], events=self.events[-12:],
            clip={'clip_id': capture['clip_id'], 'seconds': round(capture['duration_s'], 1), 'clock_valid': capture['clock_valid'],
                  'frames': capture['video_frames'], 'audio_gaps': len(capture['audio_gaps'])} if capture else None)

    def trace(self):
        return {'mode': 'normal', 'state': self.state(), 'events': list(self.events),
                'capture': self.completed[1] if self.completed else None,
                'packet': self.packet, 'assessment': self.result, 'confirmed': self.said}


class ASLSession(Session):
    """Fingerspelling: a ranked list of the letters the hand might be, add any of them."""
    mode = 'asl'
    LIST = 8

    def __init__(self, opts):
        super().__init__(opts)
        self.message = ''
        self.stab = None

    def ranking(self):
        history = list(self.stab.history)
        totals = {}
        for snapshot in history:
            for label, prob in snapshot.items():
                totals[label] = totals.get(label, 0.) + prob / len(history)
        return sorted(totals.items(), key=lambda z: z[1], reverse=True)[:self.LIST]

    def command(self, name, data):
        if self.phase != 'running':
            return {'error': 'Tink is still starting.' if self.phase == 'loading' else 'The session is not running.'}, 409
        with self.lock:
            if name == 'add':
                letter = str(data.get('letter') or '').strip().upper()
                if len(letter) != 1 or letter not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    return {'ok': False, 'added': None}, 200
                self.message += letter
                self.stab.clear()
                self.status = f'Added {letter}'
                return {'ok': True, 'added': letter, 'message': self.message}, 200
            if name == 'space':
                self.message += ' '
            elif name == 'delete':
                self.message = self.message[:-1]
            elif name == 'clear':
                self.message = ''
            else:
                return {'error': 'unknown action'}, 404
            return {'ok': True, 'message': self.message}, 200

    def run(self):
        import cv2
        import numpy as np
        import mediapipe as mp
        from asl import LetterClassifier, LetterStabilizer, hand_box

        self.status = 'Loading the letter model (first run downloads it)…'
        log('ASL:', self.status)
        hands_api = mp.solutions.hands
        classifier = LetterClassifier()
        camera = self.camera = SwitchCamera(self.opts.camera, mirror=True)
        self.stab = LetterStabilizer(6, .72, .16)
        last_infer, last_box = 0., None
        self.status = 'Show one hand to the camera.'
        try:
            with hands_api.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=.65,
                                 min_tracking_confidence=.6) as hands:
                while not self.stop.is_set():
                    ok, frame = camera.read()
                    if not ok:
                        raise RuntimeError('The camera stopped delivering frames.')
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    found = hands.process(rgb).multi_hand_landmarks or []
                    box, status = None, None
                    if len(found) == 1:
                        pts = found[0].landmark
                        box = hand_box(np.array([p.x * w for p in pts]), np.array([p.y * h for p in pts]), w, h)
                    with self.lock:
                        if box is None:
                            self.stab.clear()
                            last_box = None
                            status = 'Show exactly one hand' if len(found) != 1 else 'Move your hand into view'
                        else:
                            if last_box is not None and np.linalg.norm(np.array(box, float) - np.array(last_box, float)) > max(h, w) * .20:
                                self.stab.clear()           # a big jump must not carry confidence from an old gesture
                            last_box = box
                            x1, y1, x2, y2 = box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 214, 233), 2)
                            if time.monotonic() - last_infer >= .12:
                                last_infer = time.monotonic()
                                try:
                                    self.stab.add(classifier.scores(rgb[y1:y2, x1:x2]))
                                    status = 'Reading your hand'
                                except Exception as exc:
                                    self.stab.clear()
                                    status = 'Could not read the hand: ' + str(exc)[:60]
                        ranked = self.ranking()
                        if status:
                            self.status = status
                    self.frames.publish(frame)
                    self.publish(message=self.message, ranked=[{'label': l, 'p': round(p, 3)} for l, p in ranked],
                                 camera={'on': True, 'wanted': True, 'error': None})
                    self.phase = 'running'
        finally:
            camera.release()

    def trace(self):
        return {'mode': 'asl', 'state': self.state()}


SESSIONS = {'normal': NormalSession, 'asl': ASLSession}


# ---------------------------------------------------------------------------------------------------------------
# Manager and HTTP
# ---------------------------------------------------------------------------------------------------------------
class Manager:
    def __init__(self, opts):
        self.opts, self.lock = opts, threading.RLock()
        self.session, self.owner = None, None
        self.jobs = queue.Queue()

    def start(self, mode, sid):
        with self.lock:
            if mode not in SESSIONS or not valid_sid(sid):
                return None
            old = self.session
            if old is not None and old.mode == mode and not old.done.is_set() and not old.stop.is_set() and self.owner == sid:
                return old
            if old is not None:
                old.request_stop()
            self.session, self.owner = SESSIONS[mode](self.opts), sid
            self.jobs.put(self.session)
            return self.session

    def stop(self, sid):
        with self.lock:
            if not valid_sid(sid) or self.session is None or self.owner != sid:
                return False
            self.session.request_stop()
            return True

    def owns(self, sid):
        with self.lock:
            return valid_sid(sid) and self.session is not None and sid == self.owner

    def session_for(self, sid):
        with self.lock:
            return self.session if valid_sid(sid) and sid == self.owner else None

    def state(self):
        with self.lock:
            data = self.session.state() if self.session else {'mode': None, 'phase': 'idle', 'error': None}
            return {**data, 'owner_sid': self.owner}


def make_handler(manager):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def _allowed(self):
            try:
                if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                    return False
                host = urlparse('//' + (self.headers.get('Host') or ''))
                if (not loopback_host(host.hostname or '') or host.username or host.password
                        or (host.port or 80) != self.server.server_port):
                    return False
                origin = self.headers.get('Origin')
                if origin:
                    parsed = urlparse(origin)
                    if (parsed.scheme != 'http' or parsed.username or parsed.password
                            or parsed.hostname != host.hostname or (parsed.port or 80) != (host.port or 80)
                            or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
                        return False
            except ValueError:
                return False
            if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                return False
            return True

        def _json(self, data, status=200):
            body = json.dumps(data, default=str).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            try:
                size = int(self.headers.get('Content-Length') or 0)
                if not 0 <= size <= 65536:
                    self.close_connection = True
                    return None
                data = json.loads(self.rfile.read(size) or b'{}')
                return data if isinstance(data, dict) else None
            except (ValueError, UnicodeError):
                return None

        def do_GET(self):
            if not self._allowed():
                return self._json({'error': 'forbidden'}, 403)
            path = urlparse(self.path).path
            if path == '/api/state':
                return self._json(manager.state())
            if path == '/api/trace':
                session = manager.session
                return self._json(session.trace() if session else {'error': 'no session'}, 200 if session else 409)
            if path == '/api/stream':
                return self._stream()
            self._json({'error': 'not found'}, 404)

        def _stream(self):
            session = manager.session
            if session is None or session.done.is_set():
                return self._json({'error': 'no active session'}, 409)
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            count = 0
            try:
                while not session.done.is_set():
                    jpeg, count = session.frames.wait(count)
                    if jpeg is None:
                        continue
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpeg) + jpeg + b'\r\n')
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            self.close_connection = True

        def do_POST(self):
            if not self._allowed():
                return self._json({'error': 'forbidden'}, 403)
            path, data = urlparse(self.path).path, self._body()
            if data is None or not valid_sid(data.get('sid')):
                return self._json({'error': 'A valid session id and JSON body are required.'}, 400)
            if path == '/api/session/start':
                mode = data.get('mode')
                if mode not in SESSIONS:
                    return self._json({'error': 'mode must be "normal" or "asl"'}, 400)
                manager.start(mode, str(data.get('sid') or ''))
                return self._json(manager.state())
            if path == '/api/session/stop':
                if not manager.stop(data['sid']):
                    return self._json({'error': 'This page does not own the active session.'}, 409)
                return self._json(manager.state())
            session = manager.session_for(data['sid'])
            if session is None:
                return self._json({'error': 'This page does not own the active session.'}, 409)
            if session is None or session.done.is_set():
                return self._json({'error': 'no active session'}, 409)
            if path == '/api/camera':
                if not isinstance(session, NormalSession):
                    return self._json({'error': 'the camera switch is on the Normal page'}, 400)
                payload, status = session.call('camera', data)
                return self._json(payload, status)
            for prefix, kind in (('/api/normal/', NormalSession), ('/api/asl/', ASLSession)):
                if path.startswith(prefix):
                    if not isinstance(session, kind):
                        return self._json({'error': 'that page is not running'}, 409)
                    payload, status = session.command(path[len(prefix):], data)
                    return self._json(payload, status)
            self._json({'error': 'not found'}, 404)

    return Handler


def _windows_parent_waiter(parent):
    """Hold the actual supervisor, not the Windows venv trampoline or a reused PID."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x00100000, False, parent)  # SYNCHRONIZE only
    if not handle:
        return None
    return (lambda milliseconds: kernel.WaitForSingleObject(handle, milliseconds),
            lambda: kernel.CloseHandle(handle))


def watch_parent(parent=None):
    parent = parent if parent is not None else os.getppid()
    if os.name == 'nt':
        watcher = _windows_parent_waiter(parent)
        if watcher is None:
            os._exit(0)  # The supervisor has gone or cannot be safely monitored.
        wait, close = watcher
        try:
            while wait(1000) == 258:  # WAIT_TIMEOUT; zero means process exit.
                pass
        finally:
            close()
        os._exit(0)
    while True:
        time.sleep(1.)
        try:
            os.kill(parent, 0)  # Check the explicitly supplied supervisor, not getppid().
        except ProcessLookupError:
            os._exit(0)
        except PermissionError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', required=True, help='folder holding the backend modules')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--device', type=int, default=None, help='microphone device index (default: system default)')
    parser.add_argument('--no-audio', dest='audio', action='store_false')
    parser.add_argument('--exit-with-parent', action='store_true')
    parser.add_argument('--parent-pid', type=int, help='Django supervisor PID, passed through Windows venv launchers')
    opts = parser.parse_args()
    if not loopback_host(opts.host):
        parser.error('The engine may only bind to a loopback address.')
    if opts.parent_pid is not None and opts.parent_pid <= 0:
        parser.error('--parent-pid must be a positive process id.')
    faulthandler.enable()
    backend = Path(opts.backend).resolve()
    os.chdir(backend)                     # the backend finds its config/ and data/ relative to itself
    sys.path.insert(0, str(backend))
    manager = Manager(opts)
    server = ThreadingHTTPServer((opts.host, opts.port), make_handler(manager))
    if not ipaddress.ip_address(server.server_address[0]).is_loopback:
        server.server_close()
        parser.error('The engine may only bind to a loopback address.')
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name='tink-http', daemon=True).start()
    if opts.exit_with_parent:
        threading.Thread(target=watch_parent, args=(opts.parent_pid,), daemon=True).start()
    log(f'Tink engine listening on http://{opts.host}:{opts.port} (backend: {backend})')
    try:
        while True:
            try:
                session = manager.jobs.get(timeout=.25)
            except queue.Empty:
                continue
            session.execute()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
