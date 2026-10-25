"""Video evidence for Tink: tracked people (YOLO11n-pose + ByteTrack), MediaPipe gestures + face blendshapes,
and EnvisionHGdetector's LightGBM Gesture/NoGesture score.

Self-test: python video.py --video sample.mp4 [--frames N] [--no-yolo] [--no-envision]
"""
import argparse
import contextlib
import importlib
import importlib.util
import io
import sys
import threading
import types
import math
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
TORCH_THREADS = 2   # YOLO models run in separate threads; uncapped, each grabs every core and they starve
MODELS.mkdir(exist_ok=True)
GESTURE_URL = ('https://storage.googleapis.com/mediapipe-models/gesture_recognizer/'
               'gesture_recognizer/float16/1/gesture_recognizer.task')
FACE_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
            'face_landmarker/float16/1/face_landmarker.task')


def model_bytes(url, path):
    # read the .task into memory: after a mp.solutions graph exists, mediapipe 0.10.7 prefixes file paths with site-packages
    return Path(download_once(url, path)).read_bytes()


def download_once(url, path):
    if path.exists() and path.stat().st_size > 100_000:
        return str(path)
    print('Downloading:', path.name)
    temp = path.with_suffix('.part')
    try:
        urllib.request.urlretrieve(url, temp)
        if temp.stat().st_size < 100_000:
            raise RuntimeError('Model download unexpectedly small')
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    return str(path)


def envision_lightgbm_class():
    """envisionhgdetector's real-time LightGBM model, loaded without the package __init__.

    The __init__ imports the TensorFlow CNN too, and its pinned tensorflow-cpu has no macOS build. The LightGBM
    path only needs config.py and model_lightgbm.py (OpenCV, MediaPipe, NumPy, joblib), so a bare package
    stub lets just those two modules run. Install the package with pip --no-deps.
    """
    if 'envisionhgdetector' not in sys.modules:
        spec = importlib.util.find_spec('envisionhgdetector')
        if spec is None:
            raise ImportError('envisionhgdetector is not installed (pip install --no-deps envisionhgdetector==3.0.8)')
        pkg = types.ModuleType('envisionhgdetector')
        pkg.__path__ = list(spec.submodule_search_locations)
        pkg.__spec__ = spec
        pkg.__file__ = spec.origin
        sys.modules['envisionhgdetector'] = pkg
    return importlib.import_module('envisionhgdetector.model_lightgbm').LightGBMGestureModel


class EnvisionGesture:
    # Package assumptions: BGR frames in (it converts to RGB itself), one person, front view,
    # 5-frame world-landmark window, trained on 25-30 fps video (velocity features are per frame).
    def __init__(self, smooth_frames=5):
        LightGBMGestureModel = envision_lightgbm_class()
        with contextlib.redirect_stdout(io.StringIO()):  # its load message has a unicode check mark that breaks cp1252 stdout
            self.model = LightGBMGestureModel()
        if self.model.model is None:
            raise RuntimeError('envisionhgdetector LightGBM model file not found')
        names = [self.model.standardize_gesture_name(str(c)) for c in self.model.label_encoder.classes_]
        if 'GESTURE' not in names:
            raise RuntimeError(f'no Gesture class in {list(self.model.label_encoder.classes_)}')
        self.gesture_idx = names.index('GESTURE')  # every other class (NoGesture, Move) counts as NoGesture
        self.history = deque(maxlen=max(1, int(smooth_frames)))
        self.warned = False

    def analyze(self, frame_bgr):
        none = {'probs': None, 'label': None}
        try:
            buffer = self.model.landmarks_buffer
            before = len(buffer)
            features = self.model.extract_features_from_frame(frame_bgr)
            if features is None:
                if len(buffer) == before:  # pose lost: drop stale window and averages
                    self.model.reset_buffer()
                    self.history.clear()
                    return {'status': 'no pose', **none}
                return {'status': 'warming up', **none}
            p = float(self.model.predict(features)[0][self.gesture_idx])
            self.history.append(p)
            p = float(np.mean(self.history))
            return {'status': 'ok', 'probs': {'Gesture': p, 'NoGesture': 1.0 - p},
                    'label': 'Gesture' if p >= 0.5 else 'NoGesture'}
        except Exception as e:
            if not self.warned:
                print('Envision error:', e)
                self.warned = True
            return {'status': f'error: {e}', **none}

    def reset(self):
        self.model.reset_buffer()
        self.history.clear()

    def close(self):
        self.model.holistic.close()
        self.model.holistic = None  # its __del__ would close it a second time


KP = {'nose': 0, 'l_shoulder': 5, 'r_shoulder': 6, 'l_elbow': 7, 'r_elbow': 8, 'l_wrist': 9, 'r_wrist': 10}


def reversals(values, step):
    """Direction reversals in a 1-D series, with hysteresis: a swing counts only once the value has moved at
    least `step` back from the last extreme, so tracking jitter below `step` never counts."""
    count, direction, extreme = 0, 0, values[0]
    for v in values[1:]:
        if direction == 0:
            if abs(v - extreme) >= step:
                direction, extreme = (1 if v > extreme else -1), v
        elif direction > 0:
            if v > extreme:
                extreme = v
            elif extreme - v >= step:
                count, direction, extreme = count + 1, -1, v
        else:
            if v < extreme:
                extreme = v
            elif v - extreme >= step:
                count, direction, extreme = count + 1, 1, v
    return count


def head_angles(matrix):
    """Pitch and yaw in degrees from MediaPipe's 4x4 facial transformation matrix (rotation part)."""
    r = np.asarray(matrix)[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    return pitch, yaw


class HeadGesture:
    """Nod / shake from head pitch and yaw over the last window_s seconds.

    Count reversals in pitch and in yaw (hysteresis step_deg). Dominance d = pitch range / (pitch + yaw range)
    separates the axes: nod = min(1, pitch reversals / 2) * d, shake = min(1, yaw reversals / 2) * (1 - d),
    still = the rest. Two reversals is one full down-up-down (or left-right-left). The 4 degree step and
    1.5 s window are heuristics; the sources we found suggest about 5 degrees within 1.5 s.
    """

    def __init__(self, window_s=1.5, step_deg=4.0):
        self.window_s, self.step = window_s, step_deg
        self.hist = deque()

    def reset(self):
        self.hist.clear()

    def update(self, t, pitch, yaw):
        self.hist.append((t, pitch, yaw))
        while t - self.hist[0][0] > self.window_s:
            self.hist.popleft()
        if len(self.hist) < 4 or self.hist[-1][0] - self.hist[0][0] < 0.5:
            return {'status': 'warming up', 'q': None, 'pitch': pitch, 'yaw': yaw}
        ps, ys = [h[1] for h in self.hist], [h[2] for h in self.hist]
        n_p, n_y = reversals(ps, self.step), reversals(ys, self.step)
        d = (max(ps) - min(ps)) / (max(ps) - min(ps) + max(ys) - min(ys) + 1e-6)
        nod, shake = min(1.0, n_p / 2) * d, min(1.0, n_y / 2) * (1 - d)
        return {'status': 'ok', 'pitch': round(pitch, 1), 'yaw': round(yaw, 1),
                'q': {'nod': round(nod, 3), 'shake': round(shake, 3), 'still': round(max(0.0, 1 - nod - shake), 3)}}


def crop_region(frame, bbox, margin=0.15):
    """(x1, y1, x2, y2) of the person box grown by margin on each side, clipped to the frame; None if tiny."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return None if x2 - x1 < 32 or y2 - y1 < 32 else (x1, y1, x2, y2)


def crop_box(frame, bbox, margin=0.15):
    """Person box grown by margin on each side, clipped to the frame (the whole frame if the box is tiny)."""
    r = crop_region(frame, bbox, margin)
    return frame if r is None else np.ascontiguousarray(frame[r[1]:r[3], r[0]:r[2]])


class PersonTracker:
    """Every person: box, persistent ID (ByteTrack) and COCO body keypoints (YOLO11n-pose). One person is the
    target (the user); only the target's crop goes to the single-person models (Envision, hands, face).

    Hand kinematics per person, from keypoints. Wrists are measured relative to the shoulder midpoint in
    shoulder widths, so walking or leaning does not count as hand motion and distance to the camera cancels:
    - speed: wrist displacement over the last 0.6 s, shoulder widths per second;
    - raised: wrist above its shoulder;
    - hold (HOLD_S window): raised fraction * exp(-path speed / HOLD_SPEED) * min(1, span / HOLD_S);
    - wave (WAVE_S window): raised fraction * min(1, horizontal reversals / 2), hysteresis WAVE_STEP.
    hold and wave are 0..1 scores per person (max over the two hands). The constants are heuristics.
    """
    HOLD_S, HOLD_SPEED = 1.0, 0.5
    WAVE_S, WAVE_STEP = 1.5, 0.15

    def __init__(self, model='yolo11n-pose.pt', imgsz=320, conf=0.4, history_s=2.0):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(TORCH_THREADS)
        self.model = YOLO(model)
        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, verbose=False)   # warm-up
        self.imgsz, self.conf, self.history_s = imgsz, conf, history_s
        self.history = {}           # track id -> deque of (t, {'l_wrist': (x, y), 'r_wrist': ..., 'scale': sw})
        self.target_id = None
        self.manual = False         # True once the user picked a target; keep it while it is visible
        self.people = []

    def update(self, frame, t=None):
        t = time.monotonic() if t is None else t
        r = self.model.track(frame, persist=True, tracker='bytetrack.yaml', classes=[0], conf=self.conf,
                             imgsz=self.imgsz, verbose=False)[0]
        people = []
        if r.boxes is not None and r.boxes.id is not None:
            kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            for i, (box, tid, score) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy(),
                                                      r.boxes.conf.cpu().numpy())):
                person = {'id': int(tid), 'bbox': [int(v) for v in box], 'score': float(score)}
                if kps is not None:
                    person.update(self.kinematics(int(tid), kps[i], t))
                people.append(person)
        ids = {p['id'] for p in people}
        for tid in list(self.history):
            if tid not in ids and t - self.history[tid][-1][0] > 3.0:
                del self.history[tid]
        if self.target_id not in ids:
            self.manual = False
            area = lambda p: (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1])
            self.target_id = max(people, key=area)['id'] if people else None
        self.people = people
        return people

    def kinematics(self, tid, kp, t):
        pt = lambda name: kp[KP[name]] if kp[KP[name]][2] > 0.3 else None   # (x, y, conf)
        ls, rs = pt('l_shoulder'), pt('r_shoulder')
        out = {'hands': {}, 'signals': {'hold': 0.0, 'wave': 0.0}}
        if ls is None or rs is None:
            return out
        scale = float(np.hypot(*(ls[:2] - rs[:2])))
        if scale <= 1:
            return out
        mid = (ls[:2] + rs[:2]) / 2
        now = {}
        for side, wrist, shoulder in (('left', 'l_wrist', ls), ('right', 'r_wrist', rs)):
            w = pt(wrist)
            if w is None:
                continue
            raised = bool(w[1] < shoulder[1])
            now[side] = ((w[:2] - mid) / scale, raised)
            out['hands'][side] = {'raised': raised, 'speed': None}
        hist = self.history.setdefault(tid, deque())
        hist.append((t, now))
        while hist and t - hist[0][0] > self.history_s:
            hist.popleft()
        for side in out['hands']:
            pts = [(ti, s[side][0], s[side][1]) for ti, s in hist if side in s]
            recent = [p for p in pts if t - p[0] <= 0.6]
            if len(recent) >= 2 and recent[-1][0] > recent[0][0]:
                d = float(np.hypot(*(recent[-1][1] - recent[0][1])))
                out['hands'][side]['speed'] = round(d / (recent[-1][0] - recent[0][0]), 2)
            hold, wave = self.arm_signals(pts, t)
            out['signals']['hold'] = max(out['signals']['hold'], hold)
            out['signals']['wave'] = max(out['signals']['wave'], wave)
        out['pointing'] = self.pointing(pt, ls, rs, scale, out['hands'])
        return out

    @staticmethod
    def pointing(pt, ls, rs, scale, hands):
        """The most point-like arm: origin (wrist, frame pixels), unit direction (elbow -> wrist) and a 0..1
        score = straight arm * extended * not hanging down * steady. Each factor is a clipped ramp:
          straight: cos(upper arm, forearm) from 0.3 to 0.8; extended: wrist-shoulder from 0.9 to 1.4 shoulder
          widths; not down: forearm y-component from 0.8 down to 0.2 (image y points down); steady:
          exp(-wrist speed / 1.0 shoulder widths per second)."""
        ramp = lambda v, lo, hi: float(min(1.0, max(0.0, (v - lo) / (hi - lo))))
        best = {'score': 0.0}
        for side, shoulder, (wk, ek) in (('left', ls, ('l_wrist', 'l_elbow')), ('right', rs, ('r_wrist', 'r_elbow'))):
            w, e = pt(wk), pt(ek)
            if w is None or e is None:
                continue
            upper, fore = e[:2] - shoulder[:2], w[:2] - e[:2]
            nu, nf = float(np.hypot(*upper)), float(np.hypot(*fore))
            if nu < 1 or nf < 1:
                continue
            d = fore / nf
            straight = ramp(float(upper @ fore) / (nu * nf), 0.3, 0.8)
            extended = ramp(float(np.hypot(*(w[:2] - shoulder[:2]))) / scale, 0.9, 1.4)
            not_down = ramp(-float(d[1]), -0.8, -0.2)
            speed = hands.get(side, {}).get('speed') or 0.0
            score = straight * extended * not_down * math.exp(-speed / 1.0)
            if score > best['score']:
                best = {'score': round(score, 3), 'side': side, 'origin': [float(w[0]), float(w[1])],
                        'dir': [float(d[0]), float(d[1])]}
        return best

    def arm_signals(self, pts, t):
        """(hold, wave) scores for one wrist from [(t, xy relative to shoulders, raised)]."""
        hold = wave = 0.0
        win = [p for p in pts if t - p[0] <= self.HOLD_S]
        if len(win) >= 3 and win[-1][0] - win[0][0] >= 0.3:
            span = win[-1][0] - win[0][0]
            path = sum(float(np.hypot(*(b[1] - a[1]))) for a, b in zip(win, win[1:])) / span
            hold = np.mean([p[2] for p in win]) * math.exp(-path / self.HOLD_SPEED) * min(1.0, span / self.HOLD_S)
        win = [p for p in pts if t - p[0] <= self.WAVE_S]
        if len(win) >= 4:
            wave = np.mean([p[2] for p in win]) * min(1.0, reversals([p[1][0] for p in win], self.WAVE_STEP) / 2)
        return round(float(hold), 3), round(float(wave), 3)

    def target(self):
        return next((p for p in self.people if p['id'] == self.target_id), None)

    def select_next(self):
        """Cycle the target to the next visible person (left to right)."""
        ordered = sorted(self.people, key=lambda p: p['bbox'][0])
        if not ordered:
            return None
        ids = [p['id'] for p in ordered]
        self.target_id = ids[(ids.index(self.target_id) + 1) % len(ids)] if self.target_id in ids else ids[0]
        self.manual = True
        return self.target_id


class ObjectDetector:
    """Everyday objects in the scene (YOLO11n, the 80 COCO classes, people excluded): pointing targets."""

    def __init__(self, model='yolo11n.pt', imgsz=416, conf=0.35):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.imgsz, self.conf = imgsz, conf
        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, verbose=False)   # warm-up

    def detect(self, frame):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return [{'label': self.model.names[int(c)], 'bbox': [int(v) for v in box], 'score': round(float(s), 3)}
                for box, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                     r.boxes.conf.cpu().numpy()) if int(c) != 0]


class VideoAnalyzer:
    def __init__(self, use_tracker=True, use_envision=True, use_objects=True):
        import mediapipe as mp
        self.mp = mp
        vision = mp.tasks.vision
        opts = mp.tasks.BaseOptions
        self.gesture = vision.GestureRecognizer.create_from_options(
            vision.GestureRecognizerOptions(
                base_options=opts(model_asset_buffer=model_bytes(GESTURE_URL, MODELS/'gesture_recognizer.task')),
                running_mode=vision.RunningMode.VIDEO, num_hands=2))
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=opts(model_asset_buffer=model_bytes(FACE_URL, MODELS/'face_landmarker.task')),
                running_mode=vision.RunningMode.VIDEO, num_faces=1,
                output_face_blendshapes=True, output_facial_transformation_matrixes=True))
        self.head = HeadGesture()
        self.tracker = PersonTracker() if use_tracker else None
        self.tracker_warned = False
        self.objects = ObjectDetector() if use_objects else None
        self.objects_warned = False
        self.envision = None
        self.envision_status = 'disabled'
        if use_envision:
            try:
                self.envision = EnvisionGesture()
            except Exception as e:
                print('Envision init error:', e)
                self.envision_status = f'error: {e}'
        self.envision_person = None
        self.hands_person = self.face_person = None
        self.hands_ts = self.face_ts = -1     # each MediaPipe task needs its own increasing timestamps
        self.hand_history = deque(maxlen=12)

    def analyze(self, frame, ts_ms):
        tracked = self.track(frame, ts_ms)
        crop, pid = self.target_crop(frame, tracked)
        result = self.observe(crop, ts_ms, pid)
        result['envision'] = self.envision_result(crop, pid)
        result.update(tracked)
        result.update(self.detect_objects(frame))
        return result

    def detect_objects(self, frame):
        if not self.objects:
            return {'objects': []}
        try:
            return {'objects': self.objects.detect(frame)}
        except Exception as e:
            if not self.objects_warned:
                print('Object detector error:', e)
                self.objects_warned = True
            return {'objects': []}

    def observe(self, frame, ts_ms, person_id=None):
        """Hands and face together (the unthreaded path)."""
        return {**self.observe_hands(frame, ts_ms, person_id), **self.observe_face(frame, ts_ms, person_id)}

    def track(self, frame, ts_ms=None):
        """{'people': [{'id', 'bbox', 'score', 'hands'}], 'target_id'}; empty when the tracker is off."""
        if not self.tracker:
            return {'people': [], 'target_id': None}
        try:
            people = self.tracker.update(frame, None if ts_ms is None else ts_ms / 1000)
        except Exception as e:
            if not self.tracker_warned:
                print('Tracker error:', e)
                self.tracker_warned = True
            people = self.tracker.people
        return {'people': people, 'target_id': self.tracker.target_id}

    @staticmethod
    def target_crop(frame, tracked):
        """The target person's crop and id, or the whole frame when nobody is tracked."""
        target = next((p for p in tracked['people'] if p['id'] == tracked['target_id']), None)
        return (crop_box(frame, target['bbox']), target['id']) if target else (frame, None)

    def envision_result(self, frame, person_id=None):
        if self.envision:
            if person_id != self.envision_person:   # another person: their motion window must not carry over
                self.envision.reset()
                self.envision_person = person_id
            return self.envision.analyze(frame)
        return {'status': self.envision_status, 'probs': None, 'label': None}

    def mp_image(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def observe_face(self, frame, ts_ms, person_id=None):
        """Face blendshapes and head pose -> nod/shake, on one person's crop (or the frame)."""
        if person_id != self.face_person:
            self.head.reset()
            self.face_person = person_id
        self.face_ts = ts_ms = max(int(ts_ms), self.face_ts + 1)
        f = self.face.detect_for_video(self.mp_image(frame), ts_ms)
        shapes = {}
        if f.face_blendshapes:
            shapes = {c.category_name: round(float(c.score), 4) for c in f.face_blendshapes[0]}
        # Observable facial movements only; do not infer emotion from these.
        observable = {k: shapes.get(k, 0.0) for k in (
            'mouthSmileLeft', 'mouthSmileRight', 'jawOpen',
            'browInnerUp', 'eyeBlinkLeft', 'eyeBlinkRight',
            'mouthFrownLeft', 'mouthFrownRight')}
        if f.facial_transformation_matrixes:
            head = self.head.update(ts_ms / 1000, *head_angles(f.facial_transformation_matrixes[0]))
        else:
            self.head.reset()
            head = {'status': 'no face', 'q': None}
        return {'face_movements': observable, 'head': head}

    def observe_hands(self, frame, ts_ms, person_id=None):
        """MediaPipe hand gestures and hand motion on one person's crop (or the frame)."""
        if person_id != self.hands_person:
            self.hand_history.clear()
            self.hands_person = person_id
        self.hands_ts = ts_ms = max(int(ts_ms), self.hands_ts + 1)
        g = self.gesture.recognize_for_video(self.mp_image(frame), ts_ms)
        gestures = []
        for i, choices in enumerate(g.gestures):
            if not choices:
                continue
            gestures.append({'label': choices[0].category_name,
                             'score': float(choices[0].score),
                             'alternatives': [
                                 {'label': c.category_name, 'score': float(c.score)}
                                 for c in choices[:5]]})
        hand_centers = []
        for hand in g.hand_landmarks:
            hand_centers.append((float(hand[0].x), float(hand[0].y)))
        self.hand_history.append((ts_ms, hand_centers))
        motion = 'insufficient history'
        if len(self.hand_history) >= 5 and hand_centers:
            recent = [p[1][0] for p in self.hand_history if p[1]]
            if len(recent) >= 5:
                displacement = sum(math.dist(a, b) for a, b in zip(recent, recent[1:]))
                motion = 'hand moving' if displacement > 0.12 else 'hand relatively still'
        # raw hand landmarks for sign spotting: 21 (x, y) per hand in crop-normalized coordinates, with the
        # handedness MediaPipe reports (as seen in the image)
        hands = []
        for i, hand in enumerate(g.hand_landmarks):
            side = g.handedness[i][0].category_name if i < len(g.handedness) and g.handedness[i] else 'Unknown'
            hands.append({'side': side, 'points': [[float(p.x), float(p.y)] for p in hand]})
        return {'timestamp_ms': ts_ms, 'gestures': gestures, 'hand_motion': motion, 'hand_landmarks': hands}

    def close(self):
        self.gesture.close()
        self.face.close()
        if self.envision:
            self.envision.close()


class LatestFrameWorker:
    """Runs fn on the newest submitted frame in a background thread; frames that arrive while it is busy are
    dropped, so the caller never waits on it."""

    def __init__(self, fn, name, min_interval=0.0):
        self.fn = fn
        self.min_interval = min_interval   # seconds; for slow-changing things (objects in a room)
        self.cond = threading.Condition()
        self.pending = None
        self.result = None
        self.done = deque(maxlen=30)
        self.stopped = False
        self.warned = False
        self.thread = threading.Thread(target=self.loop, name=name, daemon=True)
        self.thread.start()

    def submit(self, *args):
        with self.cond:
            self.pending = args
            self.cond.notify()

    def loop(self):
        while True:
            with self.cond:
                while self.pending is None and not self.stopped:
                    self.cond.wait()
                if self.stopped:
                    return
                args, self.pending = self.pending, None
            if self.min_interval and self.done:
                time.sleep(max(0.0, self.done[-1] + self.min_interval - time.monotonic()))
            try:
                out = self.fn(*args)
            except Exception as e:
                if not self.warned:
                    print(f'{self.thread.name} error:', e)
                    self.warned = True
                continue
            with self.cond:
                self.result = out
                self.done.append(time.monotonic())

    def fps(self):
        with self.cond:
            if len(self.done) < 2:
                return 0.0
            return (len(self.done) - 1) / max(self.done[-1] - self.done[0], 1e-6)

    def latest(self):
        with self.cond:
            return self.result

    def close(self):
        with self.cond:
            self.stopped = True
            self.cond.notify()
        self.thread.join(timeout=5)


class TargetSelector:
    """Which tracked person is the user: kept while visible, else the largest box; select_next cycles."""

    def __init__(self):
        self.target_id = None

    def update(self, people):
        if self.target_id not in {p['id'] for p in people}:
            area = lambda p: (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1])
            self.target_id = max(people, key=area)['id'] if people else None
        return self.target_id

    def select_next(self, people):
        ordered = sorted(people, key=lambda p: p['bbox'][0])
        if not ordered:
            return None
        ids = [p['id'] for p in ordered]
        self.target_id = ids[(ids.index(self.target_id) + 1) % len(ids)] if self.target_id in ids else ids[0]
        return self.target_id


def _tracker_process(frames, results, use_objects):
    """Child process: person tracking, plus objects every 2 s. Its own interpreter, so MediaPipe holding the
    GIL in the parent's threads cannot starve it (alone it runs ~80 ms/frame; as a thread it got ~1.5 fps)."""
    tracker = PersonTracker()
    detector = ObjectDetector() if use_objects else None
    objects, last_objects = [], 0.0
    while True:
        item = frames.get()
        if item is None:
            break
        frame, ts_ms = item
        try:
            people = tracker.update(frame, ts_ms / 1000)
            if detector and time.monotonic() - last_objects >= 2.0:
                objects, last_objects = detector.detect(frame), time.monotonic()
            results.put({'people': people, 'objects': objects})
        except Exception as e:
            results.put({'error': str(e)})


class TrackerProcess:
    """Parent side of _tracker_process. submit() never blocks (one-slot queue; a stale frame is replaced),
    latest() is the newest {'people', 'objects'}, fps() the processing rate."""

    def __init__(self, use_objects=True):
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        self.frames, self.results = ctx.Queue(maxsize=1), ctx.Queue()
        self.proc = ctx.Process(target=_tracker_process, args=(self.frames, self.results, use_objects),
                                daemon=True)
        self.proc.start()
        self.result, self.done, self.warned = None, deque(maxlen=30), False
        self.lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            try:
                out = self.results.get()
            except (EOFError, OSError):
                return
            if out is None:
                return
            if 'error' in out:
                if not self.warned:
                    print('Tracker process error:', out['error'])
                    self.warned = True
                continue
            with self.lock:
                self.result = out
                self.done.append(time.monotonic())

    def submit(self, frame, ts_ms):
        import queue
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frames.put_nowait((frame, ts_ms))
        except queue.Full:
            pass

    def latest(self):
        with self.lock:
            return self.result

    def fps(self):
        with self.lock:
            return (len(self.done) - 1) / max(self.done[-1] - self.done[0], 1e-6) if len(self.done) > 1 else 0.0

    def close(self):
        try:
            self.frames.put_nowait(None)
        except Exception:
            pass
        self.proc.join(timeout=3)
        if self.proc.is_alive():
            self.proc.terminate()
        self.results.put(None)


class ThreadedVideoAnalyzer:
    """Live analysis without blocking the display. Person tracking and object detection run in a separate
    process (TrackerProcess); hands, face and Envision each run in their own thread on the newest frame,
    cropped to the user's latest box. Each part runs as fast as the CPU allows; Envision's rate is reported
    because its velocity features were trained at 25-30 fps. Build the VideoAnalyzer with use_tracker=False
    and use_objects=False: tracking lives in the child process here."""

    EMPTY_HANDS = {'timestamp_ms': 0, 'gestures': [], 'hand_motion': 'insufficient history'}
    EMPTY_FACE = {'face_movements': {}, 'head': {'status': 'starting', 'q': None}}

    def __init__(self, analyzer, track=True, objects=True, tracker=None):
        self.analyzer = analyzer
        # pass a TrackerProcess started early so its ~10 s model load overlaps the parent's own loading
        self.tracker = tracker or (TrackerProcess(use_objects=objects) if track else None)
        self.selector = TargetSelector()
        self.mediapipe = LatestFrameWorker(self._hands, 'hands')
        self.face = LatestFrameWorker(self._face, 'face')   # own thread: nod/shake needs head pose at ~10+ fps
        self.envision = LatestFrameWorker(self._envision, 'envision') if analyzer.envision else None

    def tracked(self):
        out = (self.tracker.latest() if self.tracker else None) or {'people': [], 'objects': []}
        return {'people': out['people'], 'objects': out['objects'],
                'target_id': self.selector.update(out['people'])}

    def _hands(self, frame, ts_ms):
        crop, pid = self.analyzer.target_crop(frame, self.tracked())
        return self.analyzer.observe_hands(crop, ts_ms, pid)

    def _face(self, frame, ts_ms):
        crop, pid = self.analyzer.target_crop(frame, self.tracked())
        return self.analyzer.observe_face(crop, ts_ms, pid)

    def _envision(self, frame):
        crop, pid = self.analyzer.target_crop(frame, self.tracked())
        return self.analyzer.envision_result(crop, pid)

    def submit(self, frame, ts_ms):
        if self.tracker:
            self.tracker.submit(frame, ts_ms)
        self.mediapipe.submit(frame, ts_ms)
        self.face.submit(frame, ts_ms)
        if self.envision:
            self.envision.submit(frame)

    def select_next(self):
        return self.selector.select_next(self.tracked()['people'])

    def latest(self):
        result = {**(self.mediapipe.latest() or self.EMPTY_HANDS), **(self.face.latest() or self.EMPTY_FACE)}
        result.update(self.tracked())
        if self.envision:
            result['envision'] = self.envision.latest() or {'status': 'starting', 'probs': None, 'label': None}
        else:
            result['envision'] = self.analyzer.envision_result(None)
        result['rates'] = {'tracker': self.tracker.fps() if self.tracker else 0.0,
                           'mediapipe': self.mediapipe.fps(), 'face': self.face.fps(),
                           'envision': self.envision.fps() if self.envision else 0.0}
        return result

    def close(self):
        for worker in (self.mediapipe, self.face, self.envision):
            if worker:
                worker.close()
        if self.tracker:
            self.tracker.close()
        self.analyzer.close()


def second_summary(sec, rows):
    labels = Counter(g['label'] for r in rows for g in r['gestures'][:1])
    status = Counter(r['envision']['status'] for r in rows)
    ps = [r['envision']['probs']['Gesture'] for r in rows if r['envision']['probs']]
    people = [len(r['people']) for r in rows]
    env = f'mean p(Gesture) {np.mean(ps):.2f} max {max(ps):.2f}' if ps else 'no probs'
    print(f"t={sec:3d}s n={len(rows):2d} people={max(people)} gestures={dict(labels)} "
          f"envision={dict(status)} {env}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--no-tracker', action='store_true')
    parser.add_argument('--no-envision', action='store_true')
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    analyzer = VideoAnalyzer(use_tracker=not args.no_tracker, use_envision=not args.no_envision)
    rows, total, n = [], 0.0, 0
    current = 0
    try:
        while n < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = int(pos_ms) if pos_ms > 0 else n * 33
            start = time.perf_counter()
            result = analyzer.analyze(frame, ts)
            total += time.perf_counter() - start
            if result['timestamp_ms'] // 1000 != current and rows:
                second_summary(current, rows)
                rows = []
            current = result['timestamp_ms'] // 1000
            rows.append(result)
        if rows:
            second_summary(current, rows)
    finally:
        cap.release()
        analyzer.close()
    if n:
        print(f'{n} frames, {total / n * 1000:.1f} ms/frame')


if __name__ == '__main__':
    main()
