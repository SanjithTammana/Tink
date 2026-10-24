"""Video evidence for Tink: YOLO people, MediaPipe gestures + face blendshapes,
and EnvisionHGdetector's LightGBM Gesture/NoGesture score.

Self-test: python video.py --video sample.mp4 [--frames N] [--no-yolo] [--no-envision]
"""
import argparse
import contextlib
import importlib
import importlib.util
import io
import sys
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

    def close(self):
        self.model.holistic.close()
        self.model.holistic = None  # its __del__ would close it a second time


class VideoAnalyzer:
    def __init__(self, use_yolo=True, use_envision=True):
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
                output_face_blendshapes=True))
        self.detector = None
        if use_yolo:
            from ultralytics import YOLO
            self.detector = YOLO('yolo11n.pt')
        self.envision = None
        self.envision_status = 'disabled'
        if use_envision:
            try:
                self.envision = EnvisionGesture()
            except Exception as e:
                print('Envision init error:', e)
                self.envision_status = f'error: {e}'
        self.prev_ts = -1
        self.last_boxes = []
        self.last_people_time = 0.0
        self.hand_history = deque(maxlen=12)

    def analyze(self, frame, ts_ms):
        ts_ms = max(int(ts_ms), self.prev_ts + 1)
        self.prev_ts = ts_ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb))
        g = self.gesture.recognize_for_video(image, ts_ms)
        f = self.face.detect_for_video(image, ts_ms)
        gestures = []
        for i, choices in enumerate(g.gestures):
            if not choices:
                continue
            gestures.append({'label': choices[0].category_name,
                             'score': float(choices[0].score),
                             'alternatives': [
                                 {'label': c.category_name, 'score': float(c.score)}
                                 for c in choices[:5]]})
        shapes = {}
        if f.face_blendshapes:
            shapes = {c.category_name: round(float(c.score), 4)
                      for c in f.face_blendshapes[0]}
        # Observable facial movements only; do not infer emotion from these.
        observable = {k: shapes.get(k, 0.0) for k in (
            'mouthSmileLeft', 'mouthSmileRight', 'jawOpen',
            'browInnerUp', 'eyeBlinkLeft', 'eyeBlinkRight',
            'mouthFrownLeft', 'mouthFrownRight')}
        now = time.monotonic()
        if self.detector and now - self.last_people_time >= 0.45:
            self.last_people_time = now
            try:
                prediction = self.detector.predict(frame, classes=[0], conf=0.4,
                                                   verbose=False, imgsz=416)[0]
                self.last_boxes = [
                    {'bbox': [int(v) for v in b.xyxy[0].tolist()],
                     'score': float(b.conf[0])}
                    for b in prediction.boxes]
            except Exception as e:
                print('YOLO error:', e)
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
        if self.envision:
            envision = self.envision.analyze(frame)
        else:
            envision = {'status': self.envision_status, 'probs': None, 'label': None}
        return {'timestamp_ms': ts_ms, 'people': self.last_boxes,
                'gestures': gestures, 'face_movements': observable,
                'hand_motion': motion, 'envision': envision}

    def close(self):
        self.gesture.close()
        self.face.close()
        if self.envision:
            self.envision.close()


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
    parser.add_argument('--no-yolo', action='store_true')
    parser.add_argument('--no-envision', action='store_true')
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    analyzer = VideoAnalyzer(use_yolo=not args.no_yolo, use_envision=not args.no_envision)
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
