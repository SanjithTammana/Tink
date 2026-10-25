"""Reusable selected-user measurements, anonymous temporal motifs and prior ASL reader."""
from collections import deque
from pathlib import Path
import os
import uuid

import cv2
import numpy as np

from automatic_target import AutomaticTarget
from person_selection import crop_bounds, full_image_landmarks
from video_layer import PoseClock, Segment, cnn_distribution, face_features, hand_features, landmarks_array, pose_features

ROOT = Path(__file__).resolve().parent
FACE_KEYS = ['eyeBlinkLeft', 'eyeBlinkRight', 'browInnerUp', 'jawOpen', 'mouthSmileLeft',
             'mouthSmileRight', 'mouthPucker', 'cheekPuff', 'eyeWideLeft', 'eyeWideRight',
             'browOuterUpLeft', 'browOuterUpRight']


def visual_vector(channels):
    f = channels.get('face', {}).get('features', {})
    body = channels.get('pose', {}).get('features', {})
    vec = [f.get('blendshapes', {}).get(k, 0.) for k in FACE_KEYS]
    vec += [f.get(k, 0.) / 90 for k in ['pitch_deg', 'yaw_deg', 'roll_deg']]
    for side in ['left', 'right']:
        vec += [np.clip(body.get(side + '_raised_sw', 0.), -2, 2) / 2,
                (body.get(side + '_elbow_deg') or 0.) / 180, body.get(side + '_extension') or 0.]
    vec += [body.get('upper_visibility_min', 0.)]
    hands = channels.get('hands', {}).get('hands', [])
    for i in range(2):
        values = hands[i]['features'] if i < len(hands) else {}
        vec += [values.get(finger + '_extension') or 0. for finger in ['thumb', 'index', 'middle', 'ring', 'little']]
    return np.asarray(vec, np.float32)


class VisualSession:
    def __init__(self, session_id=None, profile=None, asl=True):
        import mediapipe as mp
        from envision_port import EnvisionCNN, EnvisionLGBM
        from video import PersonTracker
        os.environ['YOLO_AUTOINSTALL'] = 'False'
        person_model = next((p for p in [ROOT / 'yolo11n-pose.pt', ROOT.parent / 'tink/yolo11n-pose.pt'] if p.is_file()), None)
        if person_model is None:
            raise RuntimeError('local YOLO person model missing; no automatic download')
        self.mp = mp
        self.tracker = PersonTracker(model=str(person_model))
        self.target = AutomaticTarget(profile=profile)
        self.session_id = session_id or uuid.uuid4().hex
        self.tasks = {}
        self.cnn, self.lgbm = EnvisionCNN(), EnvisionLGBM()
        self.pose_clock, self.history = PoseClock(), deque(maxlen=300)
        self.neutral, self.neutral_samples = None, []
        self.token, self.segments, self.role = None, {}, None
        self.reader, self.asl_status = None, 'disabled'
        if asl:
            try:
                from huggingface_hub import try_to_load_from_cache
                from asl import MODEL, LetterClassifier, FingerspellReader
                cached = try_to_load_from_cache(MODEL, 'config.json')
                if not isinstance(cached, str):
                    raise RuntimeError('ASL checkpoint not cached')
                self.reader = FingerspellReader(LetterClassifier(model=str(Path(cached).parent)))
                self.asl_status = 'static alphabet model loaded locally'
            except (OSError, RuntimeError, ValueError, ImportError):
                self.asl_status = 'ASL unavailable: local cached alphabet model required'
        self._reset_tasks()

    def _reset_tasks(self):
        self.close_tasks()
        v, base = self.mp.tasks.vision, self.mp.tasks.BaseOptions
        assets = ROOT / 'models'
        for name in ['pose_landmarker_full.task', 'face_landmarker.task', 'gesture_recognizer.task']:
            if not (assets / name).is_file():
                raise RuntimeError(f'local model missing: {name}')
        self.tasks['pose'] = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(assets / 'pose_landmarker_full.task')), running_mode=v.RunningMode.VIDEO))
        self.tasks['face'] = v.FaceLandmarker.create_from_options(v.FaceLandmarkerOptions(
            base_options=base(model_asset_path=str(assets / 'face_landmarker.task')), running_mode=v.RunningMode.VIDEO,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True))
        self.tasks['hands'] = v.GestureRecognizer.create_from_options(v.GestureRecognizerOptions(
            base_options=base(model_asset_path=str(assets / 'gesture_recognizer.task')), running_mode=v.RunningMode.VIDEO,
            num_hands=2, canned_gesture_classifier_options=self.mp.tasks.components.processors.ClassifierOptions(max_results=-1, score_threshold=0.)))

    def close_tasks(self):
        for task in self.tasks.values():
            task.close()
        self.tasks = {}

    def close(self):
        self.close_tasks()

    def observe(self, frame, at_s):
        import time
        start = time.perf_counter()
        people = self.tracker.update(frame, at_s)
        selected = self.target.update(people, frame, at_s)
        bounds = crop_bounds(selected['bbox'], frame.shape) if selected else None
        token = (selected['id'] if bounds else None, self.target.generation)
        events = []
        if self.role != self.target.logical_id:
            self.neutral, self.neutral_samples = None, []
            self.role = self.target.logical_id
        if token != self.token:
            for segment in self.segments.values():
                segment.finish(at_s, 'target_changed')  # incomplete: not training evidence
            self.pose_clock, self.history = PoseClock(), deque(maxlen=300)
            self.segments = {name: Segment(name, f'{self.session_id}-g{self.target.generation}', self.role)
                             for name in ['body_motion', 'face', 'head_posture', 'hands']}
            if self.reader:
                self.reader.stab.clear()
                self.reader.last_box = None
            if self.token is not None:
                self._reset_tasks()
            self.token = token
        obs = {'type': 'observation', 'session_id': self.session_id, 't_capture': at_s,
               'image_size': [frame.shape[1], frame.shape[0]], 'people': people,
               'target_id': self.target.target_id, 'person_id': selected['id'] if bounds else None,
               'logical_person_id': self.role if bounds else None, 'ownership_valid': self.target.ownership_valid,
               'target_status': self.target.status, 'person_tracker_ms': 1000 * (time.perf_counter() - start),
               'channels': {}, 'asl': {'status': self.asl_status, 'ready': False}}
        if bounds is None:
            obs['channels'] = {name: {'status': 'user lost/ambiguous'} for name in self.tasks}
            return obs, events
        x1, y1, x2, y2 = bounds
        crop = frame[y1:y2, x1:x2]
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        activation = {}
        for name, task in self.tasks.items():
            begin = time.perf_counter()
            result = task.recognize_for_video(image, round(at_s * 1000)) if name == 'hands' else task.detect_for_video(image, round(at_s * 1000))
            value = {'status': 'missing', 'model_ms': 1000 * (time.perf_counter() - begin)}
            if name == 'pose':
                p = landmarks_array(result.pose_world_landmarks[0]) if result.pose_world_landmarks else None
                grid = self.pose_clock.push(at_s, p[:23].reshape(92) if p is not None else None)
                if p is not None:
                    value.update(status='ok', features=pose_features(p), pose_image=full_image_landmarks(landmarks_array(result.pose_landmarks[0]), bounds, frame.shape))
                    if len(self.pose_clock.history) == 25 and grid:
                        seq = np.asarray([p for _, p in self.pose_clock.history])
                        heads = self.cnn(seq)
                        value.update(cnn_heads=heads, cnn_distribution=cnn_distribution(heads), lgbm_gesture=self.lgbm(seq))
                        activation['body_motion'] = heads['motion']
            elif name == 'face' and result.face_landmarks:
                blends = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
                matrix = result.facial_transformation_matrixes[0] if len(result.facial_transformation_matrixes) else None
                feats = face_features(blends, matrix, landmarks_array(result.face_landmarks[0]), crop.shape[1] / crop.shape[0])
                value.update(status='ok', features=feats, landmarks_image=full_image_landmarks(landmarks_array(result.face_landmarks[0]), bounds, frame.shape))
                if self.neutral is None:
                    self.neutral_samples.append((at_s, blends))
                    if len(self.neutral_samples) >= 5 and at_s - self.neutral_samples[0][0] >= 1.5:
                        self.neutral = {k: float(np.median([v.get(k, 0.) for _, v in self.neutral_samples])) for k in FACE_KEYS}
                else:
                    activation['face'] = min(1., max((max(0., (blends.get(k, 0.) - self.neutral.get(k, 0.)) / max(.1, 1 - self.neutral.get(k, 0.))) for k in FACE_KEYS), default=0.))
                    activation['head_posture'] = min(1., max(abs(feats.get(k, 0.)) for k in ['pitch_deg', 'yaw_deg']) / 30)
                value['calibrating_neutral'] = self.neutral is None
            elif name == 'hands':
                hands, scores, local = [], {}, []
                for i, hand in enumerate(result.hand_world_landmarks):
                    cats = {c.category_name: float(c.score) for c in result.gestures[i]}
                    points = landmarks_array(result.hand_landmarks[i])
                    local.append({'points': points[:, :2].tolist()})
                    hands.append({'features': hand_features(landmarks_array(hand)), 'scores': cats,
                                  'landmarks_image': full_image_landmarks(points, bounds, frame.shape)})
                    for key, score in cats.items():
                        scores[key] = max(scores.get(key, 0.), score)
                if hands:
                    value.update(status='ok', hands=hands)
                    activation['hands'] = max((p for k, p in scores.items() if k != 'None'), default=0.)
                if self.reader:
                    obs['asl'] = self.reader.update(crop, local)
            obs['channels'][name] = value
        vector = visual_vector(obs['channels'])
        self.history.append((at_s, vector))
        for name, segment in self.segments.items():
            for raw in segment.update(at_s, activation.get(name), {}):
                if not raw['complete']:
                    continue
                points = np.asarray([p for t, p in self.history if raw['t_start'] <= t <= raw['t_end']])
                if not len(points):
                    continue
                features = np.r_[points.mean(0), points.std(0)].tolist()
                events.append({'event_id': raw['event_id'], 'session_id': self.session_id, 'modality': 'video',
                    't_start': raw['t_start'], 't_end': raw['t_end'], 'person_id': self.role,
                    'ownership_valid': self.target.ownership_valid, 'clock_valid': True, 'features': features,
                    'quality': {'source': name, 'role_source': self.target.role_source, 'overlap': False}, 'context_id': 0})
        return obs, events
