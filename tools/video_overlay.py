"""Draw measured image landmarks and a feature panel on their inference frame.

This renderer owns no capture, model, window, recording or fusion state. Face
contour topology follows MediaPipe's Apache-2.0 face_mesh_connections.py (2021
The MediaPipe Authors); the few selected contour paths are reproduced below.
"""
import math

import cv2
import numpy as np


POSE_EDGES = [(11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (11, 23),
              (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
              (24, 26), (26, 28), (28, 30), (30, 32)]
HAND_PATHS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [5, 9, 10, 11, 12],
              [9, 13, 14, 15, 16], [13, 17, 18, 19, 20], [0, 17]]
FACE_PATHS = [
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33],
    [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263],
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61],
    [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78],
    [46, 53, 52, 65, 55], [70, 63, 105, 66, 107],
    [276, 283, 282, 295, 285], [300, 293, 334, 296, 336],
    [469, 470, 471, 472, 469], [474, 475, 476, 477, 474],
]
COLORS = {'pose': (98, 225, 133), 'face': (245, 197, 100),
          'hands': (104, 174, 255), 'text': (226, 232, 239),
          'muted': (144, 159, 178), 'bad': (99, 120, 255)}


def _number(value, digits=2):
    try:
        value = float(value)
        return f'{value:.{digits}f}' if math.isfinite(value) else '--'
    except (TypeError, ValueError):
        return '--'


def _points(landmarks, width, height, visibility=False):
    out = []
    for p in landmarks if landmarks is not None else []:
        try:
            x, y = float(p[0]), float(p[1])
            visible = not visibility or (len(p) > 3 and float(p[3]) >= .3)
            good = visible and math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1
            out.append((int(x * (width - 1)), int(y * (height - 1))) if good else None)
        except (IndexError, TypeError, ValueError):
            out.append(None)
    return out


def _bones(image, points, edges, color, thickness=2):
    for a, b in edges:
        if a < len(points) and b < len(points) and points[a] is not None and points[b] is not None:
            cv2.line(image, points[a], points[b], color, thickness, cv2.LINE_AA)


def _paths(image, points, paths, color):
    for path in paths:
        _bones(image, points, zip(path, path[1:]), color, 1)


def _draw_landmarks(image, channels):
    height, width = image.shape[:2]
    pose = channels.get('pose', {})
    if pose.get('status') == 'ok':
        points = _points(pose.get('pose_image'), width, height, visibility=True)
        _bones(image, points, POSE_EDGES, COLORS['pose'])
        for point in points:
            if point is not None:
                cv2.circle(image, point, 3, COLORS['pose'], -1, cv2.LINE_AA)
    face = channels.get('face', {})
    if face.get('status') == 'ok':
        points = _points(face.get('landmarks_image'), width, height)
        for point in points:
            if point is not None:
                cv2.circle(image, point, 1, (112, 117, 86), -1, cv2.LINE_AA)
        _paths(image, points, FACE_PATHS, COLORS['face'])
        for index in (468, 473):
            if index < len(points) and points[index] is not None:
                cv2.circle(image, points[index], 3, (255, 245, 110), -1, cv2.LINE_AA)
    hand_channel = channels.get('hands', {})
    if hand_channel.get('status') == 'ok':
        for hand in hand_channel.get('hands', []):
            points = _points(hand.get('landmarks_image'), width, height)
            _paths(image, points, HAND_PATHS, COLORS['hands'])
            for i, point in enumerate(points):
                if point is not None:
                    cv2.circle(image, point, 4 if i in (4, 8, 12, 16, 20) else 2,
                               COLORS['hands'], -1, cv2.LINE_AA)


class _Panel:
    def __init__(self, canvas, x):
        self.canvas, self.x, self.y = canvas, x + 18, 26

    def line(self, text, color=None, scale=.43, gap=19):
        cv2.putText(self.canvas, str(text)[:57], (self.x, self.y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color or COLORS['text'], 1, cv2.LINE_AA)
        self.y += gap

    def section(self, name, value, color):
        self.y += 6
        cv2.line(self.canvas, (self.x, self.y - 3), (self.x + 382, self.y - 3), (57, 68, 82), 1)
        self.y += 16
        self.line(name.upper(), color, .5, 21)
        status = value.get('status', 'not enabled')
        self.line(('TRACKING' if status == 'ok' else 'NO TRACKING: ' + status)
                  + ('  |  ' + _number(value.get('model_ms'), 1) + ' ms' if value.get('model_ms') is not None else ''),
                  COLORS['muted'] if status == 'ok' else COLORS['bad'], gap=20)
        return status == 'ok'

    def bar(self, label, value, color):
        y = self.y - 8
        self.line(label, gap=18)
        x = self.x + 191
        cv2.rectangle(self.canvas, (x, y - 3), (x + 137, y + 5), (57, 68, 82), -1)
        try:
            score = float(value)
            if math.isfinite(score):
                cv2.rectangle(self.canvas, (x, y - 3), (x + int(137 * np.clip(score, 0, 1)), y + 5), color, -1)
        except (TypeError, ValueError):
            pass
        cv2.putText(self.canvas, _number(value), (x + 147, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    .4, COLORS['text'], 1, cv2.LINE_AA)


def render_tracking(frame_bgr, observation, event_count=0, neutral_ready=False, recording=False):
    """Return a BGR preview with image-space tracks and raw measured features.

    Coordinates must come from this exact frame. Missing channels render unknown,
    never cached detections. Coefficient bars are independent raw scores, not
    communication probabilities. ``model_ms`` is model time, not whole-loop FPS.
    """
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError('frame_bgr must be H x W x 3')
    source_h, source_w = frame_bgr.shape[:2]
    if not source_h or not source_w:
        raise ValueError('frame dimensions must be nonzero')
    scale = min(960 / source_w, 720 / source_h, 1.)
    width, height = max(1, round(source_w * scale)), max(1, round(source_h * scale))
    shown = cv2.resize(frame_bgr, (width, height)) if (width, height) != (source_w, source_h) else frame_bgr.copy()
    channels = observation.get('channels', {})
    source_size = observation.get('image_size', [source_w, source_h])
    for person in observation.get('people', []):
        selected = person['id'] == observation.get('person_id')
        color = COLORS['pose'] if selected else COLORS['muted']
        x1, y1, x2, y2 = person['bbox']
        a = (int(x1 * width / source_size[0]), int(y1 * height / source_size[1]))
        b = (int(x2 * width / source_size[0]), int(y2 * height / source_size[1]))
        cv2.rectangle(shown, a, b, color, 2 if selected else 1)
        role = ('Candidate' if observation.get('ownership_valid') is False else 'USER') if selected else 'Person'
        cv2.putText(shown, f'{role} {person["id"]}', (a[0] + 4, max(16, a[1] + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    _draw_landmarks(shown, channels)
    canvas = np.full((max(720, height), width + 420, 3), (20, 25, 32), dtype=np.uint8)
    canvas[:height, :width] = shown
    panel = _Panel(canvas, width)
    panel.line('TINK / VISUAL FEATURES', COLORS['text'], .56, 23)
    panel.line(('REC  |  ' if recording else 'PREVIEW  |  ') + _number(observation.get('t_capture'), 1)
               + ' s  |  events ' + str(event_count)
               + ('  |  ' + _number(observation['tracking_fps'], 1) + ' fps' if observation.get('tracking_fps') is not None else ''),
               COLORS['bad'] if recording else COLORS['muted'])
    panel.line('Trial: ' + str(observation.get('label', 'unlabeled')), gap=21)
    if 'target_status' in observation:
        panel.line('User: ' + str(observation.get('target_id')) + ' | ' + observation['target_status'],
                   COLORS['pose'] if observation.get('person_id') is not None else COLORS['bad'], gap=20)
        controls = 'Auto follow | Y role / N correct' if observation.get('session_mode') == 'integrated' else 'Click user / N next'
        panel.line(controls + ' | tracker ' + _number(observation.get('person_tracker_ms'), 0) + ' ms', COLORS['muted'])

    if 'face' in channels:
        face = channels['face']
        if panel.section('Face', face, COLORS['face']):
            f = face.get('features', {})
            b = f.get('blendshapes', {})
            panel.line('Head deg  P ' + _number(f.get('pitch_deg'), 1) + '  Y '
                       + _number(f.get('yaw_deg'), 1) + '  R ' + _number(f.get('roll_deg'), 1))
            panel.line('EAR L/R ' + _number(f.get('left_ear')) + ' / ' + _number(f.get('right_ear'))
                       + '   iris ' + _number(f.get('left_iris_ratio')) + ' / ' + _number(f.get('right_iris_ratio')))
            for label, key in [('Blink left', 'eyeBlinkLeft'), ('Blink right', 'eyeBlinkRight'),
                               ('Jaw open', 'jawOpen'), ('Brow inner up', 'browInnerUp'),
                               ('Smile left', 'mouthSmileLeft'), ('Smile right', 'mouthSmileRight'),
                               ('Mouth pucker', 'mouthPucker'), ('Cheek puff', 'cheekPuff')]:
                panel.bar(label, b.get(key), COLORS['face'])
            calibrating = face.get('calibrating_neutral', not neutral_ready)
            panel.line('Keep NEUTRAL: baseline calibration' if calibrating else 'Neutral baseline captured',
                       COLORS['bad'] if calibrating else COLORS['muted'])

    if 'hands' in channels:
        hands = channels['hands']
        if panel.section('Hands', hands, COLORS['hands']):
            for i, hand in enumerate(hands.get('hands', [])[:2]):
                scores = hand.get('scores', {})
                finite = [(str(k), v) for k, v in scores.items() if _number(v) != '--']
                top = max(finite, key=lambda kv: float(kv[1])) if finite else ('unknown', None)
                panel.line('Hand ' + str(i + 1) + ': ' + top[0] + '  score ' + _number(top[1]))
                f = hand.get('features', {})
                panel.line('  pinch ' + _number(f.get('pinch_palm_widths')) + ' palms | index ext '
                           + _number(f.get('index_extension')), COLORS['muted'])

    if 'pose' in channels:
        pose = channels['pose']
        if panel.section('Body / Envision', pose, COLORS['pose']):
            f, heads = pose.get('features', {}), pose.get('cnn_heads', {})
            panel.line('Observed source ' + _number(pose.get('observed_hz'), 1)
                       + ' Hz | min visibility ' + _number(f.get('upper_visibility_min')))
            for side in ['left', 'right']:
                panel.line(side.capitalize() + ': elbow ' + _number(f.get(side + '_elbow_deg'), 0)
                           + ' deg | raise ' + _number(f.get(side + '_raised_sw'))
                           + ' SW | ext ' + _number(f.get(side + '_extension')))
            if heads:
                panel.line('CNN motion ' + _number(heads.get('motion'))
                           + ' | Gesture|motion ' + _number(heads.get('gesture')))
                panel.line('LGBM P(Gesture) ' + _number(pose.get('lgbm_gesture')))
            else:
                panel.line('Envision: warming / source gaps', COLORS['muted'])

    # This stays below the last section for the all-channel 720px layout.
    footer_y = max(panel.y + 7, canvas.shape[0] - 45)
    if footer_y + 28 > canvas.shape[0]:
        canvas = cv2.copyMakeBorder(canvas, 0, footer_y + 35 - canvas.shape[0], 0, 0,
                                    cv2.BORDER_CONSTANT, value=(20, 25, 32))
    panel.canvas, panel.y = canvas, footer_y
    panel.line('RAW / UNCALIBRATED - no intent inference', COLORS['muted'], .39, 18)
    controls = 'Auto follow | controls below' if observation.get('session_mode') == 'integrated' else 'Click/N select user | 0-9 label | Q quit'
    panel.line(controls, COLORS['text'], .4, 18)
    return canvas
