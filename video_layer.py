"""Standalone visual measurements and candidate events, before intent/fusion.

Scores in this module are UNCALIBRATED. A face-only switch can activate without
body motion. Missing tracking is not negative evidence. Capture timestamps,
not inference-completion times, drive windows. Nothing imports the live app.
"""
from collections import deque
import math
import uuid

import numpy as np


def cnn_distribution(heads):
    """Preserve both heads, and derive the three unconditional model scores."""
    motion, gesture, move = (float(heads[k]) for k in ('motion', 'gesture', 'move'))
    return {'NoGesture': 1 - motion, 'Gesture': motion * gesture, 'Move': motion * move}


class PoseClock:
    """Bounded linear interpolation onto 25 Hz, never extrapolation across loss.

    Interpolation preserves units/window duration, not missing high frequencies.
    The actual observed rate must be reported separately from this grid rate.
    """
    def __init__(self, hz=25, max_gap=.2):
        if hz <= 0 or max_gap <= 0:
            raise ValueError('hz and max_gap must be positive')
        self.step, self.max_gap = 1 / hz, max_gap
        self.history = deque(maxlen=25)
        self.previous = self.next_t = self.last_t = None
        self.discontinuity = False

    def push(self, t, values):
        if not math.isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError('capture timestamps must increase strictly')
        self.last_t, self.discontinuity = t, False
        if values is not None:
            values = np.asarray(values, dtype=np.float32).reshape(92)
            if not np.isfinite(values).all():
                values = None
        if values is None:
            self.history.clear()
            self.previous = self.next_t = None
            self.discontinuity = True
            return []
        if self.previous is None or t - self.previous[0] > self.max_gap:
            self.discontinuity = self.previous is not None
            self.history.clear()
            self.previous, self.next_t = (t, values.copy()), t + self.step
            samples = [(t, values.copy())]
        else:
            before, old = self.previous
            samples = []
            while self.next_t <= t + 1e-9:
                alpha = min(1., (self.next_t - before) / (t - before))
                samples.append((self.next_t, old + alpha * (values - old)))
                self.next_t += self.step
            self.previous = t, values.copy()
        self.history.extend(samples)
        return samples


class Segment:
    """One candidate per sustained activation, with hysteresis and delayed offset.

    Thresholds are lab defaults, not validated sensitivity settings. A held signal
    emits once on release; finish() marks EOF/tracking loss as incomplete.
    """
    def __init__(self, source, session=None, person_id=None, on=.6, off=.35,
                 min_duration=.25, release=.3, max_gap=.3):
        if not 0 <= off < on <= 1:
            raise ValueError('need 0 <= off < on <= 1')
        self.source, self.session = source, session or uuid.uuid4().hex
        self.person_id = person_id
        self.on, self.off = on, off
        self.min_duration, self.release, self.max_gap = min_duration, release, max_gap
        self.active = None
        self.last_t = None
        self.counter = 0

    def update(self, t, activation, scores):
        if not math.isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError('capture timestamps must increase strictly')
        events = []
        if self.last_t is not None and t - self.last_t > self.max_gap:
            event = self.finish(t, 'tracking_lost')
            if event:
                events.append(event)
        self.last_t = t
        if activation is None:
            event = self.finish(t, 'tracking_lost')
            if event:
                events.append(event)
            return events
        if not math.isfinite(activation) or not 0 <= activation <= 1:
            raise ValueError('activation must be a finite score in [0, 1]')
        if self.active is None:
            if activation < self.on:
                return events
            self.active = {'start': t, 'last': t, 'sum': {}, 'count': {}, 'peak': {}, 'n': 0}
        if activation >= self.off:
            a = self.active
            a['last'], a['n'] = t, a['n'] + 1
            for key, value in scores.items():
                if value is not None and math.isfinite(float(value)):
                    a['sum'][key] = a['sum'].get(key, 0.) + float(value)
                    a['count'][key] = a['count'].get(key, 0) + 1
                    a['peak'][key] = max(a['peak'].get(key, -math.inf), float(value))
        elif t - self.active['last'] >= self.release:
            event = self.finish(t, 'release')
            if event:
                events.append(event)
        return events

    def finish(self, t, reason='eof'):
        a, self.active = self.active, None
        if a is None or a['last'] - a['start'] < self.min_duration - 1e-9:
            return None
        self.counter += 1
        return {'type': 'event', 'event_id': f'{self.session}:{self.source}:{self.counter}',
                'source': self.source, 'person_id': self.person_id,
                't_start': a['start'], 't_end': a['last'], 'emitted_at': t,
                'end_reason': reason, 'complete': reason == 'release',
                'calibrated': False, 'samples': a['n'],
                'mean_scores': {k: v / a['count'][k] for k, v in a['sum'].items()},
                'peak_scores': a['peak']}


def landmarks_array(landmarks):
    return np.array([[p.x, p.y, p.z, getattr(p, 'visibility', 0.) or 0.]
                     for p in landmarks], dtype=np.float32)


def _angle(a, b, c):
    u, v = a - b, c - b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return None if denom < 1e-8 else float(np.degrees(np.arccos(np.clip(u @ v / denom, -1, 1))))


def pose_features(pose):
    """Metric geometry; shoulder-normalized height is positive when raised."""
    p = np.asarray(pose)
    xyz = p[:, :3]
    width = float(np.linalg.norm(xyz[11] - xyz[12]))
    if width < 1e-6:
        return {}
    out = {'shoulder_width_m': width, 'upper_visibility_min': float(p[[11, 12, 13, 14, 15, 16], 3].min())}
    for name, sh, el, wr in [('left', 11, 13, 15), ('right', 12, 14, 16)]:
        out[name + '_raised_sw'] = float((xyz[sh, 1] - xyz[wr, 1]) / width)
        out[name + '_elbow_deg'] = _angle(xyz[sh], xyz[el], xyz[wr])
        reach = np.linalg.norm(xyz[wr] - xyz[sh])
        arm = np.linalg.norm(xyz[el] - xyz[sh]) + np.linalg.norm(xyz[wr] - xyz[el])
        out[name + '_extension'] = float(reach / arm) if arm > 1e-6 else None
        out[name + '_wrist_relative_sw'] = ((xyz[wr] - (xyz[11] + xyz[12]) / 2) / width).tolist()
    return out


def hand_features(landmarks):
    p = np.asarray(landmarks)[:, :3]
    scale = float(np.linalg.norm(p[5] - p[17]))
    if scale < 1e-6:
        return {}
    out = {'pinch_palm_widths': float(np.linalg.norm(p[4] - p[8]) / scale)}
    for name, chain in [('thumb', [1, 2, 3, 4]), ('index', [5, 6, 7, 8]),
                        ('middle', [9, 10, 11, 12]), ('ring', [13, 14, 15, 16]),
                        ('little', [17, 18, 19, 20])]:
        a, b, c, d = chain
        out[name + '_proximal_deg'] = _angle(p[a], p[b], p[c])
        out[name + '_distal_deg'] = _angle(p[b], p[c], p[d])
        length = sum(np.linalg.norm(p[j] - p[i]) for i, j in zip(chain, chain[1:]))
        out[name + '_extension'] = float(np.linalg.norm(p[d] - p[a]) / length) if length > 1e-8 else None
    return out


def face_features(blendshapes, matrix, landmarks, aspect=1.):
    """Blendshape coefficients are independent, not a categorical distribution.

    Euler convention: Rz(roll) Ry(yaw) Rx(pitch), degrees, anatomical directions
    must be checked against the chosen camera mirroring before assigning names.
    """
    out = {'blendshapes': dict(blendshapes)}
    if matrix is not None:
        r = np.asarray(matrix, dtype=float)[:3, :3]
        out['rotation_valid'] = False
        if r.shape == (3, 3) and np.isfinite(r).all():
            u, singular, vt = np.linalg.svd(r)
            if singular.min() > 1e-8:
                fix = np.eye(3)
                fix[2, 2] = np.linalg.det(u @ vt)
                r = u @ fix @ vt
                out.update(rotation_valid=True,
                           pitch_deg=math.degrees(math.atan2(r[2, 1], r[2, 2])),
                           yaw_deg=math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))),
                           roll_deg=math.degrees(math.atan2(r[1, 0], r[0, 0])))
    if landmarks is not None:
        # Image x/y distances need aspect correction; normalized x/y have different units.
        xy = np.asarray(landmarks)[:, :2].copy()
        xy[:, 0] *= aspect
        for name, eye in [('right', [33, 160, 158, 133, 153, 144]),
                          ('left', [362, 385, 387, 263, 373, 380])]:
            a, b, c, d, e, f = xy[eye]
            den = 2 * np.linalg.norm(a - d)
            out[name + '_ear'] = float((np.linalg.norm(b - f) + np.linalg.norm(c - e)) / den) if den > 1e-8 else None
        for name, center, inner, outer in [('right', 468, 133, 33), ('left', 473, 362, 263)]:
            axis = xy[outer] - xy[inner]
            den = axis @ axis
            out[name + '_iris_ratio'] = float((xy[center] - xy[inner]) @ axis / den) if den > 1e-8 else None
    return out
