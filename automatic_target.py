"""Persistent local appearance/motion lock; never infer disability or speaker role.

A sole visible person initializes automatically. Several unfamiliar people yield
a visible provisional candidate; one explicit role confirmation is needed in that
ambiguous case. Overlap pauses observations without erasing the saved identity.
"""
import json
import math
import uuid
from pathlib import Path

import cv2
import numpy as np

from person_selection import TargetLock, box_overlap, crop_bounds


def appearance(frame, bbox):
    bounds = crop_bounds(bbox, frame.shape)
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    h, w = y2 - y1, x2 - x1
    torso = frame[y1 + int(.2 * h):y1 + int(.85 * h), x1 + int(.15 * w):x2 - int(.15 * w)]
    if not torso.size:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 1e-8 else None


class AutomaticTarget(TargetLock):
    def __init__(self, target_id=None, profile=None):
        super().__init__(target_id, overlap_limit=0.)
        self.profile = Path(profile) if profile else None
        self.signature = None
        self.confirmed = target_id is not None
        self.pending_manual = target_id is not None
        self.logical_id = uuid.uuid4().hex
        self._saved_at = None
        self.last_visible_s = None
        self.role_source = 'uninitialized'
        if self.profile and self.profile.is_file() and not self.profile.is_symlink():
            try:
                data = json.loads(self.profile.read_text(encoding='utf-8'))
                vec = np.asarray(data['signature'], dtype=float)
                if data.get('schema') == 1 and vec.shape == (128,) and np.isfinite(vec).all() and np.linalg.norm(vec) > 0:
                    self.signature = vec / np.linalg.norm(vec)
                    self.confirmed = True
                    self.logical_id = str(data.get('person_id') or self.logical_id)
                    self.role_source = 'remembered_local_appearance'
            except (ValueError, KeyError, OSError):
                pass

    @property
    def ownership_valid(self):
        return self.confirmed and self.visible

    def select(self, person_id):
        super().select(person_id)
        self.pending_manual = person_id is not None
        self.confirmed = person_id is not None
        self.signature = None
        self.logical_id = uuid.uuid4().hex
        self._saved_at = None
        self.role_source = 'operator_override' if person_id is not None else 'uninitialized'

    def confirm_role(self):
        if self.target_id is not None and self.visible:
            self.confirmed = True
            self.role_source = 'operator_confirmed_provisional_target'
            self._save()

    def _save(self):
        if self.profile is None or self.signature is None or not self.confirmed:
            return
        if self.profile.is_symlink():
            raise ValueError('person profile must be a regular local file')
        if self._saved_at is not None and self.last_visible_s is not None and self.last_visible_s - self._saved_at < 5:
            return
        self.profile.parent.mkdir(parents=True, exist_ok=True)
        (self.profile.parent / '.gitignore').write_text('*\n', encoding='utf-8')
        temp = self.profile.with_suffix('.pending')
        temp.write_text(json.dumps({'schema': 1, 'person_id': self.logical_id, 'signature': self.signature.tolist(),
                                    'note': 'local appearance heuristic, not biometric identity'}), encoding='utf-8')
        temp.replace(self.profile)
        self._saved_at = self.last_visible_s

    def update(self, people, frame=None, at_s=None):
        if frame is None:
            return super().update(people)
        candidates = [(p, appearance(frame, p['bbox'])) for p in people]
        candidates = [(p, a) for p, a in candidates if a is not None]
        if not candidates:
            self.visible = False
            self.status = 'user temporarily lost; waiting for automatic recovery'
            return None
        found = next(((p, a) for p, a in candidates if p['id'] == self.target_id), None)
        if self.signature is None:
            if found is None:
                if self.pending_manual:
                    self.visible = False
                    self.status = 'waiting for explicitly selected person; no fallback'
                    return None
                found = max(candidates, key=lambda item: (item[0]['bbox'][2] - item[0]['bbox'][0]) * (item[0]['bbox'][3] - item[0]['bbox'][1]))
                self.target_id = found[0]['id']
                self.generation += 1
                self.confirmed = len(candidates) == 1
                self.role_source = 'sole_visible_person_assumption' if self.confirmed else 'provisional_multiple_people'
        else:
            jump = False
            if found is not None and self.last_box is not None:
                a, b = self.last_box, found[0]['bbox']
                distance = math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)
                scale = max(math.hypot(a[2]-a[0], a[3]-a[1]), math.hypot(b[2]-b[0], b[3]-b[1]), 1.)
                jump = distance > .75 * scale
            if found is not None and float(found[1] @ self.signature) >= .65 and self.visible and not jump:
                return self._follow(found, people, at_s)
            ranked = sorted(candidates, key=lambda item: float(item[1] @ self.signature), reverse=True)
            best_score = float(ranked[0][1] @ self.signature)
            margin = best_score - (float(ranked[1][1] @ self.signature) if len(ranked) > 1 else 0.)
            if best_score < .9 or margin < .08:
                self.visible = False
                self.status = 'identity ambiguous; retaining profile, N/click corrects'
                return None
            found = ranked[0]
        return self._follow(found, people, at_s)

    def _follow(self, found, people, at_s):
        person, signature = found
        if any(other['id'] != person['id'] and box_overlap(person['bbox'], other['bbox']) > 0 for other in people):
            self.visible = False
            self.status = 'overlap: identity retained, features paused'
            return None
        if person['id'] != self.target_id or not self.visible:
            self.generation += 1
        self.target_id = person['id']
        self.visible, self.last_box, self.last_visible_s = True, person['bbox'], at_s
        if self.signature is None or self.pending_manual:
            self.signature = signature.copy()
        elif self.confirmed:
            self.signature = .98 * self.signature + .02 * signature
            self.signature /= max(np.linalg.norm(self.signature), 1e-8)
        self.pending_manual = False
        self.status = ('following user automatically' if self.confirmed else 'provisional target: Y confirms role, N changes')
        self._save()
        return person
