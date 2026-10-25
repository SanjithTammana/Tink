"""Explicit target lock for a multi-person video test. No biometric identity.

No automatic largest-person fallback. Loss withholds evidence. Ambiguous overlap
or a large discontinuous jump clears selection and requires the operator again.
"""
import math

import numpy as np


def box_overlap(a, b):
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    overlap = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return area(overlap) / max(1., min(area(a), area(b)))


def crop_bounds(box, shape):
    height, width = shape[:2]
    x1, y1 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x2, y2 = min(width, math.ceil(box[2])), min(height, math.ceil(box[3]))
    return (x1, y1, x2, y2) if x2 - x1 >= 32 and y2 - y1 >= 32 else None


def full_image_landmarks(landmarks, bounds, shape):
    """Map crop-normalized image x/y to full-frame normalized x/y. z stays crop-native."""
    p = np.asarray(landmarks, dtype=float).copy()
    x1, y1, x2, y2 = bounds
    p[:, 0] = (x1 + p[:, 0] * (x2 - x1)) / shape[1]
    p[:, 1] = (y1 + p[:, 1] * (y2 - y1)) / shape[0]
    return p.tolist()


class TargetLock:
    def __init__(self, target_id=None, overlap_limit=0.):
        self.target_id = target_id
        self.overlap_limit = overlap_limit
        self.generation = 0
        self.last_box = None
        self.visible = False
        self.status = 'selected' if target_id is not None else 'click a person or press N'

    def select(self, person_id):
        self.target_id = person_id
        self.last_box, self.visible = None, False
        self.generation += 1
        self.status = 'selected' if person_id is not None else 'click a person or press N'

    def select_next(self, people):
        ids = [p['id'] for p in sorted(people, key=lambda p: (p['bbox'][0], p['id']))]
        if ids:
            self.select(ids[(ids.index(self.target_id) + 1) % len(ids)] if self.target_id in ids else ids[0])
        return self.target_id

    def select_at(self, x, y, people):
        hits = [p for p in people if p['bbox'][0] <= x <= p['bbox'][2] and p['bbox'][1] <= y <= p['bbox'][3]]
        if len(hits) == 1:
            self.select(hits[0]['id'])
        else:
            self.select(None)
            self.status = 'ambiguous click' if hits else 'click a person or press N'
        return self.target_id

    def update(self, people):
        person = next((p for p in people if p['id'] == self.target_id), None)
        if person is None:
            self.visible = False
            self.status = 'selected person not visible' if self.target_id is not None else self.status
            return None
        if any(p['id'] != self.target_id and box_overlap(person['bbox'], p['bbox']) > self.overlap_limit for p in people):
            self.select(None)
            self.status = 'overlap ambiguous: select again when separated'
            return None
        if self.last_box is not None:
            a, b = self.last_box, person['bbox']
            distance = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
            size = max(math.hypot(a[2] - a[0], a[3] - a[1]), math.hypot(b[2] - b[0], b[3] - b[1]), 1)
            if distance > .75 * size:
                self.select(None)
                self.status = 'track jumped: select user again'
                return None
        if not self.visible:
            self.generation += 1
        self.visible, self.last_box, self.status = True, person['bbox'], 'tracking selected user'
        return person
