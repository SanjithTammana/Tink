"""Enrolled keyword signs: the user teaches Tink their own signs, and Tink spots them in the live stream.

Not ASL interpretation: a small closed set of signs each recorded by the user 3-5 times. Each frame becomes a
feature vector per hand: the 21 hand landmarks relative to the wrist and scaled by palm size (hand shape),
plus the wrist position in the person crop (where the hand is relative to the body). A take is a sequence of
those vectors; a live window is matched to every take with dynamic time warping (DTW), so signing faster or
slower than at enrollment still matches.

Probabilities: d_s = smallest DTW distance to sign s's takes. P(s) ∝ exp(-d_s / T) with a 'no sign' option at
pseudo-distance d0, so P(no sign) is large whenever nothing matches well. T and d0 come from enrollment by
leave-one-take-out: d0 sits above the typical distance between takes of the same sign and T is scaled to the
spread of those distances. Windows without a visible hand are 'no sign' outright.

Run: python signs.py enroll [--signs YES NO HELP ...] [--takes 4] [--camera 0]
     python signs.py check            (leave-one-take-out accuracy on your recordings)
     python signs.py selftest         (synthetic motions; checks the maths, not real accuracy)
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'signs'
CONFIG = ROOT / 'config' / 'signs.json'
NO_SIGN = '(no sign)'
DIM = 2 * (21 * 2 + 2)     # per hand: 21 landmarks relative to the wrist (42) + wrist position (2)


def frame_features(hands):
    """[{'side', 'points': 21 x [x, y]}] -> feature vector (DIM,) and whether any hand was present.
    Slot 0 = the hand MediaPipe calls 'Left', slot 1 = 'Right'; a missing hand is zeros."""
    out = np.zeros(DIM, np.float32)
    present = False
    for h in hands or []:
        pts = np.asarray(h['points'], np.float32)
        if pts.shape != (21, 2):
            continue
        slot = 1 if h.get('side') == 'Right' else 0
        wrist = pts[0]
        palm = float(np.linalg.norm(pts[9] - wrist)) or 1e-3          # wrist -> middle-finger knuckle
        shape = ((pts - wrist) / palm).reshape(-1)
        out[slot * 44: slot * 44 + 42] = shape * 0.5                    # shape weighted against location
        out[slot * 44 + 42: slot * 44 + 44] = (wrist - 0.5) * 4.0       # location in the crop, centred
        present = True
    return out, present


def dtw(a, b):
    """DTW distance between sequences a (n, d) and b (m, d), normalised by path length."""
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    steps = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            prev = min((acc[i - 1, j - 1], 0), (acc[i - 1, j], 1), (acc[i, j - 1], 2))
            acc[i, j] = cost[i - 1, j - 1] + prev[0]
            pi, pj = (i - 1, j - 1) if prev[1] == 0 else ((i - 1, j) if prev[1] == 1 else (i, j - 1))
            steps[i, j] = steps[pi, pj] + 1
    return float(acc[n, m] / max(steps[n, m], 1))


def resample(seq, n=12):
    """Linear resample to n frames: keeps DTW cheap and makes takes recorded at different fps comparable."""
    seq = np.asarray(seq, np.float32)
    if len(seq) == n:
        return seq
    x = np.linspace(0, len(seq) - 1, n)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    f = (x - lo)[:, None]
    return seq[lo] * (1 - f) + seq[hi] * f


class SignLibrary:
    """Enrolled takes per sign, plus the calibration (T, d0) from leave-one-take-out distances."""

    def __init__(self, user='default'):
        self.dir = DATA / user
        self.takes = {}
        for d in sorted(self.dir.glob('*')) if self.dir.exists() else []:
            seqs = [resample(np.load(f)) for f in sorted(d.glob('take_*.npy'))]
            if len(seqs) >= 2:
                self.takes[d.name] = seqs
        self.T, self.d0 = self.calibrate()

    @property
    def signs(self):
        return list(self.takes)

    def distances(self, seq, skip=None):
        """{sign: min DTW distance from seq to that sign's takes}; skip=(sign, index) leaves one take out."""
        out = {}
        for s, seqs in self.takes.items():
            ds = [dtw(seq, t) for k, t in enumerate(seqs) if skip != (s, k)]
            if ds:
                out[s] = min(ds)
        return out

    def calibrate(self):
        same, other = [], []
        for s, seqs in self.takes.items():
            for k, t in enumerate(seqs):
                d = self.distances(t, skip=(s, k))
                if s in d:
                    same.append(d[s])
                other += [v for o, v in d.items() if o != s]
        if not same:
            return 1.0, 1.0
        same, other = np.array(same), np.array(other) if other else np.array([np.max(same) * 2])
        d0 = float(min(np.percentile(same, 90) * 1.2, np.median(other)))   # 'no sign' sits above same-sign spread
        T = float(max(np.std(same), 0.05 * d0, 1e-3))
        return T, d0

    def probs(self, seq):
        """{sign: p, NO_SIGN: p} for a window."""
        d = self.distances(resample(seq))
        names = list(d) + [NO_SIGN]
        logits = -np.array(list(d.values()) + [self.d0]) / self.T
        p = np.exp(logits - logits.max())
        return dict(zip(names, (p / p.sum()).tolist()))

    def leave_one_out(self):
        rows = []
        for s, seqs in self.takes.items():
            for k, t in enumerate(seqs):
                d = self.distances(t, skip=(s, k))
                names = list(d) + [NO_SIGN]
                logits = -np.array(list(d.values()) + [self.d0]) / self.T
                rows.append((s, names[int(np.argmax(logits))]))
        return rows


class SignSpotter:
    """Live: feed hand landmarks as they arrive; probs() gives the current window's distribution, smoothed
    with an exponential moving average (half-life 0.3 s) so a single odd window does not flip it."""

    def __init__(self, library, window_s=1.5, min_frames=4, half_life_s=0.3):
        self.lib, self.window_s, self.min_frames, self.half_life_s = library, window_s, min_frames, half_life_s
        self.buf = deque()
        self.smooth, self.t = None, None

    def reset(self):
        self.buf.clear()
        self.smooth, self.t = None, None

    def update(self, t, hands):
        feat, present = frame_features(hands)
        self.buf.append((t, feat, present))
        while self.buf and t - self.buf[0][0] > self.window_s:
            self.buf.popleft()
        if not self.lib.signs:
            return {'status': 'no signs enrolled', 'probs': None}
        frames = [f for _, f, pres in self.buf if pres]
        if len(frames) < self.min_frames:
            raw = {s: 0.0 for s in self.lib.signs} | {NO_SIGN: 1.0}
        else:
            raw = self.lib.probs(np.array([f for _, f, _ in self.buf]))
        dt = 0.0 if self.t is None else max(t - self.t, 0.0)
        self.t = t
        a = 0.5 ** (dt / self.half_life_s) if self.smooth else 0.0
        self.smooth = {k: a * (self.smooth or {}).get(k, 0.0) + (1 - a) * v for k, v in raw.items()}
        top = max(self.smooth, key=self.smooth.get)
        return {'status': 'ok', 'probs': dict(self.smooth), 'label': top}


def load_meanings():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def enroll(signs, takes, camera, user='default', seconds=2.0):
    """Guided recording: for each sign, a countdown, then `seconds` of capture, `takes` times."""
    import cv2
    from video import GESTURE_URL, MODELS, PersonTracker, TargetSelector, crop_region, model_bytes
    import mediapipe as mp
    tracker, selector = PersonTracker(), TargetSelector()   # same person crop as the live app
    vision = mp.tasks.vision
    rec = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=model_bytes(GESTURE_URL, MODELS / 'gesture_recognizer.task')),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera')
    start, last_ts = time.monotonic(), -1
    region, last_track = None, -1.0

    def grab():
        nonlocal last_ts, region, last_track
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('camera read failed')
        ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
        last_ts = ts
        if ts / 1000 - last_track >= 0.5:            # the person barely moves while enrolling: re-track at 2 Hz
            people = tracker.update(frame, ts / 1000)
            person = next((p for p in people if p['id'] == selector.update(people)), None)
            region, last_track = (crop_region(frame, person['bbox']) if person else None), ts / 1000
        x1, y1, x2, y2 = region or (0, 0, frame.shape[1], frame.shape[0])
        crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        g = rec.recognize_for_video(image, ts)
        hands = [{'side': g.handedness[i][0].category_name if g.handedness[i] else 'Unknown',
                  'points': [[p.x, p.y] for p in h]} for i, h in enumerate(g.hand_landmarks)]
        view = cv2.flip(frame, 1)                       # features unmirrored (as live); the view mirrored
        W = frame.shape[1]
        cv2.rectangle(view, (W - x2, y1), (W - x1, y2), (90, 230, 190), 1)
        for h in hands:                                 # green dots: the hand landmarks being recorded
            for px, py in h['points']:
                cv2.circle(view, (int(W - (x1 + px * (x2 - x1))), int(y1 + py * (y2 - y1))), 3, (90, 220, 90), -1)
        return view, hands

    def show(frame, lines, color=(90, 230, 190)):
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (20, 40 + 36 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.imshow('Tink - teach your signs', frame)
        return cv2.waitKey(1) & 0xFF

    try:
        for sign in signs:
            out_dir = DATA / user / sign
            out_dir.mkdir(parents=True, exist_ok=True)
            k, skip = 0, False
            while k < takes and not skip:
                # start when the hands are up: at least 1 s to read the prompt, then hands visible for 3 frames in
                # a row (a fixed countdown made the first takes of each sign miss the hands)
                t0, streak = time.monotonic(), 0
                while True:
                    frame, hands = grab()
                    streak = streak + 1 if hands else 0
                    ready = time.monotonic() - t0 >= 1.0 and streak >= 3
                    key = show(frame, [f'Sign: {sign}   take {k + 1}/{takes}',
                                       'Starting...' if ready else 'Raise your hands to start',
                                       'S = skip sign, Q = quit'])
                    if key == ord('q'):
                        raise KeyboardInterrupt
                    if key == ord('s'):
                        skip = True
                        break
                    if ready:
                        break
                if skip:
                    break
                seq, seen = [], 0
                t0 = time.monotonic()
                while time.monotonic() - t0 < seconds:
                    frame, hands = grab()
                    feat, present = frame_features(hands)
                    seq.append(feat)
                    seen += present
                    show(frame, [f'Sign: {sign}   RECORDING', f'hands seen in {seen} frames'], (60, 60, 255))
                if seen < 3:
                    print(f'{sign} take {k + 1}: hands in {seen}/{len(seq)} frames - not saved, again')
                    show(frame, ['No hands seen - keep both hands in the box'], (60, 160, 255))
                    time.sleep(0.8)
                    continue
                np.save(out_dir / f'take_{k}.npy', np.array(seq, np.float32))
                print(f'{sign} take {k + 1}: hands in {seen}/{len(seq)} frames - saved')
                k += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rec.close()
    print('saved under', DATA / user)


def selftest():
    """Synthetic motions: each 'sign' is a smooth random trajectory of hand features; takes and test windows
    add speed changes and noise. Checks DTW, calibration and the no-sign option, not real-world accuracy."""
    import tempfile
    global DATA
    rng = np.random.default_rng(0)
    base = {s: np.cumsum(rng.normal(0, 0.15, (14, DIM)), axis=0) for s in ['YES', 'NO', 'HELP', 'MORE', 'WATER']}

    def variant(seq, speed, noise):
        n = max(4, int(len(seq) * speed))
        return resample(seq, n) + rng.normal(0, noise, (n, DIM))

    with tempfile.TemporaryDirectory() as tmp:
        DATA = Path(tmp)
        for s, seq in base.items():
            (DATA / 'default' / s).mkdir(parents=True)
            for k in range(4):
                np.save(DATA / 'default' / s / f'take_{k}.npy', variant(seq, rng.uniform(0.7, 1.4), 0.08))
        lib = SignLibrary()
        rows = lib.leave_one_out()
        print(f'enrolled {lib.signs}; T={lib.T:.3f} d0={lib.d0:.3f}')
        print(f'leave-one-take-out accuracy: {np.mean([a == b for a, b in rows]):.2f} ({len(rows)} takes)')
        hits = sum(max(lib.probs(variant(seq, rng.uniform(0.7, 1.4), 0.12)).items(), key=lambda x: x[1])[0] == s
                   for s, seq in base.items() for _ in range(10))
        print(f'new noisier performances recognised: {hits}/50')
        junk = [max(lib.probs(np.cumsum(rng.normal(0, 0.15, (12, DIM)), axis=0)).items(), key=lambda x: x[1])[0]
                for _ in range(30)]
        print(f'unrelated motions called no-sign: {junk.count(NO_SIGN)}/30')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('enroll')
    e.add_argument('--signs', nargs='+', default=[s for s in load_meanings()['signs']])
    e.add_argument('--takes', type=int, default=4)
    e.add_argument('--camera', type=int, default=0)
    e.add_argument('--user', default='default')
    c = sub.add_parser('check')
    c.add_argument('--user', default='default')
    sub.add_parser('selftest')
    a = p.parse_args()
    if a.cmd == 'enroll':
        enroll(a.signs, a.takes, a.camera, a.user)
    elif a.cmd == 'check':
        lib = SignLibrary(a.user)
        rows = lib.leave_one_out()
        print(f'{len(lib.signs)} signs, T={lib.T:.3f}, d0={lib.d0:.3f}')
        print(f'leave-one-take-out accuracy: {np.mean([x == y for x, y in rows]):.2f} over {len(rows)} takes')
        for x, y in rows:
            if x != y:
                print(f'  {x} take read as {y}')
    else:
        selftest()


if __name__ == '__main__':
    main()
