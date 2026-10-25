"""Webcam ASL fingerspelling aid. Static alphabet classifier, not ASL translation.

Install: pip install opencv-python mediapipe transformers torch pillow numpy
Run: python asl.py            (standalone fingerspelling app)
Inside Tink, FingerspellReader runs the same box / classifier / stability logic on the tracked user's hand.
"""
import argparse
import queue
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image

MODEL = 'abdollahhh/asl-sign-language-efficientnet-b0'
LETTERS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def speak_worker(q):
    while True:
        item = q.get()
        if item is None:
            return
        try:
            if sys.platform == 'darwin':
                subprocess.run(['say', item], timeout=30, check=False)
            elif sys.platform.startswith('win'):
                import win32com.client
                win32com.client.Dispatch('SAPI.SpVoice').Speak(item)
            else:
                subprocess.run(['espeak', item], timeout=30, check=False)
        except Exception as exc:
            print('Speech unavailable:', exc)


def draw(im, text, pos, scale=.55, color=(240, 240, 240)):
    cv2.putText(im, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def hand_box(xx, yy, w, h):
    """Square box around one hand's landmark pixels: 1.8x the larger side, at least 110 px, clipped."""
    xmin, xmax = xx.min(), xx.max()
    ymin, ymax = yy.min(), yy.max()
    side = max(xmax - xmin, ymax - ymin) * 1.8
    side = max(side, 110)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side / 2))
    x2 = min(w, int(cx + side / 2))
    y2 = min(h, int(cy + side / 2))
    return (x1, y1, x2, y2) if x2 > x1 + 32 and y2 > y1 + 32 else None


class LetterClassifier:
    """The EfficientNet-B0 A-Z classifier: an RGB hand crop -> normalized letter scores."""

    def __init__(self, model=MODEL):
        from transformers import pipeline
        self.pipe = pipeline('image-classification', model=model, device=-1)
        # every label (26, A-Z): newer transformers reads top_k=None as its default of 5
        self.top_k = int(self.pipe.model.config.num_labels)

    def scores(self, crop_rgb):
        pred = self.pipe(Image.fromarray(crop_rgb), top_k=self.top_k)
        scores = {}
        for item in pred:
            name = str(item['label']).strip().upper()
            if name in LETTERS:
                scores[name] = float(item['score'])
        if not scores:
            raise ValueError('Unexpected model labels: ' + str(pred[:2])[:65])
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}


class LetterStabilizer:
    """A letter is 'ready' only when recent predictions agree: at least stable_frames snapshots, the leader
    winning >= 75% of them, averaged confidence >= threshold and >= margin above the runner-up."""

    def __init__(self, stable_frames=6, threshold=.72, margin=.16):
        self.stable_frames, self.threshold, self.margin = stable_frames, threshold, margin
        self.history = deque(maxlen=max(stable_frames * 2, 12))

    def clear(self):
        self.history.clear()

    def add(self, scores):
        self.history.append(scores)

    def state(self):
        history = self.history
        avg = {}
        for snapshot in history:
            for label, prob in snapshot.items():
                avg[label] = avg.get(label, 0.) + prob / len(history)
        ranked = sorted(avg.items(), key=lambda z: z[1], reverse=True)
        best, confidence = ranked[0] if ranked else ('-', 0.)
        second = ranked[1][1] if len(ranked) > 1 else 0.
        votes = sum(max(s, key=s.get) == best for s in history) if ranked else 0
        ready = (len(history) >= self.stable_frames and
                 votes >= max(self.stable_frames, int(len(history) * .75)) and
                 confidence >= self.threshold and
                 confidence - second >= self.margin)
        return {'best': best, 'confidence': confidence, 'second': second, 'votes': votes, 'n': len(history),
                'ranked': ranked[:6], 'ready': ready}


class FingerspellReader:
    """For Tink: one hand's landmarks (normalized to the user crop) + that crop -> the letter state. Same rules
    as the standalone app: exactly one hand, the square box, the classifier on the mirrored crop (the app
    mirrors the view by default), history cleared on a big box jump or when the hand is lost."""

    def __init__(self, classifier=None, mirror=True, **stability):
        self.classifier = classifier or LetterClassifier()
        self.stab = LetterStabilizer(**stability)
        self.mirror = mirror
        self.last_box = None

    def update(self, crop_bgr, hands):
        h, w = crop_bgr.shape[:2]
        if len(hands or []) != 1:
            self.stab.clear()
            self.last_box = None
            return {**self.stab.state(), 'status': 'show exactly one hand' if hands else 'no hand'}
        pts = np.asarray(hands[0]['points'], float)
        box = hand_box(pts[:, 0] * w, pts[:, 1] * h, w, h)
        if box is None:
            self.stab.clear()
            self.last_box = None
            return {**self.stab.state(), 'status': 'move hand into view'}
        if self.last_box is not None and np.linalg.norm(np.array(box, float) - np.array(self.last_box, float)) > max(h, w) * .20:
            self.stab.clear()                      # a big jump must not carry confidence from an old gesture
        self.last_box = box
        x1, y1, x2, y2 = box
        crop = cv2.cvtColor(crop_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        if self.mirror:
            crop = np.ascontiguousarray(crop[:, ::-1])
        try:
            self.stab.add(self.classifier.scores(crop))
            status = 'ok'
        except Exception as exc:
            self.stab.clear()
            status = 'error: ' + str(exc)[:60]
        return {**self.stab.state(), 'status': status}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--interval', type=float, default=.12)
    p.add_argument('--threshold', type=float, default=.72)
    p.add_argument('--margin', type=float, default=.16,
                   help='Minimum gap between first and second predictions')
    p.add_argument('--stable-frames', type=int, default=6)
    p.add_argument('--no-mirror', action='store_true')
    args = p.parse_args()

    try:
        import mediapipe as mp
        # MediaPipe 0.10.21 and earlier provide mp.solutions; newer builds may not.
        hands_api = mp.solutions.hands
    except (ImportError, AttributeError) as exc:
        raise RuntimeError('Install compatible hand tracking: pip install "mediapipe==0.10.21"') from exc

    print('Loading classifier (first run may download weights)...')
    classifier = LetterClassifier()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera; check camera permissions and --camera index.')

    q = queue.Queue()
    threading.Thread(target=speak_worker, args=(q,), daemon=True).start()
    stab = LetterStabilizer(args.stable_frames, args.threshold, args.margin)
    message = ''
    last_infer = 0.0
    status = 'Hold one hand steady. Press A to confirm a letter.'
    last_box = None
    with hands_api.Hands(static_image_mode=False, max_num_hands=2,
                         min_detection_confidence=.65,
                         min_tracking_confidence=.6) as hands:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                found = result.multi_hand_landmarks or []
                box = None
                if len(found) == 1:
                    pts = found[0].landmark
                    box = hand_box(np.array([lm.x * w for lm in pts]), np.array([lm.y * h for lm in pts]), w, h)
                if box is None:
                    stab.clear()
                    last_box = None
                    status = 'Show exactly one hand' if len(found) != 1 else 'Move hand into camera view'
                else:
                    if last_box is not None:
                        a = np.array(box, dtype=float)
                        b = np.array(last_box, dtype=float)
                        # Large ROI jumps should not carry confidence from an old gesture.
                        if np.linalg.norm(a - b) > max(h, w) * .20:
                            stab.clear()
                    last_box = box
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 230, 140), 2)
                    if time.monotonic() - last_infer >= args.interval:
                        last_infer = time.monotonic()
                        try:
                            stab.add(classifier.scores(rgb[y1:y2, x1:x2]))
                        except Exception as exc:
                            stab.clear()
                            status = 'Inference failed: ' + str(exc)[:65]

                st = stab.state()
                best, confidence, second = st['best'], st['confidence'], st['second']
                votes, ranked, ready = st['votes'], st['ranked'], st['ready']

                panel = np.zeros((max(h, 580), w + 420, 3), dtype=np.uint8)
                panel[:h, :w] = frame
                px = w + 15
                draw(panel, 'ASL FINGERSPELLING', (px, 35), .7, (100, 215, 255))
                draw(panel, 'Candidate:', (px, 80))
                draw(panel, f'{best}   {confidence * 100:.1f}%', (px, 119), 1.,
                     (90, 240, 120) if ready else (120, 160, 255))
                draw(panel, 'STABLE - press A' if ready else 'WAIT / UNCERTAIN', (px, 150), .55)
                draw(panel, f'Votes: {votes}/{st["n"]}  Gap: {(confidence-second)*100:.1f}%',
                     (px, 179), .46)
                for i, (label, prob) in enumerate(ranked[:6]):
                    yy = 215 + 30 * i
                    draw(panel, f'{label}: {prob * 100:5.1f}%', (px, yy), .52)
                    cv2.rectangle(panel, (px + 145, yy - 13),
                                  (px + 145 + int(190 * prob), yy - 4), (80, 190, 240), -1)
                draw(panel, 'Text:', (px, 424), .65)
                draw(panel, message[-31:] or '(empty)', (px, 459), .57, (120, 235, 160))
                draw(panel, 'A add  SPACE space  D delete', (px, 496), .48)
                draw(panel, 'C clear  S speak  Q quit', (px, 525), .48)
                draw(panel, status[:55], (px, 558), .42, (155, 180, 245))
                cv2.imshow('ASL fingerspelling', panel)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    if ready:
                        message += best
                        stab.clear()  # force a new stable prediction before next confirmation
                        status = f'Added {best}'
                    else:
                        status = 'Not stable/confident enough; adjust pose or lighting'
                elif key == 32:
                    message += ' '
                elif key == ord('d'):
                    message = message[:-1]
                elif key == ord('c'):
                    message = ''
                elif key == ord('s') and message.strip():
                    q.put(message)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            q.put(None)


if __name__ == '__main__':
    main()
