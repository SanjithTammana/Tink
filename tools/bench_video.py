"""Per-model CPU cost on a recorded clip, and whether Pose Landmarker can replace Holistic for Envision.

Runs every model on every frame, one after another in one process (as the live loop would), and reports
median and p90 milliseconds per frame. Then compares the Envision outputs computed from Holistic world landmarks
(what the package was trained on) with those from Pose Landmarker lite and full.

    python tools/bench_video.py .scratch_video/webcam_session.mp4 [--max-frames 600] [--holistic]
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envision_port import EnvisionCNN, EnvisionLGBM, SEQ_CNN  # noqa: E402

MODELS = ROOT / 'models'


def world92(landmarks):
    """Pose world landmarks 0-22 -> 92 values (x, y, z, visibility), as the package builds them."""
    return np.array([[p.x, p.y, p.z, p.visibility] for p in landmarks[:23]], dtype=np.float32).reshape(-1)


def tasks():
    import mediapipe as mp
    v, base = mp.tasks.vision, mp.tasks.BaseOptions
    video = v.RunningMode.VIDEO
    out = {}
    for name in ('lite', 'full'):
        out[f'pose_{name}'] = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(MODELS / f'pose_landmarker_{name}.task')), running_mode=video))
    out['face'] = v.FaceLandmarker.create_from_options(v.FaceLandmarkerOptions(
        base_options=base(model_asset_path=str(MODELS / 'face_landmarker.task')), running_mode=video, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    out['hands'] = v.GestureRecognizer.create_from_options(v.GestureRecognizerOptions(
        base_options=base(model_asset_path=str(MODELS / 'gesture_recognizer.task')), running_mode=video,
        num_hands=2))
    return mp, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--max-frames', type=int, default=600)
    ap.add_argument('--holistic', action='store_true', help='also run legacy Holistic (needs mp.solutions)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    mp, models = tasks()
    holistic = None
    if args.holistic:
        holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1,
                                                  enable_segmentation=False, smooth_landmarks=True,
                                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cnn, lgbm = EnvisionCNN(), EnvisionLGBM()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ms = {k: [] for k in ['holistic', 'pose_lite', 'pose_full', 'face', 'hands', 'cnn', 'lgbm']}
    sources = ['pose_lite', 'pose_full'] + (['holistic'] if holistic else [])
    buffers = {s: deque(maxlen=SEQ_CNN) for s in sources}
    preds = {s: [] for s in sources}
    i = 0
    while i < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(i * 1000 / fps)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        landmarks = {}
        for name in ('pose_lite', 'pose_full'):
            t = time.perf_counter()
            r = models[name].detect_for_video(image, ts)
            ms[name].append(1000 * (time.perf_counter() - t))
            landmarks[name] = world92(r.pose_world_landmarks[0]) if r.pose_world_landmarks else None
        if holistic:
            t = time.perf_counter()
            r = holistic.process(rgb)
            ms['holistic'].append(1000 * (time.perf_counter() - t))
            landmarks['holistic'] = world92(r.pose_world_landmarks.landmark) if r.pose_world_landmarks else None
        for name, call in (('face', models['face'].detect_for_video), ('hands', models['hands'].recognize_for_video)):
            t = time.perf_counter()
            call(image, ts)
            ms[name].append(1000 * (time.perf_counter() - t))
        for s in sources:
            if landmarks[s] is None:
                buffers[s].clear()      # the package drops its window when pose is lost
                preds[s].append(None)
                continue
            buffers[s].append(landmarks[s])
            if len(buffers[s]) < SEQ_CNN:
                preds[s].append(None)
                continue
            seq = np.array(buffers[s])
            t = time.perf_counter()
            c = cnn(seq)
            ms['cnn'].append(1000 * (time.perf_counter() - t))
            t = time.perf_counter()
            g = lgbm(seq)
            ms['lgbm'].append(1000 * (time.perf_counter() - t))
            preds[s].append([c['motion'], c['gesture'], c['move'], g])
        i += 1
    report = {'frames': i, 'video_fps': fps,
              'ms_per_frame': {k: {'median': round(float(np.median(v)), 2), 'p90': round(float(np.percentile(v, 90)), 2)}
                               for k, v in ms.items() if v}}
    ref = 'holistic' if holistic else 'pose_full'
    agree = {}
    for s in sources:
        if s == ref:
            continue
        pairs = [(a, b) for a, b in zip(preds[ref], preds[s]) if a is not None and b is not None]
        if not pairs:
            continue
        a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        agree[f'{s}_vs_{ref}'] = {
            'windows': len(pairs),
            **{name: {'mean_abs_diff': round(float(np.abs(a[:, j] - b[:, j]).mean()), 3),
                      'corr': round(float(np.corrcoef(a[:, j], b[:, j])[0, 1]), 3)}
               for j, name in enumerate(['cnn_motion', 'cnn_gesture', 'cnn_move', 'lgbm_gesture'])}}
    report['agreement'] = agree
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({'report': report, 'preds': preds}), encoding='utf-8')


if __name__ == '__main__':
    main()
