"""Time one MediaPipe task alone in a fresh process: python tools/bench_one.py <pose_lite|pose_full|face|hands> <video>

Separates a model's own cost from contention with other models in the same process.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'


def build(name):
    import mediapipe as mp
    v, base = mp.tasks.vision, mp.tasks.BaseOptions
    video = v.RunningMode.VIDEO
    if name.startswith('pose'):
        task = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(MODELS / f'pose_landmarker_{name[5:]}.task')), running_mode=video))
        return mp, task.detect_for_video
    if name == 'face':
        task = v.FaceLandmarker.create_from_options(v.FaceLandmarkerOptions(
            base_options=base(model_asset_path=str(MODELS / 'face_landmarker.task')), running_mode=video,
            num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True))
        return mp, task.detect_for_video
    task = v.GestureRecognizer.create_from_options(v.GestureRecognizerOptions(
        base_options=base(model_asset_path=str(MODELS / 'gesture_recognizer.task')), running_mode=video, num_hands=2))
    return mp, task.recognize_for_video


def main():
    name, path = sys.argv[1], sys.argv[2]
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    mp, call = build(name)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ms, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        t = time.perf_counter()
        call(image, int(i * 1000 / fps))
        ms.append(1000 * (time.perf_counter() - t))
        i += 1
    ms = np.array(ms[10:])          # skip warm-up
    print(f'{name} scale={scale}: median {np.median(ms):.1f} ms, p90 {np.percentile(ms, 90):.1f} ms over {len(ms)} frames')


if __name__ == '__main__':
    main()
