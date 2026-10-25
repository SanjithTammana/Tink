"""Run one video channel, or the complete visual layer, without fusion or TTS.

Examples (from tink-alt, using the original checkout's existing environment):
  ../tink/.venv/Scripts/python tools/video_layer_lab.py --video ../tink/.scratch_video/webcam_session.mp4 --target-id 1 --channels pose --headless
  ../tink/.venv/Scripts/python tools/video_layer_lab.py --camera 0 --record-only
  ../tink/.venv/Scripts/python tools/video_layer_lab.py --camera 0 --channels face

Click a person's box or press N to select the user. No automatic target fallback.
No package installs or model downloads. Output is private under
.scratch_video_layer/. Press 0-9 to label a trial; Q quits. Labels describe what
the operator performed, not a model's prediction.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video_layer import (PoseClock, Segment, cnn_distribution, face_features,
                         hand_features, landmarks_array, pose_features)
from tools.video_overlay import render_tracking
from person_selection import TargetLock, crop_bounds, full_image_landmarks

LABELS = {str(i): name for i, name in enumerate([
    'idle', 'thumb_up', 'thumb_down', 'nod', 'shake', 'eyebrow_raise',
    'mouth_open', 'wave', 'point', 'other_voluntary'])}
FACE_SWITCHES = ['eyeBlinkLeft', 'eyeBlinkRight', 'browInnerUp', 'browOuterUpLeft',
                 'browOuterUpRight', 'jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'mouthPucker']


class LiveCapture:
    """Record continuously while inference reads the latest timestamped frame.

    Preview landmarks belong to the analyzed frame. Raw recording includes frames
    inference skips. The clock is host camera-read time, not sensor exposure time.
    """
    def __init__(self, cap, folder=None, label='unlabeled', time_source=None, frame_callback=None):
        self.cap, self.folder, self.label = cap, folder, label
        self.condition, self.stop = threading.Condition(), threading.Event()
        self.latest, self.done, self.error = None, False, None
        self.captured = 0
        self.time_source = time_source
        self.frame_callback = frame_callback
        self.thread = threading.Thread(target=self._run, name='video-lab-capture', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        writer = times = None
        origin = time.perf_counter()
        try:
            if self.folder is not None:
                times = (self.folder / 'capture_times.jsonl').open('w', encoding='utf-8')
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.
            while not self.stop.is_set():
                ok, frame = self.cap.read()
                t = self.time_source() if self.time_source is not None else time.perf_counter() - origin
                if not ok:
                    break
                if self.frame_callback is not None:
                    self.frame_callback(frame, t)
                with self.condition:
                    label = self.label
                index = self.captured
                if self.folder is not None:
                    if writer is None:
                        writer = cv2.VideoWriter(str(self.folder / 'capture.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                 (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('could not create camera recording')
                    writer.write(frame)
                    times.write(json.dumps({'frame': index, 't_capture': t, 'label': label}) + '\n')
                with self.condition:
                    self.captured += 1
                    self.latest = index, t, frame, label
                    self.condition.notify_all()
        except Exception as error:
            self.error = error
        finally:
            if times:
                times.close()
            if writer:
                writer.release()
            self.cap.release()
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def read_after(self, index):
        with self.condition:
            self.condition.wait_for(lambda: self.done or (self.latest is not None and self.latest[0] > index))
            if self.error:
                raise RuntimeError('camera capture failed') from self.error
            return self.latest if self.latest is not None and self.latest[0] > index else None

    def set_label(self, label):
        with self.condition:
            self.label = label

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3.)
        if self.thread.is_alive():
            self.cap.release()
            self.thread.join(timeout=2.)
        if self.thread.is_alive():
            raise RuntimeError('camera did not stop; recording finalization is unconfirmed')
        if self.error:
            raise RuntimeError('camera recording failed') from self.error


def record_trial(args, folder):
    """Capture without inference; host read timestamps are authoritative, not MP4 FPS."""
    cap, writer = cv2.VideoCapture(args.camera), None
    count, label, start = 0, args.label, time.perf_counter()
    try:
        if not cap.isOpened():
            raise RuntimeError('could not open camera')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.  # encoder only; analysis uses the timestamp sidecar
        with (folder / 'capture_times.jsonl').open('w', encoding='utf-8') as times:
            while not args.max_frames or count < args.max_frames:
                ok, frame = cap.read()
                t = time.perf_counter() - start
                if not ok:
                    break
                if writer is None:
                    writer = cv2.VideoWriter(str(folder / 'capture.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                             (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('could not create recording')
                writer.write(frame)
                times.write(json.dumps({'frame': count, 't_capture': t, 'label': label}) + '\n')
                count += 1
                if not args.headless:
                    shown = frame.copy()
                    cv2.putText(shown, f'label={label} | 0-9 labels, Q quit | {t:.1f}s', (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                    cv2.imshow('Tink record only (no inference)', shown)
                    key = cv2.waitKey(1) & 255
                    if key in (ord('q'), ord('Q'), 27):
                        break
                    if chr(key) in LABELS:
                        label = LABELS[chr(key)]
        (folder / 'manifest.json').write_text(json.dumps({'mode': 'record-only', 'frames': count,
            'timestamp_basis': 'host_after_camera_read; sensor exposure timestamp unavailable',
            'encoded_fps': fps, 'labels': LABELS,
            'notes': 'Analyze capture.mp4 with capture_times.jsonl beside it. MP4 playback timing is nominal; the sidecar records read times.'}, indent=2), encoding='utf-8')
        print(f'Recorded {count} frames. PowerShell test command:\n& "{sys.executable}" "{Path(__file__).resolve()}" --video "{folder / "capture.mp4"}"', flush=True)
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--camera', type=int)
    ap.add_argument('--channels', nargs='+', choices=['pose', 'face', 'hands'])
    ap.add_argument('--pose-model', choices=['lite', 'full'], default='full')
    ap.add_argument('--models', type=Path, default=ROOT / 'models')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--record', action='store_true', help='also save original camera/video frames locally')
    ap.add_argument('--save-preview', action='store_true', help='save an annotated preview PNG for inspection')
    ap.add_argument('--record-only', action='store_true', help='record a labeled camera trial without running any model')
    ap.add_argument('--label', default='unlabeled')
    ap.add_argument('--person-id', default=None, help='optional operator label; actual ownership uses the selected local tracker ID')
    ap.add_argument('--target-id', type=int, help='explicit local tracker ID for an unattended clip test')
    ap.add_argument('--person-model', type=Path, default=ROOT.parent / 'tink' / 'yolo11n-pose.pt')
    ap.add_argument('--neutral-seconds', type=float, default=2., help='remain neutral at the start of a face trial')
    args = ap.parse_args()
    args.channels = args.channels or ['pose', 'face', 'hands']
    if args.record_only and args.camera is None:
        ap.error('--record-only needs --camera')
    if args.neutral_seconds < 0 or args.max_frames < 0:
        ap.error('durations/frame limits must be nonnegative')
    if args.camera is not None and args.headless and not args.max_frames:
        ap.error('headless camera needs --max-frames')
    folder = args.out or ROOT / '.scratch_video_layer' / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    if args.record_only:
        folder.mkdir(parents=True, exist_ok=False)
        (folder / '.gitignore').write_text('*\n', encoding='utf-8')
        print(f'Output: {folder}', flush=True)
        record_trial(args, folder)
        return
    if args.headless and args.target_id is None:
        ap.error('headless inference needs --target-id; interactive mode uses click/N selection')
    if not args.person_model.is_file():
        ap.error(f'missing local person model: {args.person_model}; pass --person-model (no download)')
    assets = {'pose': f'pose_landmarker_{args.pose_model}.task',
              'face': 'face_landmarker.task', 'hands': 'gesture_recognizer.task'}
    for channel in args.channels:
        if not (args.models / assets[channel]).is_file():
            ap.error(f'missing model: {args.models / assets[channel]} (no automatic download)')
    if 'pose' in args.channels:
        for name in ['cnn_default.npz', 'lightgbm_default.txt', 'lightgbm_default.json']:
            if not (ROOT / 'models' / 'envision' / name).is_file():
                ap.error(f'missing exported Envision asset: models/envision/{name}')
    folder.mkdir(parents=True, exist_ok=False)
    # Keep raw measurements/recordings private even if --out overrides the default.
    (folder / '.gitignore').write_text('*\n', encoding='utf-8')
    import mediapipe as mp
    os.environ['YOLO_AUTOINSTALL'] = 'False'
    from video import PersonTracker
    v, base = mp.tasks.vision, mp.tasks.BaseOptions
    models = {}
    cap = writer = stream = recorded_times = camera_source = None
    frame_count, event_count = 0, 0
    costs = {c: [] for c in args.channels}
    costs['person_tracker'] = []
    target_lock = TargetLock(args.target_id)
    active_target = (None, target_lock.generation)
    mouse_view = {}

    def click_target(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or not mouse_view:
            return
        width, height = mouse_view['size']
        scale = min(960 / width, 720 / height, 1.)
        shown_w, shown_h = round(width * scale), round(height * scale)
        if x < shown_w and y < shown_h:
            target_lock.select_at(x * width / shown_w, y * height / shown_h, mouse_view['people'])
    clock, captured = PoseClock(), deque(maxlen=50)
    session = uuid.uuid4().hex
    segments = {name: Segment(name, session) for name in ['body_motion', 'face', 'head_posture', 'hands']}
    neutral_samples, neutral = [], None
    calibration_start = None
    cnn = lgbm = None
    label, last_t, started = args.label, 0., time.perf_counter()
    preview_times = deque(maxlen=30)

    def build_channel(channel):
        opts = dict(base_options=base(model_asset_path=str(args.models / assets[channel])), running_mode=v.RunningMode.VIDEO)
        if channel == 'pose':
            return v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(num_poses=1, **opts))
        if channel == 'face':
            return v.FaceLandmarker.create_from_options(v.FaceLandmarkerOptions(
                num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True, **opts))
        classifiers = mp.tasks.components.processors.ClassifierOptions(max_results=-1, score_threshold=0.)
        return v.GestureRecognizer.create_from_options(v.GestureRecognizerOptions(
            num_hands=2, canned_gesture_classifier_options=classifiers, **opts))

    try:
        tracker = PersonTracker(model=str(args.person_model.resolve()))
        for channel in args.channels:
            models[channel] = build_channel(channel)
            if channel == 'pose':
                from envision_port import EnvisionCNN, EnvisionLGBM
                cnn, lgbm = EnvisionCNN(), EnvisionLGBM()
        cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
        if not cap.isOpened():
            raise RuntimeError('could not open the requested video/camera')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            if args.video:
                raise RuntimeError('source FPS is unavailable; cannot construct a trustworthy capture clock')
            fps = 30.  # recording encoder only; camera uses host-read timestamps
        times_path = args.video.with_name('capture_times.jsonl') if args.video and args.video.name == 'capture.mp4' else None
        capture_times = [json.loads(line) for line in times_path.read_text(encoding='utf-8').splitlines()] if times_path and times_path.is_file() else None
        stream = (folder / 'observations.jsonl').open('w', encoding='utf-8')

        def emit(value):
            nonlocal event_count
            if value is not None:
                stream.write(json.dumps(value, allow_nan=False) + '\n')
                if value['type'] == 'event':
                    event_count += 1
                    print(f"event {value['source']} {value['t_start']:.2f}-{value['t_end']:.2f}s {value['end_reason']}", flush=True)

        manifest = {'schema': 1, 'session': session, 'channels': args.channels, 'source_fps': fps,
                    'source': str(args.video) if args.video else f'camera:{args.camera}',
                    'pose_model': args.pose_model, 'person_label': args.person_id,
                    'attribution': 'explicit selection of a local person track; no biometric identity',
                    'initial_target_id': args.target_id,
                    'calibrated': False, 'labels': LABELS,
                    'neutral_seconds': args.neutral_seconds,
                    'recording': args.record,
                    'timestamp_basis': 'host_camera_read_sidecar' if capture_times else ('constant_source_fps' if args.video else 'host_after_camera_read'),
                    'notes': 'Multiple people detected; features only from explicitly selected isolated target. Loss/overlap withholds evidence. Scores are uncalibrated.'}
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'Output: {folder}\nClick a person or N to select. Keep neutral after selection. Keys 0-9 label; Q quits.', flush=True)
        capture_origin = time.perf_counter()
        source_index, skipped_frames = -1, 0
        if args.camera is not None:
            camera_source = LiveCapture(cap, folder if args.record else None, label).start()
        while not args.max_frames or frame_count < args.max_frames:
            if camera_source is not None:
                sample = camera_source.read_after(source_index)
                if sample is None:
                    break
                index, t, frame, label = sample
                skipped_frames += index - source_index - 1
                source_index = index
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            if camera_source is not None:
                pass  # t and label belong to the captured frame, before inference
            elif capture_times is not None:
                if frame_count >= len(capture_times) or capture_times[frame_count]['frame'] != frame_count:
                    raise ValueError('capture timestamp sidecar does not match video frames')
                t, label = capture_times[frame_count]['t_capture'], capture_times[frame_count]['label']
            else:
                t = frame_count / fps if args.video else time.perf_counter() - capture_origin
            last_t = t
            ts = int(round(t * 1000))
            begin = time.perf_counter()
            people = tracker.update(frame, t)
            tracker_ms = 1000 * (time.perf_counter() - begin)
            costs['person_tracker'].append(tracker_ms)
            selected = target_lock.update(people)
            bounds = crop_bounds(selected['bbox'], frame.shape) if selected else None
            if selected is not None and bounds is None:
                target_lock.status = 'selected person crop too small'
            pid = selected['id'] if bounds else None
            token = (pid, target_lock.generation)
            if token != active_target:
                for segment in segments.values():
                    emit(segment.finish(t, 'target_changed'))
                clock, captured = PoseClock(), deque(maxlen=50)
                neutral_samples, neutral, calibration_start = [], None, None
                segments = {name: Segment(name, f'{session}-g{target_lock.generation}', pid)
                            for name in ['body_motion', 'face', 'head_posture', 'hands']}
                for name in models:
                    models[name].close()
                    models[name] = build_channel(name)
                active_target = token
            image = None
            if bounds is not None:
                x1, y1, x2, y2 = bounds
                crop = frame[y1:y2, x1:x2]
                rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            obs = {'type': 'observation', 'frame': frame_count, 't_capture': t, 'label': label,
                   'source_frame': source_index if camera_source else frame_count,
                   'skipped_inference_frames': skipped_frames,
                   'person_id': pid, 'target_id': target_lock.target_id,
                   'target_generation': target_lock.generation, 'target_status': target_lock.status,
                   'target_crop': bounds, 'image_size': [frame.shape[1], frame.shape[0]],
                   'people': [{'id': p['id'], 'bbox': p['bbox'], 'score': p['score']} for p in people],
                   'person_tracker_ms': tracker_ms, 'channels': {}}
            for name, model in models.items():
                if image is None:
                    obs['channels'][name] = {'status': 'no selected visible user'}
                    continue
                begin = time.perf_counter()
                result = model.recognize_for_video(image, ts) if name == 'hands' else model.detect_for_video(image, ts)
                cost = 1000 * (time.perf_counter() - begin)
                costs[name].append(cost)
                value = {'model_ms': cost, 'status': 'missing'}
                if name == 'pose':
                    pose = landmarks_array(result.pose_world_landmarks[0]) if result.pose_world_landmarks else None
                    grid = clock.push(t, pose[:23].reshape(92) if pose is not None else None)
                    if clock.discontinuity:
                        emit(segments['body_motion'].finish(t, 'tracking_lost'))
                        captured.clear()
                    if pose is not None:
                        captured.append(t)
                        value.update(status='ok', features=pose_features(pose), pose_world=pose.tolist(),
                                     pose_image=full_image_landmarks(landmarks_array(result.pose_landmarks[0]), bounds, frame.shape),
                                     observed_hz=(len(captured) - 1) / (t - captured[0]) if len(captured) > 1 else None,
                                     grid_hz=25, resampled=bool(grid))
                        if len(clock.history) == 25 and grid:
                            seq = np.array([p for _, p in clock.history])
                            begin = time.perf_counter()
                            heads = cnn(seq)
                            features = lgbm.features(seq)
                            score = lgbm(features=features)
                            value.update(cnn_heads=heads, cnn_distribution=cnn_distribution(heads),
                                         lgbm_gesture=score, lgbm_features=features.tolist(), classifier_ms=1000 * (time.perf_counter() - begin))
                            for event in segments['body_motion'].update(t, heads['motion'],
                                    {**{'cnn_' + k: val for k, val in heads.items()}, 'lgbm_gesture': score}):
                                emit(event)
                    else:
                        emit(segments['body_motion'].finish(t, 'tracking_lost'))
                elif name == 'face':
                    if result.face_landmarks:
                        blends = {c.category_name: float(c.score) for c in result.face_blendshapes[0]} if result.face_blendshapes else {}
                        matrix = result.facial_transformation_matrixes[0] if len(result.facial_transformation_matrixes) else None
                        feats = face_features(blends, matrix, landmarks_array(result.face_landmarks[0]), crop.shape[1] / crop.shape[0])
                        flat = {**blends, **{k: feats[k] for k in ['pitch_deg', 'yaw_deg', 'roll_deg'] if k in feats}}
                        value.update(status='ok', features=feats,
                                     landmarks_image=full_image_landmarks(landmarks_array(result.face_landmarks[0]), bounds, frame.shape))
                        if calibration_start is None:
                            calibration_start = t
                        if neutral is None:
                            neutral_samples.append(flat)
                            if t - calibration_start >= args.neutral_seconds and len(neutral_samples) >= 5:
                                neutral = {k: float(np.median([s[k] for s in neutral_samples if k in s])) for k in flat}
                                (folder / 'neutral.json').write_text(json.dumps(neutral, indent=2), encoding='utf-8')
                                (folder / f'neutral-track{pid}-g{target_lock.generation}.json').write_text(
                                    json.dumps(neutral, indent=2), encoding='utf-8')
                        else:
                            deltas = {k: max(0., (blends.get(k, 0.) - neutral.get(k, 0.)) / max(.1, 1 - neutral.get(k, 0.))) for k in FACE_SWITCHES}
                            activation = min(1., max(deltas.values(), default=0.))
                            head = min(1., max([abs(flat[k] - neutral.get(k, flat[k])) / 15 for k in ['pitch_deg', 'yaw_deg', 'roll_deg'] if k in flat], default=0.))
                            value.update(neutral_deltas=deltas, activation=activation)
                            for source_name, strength in [('face', activation), ('head_posture', head)]:
                                for event in segments[source_name].update(t, strength, flat):
                                    emit(event)
                        value['calibrating_neutral'] = neutral is None
                    else:
                        for source_name in ['face', 'head_posture']:
                            emit(segments[source_name].finish(t, 'tracking_lost'))
                        if neutral is None:
                            neutral_samples, calibration_start = [], None
                else:
                    hands, scores = [], {}
                    for i, points in enumerate(result.hand_world_landmarks):
                        categories = {c.category_name: float(c.score) for c in result.gestures[i]}
                        hands.append({'features': hand_features(landmarks_array(points)), 'scores': categories,
                                      'landmarks_world': landmarks_array(points)[:, :3].tolist(),
                                      'landmarks_image': full_image_landmarks(landmarks_array(result.hand_landmarks[i]), bounds, frame.shape),
                                      'handedness': [{'label': c.category_name, 'score': float(c.score)} for c in result.handedness[i]]})
                        for k, val in categories.items():
                            scores[k] = max(scores.get(k, 0.), val)
                    if hands:
                        activation = max([val for k, val in scores.items() if k != 'None'], default=0.)
                        value.update(status='ok', hands=hands, activation=activation)
                        for event in segments['hands'].update(t, activation, scores):
                            emit(event)
                    else:
                        emit(segments['hands'].finish(t, 'tracking_lost'))
                obs['channels'][name] = value
            preview_times.append(time.perf_counter())
            obs['tracking_fps'] = ((len(preview_times) - 1) / (preview_times[-1] - preview_times[0])
                                   if len(preview_times) > 1 else None)
            emit(obs)
            if args.record and camera_source is None:
                if writer is None:
                    writer = cv2.VideoWriter(str(folder / 'capture.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                             (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('could not create recording')
                    recorded_times = (folder / 'capture_times.jsonl').open('w', encoding='utf-8')
                writer.write(frame)
                recorded_times.write(json.dumps({'frame': frame_count, 't_capture': t, 'label': label}) + '\n')
            frame_count += 1
            if not args.headless or args.save_preview:
                display = render_tracking(frame, obs, event_count, neutral is not None, args.record)
                if args.save_preview and frame_count == 1:
                    if not cv2.imwrite(str(folder / 'tracking_preview.png'), display):
                        raise RuntimeError('could not save tracking preview')
            if not args.headless:
                mouse_view.update(size=(frame.shape[1], frame.shape[0]), people=people)
                cv2.imshow('Tink visual layer lab (uncalibrated)', display)
                cv2.setMouseCallback('Tink visual layer lab (uncalibrated)', click_target)
                key = cv2.waitKey(1) & 255
                if key in (ord('q'), ord('Q'), 27):
                    break
                if key in (ord('n'), ord('N')):
                    target_lock.select_next(people)
                if chr(key) in LABELS:
                    label = LABELS[chr(key)]
                    if camera_source:
                        camera_source.set_label(label)
        for seg in segments.values():
            emit(seg.finish(last_t, 'eof'))
        if camera_source:
            camera_source.close()
        summary = {'frames': frame_count, 'events': event_count, 'wall_s': time.perf_counter() - started,
                   'captured_frames': camera_source.captured if camera_source else frame_count,
                   'skipped_inference_frames': skipped_frames,
                   'model_ms': {k: {'median': float(np.median(ms)), 'p90': float(np.percentile(ms, 90))} for k, ms in costs.items() if ms},
                   'neutral_ready': neutral is not None if 'face' in args.channels else None,
                   'calibrated': False, 'intent_accuracy': None}
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)
        if camera_source and args.record:
            print(f'Full recording analysis:\n& "{sys.executable}" "{Path(__file__).resolve()}" --video "{folder / "capture.mp4"}"', flush=True)
    finally:
        try:
            if camera_source:
                camera_source.close()
        finally:
            if stream:
                stream.close()
            if recorded_times:
                recorded_times.close()
            if cap and camera_source is None:
                cap.release()
            if writer:
                writer.release()
            for model in models.values():
                model.close()
            if not args.headless:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
