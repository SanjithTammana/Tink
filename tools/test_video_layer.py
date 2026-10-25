"""Contract checks for the standalone video layer; no camera or model downloads."""
import sys
import unittest
from pathlib import Path
import importlib.util
import ast
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from video_layer import PoseClock, Segment, cnn_distribution, face_features, pose_features
from person_selection import TargetLock, crop_bounds, full_image_landmarks


class VideoContract(unittest.TestCase):
    def test_target_never_falls_back_to_largest_bystander(self):
        left = {'id': 1, 'bbox': [0, 0, 100, 200]}
        right = {'id': 2, 'bbox': [200, 0, 500, 500]}
        lock = TargetLock()
        self.assertIsNone(lock.update([left, right]))
        lock.select_at(50, 50, [left, right])
        self.assertEqual(lock.update([left, right])['id'], 1)
        generation = lock.generation
        self.assertIsNone(lock.update([right]))
        self.assertEqual(lock.target_id, 1)
        self.assertEqual(lock.update([left, right])['id'], 1)
        self.assertGreater(lock.generation, generation)
        self.assertEqual(lock.select_next([left, right]), 2)

    def test_overlap_and_jump_require_new_selection(self):
        one = {'id': 1, 'bbox': [0, 0, 100, 200]}
        lock = TargetLock(1)
        self.assertEqual(lock.update([one])['id'], 1)
        self.assertIsNone(lock.update([one, {'id': 2, 'bbox': [40, 0, 140, 200]}]))
        self.assertIsNone(lock.target_id)
        lock.select(1)
        lock.update([one])
        self.assertIsNone(lock.update([{'id': 1, 'bbox': [600, 0, 700, 200]}]))
        self.assertIsNone(lock.target_id)

    def test_partial_bystander_strip_is_not_attributed_to_user(self):
        lock = TargetLock(1)
        selected = {'id': 1, 'bbox': [0, 0, 300, 600]}
        bystander = {'id': 2, 'bbox': [250, 0, 550, 600]}
        self.assertIsNone(lock.update([selected, bystander]))
        self.assertIsNone(lock.target_id)

    def test_crop_geometry_maps_back_to_the_same_person_box(self):
        bounds = crop_bounds([100, 50, 300, 250], (400, 800, 3))
        points = np.array([[0, 0, .1, 1], [1, 1, .2, .8]])
        result = full_image_landmarks(points, bounds, (400, 800, 3))
        np.testing.assert_allclose(np.asarray(result)[:, :2], [[.125, .125], [.375, .625]])
        np.testing.assert_allclose(np.asarray(result)[:, 2:], points[:, 2:])
        np.testing.assert_allclose(points[:, :2], [[0, 0], [1, 1]])

    def test_resample_uses_capture_time(self):
        clock = PoseClock()
        a = np.zeros(92, np.float32)
        self.assertEqual(len(clock.push(0.0, a)), 1)
        samples = clock.push(0.1, a + 1)
        np.testing.assert_allclose([t for t, _ in samples], [.04, .08])
        np.testing.assert_allclose([x[0] for _, x in samples], [.4, .8])

    def test_gap_and_missing_pose_do_not_bridge(self):
        clock = PoseClock()
        clock.push(0, np.zeros(92))
        self.assertEqual(len(clock.push(.5, np.ones(92))), 1)
        self.assertEqual(len(clock.history), 1)
        clock.push(.6, None)
        self.assertEqual(len(clock.history), 0)
        self.assertEqual(len(clock.push(.7, np.ones(92))), 1)
        with self.assertRaises(ValueError):
            clock.push(.69, np.ones(92))

    def test_cnn_heads_are_conditional(self):
        result = cnn_distribution({'motion': .1, 'gesture': .9, 'move': .1})
        self.assertAlmostEqual(result['Gesture'], .09)
        self.assertAlmostEqual(sum(result.values()), 1)

    def test_one_event_per_held_signal(self):
        seg = Segment('face', 'trial')
        events = []
        for i in range(30):
            events += seg.update(i / 10, .9, {'jawOpen': .9})
        self.assertEqual(events, [])
        for i in range(30, 36):
            events += seg.update(i / 10, .1, {'jawOpen': .1})
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event['source'], 'face')
        self.assertEqual(event['calibrated'], False)
        self.assertEqual(event['person_id'], None)
        self.assertAlmostEqual(event['t_end'], 2.9)
        self.assertGreater(event['emitted_at'], event['t_end'])
        self.assertIsNone(seg.finish(4.0))

    def test_tracking_loss_is_explicit_and_not_silence(self):
        seg = Segment('hands', 'trial')
        for i in range(6):
            seg.update(i / 10, .9, {'Thumb_Up': .9})
        event = seg.finish(.6, 'tracking_lost')
        self.assertFalse(event['complete'])
        self.assertEqual(event['end_reason'], 'tracking_lost')

    def test_short_spike_rejected_and_ids_do_not_repeat(self):
        seg = Segment('face', 'trial')
        seg.update(0, .9, {})
        self.assertEqual(seg.update(.1, .1, {}), [])
        self.assertEqual(seg.update(.5, .1, {}), [])
        ids = []
        for offset in [1, 3]:
            for i in range(6):
                seg.update(offset + i / 10, .9, {})
            ids.append(seg.finish(offset + .6, 'eof')['event_id'])
        self.assertEqual(len(set(ids)), 2)

    def test_rotation_scale_does_not_change_angles(self):
        angle = np.deg2rad(20)
        r = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle), np.cos(angle)]])
        matrix = np.eye(4)
        matrix[:3, :3] = 3 * r
        result = face_features({}, matrix, None)
        self.assertAlmostEqual(result['pitch_deg'], 20)
        self.assertAlmostEqual(result['yaw_deg'], 0)

    def test_invalid_rotation_is_unknown_not_neutral(self):
        for matrix in [np.zeros((4, 4)), np.full((4, 4), np.nan)]:
            result = face_features({}, matrix, None)
            self.assertFalse(result['rotation_valid'])
            self.assertNotIn('pitch_deg', result)

    def test_slow_sampling_abstains_instead_of_fabricating_pose(self):
        clock = PoseClock()
        for i in range(20):
            clock.push(i * .35, np.ones(92) * i)
        self.assertEqual(len(clock.history), 1)
        self.assertTrue(clock.discontinuity)

    def test_body_coordinates_are_relative_to_shoulders(self):
        pose = np.zeros((33, 4))
        pose[:, 3] = 1
        pose[11, :3], pose[12, :3] = [-.2, 0, 0], [.2, 0, 0]
        pose[15, :3], pose[16, :3] = [-.2, -.4, 0], [.2, .4, 0]
        first = pose_features(pose)
        shifted = pose.copy()
        shifted[:, :3] += [10, -4, 2]
        second = pose_features(shifted)
        self.assertAlmostEqual(first['left_raised_sw'], 1)
        self.assertAlmostEqual(first['left_raised_sw'], second['left_raised_sw'])

    def test_capture_only_writes_matching_timestamp_sidecar(self):
        path = Path(__file__).with_name('video_layer_lab.py')
        spec = importlib.util.spec_from_file_location('video_lab_test', path)
        lab = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lab)

        class FakeCamera:
            def __init__(self):
                self.i, self.released = 0, False

            def isOpened(self):
                return True

            def get(self, prop):
                return 20.

            def read(self):
                self.i += 1
                return True, np.full((64, 64, 3), self.i * 30, np.uint8)

            def release(self):
                self.released = True

        camera = FakeCamera()
        args = SimpleNamespace(camera=0, label='synthetic_test', max_frames=3, headless=True)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(lab.cv2, 'VideoCapture', return_value=camera):
                lab.record_trial(args, Path(folder))
            rows = [json.loads(line) for line in (Path(folder) / 'capture_times.jsonl').read_text().splitlines()]
            self.assertEqual([r['frame'] for r in rows], [0, 1, 2])
            self.assertTrue(all(r['label'] == 'synthetic_test' for r in rows))
            self.assertTrue(all(b['t_capture'] > a['t_capture'] for a, b in zip(rows, rows[1:])))
            saved = lab.cv2.VideoCapture(str(Path(folder) / 'capture.mp4'))
            try:
                self.assertEqual(int(saved.get(lab.cv2.CAP_PROP_FRAME_COUNT)), 3)
            finally:
                saved.release()
        self.assertTrue(camera.released)

    def test_port_feature_vector_matches_installed_reference(self):
        # Execute only the installed package's pure feature method, not its imports,
        # constructors, TensorFlow, Holistic or model loading. This checks feature
        # order/units as well as classifier parity on already-computed features.
        spec = importlib.util.find_spec('envisionhgdetector')
        if spec is None:
            self.skipTest('reference package not installed; portable runtime does not require it')
        source = Path(next(iter(spec.submodule_search_locations))) / 'model_lightgbm.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        namespace = {'np': np}
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                try:
                    value = ast.literal_eval(statement.value)
                except (ValueError, TypeError):
                    call = statement.value
                    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                            and isinstance(call.func.value, ast.Name) and call.func.value.id == 'np'
                            and call.func.attr == 'array' and len(call.args) == 1 and not call.keywords):
                        value = np.array(ast.literal_eval(call.args[0]))
                    else:
                        continue
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        namespace[target.id] = value
        owner = next(s for s in tree.body if isinstance(s, ast.ClassDef) and s.name == 'LightGBMGestureModel')
        method = next(s for s in owner.body if isinstance(s, ast.FunctionDef) and s.name == 'extract_sequence_features')
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        from envision_port import lgbm_features
        rng = np.random.default_rng(8)
        for size in [0, 1, 4, 5, 25, 59]:
            points = rng.normal(0, .2, (size, 92)).astype(np.float32)
            points[:, 3::4] = rng.uniform(0, 1, (size, 23))
            for stride in [1, 2, 3]:
                reference = namespace['extract_sequence_features'](None, points, stride=stride)
                np.testing.assert_allclose(lgbm_features(points, stride=stride), reference, rtol=1e-6, atol=1e-7)

    def test_overlay_tracks_only_current_visible_points(self):
        from video_overlay import render_tracking
        frame = np.zeros((480, 640, 3), np.uint8)
        pose = np.tile([.5, .5, 0, 1.], (33, 1)).tolist()
        observed = {'channels': {'pose': {'status': 'ok', 'pose_image': pose}}}
        image = render_tracking(frame, observed)
        self.assertGreater(np.count_nonzero(image[225:255, 305:335]), 0)
        missing = render_tracking(frame, {'channels': {'pose': {'status': 'missing', 'pose_image': pose}}})
        self.assertEqual(np.count_nonzero(missing[:480, :640]), 0)
        self.assertEqual(np.count_nonzero(frame), 0)
        self.assertGreaterEqual(image.shape[0], 720)
        self.assertEqual(image.shape[1], 1060)

    def test_live_capture_records_frames_skipped_by_inference(self):
        path = Path(__file__).with_name('video_layer_lab.py')
        spec = importlib.util.spec_from_file_location('video_live_test', path)
        lab = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lab)
        gate = lab.threading.Event()

        class FakeCamera:
            def __init__(self):
                self.i, self.released = 0, False

            def get(self, prop):
                return 20.

            def read(self):
                if self.i == 1:
                    if not gate.wait(2):
                        raise RuntimeError('test capture gate timed out')
                if self.i == 6:
                    return False, None
                self.i += 1
                return True, np.full((64, 64, 3), self.i * 20, np.uint8)

            def release(self):
                self.released = True

        camera = FakeCamera()
        with tempfile.TemporaryDirectory() as folder:
            capture = lab.LiveCapture(camera, Path(folder), 'idle').start()
            try:
                first = capture.read_after(-1)
                self.assertEqual(first[0], 0)
                capture.set_label('thumb_up')
                gate.set()
                capture.thread.join(2)
                self.assertFalse(capture.thread.is_alive())
                last = capture.read_after(first[0])
                self.assertEqual(last[0], 5)
                self.assertEqual(last[3], 'thumb_up')
                self.assertIsNone(capture.read_after(last[0]))
                rows = [json.loads(line) for line in (Path(folder) / 'capture_times.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 6)
                self.assertEqual(rows[-1]['label'], 'thumb_up')
                self.assertEqual(rows[-1]['t_capture'], last[1])
                saved = lab.cv2.VideoCapture(str(Path(folder) / 'capture.mp4'))
                try:
                    self.assertEqual(int(saved.get(lab.cv2.CAP_PROP_FRAME_COUNT)), 6)
                finally:
                    saved.release()
            finally:
                gate.set()
                capture.close()
        self.assertTrue(camera.released)


if __name__ == '__main__':
    unittest.main()
