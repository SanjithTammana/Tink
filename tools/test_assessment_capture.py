"""Recording/assessment packet gates, without sensors or real Gemini calls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assessment_capture import ClipRecorder, compact_observation, RecordedAudioCapture
from types import SimpleNamespace


class ClipTests(unittest.TestCase):
    def test_earlier_overflow_does_not_erase_later_recorded_audio(self):
        now = [1.]
        capture = RecordedAudioCapture(SimpleNamespace(now=lambda: now[0]))
        capture.callback(np.ones((320, 1), np.float32)*.1, 320, None, True)
        now[0] = 2.
        capture.callback(np.ones((320, 1), np.float32)*.2, 320, None, False)
        self.assertFalse(capture.clock_valid)  # legacy aggregate diagnostic remains honest
        self.assertFalse(capture.block_valid(0))
        self.assertTrue(capture.block_valid(320))
        self.assertAlmostEqual(capture.at(320), 1.98)
        with tempfile.TemporaryDirectory() as directory:
            clip = ClipRecorder(Path(directory) / 'clip', 'session', 1.98)
            for index, wave, bad in capture.drain():
                clip.add_audio(wave, capture.at(index), capture.block_valid(index))
            result = clip.finish(2.)
            import soundfile as sf
            wave, _ = sf.read(result['audio_path'])
            self.assertGreater(float(np.max(wave)), .19)
            self.assertTrue(result['clock_valid'])

    def test_media_uses_one_origin_and_gaps_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = ClipRecorder(Path(directory) / 'clip', 'session', 10., fps=8)
            frame = np.zeros((120, 160, 3), np.uint8)
            clip.add_frame(frame, 10.02)
            clip.add_frame(frame + 80, 10.2)
            clip.add_audio(np.ones(1600, np.float32) * .1, 10.1, True)
            result = clip.finish(10.5)
            import soundfile as sf
            wave, sr = sf.read(result['audio_path'])
            self.assertEqual(sr, 16000)
            self.assertEqual(len(wave), 8000)
            self.assertTrue(np.all(wave[:1600] == 0))
            self.assertGreater(np.max(wave[1600:3200]), .09)
            self.assertTrue(result['audio_gaps'])
            cap = cv2.VideoCapture(result['video_path'])
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            cap.release()
            self.assertEqual(result['start_s'], 10.)
            self.assertEqual(result['duration_s'], .5)
            self.assertTrue((Path(directory) / 'clip' / 'capture.json').is_file())

    def test_stop_is_idempotent_and_late_callbacks_do_not_change_media(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = ClipRecorder(Path(directory) / 'clip', 'session', 0.)
            clip.add_frame(np.zeros((100, 100, 3), np.uint8), 0.)
            clip.add_audio(np.ones(1600, np.float32), 0., True)
            first = clip.finish(.2)
            original = Path(first['audio_path']).read_bytes()
            clip.add_audio(np.zeros(16000), .2, True)
            second = clip.finish(2.)
            self.assertEqual(first, second)
            self.assertEqual(original, Path(first['audio_path']).read_bytes())

    def test_device_clock_failure_is_not_hidden_and_duration_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            clip = ClipRecorder(Path(directory) / 'clip', 'session', 2., max_seconds=1)
            clip.add_frame(np.zeros((100, 100, 3), np.uint8), 2.)
            clip.add_audio(np.zeros(320), None, False)
            result = clip.finish(20.)
            self.assertFalse(result['clock_valid'])
            self.assertEqual(result['duration_s'], 1.)

    def test_compact_features_preserve_measurements_without_landmark_payload(self):
        source = {'t_capture': 3., 'person_id': 2, 'ownership_valid': False,
                  'people': [{'id': 2, 'bbox': [0, 0, 50, 50], 'score': .9}],
                  'channels': {'face': {'status': 'ok', 'features': {'jawOpen': .4},
                                        'landmarks_image': [[1., 2., 3.]]}}, 'asl': {'ready': False}}
        result = compact_observation(source)
        self.assertEqual(result['channels']['face']['features'], {'jawOpen': .4})
        self.assertNotIn('landmarks_image', result['channels']['face'])
        self.assertFalse(result['ownership_valid'])
        self.assertEqual(json.loads(json.dumps(result))['person_id'], 2)


if __name__ == '__main__':
    unittest.main()
