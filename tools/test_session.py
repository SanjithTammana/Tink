"""Offline integration gates: clocks, ownership, ASL bridge and explicit confirmation."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
from concurrent.futures import Future
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from automatic_target import AutomaticTarget
from asl import FingerspellReader
from session_capture import SessionClock, SharedAudioCapture
from session_engine import SessionEngine
from tink_session import interval_owner, Lane
from visual_session import visual_vector
from tts import Speaker


def event(identity='e1', modality='video', person='u', owned=True, start=1.):
    return {'event_id': identity, 'session_id': 's', 'modality': modality,
            't_start': start, 't_end': start + .5, 'person_id': person,
            'ownership_valid': owned, 'clock_valid': True, 'features': [1., .1, .2],
            'quality': {}, 'context_id': 0}


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.spoken = []
        self.engine = SessionEngine('s', speaker=SimpleNamespace(say=self.spoken.append))
        self.engine.set_role('u', True)

    def test_no_speech_or_training_before_confirmation(self):
        self.engine.observe(event())
        self.engine.suggest_local(1.6)
        self.assertEqual(self.spoken, [])
        self.assertEqual(self.engine.learner.confirmations, [])
        outcome = self.engine.confirm(0, 2.)
        self.assertEqual(outcome['status'], 'confirmed')
        self.assertEqual(self.spoken, ['Yes.'])
        self.assertEqual(len(self.engine.context.snapshot(2.)['confirmed_user_turns']), 1)

    def test_unknown_audio_does_not_pair_or_ground(self):
        self.engine.observe(event('a', 'audio', None, False))
        self.engine.observe(event('v'))
        self.assertFalse(any(e['modality'] == 'paired' for e in self.engine.events))
        self.assertEqual(self.engine.packet(2.)['event_ids'], ['v'])

    def test_shared_clock_pair_is_one_joint_evidence_group(self):
        self.engine.observe(event('a', 'audio'))
        self.engine.observe(event('v'))
        self.assertEqual(sum(e['modality'] == 'paired' for e in self.engine.events), 1)
        self.engine.suggest_local(2.)
        outcome = self.engine.confirm(0, 2.)
        self.assertEqual(len(outcome['evidence_ids']), 1)

    def test_stale_reply_cannot_replace_choices(self):
        packet, version = self.engine.packet(2.), self.engine.version()
        self.engine.context.add_turn({'text': 'A new question?', 'start_s': 1., 'end_s': 2., 'final': True})
        self.assertFalse(self.engine.present({'options': []}, packet, version))

    def test_evidence_expiring_during_request_is_not_presented(self):
        self.engine.observe(event())
        packet, version = self.engine.packet(2.), self.engine.version()
        self.assertFalse(self.engine.present({'options': []}, packet, version, now_s=6.))

    def test_undo_invalidates_pending_and_displayed_suggestions(self):
        self.engine.observe(event())
        self.engine.suggest_local(2.)
        self.engine.confirm(0, 2.)
        self.engine.suggest_local(2.1)
        version = self.engine.version()
        self.assertTrue(self.engine.undo())
        self.assertNotEqual(version, self.engine.version())
        self.assertEqual(self.engine.options, [])

    def test_role_loss_blocks_old_selection_and_change_clears_history(self):
        self.engine.observe(event())
        self.engine.suggest_local(2.)
        self.engine.set_role('u', False)
        self.assertIsNone(self.engine.confirm(0, 2.))
        self.assertEqual(self.spoken, [])
        self.engine.set_role('different', True)
        self.assertEqual(self.engine.events, self.engine.events.__class__(maxlen=128))
        self.assertEqual(self.engine.options, [])

    def test_context_only_clarification_never_speaks(self):
        self.engine.suggest_local(2.)
        self.assertEqual(self.engine.options[0]['kind'], 'clarify')
        self.assertIsNone(self.engine.confirm(0, 2.))
        self.assertEqual(self.spoken, [])

    def test_reject_is_not_a_negative_meaning_label(self):
        self.engine.observe(event())
        self.engine.suggest_local(2.)
        rejected_text = {o['text'] for o in self.engine.options}
        self.engine.reject(2.)
        self.assertEqual(self.engine.learner.confirmations, [])
        self.assertEqual(self.engine.options, [])
        self.assertFalse(rejected_text & set(self.engine.packet(2.)['candidate_pool']))

    def test_clock_adc_offset_and_overflow_fail_closed(self):
        tick = [100.]
        clock = SessionClock(source=lambda: tick[0])
        capture = SharedAudioCapture(clock)
        tick[0] = 102.
        capture.callback(np.zeros((320, 1)), 320, SimpleNamespace(inputBufferAdcTime=50., currentTime=50.02), False)
        self.assertAlmostEqual(capture.at(320), 2.)
        capture.callback(np.zeros((320, 1)), 320, None, True)
        self.assertIsNone(capture.at(640))

    def test_owner_requires_coverage_and_rejects_overlap_loss(self):
        history = [(0., 'u', True), (.5, 'u', True), (1., 'u', True)]
        self.assertEqual(interval_owner(history, .1, .8), 'u')
        self.assertIsNone(interval_owner([(0., 'u', True)], .1, .8))
        history[1] = (.5, 'u', False)
        self.assertIsNone(interval_owner(history, .1, .8))

    def test_automatic_reacquisition_retains_identity_after_overlap(self):
        frame = np.zeros((220, 400, 3), np.uint8)
        frame[:, :150] = [0, 0, 255]
        frame[:, 220:] = [255, 0, 0]
        user = {'id': 1, 'bbox': [0, 0, 150, 220]}
        other = {'id': 2, 'bbox': [220, 0, 400, 220]}
        lock = AutomaticTarget()
        self.assertEqual(lock.update([user], frame, 0.)['id'], 1)
        identity = lock.logical_id
        overlap = {'id': 2, 'bbox': [120, 0, 280, 220]}
        self.assertIsNone(lock.update([user, overlap], frame, .1))
        self.assertEqual(lock.update([dict(user, id=9), other], frame, .2)['id'], 9)
        self.assertEqual(lock.logical_id, identity)
        self.assertTrue(lock.ownership_valid)

    def test_multiple_new_people_remain_provisional(self):
        frame = np.zeros((200, 400, 3), np.uint8)
        frame[:, :150] = [0, 0, 255]
        frame[:, 220:] = [255, 0, 0]
        people = [{'id': 1, 'bbox': [0, 0, 150, 200]}, {'id': 2, 'bbox': [220, 0, 400, 200]}]
        lock = AutomaticTarget()
        self.assertIsNotNone(lock.update(people, frame, 0.))
        self.assertFalse(lock.ownership_valid)
        lock.confirm_role()
        self.assertTrue(lock.ownership_valid)

    def test_explicit_missing_selection_never_falls_back(self):
        frame = np.full((200, 200, 3), [0, 0, 255], np.uint8)
        lock = AutomaticTarget()
        lock.select(2)
        self.assertIsNone(lock.update([{'id': 1, 'bbox': [0, 0, 200, 200]}], frame, 1.))
        self.assertEqual(lock.target_id, 2)
        self.assertFalse(lock.ownership_valid)

    def test_tts_ledger_excludes_playback_missed_between_frames(self):
        engine = SimpleNamespace(setProperty=lambda *a: None, say=lambda *a: None, runAndWait=lambda: None)
        with patch.dict(sys.modules, {'pyttsx3': SimpleNamespace(init=lambda: engine)}):
            speaker = Speaker()
            start = time.perf_counter()
            speaker.say('Offline test')
            speaker.close()
            self.assertFalse(speaker.speaking.is_set())
            self.assertTrue(speaker.overlaps(start, time.perf_counter()))

    def test_live_four_value_capture_contract_without_sensors(self):
        import tink_session as app
        frame = np.zeros((200, 200, 3), np.uint8)
        class FakeVisual:
            def __init__(self, *a, **kw):
                self.target = SimpleNamespace(logical_id='u')
                self.neutral, self.asl_status = {}, 'test'
            def observe(self, frame, now):
                return {'ownership_valid': True, 't_capture': now, 'asl': {}}, []
            def close(self):
                pass
        class FakeLive:
            captured = 1
            def __init__(self, *a, **kw):
                pass
            def start(self):
                return self
            def read_after(self, index):
                return (0, 1., frame, 'unlabeled') if index < 0 else None
            def close(self):
                pass
        cap = SimpleNamespace(isOpened=lambda: True, get=lambda prop: 30., release=lambda: None)
        with tempfile.TemporaryDirectory() as directory, patch.object(app, 'ROOT', Path(directory)), \
                patch.object(app, 'VisualSession', FakeVisual), patch.object(app, 'LiveCapture', FakeLive), \
                patch.object(app.cv2, 'VideoCapture', return_value=cap), \
                patch.object(app, 'panel', return_value=frame), patch.object(app.cv2, 'imshow'), \
                patch.object(app.cv2, 'namedWindow'), patch.object(app.cv2, 'resizeWindow'), \
                patch.object(app.cv2, 'waitKey', return_value=-1):
            app.main(['--no-audio', '--no-asr', '--no-asl', '--no-tts', '--max-frames', '1'])
            self.assertTrue(list(Path(directory).glob('.scratch_session/default/*/manifest.json')))

    def test_gemini_failure_latches_until_explicit_retry(self):
        import tink_session as app
        requests = []
        class FakeClient:
            configured = True
            def __init__(self, *a, **kw):
                pass
            def generate(self, packet):
                requests.append(packet)
                raise RuntimeError('mock outage')
        class ImmediateLane(Lane):
            def submit(self, fn, tag, *args):
                self.tag, self.future = tag, Future()
                try:
                    self.future.set_result(fn(*args))
                except Exception as error:
                    self.future.set_exception(error)
                return True
        class FakeVisual:
            def __init__(self, *a, **kw):
                self.target = SimpleNamespace(logical_id='u')
                self.neutral, self.asl_status = {}, 'test'
            def observe(self, frame, now):
                return {'ownership_valid': True, 't_capture': now, 'asl': {}}, [event(str(now), start=now)]
            def close(self):
                pass
        frames = iter([(True, np.zeros((200, 200, 3), np.uint8)) for _ in range(5)])
        cap = SimpleNamespace(isOpened=lambda: True, get=lambda prop: .5,
                              read=lambda: next(frames, (False, None)), release=lambda: None)
        with tempfile.TemporaryDirectory() as directory, patch.object(app, 'ROOT', Path(directory)), \
                patch.object(app, 'VisualSession', FakeVisual), patch.object(app, 'GeminiClient', FakeClient), \
                patch.object(app, 'Lane', ImmediateLane), patch.object(app.cv2, 'VideoCapture', return_value=cap):
            app.main(['--video', 'mock.mp4', '--headless', '--no-asr', '--no-asl', '--no-tts', '--gemini'])
        self.assertEqual(len(requests), 1)

    def test_asl_reuses_existing_stability_and_resets_after_hand_loss(self):
        classifier = SimpleNamespace(scores=lambda crop: {'A': .95, 'B': .05})
        reader = FingerspellReader(classifier=classifier)
        frame = np.zeros((240, 240, 3), np.uint8)
        hand = {'points': np.c_[np.linspace(.3, .6, 21), np.linspace(.3, .7, 21)].tolist()}
        for _ in range(6):
            result = reader.update(frame, [hand])
        self.assertTrue(result['ready'])
        self.assertEqual(result['best'], 'A')
        self.assertFalse(reader.update(frame, [])['ready'])

    def test_visual_feature_space_is_fixed(self):
        self.assertEqual(visual_vector({}).shape, (32,))
        self.assertTrue(np.isfinite(visual_vector({})).all())


if __name__ == '__main__':
    unittest.main()
