"""Backend contract checks without models, devices, credentials or cloud."""
from collections import deque
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assessment_capture import ClipRecorder
from assessment_service import ClipAssessmentService
from session_engine import SessionEngine


class ServiceTests(unittest.TestCase):
    def service(self):
        service = object.__new__(ClipAssessmentService)
        service.clock = SimpleNamespace(session_id='session')
        service.engine = SessionEngine('session')
        service.engine.set_role('u', True)
        service.warnings, service.recent_assessments = [], deque(maxlen=5)
        service.encoder, service.transcriber = None, None
        service.visual = SimpleNamespace(target=SimpleNamespace(ownership_valid=True, role_source='operator_confirmed'))
        return service

    def recording(self, directory, different=False):
        clip = ClipRecorder(Path(directory) / 'clip', 'session', 10.)
        clip.add_frame(np.zeros((100, 100, 3), np.uint8), 10.)
        clip.add_audio(np.zeros(16000), 10., True)
        clip.add_observation({'t_capture': 10.1, 'logical_person_id': 'u', 'person_id': 1,
                              'ownership_valid': True, 'channels': {}})
        if different:
            clip.add_observation({'t_capture': 10.5, 'logical_person_id': 'other', 'person_id': 2,
                                  'ownership_valid': True, 'channels': {}})
        return clip, clip.finish(11.)

    def test_packet_has_shared_identity_and_preserves_unknown_speakers(self):
        service = self.service()
        service.transcriber = SimpleNamespace(transcribe_turns=lambda wave, start: [
            {'text': 'What would you like?', 'start_s': start, 'end_s': start+.8,
             'speaker_id': 'unknown', 'speaker_role': 'unknown', 'final': True}])
        with tempfile.TemporaryDirectory() as directory:
            clip, capture = self.recording(directory)
            packet = service.packet(clip, capture)
            self.assertIn(clip.clip_id, packet['event_ids'])
            self.assertTrue(packet['communicator']['ownership_valid'])
            self.assertEqual(packet['conversation']['local_transcript'][0]['speaker_role'], 'unknown')
            self.assertNotIn('audio_path', packet['capture'])
            self.assertTrue((clip.folder / 'assessment-input.json').is_file())

    def test_mixed_identity_recording_cannot_claim_owned_communicator(self):
        service = self.service()
        with tempfile.TemporaryDirectory() as directory:
            clip, capture = self.recording(directory, different=True)
            self.assertFalse(service.packet(clip, capture)['communicator']['ownership_valid'])

    def test_packet_budget_accounts_for_actual_serialized_feature_size(self):
        import json
        service = self.service()
        with tempfile.TemporaryDirectory() as directory:
            clip, capture = self.recording(directory)
            base = clip.observations[0]
            clip.observations = [dict(base, channels={'face': {'features': {
                'coefficient_' + str(i): .1234567890123456 for i in range(200)}}}) for _ in range(64)]
            packet = service.packet(clip, capture)
            self.assertLess(len(json.dumps(packet).encode()), 256 * 1024)
            self.assertLess(packet['capture']['feature_sampling']['transmitted_observations'], 64)

    def test_local_transcriber_failure_preserves_raw_assessment_path(self):
        service = self.service()
        def fail(*args):
            raise RuntimeError('mock failure')
        service.transcriber = SimpleNamespace(transcribe_turns=fail)
        with tempfile.TemporaryDirectory() as directory:
            clip, capture = self.recording(directory)
            packet = service.packet(clip, capture)
            self.assertEqual(packet['conversation']['local_transcript'], [])
            self.assertTrue(packet['capture']['limitations'])

    def test_clip_only_option_confirmation_does_not_label_motifs(self):
        service = self.service()
        spoken = []
        service.engine.speaker = SimpleNamespace(say=spoken.append)
        outcome = service.confirm({'text': 'Please repeat that.', 'kind': 'message', 'evidence_ids': ['clip']},
            {'clip_id': 'clip', 'communicator': {'ownership_valid': True, 'person_id': 'u'}}, 10.)
        self.assertEqual(outcome['status'], 'text_only_no_training')
        self.assertEqual(service.engine.learner.confirmations, [])
        self.assertEqual(spoken, ['Please repeat that.'])

    def test_confirmed_hypothesis_can_ground_only_owned_clock_valid_events(self):
        service = self.service()
        service.engine.observe({'event_id': 'visual-1', 'session_id': 'session', 'modality': 'video',
            'person_id': 'u', 'ownership_valid': True, 'clock_valid': True, 't_start': 9., 't_end': 9.5,
            'features': [1., .2, .1], 'quality': {}, 'context_id': 0})
        self.assertEqual(service.engine.learner.confirmations, [])
        outcome = service.confirm({'text': 'I want a different option.', 'kind': 'hypothesis',
            'evidence_ids': ['clip', 'visual-1']}, {'clip_id': 'clip', 'capture': {'clock_valid': True},
            'communicator': {'ownership_valid': True, 'person_id': 'u'}}, 10.)
        self.assertEqual(outcome['status'], 'confirmed')
        self.assertEqual(outcome['evidence_ids'], ['visual-1'])

    def test_speculative_source_or_clock_does_not_train_even_after_selection(self):
        for owned, clock_valid, person in [(False, True, 'u'), (True, False, 'u'), (True, True, 'other')]:
            with self.subTest(owned=owned, clock_valid=clock_valid, person=person):
                service = self.service()
                service.engine.observe({'event_id': 'visual-1', 'session_id': 'session', 'modality': 'video',
                    'person_id': 'u', 'ownership_valid': True, 'clock_valid': True, 't_start': 9., 't_end': 9.5,
                    'features': [1., .2, .1], 'quality': {}, 'context_id': 0})
                outcome = service.confirm({'text': 'I want a different option.', 'kind': 'hypothesis',
                    'evidence_ids': ['clip', 'visual-1']}, {'clip_id': 'clip', 'capture': {'clock_valid': clock_valid},
                    'communicator': {'ownership_valid': owned, 'person_id': person}}, 10.)
                self.assertEqual(outcome['status'], 'text_only_no_training')
                self.assertEqual(service.engine.learner.confirmations, [])


if __name__ == '__main__':
    unittest.main()
