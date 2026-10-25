"""Recorded assessment contracts, with fabricated media and mocked HTTP only."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gemini_assessment import AssessmentClient
from gemini_client import ENDPOINT, GeminiError


def packet():
    return {'schema_version': 1, 'session_id': 'session-test', 'clip_id': 'clip-test',
            'event_ids': ['clip-test', 'e1'],
            'capture': {'start_s': 10., 'end_s': 12., 'duration_s': 2., 'clock_valid': True,
                        'audio_gaps': [], 'video_frames': 30},
            'communicator': {'person_id': 'u1', 'ownership_valid': True,
                             'role_source': 'operator_asserted'},
            'signals': {'visual_observations': [], 'events': [], 'audio_events': []},
            'conversation': {'prior_context': {}, 'local_transcript': [], 'confirmedmessages': []},
            'belief': {'kind': 'ungrounded_motifs', 'calibrated': False}, 'spelled': ''}


def assessment():
    return {'schema_version': 1, 'session_id': 'session-test', 'clip_id': 'clip-test',
            'assessment': {'summary': 'There are competing possible interpretations.',
                           'observations': [{'text': 'A movement is visible.', 'evidence_ids': ['e1']}],
                           'limitations': ['The meaning is not established.']},
            'partner_context': {'summary': 'A question may have been asked.', 'topic': 'Unknown',
                                'pending_question': '', 'speaker_hypotheses': [
                                    {'speaker_id': 'A', 'role': 'partner',
                                     'basis': 'The utterance sounds like a question.', 'uncertain': True}],
                                'uncertainties': ['Roles are tentative.']},
            'transcript': [{'text': 'What would you like?', 'speaker_id': 'A',
                            'role_hypothesis': 'partner', 'start_s': 10.1, 'end_s': 11.2,
                            'uncertain': True}],
            'options': [
                {'id': '1', 'text': 'I would like to continue.', 'kind': 'message',
                 'reason': 'A possible response to the question.', 'evidence_ids': ['clip-test']},
                {'id': '2', 'text': 'I would like a pause.', 'kind': 'message',
                 'reason': 'An alternative to confirm.', 'evidence_ids': ['e1']},
                {'id': '3', 'text': 'Could you clarify the question?', 'kind': 'clarify',
                 'reason': 'The intended meaning is uncertain.', 'evidence_ids': []}],
            'confirmation_required': True}


def response(result):
    return json.dumps({'status': 'completed', 'steps': [
        {'type': 'model_output', 'content': [{'type': 'text', 'text': json.dumps(result)}]}]}).encode()


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.video, self.audio = self.root / 'clip.mp4', self.root / 'clip.wav'
        self.video.write_bytes(b'fabricated-mp4-for-mock-only')
        self.audio.write_bytes(b'fabricated-wav-for-mock-only')

    def client(self, result=None, transport=None):
        return AssessmentClient(self.root, environ={'GEMINI_API_KEY': 'offline-only-test-secret'},
                                transport=transport or (lambda *_: response(result or assessment())))

    def run_assessment(self, result=None, data=None):
        return self.client(result).assess(data or packet(), self.video, self.audio)

    def test_verified_native_media_payload_stateless_header_only_and_timeout(self):
        def capture(req, timeout, response_cap):
            self.assertEqual(req.full_url, ENDPOINT)
            self.assertEqual(req.get_header('X-goog-api-key'), 'offline-only-test-secret')
            self.assertNotIn(b'offline-only-test-secret', req.data)
            body = json.loads(req.data)
            self.assertFalse(body['store'])
            self.assertNotIn('tools', body)
            self.assertNotIn('previous_interaction_id', body)
            self.assertEqual(timeout, 45.)
            self.assertLessEqual(response_cap, 131072)
            self.assertEqual(body['generation_config']['thinking_level'], 'low')
            media = {item['type']: item for item in body['input'][1:]}
            self.assertEqual(media['video']['mime_type'], 'video/mp4')
            self.assertEqual(base64.b64decode(media['video']['data']), self.video.read_bytes())
            self.assertEqual(media['audio']['mime_type'], 'audio/wav')
            schema = body['response_format']['schema']['properties']['options']
            self.assertEqual((schema['minItems'], schema['maxItems']), (3, 3))
            self.assertNotIn('maxItems', schema['items']['properties']['evidence_ids'])
            return response(assessment())
        result = self.client(transport=capture).assess(packet(), self.video, self.audio)
        self.assertEqual([o['id'] for o in result['options']], ['1', '2', '3'])
        self.assertTrue(result['confirmation_required'])

    def test_simplified_provider_schema_keeps_strict_local_array_limits(self):
        result = assessment()
        result['assessment']['limitations'] = ['bounded'] * 17
        with self.assertRaises(GeminiError):
            self.run_assessment(result)

    def test_freely_worded_hypotheses_do_not_need_candidate_whitelist(self):
        data = packet()
        data['candidate_pool'] = []
        self.assertEqual(len(self.run_assessment(data=data)['options']), 3)

    def test_evidence_linked_hypotheses_allow_unknown_role_or_invalid_crossmodal_clock(self):
        result = assessment()
        for option in result['options'][:2]:
            option['kind'] = 'hypothesis'
            option['reason'] = ('The visible hand movement could support this intent; '
                                'the person and timing are uncertain, so this needs confirmation.')
        for owned, clock_valid in [(True, True), (False, True), (True, False), (False, False)]:
            data = packet()
            data['communicator']['ownership_valid'] = owned
            data['capture']['clock_valid'] = clock_valid
            with self.subTest(owned=owned, clock_valid=clock_valid):
                accepted = self.run_assessment(result, data)
                self.assertEqual([o['kind'] for o in accepted['options'][:2]], ['hypothesis', 'hypothesis'])
                self.assertTrue(accepted['confirmation_required'])

    def test_hypotheses_require_attached_media_and_valid_refs_even_for_spelling(self):
        result = assessment()
        result['transcript'] = []
        for option in result['options']:
            option['kind'] = 'hypothesis'
            option['evidence_ids'] = ['clip-test']
        data = packet()
        data['spelled'] = result['options'][0]['text']
        with self.assertRaises(GeminiError):
            self.client(result).assess(data)
        for refs in [[], ['invented']]:
            malformed = deepcopy(result)
            malformed['options'][0]['evidence_ids'] = refs
            with self.subTest(refs=refs), self.assertRaises(GeminiError):
                self.run_assessment(malformed, data)
        accepted = self.client(result).assess(data, self.video)
        self.assertEqual(accepted['options'][0]['kind'], 'hypothesis')

    def test_message_kind_stays_strict_for_unknown_role_or_invalid_clock(self):
        for field in ['ownership', 'clock']:
            data = packet()
            if field == 'ownership':
                data['communicator']['ownership_valid'] = False
            else:
                data['capture']['clock_valid'] = False
            with self.subTest(field=field), self.assertRaises(GeminiError):
                self.run_assessment(data=data)

    def test_exactly_three_distinct_ids_and_wordings(self):
        cases = []
        result = assessment(); result['options'].pop(); cases.append(result)
        result = assessment(); result['options'].append(deepcopy(result['options'][0])); cases.append(result)
        result = assessment(); result['options'][2]['id'] = '2'; cases.append(result)
        result = assessment(); result['options'][2]['id'] = '4'; cases.append(result)
        result = assessment(); result['options'][1]['text'] = result['options'][0]['text'].upper(); cases.append(result)
        result = assessment(); result['options'][0]['text'] = 'x' * 201; cases.append(result)
        for index, result in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(GeminiError):
                self.run_assessment(result)

    def test_session_clip_evidence_and_explicit_confirmation_validation(self):
        cases = []
        result = assessment(); result['session_id'] = 'other'; cases.append(result)
        result = assessment(); result['clip_id'] = 'other'; cases.append(result)
        result = assessment(); result['confirmation_required'] = False; cases.append(result)
        result = assessment(); result['assessment']['observations'][0]['evidence_ids'] = ['invented']; cases.append(result)
        result = assessment(); result['options'][0]['evidence_ids'] = ['invented']; cases.append(result)
        result = assessment(); result['options'][0]['confidence'] = .9; cases.append(result)
        result = assessment(); result['assessment']['observations'][0]['evidence_ids'] = []; cases.append(result)
        for index, result in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(GeminiError):
                self.run_assessment(result)

    def test_role_hypotheses_remain_uncertain_and_transcript_stays_in_clip(self):
        cases = []
        result = assessment(); result['partner_context']['speaker_hypotheses'][0]['uncertain'] = False; cases.append(result)
        result = assessment(); result['transcript'][0]['uncertain'] = False; cases.append(result)
        result = assessment(); result['transcript'][0]['role_hypothesis'] = 'confirmed partner'; cases.append(result)
        result = assessment(); result['transcript'][0]['start_s'] = 9.9; cases.append(result)
        result = assessment(); result['transcript'][0]['end_s'] = 12.1; cases.append(result)
        result = assessment(); result['transcript'][0]['start_s'] = float('nan'); cases.append(result)
        for index, result in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(GeminiError):
                self.run_assessment(result)

    def test_missing_channels_or_ownership_use_clarifications(self):
        result = assessment()
        result['transcript'] = []
        result['assessment']['observations'] = []
        for index, option in enumerate(result['options']):
            option.update(text=f'Clarify option {index + 1}', kind='clarify', evidence_ids=[])
        self.assertEqual(len(self.client(result).assess(packet())['options']), 3)
        with self.assertRaises(GeminiError):
            self.client().assess(packet(), self.video)  # No fabricated transcript from silent video.
        data = packet(); data['communicator']['ownership_valid'] = False
        with self.assertRaises(GeminiError):
            self.run_assessment(data=data)
        self.assertEqual(len(self.run_assessment(result, data)['options']), 3)

    def test_clip_bounds_and_media_limits_fail_before_transport(self):
        requests = []
        client = self.client(transport=lambda *_: requests.append(True))
        for change in [{'duration_s': 31., 'end_s': 41.}, {'duration_s': -1.},
                       {'start_s': float('inf')}, {'end_s': 10.}]:
            data = packet(); data['capture'].update(change)
            with self.subTest(change=change), self.assertRaises(GeminiError):
                client.assess(data, self.video, self.audio)
        with self.assertRaises(GeminiError):
            client.assess(packet(), self.root / 'missing.mp4')
        self.audio.write_bytes(b'x' * (1024 * 1024 + 1))
        with self.assertRaises(GeminiError):
            client.assess(packet(), self.video, self.audio)
        self.assertEqual(requests, [])

    def test_untrusted_failure_is_sanitized_and_has_no_retry(self):
        requests = []
        def fail(*_):
            requests.append(True)
            raise RuntimeError('offline-only-test-secret private transcript')
        with self.assertRaises(GeminiError) as caught:
            self.client(transport=fail).assess(packet(), self.video, self.audio)
        self.assertEqual(len(requests), 1)
        self.assertNotIn('offline-only-test-secret', str(caught.exception))
        self.assertNotIn('private transcript', str(caught.exception))
        for raw in [b'{}', b'[]', b'invalid-json', b'x' * 131073]:
            with self.subTest(length=len(raw)), self.assertRaises(GeminiError):
                self.client(transport=lambda *_, raw=raw: raw).assess(packet(), self.video, self.audio)


if __name__ == '__main__':
    unittest.main(verbosity=2)
