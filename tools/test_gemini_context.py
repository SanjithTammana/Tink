"""Offline contract tests: no credentials, network, devices, or model downloads."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversation_state import ConversationState
from gemini_client import ENDPOINT, GeminiClient, GeminiError, _NoRedirect
from transcribe import MetaTranscriber, Transcriber


def packet():
    return {'session_id': 'offline', 'context_revision': 1, 'event_ids': ['e1'],
            'quality': {'clock_valid': True}, 'user_role': {'ownership_valid': True},
            'belief': {'kind': 'ungrounded_motifs', 'calibrated': False, 'items': []},
            'conversation': {}, 'candidate_pool': ['Yes', 'No'], 'spelled': '',
            'rejected_option_ids': []}


def suggestion(text='Yes', kind='message', refs=None):
    return {'options': [{'text': text, 'kind': kind, 'evidence_ids': ['e1'] if refs is None else refs}],
            'needs_clarification': False}


def response(result):
    return json.dumps({'status': 'completed', 'steps': [
        {'type': 'model_output', 'content': [{'type': 'text', 'text': json.dumps(result)}]}]}).encode()


class GeminiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def client(self, result=None, transport=None, environ=None):
        return GeminiClient(self.root, environ=environ if environ is not None else {'GEMINI_API_KEY': 'offline-secret'},
                            transport=transport or (lambda *_: response(result or suggestion())))

    def test_literal_allowlisted_dotenv_and_explicit_environment_precedence(self):
        (self.root / '.env').write_text('GEMINI_API_KEY="${NO_EXPANSION}"\nOTHER_SECRET=ignore\n'
                                      'TINK_GEMINI_MODEL=gemini-3.8-flash\n', encoding='utf-8')
        observed = []
        def capture(req, timeout, cap):
            observed.append(req.get_header('X-goog-api-key'))
            return response(suggestion())
        self.client(transport=capture, environ={}).generate(packet())
        self.assertEqual(observed, ['${NO_EXPANSION}'])
        self.assertFalse(self.client(environ={'GEMINI_API_KEY': ''}).configured)
        self.assertNotIn('offline-secret', repr(self.client()))

    def test_symlink_env_is_refused(self):
        outside = self.root / 'other'
        outside.write_text('GEMINI_API_KEY=offline', encoding='utf-8')
        try:
            (self.root / '.env').symlink_to(outside)
        except OSError:
            self.skipTest('Windows account does not permit creating test symlinks')
        with self.assertRaises(GeminiError):
            self.client()

    def test_local_demo_override_is_literal_and_never_executes(self):
        demo = self.root / 'demo_secrets.py'
        demo.write_text("GEMINI_API_KEY = 'demo-only'\nTINK_GEMINI_MODEL = 'gemini-3.8-flash'\n", encoding='utf-8')
        client = self.client(environ={'GEMINI_API_KEY': ''})
        self.assertTrue(client.configured)
        observed = []
        def capture(req, timeout, cap):
            observed.append(req.get_header('X-goog-api-key'))
            return response(suggestion())
        self.client(transport=capture).generate(packet())
        self.assertEqual(observed, ['demo-only'])
        demo.write_text("GEMINI_API_KEY = __import__('os').getcwd()\n", encoding='utf-8')
        with self.assertRaises(GeminiError):
            self.client()

    def test_native_request_is_stateless_header_only_and_bounded(self):
        def capture(req, timeout, cap):
            self.assertEqual(req.full_url, ENDPOINT)
            self.assertEqual(req.get_header('X-goog-api-key'), 'offline-secret')
            self.assertNotIn(b'offline-secret', req.data)
            body = json.loads(req.data)
            self.assertFalse(body['store'])
            self.assertEqual(body['generation_config']['thinking_level'], 'low')
            self.assertEqual(body['response_format']['schema']['properties']['options']['maxItems'], 4)
            self.assertNotIn('tools', body)
            self.assertNotIn('previous_interaction_id', body)
            self.assertEqual(timeout, 8.0)
            self.assertLessEqual(cap, 131072)
            return response(suggestion())
        result = self.client(transport=capture).generate(packet())
        self.assertTrue(result['needs_clarification'])

    def test_failure_messages_do_not_echo_transport_or_server_content(self):
        def fail(*_):
            raise RuntimeError('offline-secret private conversation')
        with self.assertRaises(GeminiError) as caught:
            self.client(transport=fail).generate(packet())
        self.assertNotIn('offline-secret', str(caught.exception))
        self.assertNotIn('private conversation', str(caught.exception))
        with self.assertRaises(GeminiError):
            _NoRedirect().redirect_request(None, None, 302, 'x', {}, 'https://other.example')

    def test_service_error_status_is_visible_without_body_or_secret(self):
        import urllib.error
        def fail(*_):
            raise urllib.error.HTTPError('https://example.test/offline-secret', 503, 'private conversation', {}, None)
        with self.assertRaises(GeminiError) as caught:
            self.client(transport=fail).generate(packet())
        self.assertIn('503', str(caught.exception))
        self.assertNotIn('offline-secret', str(caught.exception))
        self.assertNotIn('private conversation', str(caught.exception))

    def test_cold_start_and_context_only_guards(self):
        self.assertTrue(self.client().generate(packet())['needs_clarification'])
        with self.assertRaises(GeminiError):
            self.client(suggestion('Invented meaning')).generate(packet())
        context_only = packet()
        context_only['event_ids'] = []
        with self.assertRaises(GeminiError):
            self.client(suggestion(refs=[])).generate(context_only)
        result = self.client(suggestion('Could you confirm your message?', 'clarify', [])).generate(context_only)
        self.assertEqual(result['options'][0]['kind'], 'clarify')
        context_only['spelled'] = 'Hello'
        self.assertEqual(self.client(suggestion('Hello', refs=[])).generate(context_only)['options'][0]['text'], 'Hello')
        invalid_clock = packet()
        invalid_clock['quality']['clock_valid'] = False
        with self.assertRaises(GeminiError):
            self.client().generate(invalid_clock)

    def test_grounded_paraphrases_still_need_owned_events_and_valid_refs(self):
        data = packet()
        data['belief'] = {'kind': 'intent_posterior', 'calibrated': True}
        self.assertFalse(self.client(suggestion('I would like some water')).generate(data)['needs_clarification'])
        data['user_role']['ownership_valid'] = False
        with self.assertRaises(GeminiError):
            self.client(suggestion('I would like some water')).generate(data)

    def test_malformed_and_unsupported_outputs_fail_closed(self):
        variants = []
        bad = suggestion(); bad['options'] *= 5; variants.append(bad)
        bad = suggestion(); bad['options'] *= 2; variants.append(bad)
        bad = suggestion(); bad['confidence'] = .99; variants.append(bad)
        variants.append(suggestion('x' * 161))
        variants.append(suggestion(refs=['unavailable']))
        variants.append(suggestion('Yes\n'))
        for result in variants:
            with self.subTest(result=result), self.assertRaises(GeminiError):
                self.client(result).generate(packet())
        for raw in [b'invalid', b'{}', b'[]', b'X' * 131073]:
            with self.subTest(raw_length=len(raw)), self.assertRaises(GeminiError):
                self.client(transport=lambda *_, raw=raw: raw).generate(packet())

    def test_opt_in_inline_audio_schema_and_chunk_scoped_speakers(self):
        def capture(req, timeout, cap):
            import base64
            import io
            import wave
            body = json.loads(req.data)
            self.assertFalse(body['store'])
            audio = body['input'][1]
            self.assertEqual(audio['type'], 'audio')
            self.assertEqual(audio['mime_type'], 'audio/wav')
            with wave.open(io.BytesIO(base64.b64decode(audio['data']))) as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getnframes()), (16000, 1, 32000))
            return response({'turns': [{'text': 'Hello', 'speaker': 'A', 'language': 'en', 'overlap': False},
                                       {'text': 'Hi', 'speaker': 'B', 'language': 'en', 'overlap': True}]})
        client = self.client(transport=capture)
        turns = client.transcribe_audio([0] * 32000, 10)
        self.assertEqual([t['speaker_label'] for t in turns], ['A', 'B'])
        self.assertTrue(all(t['speaker_uncertain'] and t['source'] == 'cloud_audio' for t in turns))
        self.assertEqual([(t['start_s'], t['end_s']) for t in turns], [(10, 12), (10, 12)])
        with self.assertRaises(GeminiError):
            client.transcribe_audio([0] * 16000, 10, tts_active=True)
        for invalid in [[], [float('nan')], [0] * 128001, [[0, 1]]]:
            with self.subTest(length=len(invalid)), self.assertRaises(GeminiError):
                client.transcribe_audio(invalid, 10)
        with self.assertRaises(GeminiError):
            self.client({'turns': [{'text': 'oops', 'speaker': 'Named person', 'language': 'en',
                                   'overlap': False}]}).transcribe_audio([0], 10)


class ConversationTests(unittest.TestCase):
    def test_revision_dedup_finalization_and_speaker_attribution(self):
        state = ConversationState()
        turn = {'turn_id': 't1', 'text': 'Water?', 'start_s': 1, 'end_s': 2,
                'speaker_id': 'A', 'speaker_role': 'partner', 'final': False}
        self.assertTrue(state.add_turn(turn))
        self.assertFalse(state.add_turn(turn))
        self.assertIsNone(state.snapshot(3)['pending_question'])
        self.assertTrue(state.add_turn(dict(turn, final=True, revision=1)))
        self.assertFalse(state.add_turn(dict(turn, text='wrong older', revision=0)))
        self.assertEqual(state.snapshot(3)['pending_question']['speaker_id'], 'A')
        state.confirm_user('Yes', ['e1'], 4)
        self.assertIsNone(state.snapshot(5)['pending_question'])
        self.assertEqual(state.snapshot(5)['confirmed_user_turns'][0]['text'], 'Yes')

    def test_tts_echo_and_unconfirmed_suggestions_never_become_history(self):
        state = ConversationState()
        state.record_tts('Hello there', 10, 12)
        self.assertFalse(state.add_turn({'text': 'Hello, there!', 'start_s': 10, 'end_s': 12}))
        self.assertFalse(state.add_turn({'text': 'Invented', 'source': 'suggestion'}))
        self.assertTrue(state.add_turn({'text': 'Real interruption', 'start_s': 11, 'end_s': 12,
                                       'overlap': True}))
        snapshot = state.snapshot(13)
        self.assertTrue(snapshot['unknown_speakers'])
        self.assertTrue(snapshot['overlap_present'])
        self.assertEqual(snapshot['confirmed_user_turns'], [])
        self.assertEqual(snapshot['mentions'][0]['status'], 'mentioned_in_transcript')

    def test_bounded_history_and_time_decay(self):
        state = ConversationState(max_turns=2, max_age_s=10)
        for i in range(3):
            state.add_turn({'turn_id': str(i), 'text': 'mention', 'start_s': i, 'end_s': i})
        self.assertEqual(len(state.snapshot(3)['turns']), 2)
        self.assertEqual(state.snapshot(30)['turns'], [])
        state.reject_suggestions(['o1'], 30)
        self.assertEqual(state.snapshot(31)['rejections'][0]['option_ids'], ['o1'])


class StructuredAsrTests(unittest.TestCase):
    def test_whisper_segment_times_language_and_unknown_speaker(self):
        asr = Transcriber()
        asr.model = SimpleNamespace(transcribe=lambda *a, **k: (
            iter([SimpleNamespace(text=' hello ', start=.25, end=.75)]),
            SimpleNamespace(language='en')))
        turn = asr.transcribe_turns([0] * 16000, 10)[0]
        self.assertEqual((turn['start_s'], turn['end_s']), (10.25, 10.75))
        self.assertEqual(turn['speaker_id'], 'unknown')
        self.assertEqual(turn['language'], 'en')
        self.assertTrue(turn['final'])
        self.assertEqual(asr.transcribe([0]), 'hello')

    def test_meta_retains_recording_scoped_speakers_and_estimated_turn_times(self):
        asr = MetaTranscriber(api_key_env='TINK_OFFLINE_TEST_UNUSED_ENV', environ={})
        asr.key = 'offline'
        asr._call_json = lambda _: {'audioDurationMs': 2000, 'transcript': 'Hello Hi Uncertain',
            'turns': [{'turnId': i, 'transcript': text, 'speaker': label,
                       'startMs': i * 500, 'endMs': (i+1) * 500}
                      for i, (label, text) in enumerate([('A', 'Hello'), ('B', 'Hi'), ('unknown', 'Uncertain')])]}
        turns = asr.transcribe_turns([0] * 32000, 5)
        self.assertEqual([t['speaker_label'] for t in turns], ['A', 'B', 'unknown'])
        self.assertNotEqual(turns[0]['speaker_id'], turns[1]['speaker_id'])
        self.assertEqual([(t['start_s'], t['end_s']) for t in turns], [(5, 5.5), (5.5, 6), (6, 6.5)])
        self.assertTrue(all(t['timestamp_quality'] == 'asr_turn' for t in turns))
        self.assertTrue(all(t['speaker_role'] == 'unknown' and t['speaker_uncertain'] for t in turns))
        self.assertEqual(asr.call([0] * 32000), 'Hello Hi Uncertain')


if __name__ == '__main__':
    unittest.main(verbosity=2)
