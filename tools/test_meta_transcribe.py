"""Offline Muse request/response and recorded-service contracts. No credentials or devices."""
from collections import deque
from copy import deepcopy
import io
import http.client
import json
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transcribe import (MetaTranscriber, MetaTranscriptionError, META_ASR_URL,
                       META_MAX_RESPONSE_BYTES, _NoMetaRedirect, _meta_transport, meta_settings)
from assessment_capture import ClipRecorder
from assessment_service import ClipAssessmentService
from session_engine import SessionEngine


def response():
    return {'sessionId': 'provider-correlation', 'audioDurationMs': 1000, 'transcript': 'Hello. Yes.',
        'turns': [{'turnId': 1, 'startMs': 100, 'endMs': 500, 'transcript': 'Hello.', 'speaker': 'A'},
                  {'turnId': 2, 'startMs': 450, 'endMs': 900, 'transcript': 'Yes.', 'speaker': 'B'}]}


class MetaAdapterTests(unittest.TestCase):
    def client(self, payload=None, transport=None):
        return MetaTranscriber(environ={'MODEL_API_KEY': 'fake-key'}, use_fallback=False,
            transport=transport or (lambda *args: json.dumps(response() if payload is None else payload).encode()))

    def test_request_is_fixed_destination_bounded_pcm_diarization_not_chat_or_files_api(self):
        calls = []
        def transport(request, timeout, limit):
            calls.append(request)
            self.assertEqual(request.full_url, META_ASR_URL)
            self.assertEqual(request.get_method(), 'POST')
            self.assertEqual(request.get_header('Authorization'), 'Bearer fake-key')
            self.assertEqual(request.get_header('Accept'), 'application/json')
            self.assertEqual(limit, META_MAX_RESPONSE_BYTES)
            self.assertLess(len(request.data), 1_000_000)
            self.assertIn(b'"mode": "DIARIZATION"', request.data)
            self.assertIn(b'name="request"', request.data)
            self.assertIn(b'name="audio"; filename="chunk.wav"', request.data)
            self.assertNotIn(b'fake-key', request.data)
            start = request.data.index(b'RIFF')
            wav_size = int.from_bytes(request.data[start+4:start+8], 'little') + 8
            with wave.open(io.BytesIO(request.data[start:start+wav_size])) as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 16000))
                self.assertEqual(wav.getnframes(), 16000)
            return json.dumps(response()).encode()
        turns = self.client(transport=transport).transcribe_turns(np.zeros(16000), 10.)
        self.assertEqual(len(calls), 1)
        self.assertEqual((turns[0]['start_s'], turns[0]['end_s']), (10.1, 10.5))
        self.assertTrue(all(t['speaker_uncertain'] and t['speaker_role'] == 'unknown' for t in turns))
        self.assertTrue(all(t['overlap'] for t in turns))

    def test_speaker_labels_never_become_cross_request_or_verified_person_identity(self):
        client = self.client()
        a, b = client.transcribe_turns(np.zeros(16000), 10), client.transcribe_turns(np.zeros(16000), 10)
        self.assertNotEqual(a[0]['speaker_id'], b[0]['speaker_id'])
        self.assertEqual(a[0]['speaker_scope'], 'recording')
        payload = response()
        payload['turns'][0].pop('speaker')
        self.assertEqual(self.client(payload).transcribe_turns(np.zeros(16000), 0)[0]['speaker_id'], 'unknown')

    def test_silence_is_empty_and_out_of_order_turns_are_chronological(self):
        silence = {'audioDurationMs': 1000, 'transcript': '', 'turns': []}
        self.assertEqual(self.client(silence).transcribe_turns(np.zeros(16000)), [])
        payload = response()
        payload['turns'].reverse()
        turns = self.client(payload).transcribe_turns(np.zeros(16000), 2)
        self.assertEqual([t['text'] for t in turns], ['Hello.', 'Yes.'])

    def test_bad_input_or_key_cannot_reach_transport(self):
        calls = []
        client = self.client(transport=lambda *args: calls.append(args))
        for samples, start in [(np.zeros((1, 2)), 0), ([float('nan')], 0), ([], 0),
                               (np.zeros(480001), 0), ([0], -1), ([0], float('inf'))]:
            with self.subTest(start=start, size=np.asarray(samples).size):
                with self.assertRaises(MetaTranscriptionError):
                    client.transcribe_turns(samples, start)
        self.assertEqual(calls, [])
        with self.assertRaises(MetaTranscriptionError):
            MetaTranscriber(environ={'MODEL_API_KEY': 'fake\r\nInjected: value'})
        missing = MetaTranscriber(environ={}, use_fallback=False, transport=lambda *args: calls.append(args))
        with self.assertRaises(MetaTranscriptionError):
            missing.transcribe_turns([0])
        self.assertEqual(calls, [])

    def test_malformed_provider_times_text_and_ids_fail_closed(self):
        payloads = []
        for field, value in [('startMs', -1), ('endMs', 1501), ('endMs', float('nan')),
                             ('turnId', True), ('transcript', 'x' * 2001), ('speaker', 'person:name')]:
            payload = response()
            payload['turns'][0][field] = value
            payloads.append(payload)
        duplicate = response()
        duplicate['turns'][1]['turnId'] = 1
        payloads.extend([duplicate, {'transcript': 'untimed speech', 'turns': [], 'audioDurationMs': 1000}])
        for payload in payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(MetaTranscriptionError):
                    self.client(payload).transcribe_turns(np.zeros(16000))

    def test_errors_do_not_echo_response_credentials_or_redirect(self):
        def fail(*args):
            raise urllib.error.HTTPError(META_ASR_URL, 429, 'fake-key private words', {}, None)
        with self.assertRaises(MetaTranscriptionError) as raised:
            self.client(transport=fail).transcribe_turns([0])
        self.assertNotIn('fake-key', str(raised.exception))
        self.assertNotIn('private words', str(raised.exception))
        with self.assertRaises(MetaTranscriptionError):
            _NoMetaRedirect().redirect_request(None, None, 302, '', {}, 'https://untrusted.example')
        with self.assertRaises(MetaTranscriptionError):
            self.client(transport=lambda *args: b'x' * (META_MAX_RESPONSE_BYTES + 1)).transcribe_turns([0])

    def test_protocol_and_arbitrary_transport_failures_are_sanitized_at_cloud_boundary(self):
        private = 'fake-key private-response-first-line'
        errors = [http.client.BadStatusLine(private),
                  http.client.IncompleteRead(private.encode(), 100),
                  RuntimeError(private), MetaTranscriptionError(private)]
        for error in errors:
            with self.subTest(error_type=type(error).__name__):
                calls = []
                def fail(*args):
                    calls.append(args)
                    raise error
                try:
                    self.client(transport=fail).transcribe_turns([0])
                except MetaTranscriptionError as exc:
                    displayed = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    self.assertNotIn(private, str(exc))
                    self.assertNotIn(private, displayed)
                    self.assertTrue(exc.__suppress_context__)
                else:
                    self.fail('An unsanitized transport result escaped')
                self.assertEqual(len(calls), 1)

    def test_protocol_failure_falls_back_locally_without_repeating_cloud_request(self):
        calls = []
        def fail(*args):
            calls.append(args)
            raise http.client.BadStatusLine('fake-key private-response-first-line')
        client = MetaTranscriber(environ={'MODEL_API_KEY': 'fake-key'}, transport=fail)
        client.fallback = SimpleNamespace(transcribe_turns=lambda *args: [{'text': 'Local only.'}])
        self.assertEqual(client.transcribe([0]), 'Local only.')
        self.assertEqual(len(calls), 1)
        self.assertEqual(client.last_backend, 'whisper')
        self.assertNotIn('fake-key', client.last_error)

    def test_default_transport_disables_environment_proxy_and_validates_destination_and_size(self):
        fake = SimpleNamespace(geturl=lambda: META_ASR_URL, read=lambda n: b'{}',
                               __enter__=lambda self: self, __exit__=lambda *args: None)
        class Reply:
            def __enter__(self): return fake
            def __exit__(self, *args): return None
        with patch('transcribe.urllib.request.build_opener') as build:
            build.return_value.open.return_value = Reply()
            self.assertEqual(_meta_transport(object(), 1, 8), b'{}')
            self.assertEqual(build.call_args.args[0].proxies, {})
            fake.geturl = lambda: 'https://untrusted.example'
            with self.assertRaises(MetaTranscriptionError):
                _meta_transport(object(), 1, 8)
            fake.geturl = lambda: META_ASR_URL
            fake.read = lambda n: b'x' * 9
            with self.assertRaises(MetaTranscriptionError):
                _meta_transport(object(), 1, 8)

    def test_literal_project_settings_and_explicit_empty_os_key(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '.env'
            path.write_text('MODEL_API_KEY="fake-key"\nTINK_TRANSCRIPTION_PROVIDER=meta\nIGNORED=secret\n')
            settings = meta_settings(folder, {})
            self.assertEqual(settings, {'MODEL_API_KEY': 'fake-key', 'TINK_TRANSCRIPTION_PROVIDER': 'meta'})
            self.assertEqual(meta_settings(folder, {'MODEL_API_KEY': ''})['MODEL_API_KEY'], '')
            path.write_text('TINK_TRANSCRIPTION_PROVIDER=unknown')
            with self.assertRaises(MetaTranscriptionError):
                meta_settings(folder, {})
            path.write_bytes(b'x' * 16385)
            with self.assertRaises(MetaTranscriptionError):
                meta_settings(folder, {})

    def test_legacy_local_fallback_is_explicit_and_does_not_add_a_second_cloud_call(self):
        client = MetaTranscriber(environ={}, transport=lambda *args: self.fail('network'), use_fallback=True)
        client.fallback = SimpleNamespace(transcribe_turns=lambda *args: [{'text': 'Local only.'}])
        self.assertEqual(client.transcribe([0]), 'Local only.')
        self.assertEqual(client.last_backend, 'whisper')
        self.assertTrue(client.last_error)


class RecordedMuseTests(unittest.TestCase):
    def setup_recording(self, directory, provider='meta'):
        service = object.__new__(ClipAssessmentService)
        service.root = Path(directory)
        service.clock = SimpleNamespace(session_id='session')
        service.engine = SessionEngine('session')
        service.engine.set_role('u', True)
        service.asr_provider = provider
        service.warnings, service.recent_assessments = [], deque(maxlen=5)
        service.encoder = None
        service.transcriber = SimpleNamespace(transcribe_turns=lambda *args: [
            {'turn_id': 'local-1', 'text': 'Local hello.', 'start_s': 10.1, 'end_s': 10.5,
             'speaker_id': 'unknown', 'speaker_role': 'unknown', 'source': 'whisper', 'final': True}])
        service.visual = SimpleNamespace(target=SimpleNamespace(ownership_valid=True, role_source='operator_confirmed'))
        clip = ClipRecorder(Path(directory) / 'clip', 'session', 10.)
        clip.add_frame(np.zeros((10, 10, 3), np.uint8), 10.)
        clip.add_audio(np.zeros(16000), 10., True)
        clip.add_observation({'t_capture': 10.1, 'logical_person_id': 'u', 'person_id': 1,
                              'ownership_valid': True, 'channels': {}})
        return service, clip, clip.finish(11.)

    def test_packet_is_offline_and_opt_in_assess_attaches_timed_cloud_context_once(self):
        with tempfile.TemporaryDirectory() as folder:
            service, clip, capture = self.setup_recording(folder)
            with patch('transcribe.MetaTranscriber') as constructor:
                packet = service.packet(clip, capture)
                constructor.assert_not_called()
                self.assertEqual(service.engine.context.snapshot(11)['finalized_turns'], [])
                constructor.return_value.transcribe_turns.return_value = MetaAdapterTests().client().transcribe_turns(np.zeros(16000), 10.)
                with patch('assessment_service.AssessmentClient') as gemini:
                    gemini.return_value.assess.return_value = {'assessment': {'summary': 'Tentative'}, 'partner_context': {}}
                    service.assess(clip, capture, packet)
                    service.assess(clip, capture, packet)
                self.assertEqual(constructor.return_value.transcribe_turns.call_count, 1)
                self.assertEqual(packet['conversation']['cloud_transcription']['status'], 'ok')
                turns = service.engine.context.snapshot(11)['finalized_turns']
                self.assertEqual(len(turns), 2)
                self.assertTrue(all(t['backend'] == 'meta_muse' and t['speaker_role'] == 'unknown' for t in turns))
                self.assertEqual(len(packet['conversation']['local_transcript']), 1)
                self.assertEqual(len(packet['conversation']['cloud_transcript']), 2)
                saved = json.loads((clip.folder / 'assessment-input.json').read_text())
                self.assertEqual(saved['conversation']['cloud_transcription']['status'], 'ok')

    def test_default_local_assessment_never_constructs_muse(self):
        with tempfile.TemporaryDirectory() as folder:
            service, clip, capture = self.setup_recording(folder, provider='local')
            with patch('transcribe.MetaTranscriber') as constructor, patch('assessment_service.AssessmentClient') as gemini:
                gemini.return_value.assess.return_value = {'assessment': {'summary': 'Tentative'}, 'partner_context': {}}
                packet = service.packet(clip, capture)
                service.assess(clip, capture, packet)
                constructor.assert_not_called()
                self.assertNotIn('cloud_transcription', packet['conversation'])
                self.assertEqual(len(service.engine.context.snapshot(11)['finalized_turns']), 1)

    def test_missing_key_failure_uses_existing_local_context_without_provider_details(self):
        with tempfile.TemporaryDirectory() as folder:
            service, clip, capture = self.setup_recording(folder)
            packet = service.packet(clip, capture)
            with patch('transcribe.MetaTranscriber') as constructor:
                constructor.return_value.transcribe_turns.side_effect = MetaTranscriptionError('Muse unavailable.')
                service._meta_transcript(clip, capture, packet)
            self.assertEqual(packet['conversation']['cloud_transcription']['status'], 'unavailable')
            self.assertNotIn('cloud_transcript', packet['conversation'])
            self.assertEqual(len(service.engine.context.snapshot(11)['finalized_turns']), 1)

    def test_bad_clock_or_audio_gap_cannot_enter_conversation_context_as_clean_timed_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            service, clip, capture = self.setup_recording(folder)
            packet = service.packet(clip, capture)
            capture['clock_valid'] = False
            with patch('transcribe.MetaTranscriber') as constructor:
                service._meta_transcript(clip, capture, packet)
                constructor.assert_not_called()
            self.assertEqual(service.engine.context.snapshot(11)['finalized_turns'], [])
        with tempfile.TemporaryDirectory() as folder:
            service, clip, capture = self.setup_recording(folder)
            capture['audio_gaps'] = [[10.05, 10.8]]
            packet = service.packet(clip, capture)
            with patch('transcribe.MetaTranscriber') as constructor:
                constructor.return_value.transcribe_turns.return_value = MetaAdapterTests().client().transcribe_turns(np.zeros(16000), 10.)
                service._meta_transcript(clip, capture, packet)
            self.assertTrue(all(t['timestamp_quality'] == 'asr_turn_with_capture_gap'
                                for t in packet['conversation']['cloud_transcript']))
            self.assertEqual(service.engine.context.snapshot(11)['finalized_turns'], [])


if __name__ == '__main__':
    unittest.main()
