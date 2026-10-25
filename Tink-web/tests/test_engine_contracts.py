"""Plain unittest coverage: local HTTP and fake devices/models; no credentials/cloud."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


server = load('tink_engine_contract_test_server', ROOT / 'engine/server.py')
proxy = load('tink_engine_contract_test_proxy', ROOT / 'tink/engine.py')


class CameraTests(unittest.TestCase):
    def test_release_waits_for_native_read(self):
        entered, resume, released = threading.Event(), threading.Event(), threading.Event()
        class Device:
            def read(self):
                entered.set()
                resume.wait(2.)
                return True, SimpleNamespace(shape=(2, 2, 3))
            def release(self):
                released.set()
        camera = server.SwitchCamera.__new__(server.SwitchCamera)
        camera.lock, camera.cap = threading.Lock(), Device()
        camera.on, camera.error, camera.mirror = True, None, False
        reader = threading.Thread(target=camera.read)
        closer = threading.Thread(target=lambda: camera.set_on(False))
        reader.start()
        self.assertTrue(entered.wait(1.))
        closer.start()
        try:
            self.assertFalse(released.wait(.03), 'native device released during read')
        finally:
            resume.set()
            reader.join(2.)
            closer.join(2.)
        self.assertTrue(released.is_set())
        self.assertIsNone(camera.cap)


class LifecycleTests(unittest.TestCase):
    def test_spawn_passes_actual_supervisor_and_omits_unset_mic(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = proxy.Engine(ROOT, Path(directory) / 'engine.log', python='offline-unused', mic='')
            process = SimpleNamespace(poll=lambda: None)
            engine._raw = Mock(return_value=(200, 'application/json', b'{}'))
            with patch.object(proxy.subprocess, 'Popen', return_value=process) as spawn, \
                    patch.object(proxy.os, 'getpid', return_value=24680):
                engine._spawn()
            command = spawn.call_args.args[0]
            self.assertEqual(command[command.index('--parent-pid') + 1], '24680')
            self.assertFalse(any(arg.startswith('--device') for arg in command))

    def test_windows_monitor_waits_actual_supervisor_not_launcher(self):
        wait = Mock(side_effect=[258, 258, 0])
        close = Mock()
        with patch.object(server.os, 'name', 'nt'), \
                patch.object(server.os, 'getppid', return_value=13579), \
                patch.object(server, '_windows_parent_waiter', return_value=(wait, close)) as watcher, \
                patch.object(server.os, '_exit', side_effect=SystemExit) as exit_process:
            with self.assertRaises(SystemExit):
                server.watch_parent(24680)
        watcher.assert_called_once_with(24680)
        self.assertEqual(wait.call_count, 3)
        close.assert_called_once_with()
        exit_process.assert_called_once_with(0)

    def test_missing_windows_supervisor_exits_without_waiting(self):
        with patch.object(server.os, 'name', 'nt'), \
                patch.object(server, '_windows_parent_waiter', return_value=None), \
                patch.object(server.os, '_exit', side_effect=SystemExit):
            with self.assertRaises(SystemExit):
                server.watch_parent(24680)

    def scenario(self, failure, real_executor=None):
        actions = []
        class Service:
            def __init__(self, *args):
                self.clock = SimpleNamespace(now=lambda: 0.)
            def prepare(self):
                actions.append('prepare')
                if failure == 'prepare':
                    raise RuntimeError('fake prepare failure')
            def close(self):
                actions.append('service_close')
        class Camera:
            def __init__(self, *args):
                actions.append('camera_init')
                if failure == 'camera':
                    raise RuntimeError('fake camera failure')
            def release(self):
                actions.append('camera_close')
        class Live:
            def __init__(self, *args, **kwargs):
                pass
            def start(self):
                actions.append('live_start')
                raise RuntimeError('fake capture startup failure')
            def close(self):
                actions.append('live_close')
        class Executor:
            def __init__(self, *args, **kwargs):
                pass
            def shutdown(self, wait=True, cancel_futures=False):
                actions.append(('executor_close', wait, cancel_futures))
        modules = {'assessment_capture': SimpleNamespace(ClipRecorder=object, RecordedAudioCapture=lambda clock: None),
                   'assessment_service': SimpleNamespace(ClipAssessmentService=Service),
                   'gemini_client': SimpleNamespace(GeminiError=RuntimeError),
                   'tools.video_layer_lab': SimpleNamespace(LiveCapture=Live)}
        session = server.NormalSession(SimpleNamespace(camera=0, audio=False, device=None))
        return actions, session, modules, Camera, (lambda **kw: real_executor) if real_executor else Executor

    def test_partial_startup_failures_close_every_allocated_resource(self):
        for failure in ('prepare', 'camera', 'live'):
            actions, session, modules, camera, executor = self.scenario(failure)
            with self.subTest(failure=failure), patch.dict(sys.modules, modules), \
                    patch.object(server, 'SwitchCamera', camera), patch.object(server, 'ThreadPoolExecutor', executor):
                with self.assertRaises(RuntimeError):
                    session.run()
                self.assertEqual(actions[-1], 'service_close')
                if failure != 'prepare':
                    self.assertIn(('executor_close', True, True), actions)
                if failure == 'live':
                    self.assertIn('live_close', actions)
                    self.assertIn('camera_close', actions)

    def test_service_disposal_waits_for_running_worker(self):
        gate, entered = threading.Event(), threading.Event()
        pool = ThreadPoolExecutor(max_workers=1)
        def running_work():
            entered.set()
            gate.wait(2.)
        future = pool.submit(running_work)
        self.assertTrue(entered.wait(1.))
        actions, session, modules, camera, executor = self.scenario('live', real_executor=pool)
        errors = []
        def run_session():
            try:
                session.run()
            except RuntimeError:
                errors.append('expected startup error')
        with patch.dict(sys.modules, modules), patch.object(server, 'SwitchCamera', camera), \
                patch.object(server, 'ThreadPoolExecutor', executor):
            thread = threading.Thread(target=run_session)
            thread.start()
            try:
                threading.Event().wait(.03)
                self.assertNotIn('service_close', actions)
            finally:
                gate.set()
                thread.join(2.)
        self.assertTrue(future.done())
        self.assertEqual(actions[-1], 'service_close')
        self.assertEqual(errors, ['expected startup error'])

    def test_stop_during_loading_does_not_open_devices_after_models_finish(self):
        actions, session, modules, camera, executor = self.scenario('live')
        session.stop.set()
        with patch.dict(sys.modules, modules), patch.object(server, 'SwitchCamera', camera):
            session.run()
        self.assertEqual(actions, ['prepare', 'service_close'])

    def test_shutdown_requests_finalization_before_windows_termination(self):
        events = []
        class Process:
            stopped = False
            def poll(self):
                return 0 if self.stopped else None
            def terminate(self):
                events.append('terminate')
                self.stopped = True
            def wait(self, timeout):
                events.append('wait')
                return 0
        engine = proxy.Engine(ROOT, ROOT / 'unused-test.log', python='offline-unused')
        engine.proc, engine.want = Process(), {'mode': 'normal', 'sid': 'owner'}
        responses = iter(['running', 'stopped'])
        def request(method, path, body=None, timeout=10):
            if method == 'POST':
                self.assertEqual(body, {'sid': 'owner'})
                events.append('stop')
                phase = 'stopping'
            else:
                phase = next(responses)
                events.append(phase)
            return 200, 'application/json', json.dumps({'phase': phase}).encode()
        engine._raw = request
        with patch.object(proxy.time, 'sleep'):
            engine.shutdown()
        self.assertEqual(events, ['stop', 'running', 'stopped', 'terminate', 'wait'])
        self.assertIsNone(engine.proc)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.session = server.NormalSession(SimpleNamespace())
        self.session.phase = 'running'
        self.clock = [10.]
        self.confirmed = []
        self.service = SimpleNamespace(clock=SimpleNamespace(now=lambda: self.clock[0], session_id='session-test'),
                                       confirm=lambda *args: self.confirmed.append(args))

    def handle(self, name, data, recorder=None):
        return self.session._handle(name, data, self.service, None, None, None, recorder, ROOT, None, {})

    def test_timed_out_queued_command_never_executes(self):
        result, status = self.session.call('record', {}, timeout=.001)
        self.assertEqual(status, 504)
        handler = Mock()
        with patch.object(self.session, '_handle', handler):
            self.session._drain_commands(None, None, None, None, None, None, None, {})
        handler.assert_not_called()
        self.assertTrue(self.session.commands.empty())

    def test_confirmation_binds_index_clip_option_and_result(self):
        self.session.result = {'options': [{'id': '1', 'kind': 'hypothesis', 'text': 'I want a pause.'}]}
        self.session.result_id, self.session.packet = 'result-new', {'clip_id': 'clip-new'}
        valid = {'index': 0, 'clip_id': 'clip-new', 'option_id': '1', 'result_id': 'result-new'}
        for change in [{'index': -1}, {'index': True}, {'index': 1}, {'clip_id': 'old'},
                       {'option_id': '2'}, {'result_id': 'old'}, {'result_id': None}]:
            with self.subTest(change=change):
                _, status = self.handle('confirm', dict(valid, **change))
                self.assertIn(status, (400, 409))
                self.assertEqual(self.confirmed, [])
        receipt, status = self.handle('confirm', valid)
        self.assertEqual(status, 200)
        self.assertEqual(receipt, {'ok': True, 'text': 'I want a pause.', 'kind': 'hypothesis',
                                   'clip_id': 'clip-new', 'option_id': '1', 'result_id': 'result-new'})
        self.assertEqual(len(self.confirmed), 1)
        self.assertIsNone(self.session.result_id)
        self.assertEqual(self.handle('confirm', valid)[1], 400)

    def test_playback_lease_and_echo_guard_block_recording(self):
        created = []
        recorder = lambda *args: created.append(args) or SimpleNamespace()
        self.assertEqual(self.handle('playback', {'active': True})[1], 200)
        self.assertEqual(self.handle('record', {}, recorder)[1], 409)
        self.clock[0] = 20.
        self.handle('playback', {'active': True})  # renewal extends the reservation
        self.clock[0] = 30.
        self.assertEqual(self.handle('record', {}, recorder)[1], 409)
        self.handle('playback', {'active': False})
        self.assertEqual(self.handle('record', {}, recorder)[1], 409)
        self.clock[0] += .71
        with patch.object(self.session, 'note'):
            self.assertEqual(self.handle('record', {}, recorder)[1], 200)
        self.assertEqual(len(created), 1)
        self.assertEqual(self.handle('playback', {'active': True})[1], 409)


class PublicStateTests(unittest.TestCase):
    def test_result_exposes_identity_needed_by_confirmation_client(self):
        session = server.NormalSession(SimpleNamespace(audio=False))
        session.packet = {'clip_id': 'clip-1'}
        session.result_id = 'result-1'
        session.result = {'assessment': {'summary': 'A tentative interpretation'},
            'options': [{'id': '1', 'kind': 'hypothesis', 'text': 'No, thank you.', 'reason': 'Visible motion'}]}
        session._publish(SimpleNamespace(warnings=[]), {'people': []}, 1., None, None,
                         SimpleNamespace(on=False, error=None))
        state = session.state()
        self.assertEqual(state['result']['clip_id'], 'clip-1')
        self.assertEqual(state['result_id'], 'result-1')


class HttpOwnershipTests(unittest.TestCase):
    def setUp(self):
        class FakeNormal(server.NormalSession):
            def __init__(self, opts):
                super().__init__(opts)
                self.phase, self.applied = 'running', []
            def call(self, name, data):
                self.applied.append(name)
                return {'ok': True}, 200
            command = call
        self.sessions = patch.dict(server.SESSIONS, {'normal': FakeNormal}, clear=True)
        self.sessions.start()
        self.addCleanup(self.sessions.stop)
        self.manager = server.Manager(SimpleNamespace())
        self.http = server.ThreadingHTTPServer(('127.0.0.1', 0), server.make_handler(self.manager))
        self.port = self.http.server_port
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)

    def close(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(2.)

    def post(self, path, body, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=2.)
        try:
            connection.request('POST', path, json.dumps(body), {'Content-Type': 'application/json', **(headers or {})})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_missing_and_stale_owners_cannot_mutate_the_current_session(self):
        self.assertEqual(self.post('/api/session/start', {'mode': 'normal'})[0], 400)
        self.assertEqual(self.post('/api/session/start', {'mode': 'normal', 'sid': 'new'})[0], 200)
        current = self.manager.session
        for path in ['/api/normal/record', '/api/normal/confirm', '/api/camera', '/api/session/stop']:
            with self.subTest(path=path):
                self.assertEqual(self.post(path, {'sid': 'stale'})[0], 409)
        self.assertEqual(current.applied, [])
        self.assertEqual(self.post('/api/normal/record', {'sid': 'new'})[0], 200)
        self.assertEqual(current.applied, ['record'])

    def test_stop_requested_session_is_not_reused(self):
        self.manager.start('normal', 'owner')
        old = self.manager.session
        self.manager.stop('owner')
        new = self.manager.start('normal', 'owner')
        self.assertIsNot(old, new)

    def test_foreign_localhost_origin_and_nonloopback_client_are_refused(self):
        self.assertEqual(self.post('/api/session/start', {'mode': 'normal', 'sid': 'owner'},
                                   {'Origin': f'http://127.0.0.1:{self.port + 1}'})[0], 403)
        self.assertEqual(self.post('/api/session/start', {'mode': 'normal', 'sid': 'owner'},
                                   {'Origin': f'http://127.0.0.1:{self.port}'})[0], 200)
        handler = server.make_handler(self.manager)
        fake = SimpleNamespace(headers={'Host': f'127.0.0.1:{self.port}'},
                               client_address=('203.0.113.2', 1), server=self.http)
        self.assertFalse(handler._allowed(fake))
        self.assertFalse(server.loopback_host('0.0.0.0'))
        self.assertFalse(server.loopback_host('192.168.1.2'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
