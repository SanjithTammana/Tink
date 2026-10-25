"""Runs and talks to the Tink engine: Tink-web/engine/server.py (which wraps the backend folder) in its own process.

Why a separate process: the camera, MediaPipe, LightGBM and torch are native code. If one of them crashes
(a segmentation fault), the website must survive and say what happened instead of silently dying. The engine is
started automatically the first time a page needs it, restarted if it crashes, and, for the Normal page, retried
with the riskiest models switched off so the page still works.

Plain Python (no Django imports) so it can be tested on its own.
"""
import atexit
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# The engine is restarted on demand after a crash. (The old retry ladder that switched models off is gone: the new
# backend has no optional models to drop; a crash is reported with its log instead.)
LADDER = [([], None)]


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def exit_reason(code):
    if code is None:
        return 'unknown'
    if code < 0:
        try:
            return f'killed by {signal.Signals(-code).name}'
        except ValueError:
            return f'killed by signal {-code}'
    return f'exit code {code}'


class Engine:
    def __init__(self, backend_dir, log_path, python=None, camera=0, audio=True, ready_timeout=25, server=None, mic=''):
        self.backend_dir, self.log_path = Path(backend_dir), Path(log_path)
        self.server = Path(server) if server else Path(__file__).resolve().parent.parent / 'engine' / 'server.py'
        self.mic = str(mic or '')
        self.python = python or self._default_python()
        self.camera, self.audio, self.ready_timeout = camera, audio, ready_timeout
        self.lock = threading.RLock()
        self.proc = None
        self.port = None
        self.level = 0                  # index into LADDER
        self.want = None                # {'mode', 'sid'} the page that is waiting for a session
        self.failed = None              # {'reason', 'log'} until the next start
        self.crashes = []

    # -- process management ----------------------------------------------------------------------------------
    def _default_python(self):
        for folder in (self.backend_dir, self.server.parent.parent):
            for rel in ('.venv/bin/python', '.venv/Scripts/python.exe'):
                candidate = folder / rel
                if candidate.exists():
                    return str(candidate)
        return sys.executable

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _spawn(self):
        self.port = free_port()
        args = LADDER[self.level][0]
        cmd = [self.python, str(self.server), '--backend', str(self.backend_dir), '--port', str(self.port),
               '--camera', str(self.camera), '--exit-with-parent', '--parent-pid', str(os.getpid())]
        if not self.audio:
            cmd.append('--no-audio')
        if self.mic:
            cmd.append('--device=' + self.mic)
        if args:
            cmd.extend(args)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, 'ab')
        log.write(f'\n=== engine start {time.strftime("%H:%M:%S")} level {self.level}: {" ".join(cmd)}\n'.encode())
        log.flush()
        env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONFAULTHANDLER': '1'}
        self.proc = subprocess.Popen(cmd, cwd=self.backend_dir, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.close()
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f'The engine stopped while starting ({exit_reason(self.proc.returncode)}).')
            try:
                self._raw('GET', '/api/state', timeout=1)
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError('The engine did not start in time.')

    def log_tail(self, lines=14):
        try:
            data = self.log_path.read_bytes()[-6000:].decode('utf-8', 'replace')
        except OSError:
            return ''
        return '\n'.join(data.strip().splitlines()[-lines:])

    def shutdown(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                # Windows terminate() skips Python finally blocks. First let the owner
                # finalize its clip and join any assessment before stopping the server.
                deadline = time.monotonic() + 60.
                try:
                    if self.want and self.want.get('sid'):
                        self._raw('POST', '/api/session/stop', {'sid': self.want['sid']}, timeout=2)
                    while self.proc.poll() is None and time.monotonic() < deadline:
                        _, _, body = self._raw('GET', '/api/state', timeout=2)
                        if json.loads(body).get('phase') in ('idle', 'stopped', 'error'):
                            break
                        time.sleep(.05)
                except (OSError, ValueError):
                    pass
                if self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=6)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait(timeout=3)
            self.proc = None
            self.want = None

    # -- talking to it -----------------------------------------------------------------------------------------
    def _raw(self, method, path, body=None, timeout=10):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=timeout)
        try:
            payload = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
            conn.request(method, path, payload, {'Content-Type': 'application/json'})
            response = conn.getresponse()
            return response.status, response.getheader('Content-Type', 'application/json'), response.read()
        finally:
            conn.close()

    def open_stream(self):
        """(connection, response) for the camera stream, or None if no session is running."""
        if not self.alive():
            return None
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=15)
        try:
            conn.request('GET', '/api/stream')
            response = conn.getresponse()
        except OSError:
            conn.close()
            return None
        if response.status != 200:
            response.read()
            conn.close()
            return None
        return conn, response

    def passthrough(self, method, path, body=None):
        """Forward a request to the engine. Returns (status, content_type, bytes)."""
        with self.lock:
            self._check_crash()
        if not self.alive():
            return 409, 'application/json', json.dumps({'error': 'no active session'}).encode()
        try:
            return self._raw(method, path, body)
        except OSError:
            return 503, 'application/json', json.dumps({'error': 'the engine did not respond'}).encode()

    # -- crashes ---------------------------------------------------------------------------------------------
    def _info(self):
        degraded = [label for _, label in LADDER[1:self.level + 1]]
        return {'level': self.level, 'turned_off': degraded, 'crashes': len(self.crashes)}

    def _check_crash(self):
        """If the engine process died on its own, record it and (Normal page) retry with fewer models."""
        if self.proc is None or self.proc.poll() is None:
            return
        reason, tail = exit_reason(self.proc.returncode), self.log_tail()
        self.proc = None
        self.crashes.append({'at': time.strftime('%H:%M:%S'), 'reason': reason, 'log': tail})
        want = self.want
        if want and want['mode'] == 'normal' and self.level < len(LADDER) - 1:
            self.level += 1
            try:
                self._spawn()
                self._raw('POST', '/api/session/start', want)
                return
            except (OSError, RuntimeError) as exc:
                reason = f'{reason}; retry failed: {exc}'
                tail = self.log_tail()
        self.failed = {'reason': reason, 'log': tail}

    def _error_state(self):
        reason = self.failed['reason']
        return {'mode': (self.want or {}).get('mode'), 'phase': 'error',
                'error': f'The Tink engine crashed ({reason}) while running this page.',
                'engine': {**self._info(), 'log': self.failed['log']}}

    # -- the API the views use ------------------------------------------------------------------------------
    def state(self):
        with self.lock:
            self._check_crash()
            if self.failed:
                return self._error_state()
            if not self.alive():
                return {'mode': None, 'phase': 'idle', 'error': None, 'engine': self._info()}
            try:
                _, _, body = self._raw('GET', '/api/state')
                data = json.loads(body)
            except (OSError, ValueError):
                self._check_crash()
                if self.failed:
                    return self._error_state()
                return {'mode': (self.want or {}).get('mode'), 'phase': 'loading', 'error': None,
                        'engine': self._info()}
            data['engine'] = self._info()
            return data

    def start(self, mode, sid):
        with self.lock:
            self.failed = None
            self.want = {'mode': mode, 'sid': sid}
            self._check_crash()
            self.failed = None
            if not self.alive():
                try:
                    self._spawn()
                except RuntimeError as exc:
                    self.failed = {'reason': str(exc), 'log': self.log_tail()}
                    return self._error_state()
            try:
                status, _, body = self._raw('POST', '/api/session/start', self.want)
            except OSError:
                self._check_crash()
                return self.state()
            if status == 400:
                return {'mode': mode, 'phase': 'error', 'error': json.loads(body).get('error'), 'engine': self._info()}
            return self.state()

    def stop(self, sid=None):
        with self.lock:
            if sid and self.want and self.want.get('sid') != sid:
                return self.state()          # a newer page owns the session now: this stop is stale
            if self.alive():
                try:
                    self._raw('POST', '/api/session/stop', {'sid': sid} if sid else {})
                except OSError:
                    pass
            if not sid or (self.want and self.want.get('sid') == sid):
                self.want = None
            return self.state()


_engine = None
_engine_lock = threading.Lock()


def get_engine():
    """The one Engine for this website, configured from Django settings."""
    global _engine
    with _engine_lock:
        if _engine is None:
            from django.conf import settings
            _engine = Engine(settings.BACKEND_DIR, settings.ENGINE_LOG, settings.ENGINE_PYTHON or None,
                             settings.ENGINE_CAMERA, settings.ENGINE_AUDIO, server=getattr(settings, 'ENGINE_SERVER', None),
                             mic=getattr(settings, 'ENGINE_MIC', ''))
            atexit.register(_engine.shutdown)
        return _engine
