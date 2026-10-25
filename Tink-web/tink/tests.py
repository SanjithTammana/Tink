"""Run with:  python manage.py test      (no camera or AI models needed)"""
from django.test import SimpleTestCase
from unittest.mock import patch


class SiteTests(SimpleTestCase):
    def test_home_is_served(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_state_is_idle_before_any_page_opens(self):
        response = self.client.get('/api/state')
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()['phase'], ('idle', 'loading', 'running', 'error'))

    def test_unknown_mode_is_rejected(self):
        response = self.client.post('/api/session/start', {'mode': 'nope'}, content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_other_websites_are_refused(self):
        response = self.client.get('/api/state', HTTP_SEC_FETCH_SITE='cross-site')
        self.assertEqual(response.status_code, 403)

    def test_other_localhost_port_cannot_control_camera(self):
        response = self.client.post('/api/session/start', {'mode': 'normal', 'sid': 'owner-1'},
            content_type='application/json', HTTP_HOST='localhost:8000', HTTP_ORIGIN='http://localhost:9000')
        self.assertEqual(response.status_code, 403)

    def test_untrusted_host_without_origin_cannot_read_private_endpoints(self):
        for path in ('/api/trace', '/api/stream'):
            with self.subTest(path=path), patch('tink.views.get_engine') as get_engine:
                response = self.client.get(path, HTTP_HOST='untrusted.example')
                self.assertEqual(response.status_code, 400)
                get_engine.assert_not_called()

    def test_exact_own_origin_is_allowed(self):
        with patch('tink.views.get_engine') as get_engine:
            get_engine.return_value.state.return_value = {'phase': 'idle'}
            response = self.client.get('/api/state', HTTP_HOST='localhost:8000', HTTP_ORIGIN='http://localhost:8000')
            self.assertEqual(response.status_code, 200)

    def test_explicit_dev_origin_can_be_allowed(self):
        with self.settings(TINK_ALLOWED_ORIGINS=('http://localhost:5173',)):
            with patch('tink.views.get_engine') as get_engine:
                get_engine.return_value.state.return_value = {'phase': 'idle'}
                response = self.client.get('/api/state', HTTP_HOST='localhost:8000', HTTP_ORIGIN='http://localhost:5173')
                self.assertEqual(response.status_code, 200)

    def test_mutations_require_page_owner(self):
        for path, body in [('/api/session/start', {'mode': 'normal'}), ('/api/session/stop', {}),
                           ('/api/camera', {'on': True}), ('/api/asl/add', {'letter': 'A'})]:
            with self.subTest(path=path), patch('tink.views.get_engine') as get_engine:
                response = self.client.post(path, body, content_type='application/json')
                self.assertEqual(response.status_code, 400)
                get_engine.assert_not_called()

    def test_static_path_traversal_never_serves_files_outside_build(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist = root / 'dist'
            dist.mkdir()
            (dist / 'index.html').write_text('public app')
            (root / 'private.txt').write_text('must stay private')
            with self.settings(FRONTEND_DIST=dist):
                response = self.client.get('/../private.txt')
                self.assertEqual(b''.join(response.streaming_content), b'public app')
