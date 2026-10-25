"""The website's HTTP surface: the built React site plus a small /api that talks to the engine."""
import json
import mimetypes
import re
from urllib.parse import urlparse

from django.conf import settings
from django.http import FileResponse, HttpResponse, JsonResponse, StreamingHttpResponse

from .engine import get_engine

LOCAL_HOSTS = {'127.0.0.1', 'localhost', '[::1]', '::1'}


def same_site(request):
    """Only this site's own pages may control the camera: refuse requests started by other websites."""
    host = request.get_host()  # validate ALLOWED_HOSTS even when Origin is absent
    if request.headers.get('Sec-Fetch-Site') == 'cross-site':
        return False
    origin = request.headers.get('Origin')
    if not origin:
        return True
    try:
        parsed = urlparse(origin)
        if parsed.scheme not in ('http', 'https') or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
            return False
        own_origin = request.scheme + '://' + host
        return origin.rstrip('/') == own_origin or origin.rstrip('/') in settings.TINK_ALLOWED_ORIGINS
    except ValueError:
        return False


def guarded(view):
    def wrapper(request, *args, **kwargs):
        if not same_site(request):
            return JsonResponse({'error': 'forbidden'}, status=403)
        return view(request, *args, **kwargs)
    wrapper.__name__ = view.__name__
    return wrapper


def body_json(request):
    try:
        data = json.loads(request.body or b'{}')
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def valid_owner(data):
    sid = data.get('sid')
    return isinstance(sid, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', sid) is not None


def no_store(response):
    response['Cache-Control'] = 'no-store'
    return response


@guarded
def state(request):
    return no_store(JsonResponse(get_engine().state()))


@guarded
def session_start(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    data = body_json(request)
    mode = data.get('mode')
    if mode not in ('normal', 'asl'):
        return JsonResponse({'error': 'mode must be "normal" or "asl"'}, status=400)
    if not valid_owner(data):
        return JsonResponse({'error': 'a session owner id is required'}, status=400)
    return no_store(JsonResponse(get_engine().start(mode, str(data.get('sid') or ''))))


@guarded
def session_stop(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    data = body_json(request)
    if not valid_owner(data):
        return JsonResponse({'error': 'a session owner id is required'}, status=400)
    return no_store(JsonResponse(get_engine().stop(data['sid'])))


@guarded
def trace(request):
    status, ctype, body = get_engine().passthrough('GET', '/api/trace')
    return no_store(HttpResponse(body, status=status, content_type=ctype))


@guarded
def passthrough(request, rest):
    """camera, confirm, reject, switch-user, asl/*: forwarded to the engine unchanged."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    if not valid_owner(body_json(request)):
        return JsonResponse({'error': 'a session owner id is required'}, status=400)
    status, ctype, body = get_engine().passthrough('POST', f'/api/{rest}', request.body or b'{}')
    return no_store(HttpResponse(body, status=status, content_type=ctype))


@guarded
def stream(request):
    """The camera, as a never-ending MJPEG response relayed from the engine."""
    opened = get_engine().open_stream()
    if opened is None:
        return JsonResponse({'error': 'no active session'}, status=409)
    conn, upstream = opened

    def relay():
        try:
            while True:
                chunk = upstream.read1(65536)
                if not chunk:
                    break
                yield chunk
        except OSError:
            pass
        finally:
            upstream.close()
            conn.close()

    response = StreamingHttpResponse(relay(), content_type=upstream.getheader('Content-Type', 'multipart/x-mixed-replace; boundary=frame'))
    return no_store(response)


def site(request, path):
    """Serve the built website; any unknown path gets index.html so the single-page app can handle it."""
    dist = settings.FRONTEND_DIST
    if not dist.is_dir():
        return HttpResponse(
            'The Tink engine is ready, but the website has not been built yet.\n\n'
            'In a terminal:  cd web/frontend && npm install && npm run build\n'
            'then reload this page. (Or just run: python start.py)\n',
            content_type='text/plain; charset=utf-8')
    root = dist.resolve()
    target = (root / path.lstrip('/')).resolve()
    if root not in target.parents or not target.is_file():
        target = root / 'index.html'
    ctype = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
    return FileResponse(open(target, 'rb'), content_type=ctype)
