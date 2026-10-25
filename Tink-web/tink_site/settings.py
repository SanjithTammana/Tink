"""Settings for the Tink website. Local-only: no database, no accounts, no sessions."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO = BASE_DIR.parent

SECRET_KEY = 'tink-local-only-this-site-has-no-accounts'
DEBUG = os.environ.get('TINK_DEBUG') == '1'
ALLOWED_HOSTS = ['127.0.0.1', 'localhost', '[::1]']
INSTALLED_APPS = ['tink']
MIDDLEWARE = ['django.middleware.security.SecurityMiddleware']
ROOT_URLCONF = 'tink_site.urls'
WSGI_APPLICATION = 'tink_site.wsgi.application'
TEMPLATES = []
DATABASES = {}
USE_TZ = True
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Where the built React site lives (npm run build) and where the AI engine code is.
FRONTEND_DIST = Path(os.environ.get('TINK_FRONTEND_DIST', REPO / 'web' / 'frontend' / 'dist'))
BACKEND_DIR = Path(os.environ.get('TINK_BACKEND_DIR', REPO))
ENGINE_SERVER = BASE_DIR / 'engine' / 'server.py'
ENGINE_PYTHON = os.environ.get('TINK_PYTHON', '')          # default: the backend's .venv, else this Python
ENGINE_CAMERA = int(os.environ.get('TINK_CAMERA', '0'))
ENGINE_AUDIO = os.environ.get('TINK_AUDIO', '1') != '0'    # TINK_AUDIO=0 runs the Normal page without the microphone
ENGINE_MIC = os.environ.get('TINK_MIC', '')                # microphone device index (default: system default)
ENGINE_LOG = Path(os.environ.get('TINK_ENGINE_LOG', BASE_DIR / 'engine.log'))
TINK_ALLOWED_ORIGINS = tuple(value.strip() for value in os.environ.get('TINK_ALLOWED_ORIGINS', '').split(',') if value.strip())
