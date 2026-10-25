#!/usr/bin/env python
"""One command to run Tink:   python start.py

Builds the website if it is missing or out of date (needs Node.js), then starts the Django site at
http://127.0.0.1:8000 and opens it in your browser. The AI engine starts by itself when a page needs it.

Run it with the Python that has Tink's packages (the backend's .venv). If Django is missing it installs it for you.
Options:  --port 8000   --no-browser   --no-build   --no-audio
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRONTEND = HERE.parent / 'web' / 'frontend'


def newest(path, suffixes):
    files = [p for p in path.rglob('*') if p.is_file() and p.suffix in suffixes]
    return max((p.stat().st_mtime for p in files), default=0)


def build_site():
    dist = FRONTEND / 'dist' / 'index.html'
    sources = max(newest(FRONTEND / 'src', {'.js', '.jsx', '.css'}), (FRONTEND / 'index.html').stat().st_mtime,
                  (FRONTEND / 'package.json').stat().st_mtime)
    if dist.exists() and dist.stat().st_mtime >= sources:
        return True
    npm = shutil.which('npm')
    if not npm:
        print('Node.js (npm) is needed once to build the website: install it from https://nodejs.org, then rerun.')
        return False
    print('Building the website (first time only, or after changes)...')
    if not (FRONTEND / 'node_modules' / 'vite').exists():
        install = 'ci' if (FRONTEND / 'package-lock.json').exists() else 'install'
        if subprocess.call([npm, install, '--no-audit', '--no-fund'], cwd=FRONTEND) != 0:
            return False
    return subprocess.call([npm, 'run', 'build'], cwd=FRONTEND) == 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--no-browser', action='store_true')
    p.add_argument('--no-build', action='store_true')
    p.add_argument('--no-audio', action='store_true', help='Normal page without the microphone')
    args = p.parse_args()
    try:
        import django  # noqa: F401
    except ImportError:
        print(f'Django is not installed for {sys.executable}. Installing it now...')
        if subprocess.call([sys.executable, '-m', 'pip', 'install', '-r', str(HERE / 'requirements.txt')]) != 0:
            sys.exit(f'Could not install Django. Try this yourself:\n  "{sys.executable}" -m pip install django')
        os.execv(sys.executable, [sys.executable] + sys.argv)      # restart so the new package is found
    if not args.no_build and not build_site():
        sys.exit('The website could not be built. See the messages above.')
    if args.no_audio:
        os.environ['TINK_AUDIO'] = '0'
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tink_site.settings')
    url = f'http://127.0.0.1:{args.port}/'
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
    print(f'Tink is starting at {url}  (Ctrl+C to stop)')
    from django.core.management import execute_from_command_line
    # --noreload: one process, so the camera engine can never be orphaned by an auto-restart
    execute_from_command_line(['manage.py', 'runserver', f'127.0.0.1:{args.port}', '--noreload'])


if __name__ == '__main__':
    main()
