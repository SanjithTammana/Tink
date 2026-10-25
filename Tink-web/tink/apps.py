import atexit

from django.apps import AppConfig


class TinkConfig(AppConfig):
    name = 'tink'

    def ready(self):
        from .engine import get_engine
        atexit.register(lambda: get_engine().shutdown())   # never leave the camera engine running after the site stops
