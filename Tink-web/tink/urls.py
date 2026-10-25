from django.urls import path, re_path

from . import views

urlpatterns = [
    path('api/state', views.state),
    path('api/session/start', views.session_start),
    path('api/session/stop', views.session_stop),
    path('api/stream', views.stream),
    path('api/trace', views.trace),
    re_path(r'^api/(?P<rest>.+)$', views.passthrough),     # camera, confirm, reject, switch-user, asl/*
    re_path(r'^(?P<path>.*)$', views.site),                # the website (and its assets)
]
