"""
WebSocket URL routing
"""
from django.urls import re_path
from . import consumers
from apps.chat.routing import websocket_urlpatterns as chat_websocket_urlpatterns

websocket_urlpatterns = [
    # Core WebSocket endpoints
    re_path(r'ws/notifications/$', consumers.NotificationConsumer.as_asgi()),
    re_path(r'ws/cart/$', consumers.LiveCartConsumer.as_asgi()),
    re_path(r'ws/search/$', consumers.LiveSearchConsumer.as_asgi()),
    re_path(r'ws/analytics/$', consumers.AnalyticsConsumer.as_asgi()),
] + chat_websocket_urlpatterns  # Add chat WebSocket endpoints
