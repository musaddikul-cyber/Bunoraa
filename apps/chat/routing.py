"""
WebSocket URL Routing for Bunoraa Chat System
"""
from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    # Customer chat WebSocket
    # URL: /ws/chat/{conversation_id}/
    re_path(
        r'ws/chat/(?P<conversation_id>[0-9a-f-]+)/$',
        consumers.ChatConsumer.as_asgi(),
        name='chat_websocket'
    ),
    
    # Agent dashboard WebSocket
    # URL: /ws/chat/agents/dashboard/
    re_path(
        r'ws/chat/agents/dashboard/$',
        consumers.AgentDashboardConsumer.as_asgi(),
        name='agent_dashboard_websocket'
    ),
]
