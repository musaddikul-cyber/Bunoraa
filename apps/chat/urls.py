"""
URL Configuration for Bunoraa Chat API
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .api.views import (
    ChatAgentViewSet,
    ConversationViewSet,
    MessageViewSet,
    CannedResponseViewSet,
    ChatSettingsViewSet,
    ChatAnalyticsViewSet
)

app_name = 'chat'

# Create router
router = DefaultRouter()
router.register(r'agents', ChatAgentViewSet, basename='agent')
router.register(r'conversations', ConversationViewSet, basename='conversation')
router.register(r'messages', MessageViewSet, basename='message')
router.register(r'canned-responses', CannedResponseViewSet, basename='canned-response')
router.register(r'settings', ChatSettingsViewSet, basename='settings')
router.register(r'analytics', ChatAnalyticsViewSet, basename='analytics')

urlpatterns = [
    path('', include(router.urls)),
]
