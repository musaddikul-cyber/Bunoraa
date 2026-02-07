"""
Notifications API URL configuration

URL structure:
- GET /api/v1/notifications/ - List user notifications
- GET /api/v1/notifications/{id}/ - Get notification detail
- DELETE /api/v1/notifications/{id}/ - Delete notification
- GET /api/v1/notifications/unread_count/ - Get unread count
- POST /api/v1/notifications/mark_read/ - Mark notifications as read
- POST /api/v1/notifications/mark_all_read/ - Mark all as read
- GET/PUT /api/v1/notifications/preferences/ - Notification preferences
- POST /api/v1/notifications/push-tokens/ - Register push token
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import NotificationViewSet, NotificationPreferenceViewSet, PushTokenViewSet

router = DefaultRouter()
# Register at root path to avoid double 'notifications/notifications/' URL
router.register(r'', NotificationViewSet, basename='notifications')

urlpatterns = [
    # Preferences endpoint (before router to avoid conflicts)
    path('preferences/', NotificationPreferenceViewSet.as_view({
        'get': 'list',
        'put': 'update',
        'patch': 'update'
    }), name='notification-preferences'),
    path('push-tokens/', PushTokenViewSet.as_view({
        'post': 'create'
    }), name='push-token-create'),
    path('push-tokens/<str:pk>/', PushTokenViewSet.as_view({
        'delete': 'destroy'
    }), name='push-token-delete'),
    # Router URLs
    path('', include(router.urls)),
]
