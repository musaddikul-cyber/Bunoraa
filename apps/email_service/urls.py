"""
Email Service URL Configuration
================================

URL patterns for the email service API and tracking endpoints.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

from . import views

app_name = 'email_service'

# API Router
router = DefaultRouter()
router.register(r'api-keys', views.APIKeyViewSet, basename='apikey')
router.register(r'domains', views.SenderDomainViewSet, basename='domain')
router.register(r'identities', views.SenderIdentityViewSet, basename='identity')
router.register(r'templates', views.EmailTemplateViewSet, basename='template')
router.register(r'messages', views.EmailMessageViewSet, basename='message')
router.register(r'suppressions', views.SuppressionViewSet, basename='suppression')
router.register(r'unsubscribe-groups', views.UnsubscribeGroupViewSet, basename='unsubscribe-group')
router.register(r'webhooks', views.WebhookViewSet, basename='webhook')

urlpatterns = [
    # Main API endpoints
    path('v1/', include(router.urls)),
    
    # Mail send endpoint (primary API)
    path('v1/mail/send/', views.MailSendView.as_view(), name='mail_send'),
    
    # Statistics
    path('v1/stats/', views.StatsView.as_view(), name='stats'),
    
    # Tracking endpoints (public, no auth)
    path('track/open/<str:message_id>/pixel.gif', views.TrackOpenView.as_view(), name='track_open'),
    path('track/click/<str:message_id>/<str:url>/', views.TrackClickView.as_view(), name='track_click'),
    
    # Unsubscribe endpoint (public)
    path('unsubscribe/<str:message_id>/', views.UnsubscribeView.as_view(), name='unsubscribe'),
]
