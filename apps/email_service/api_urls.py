"""
Email Service API URL Configuration (v1)
=======================================
Versioned API endpoints for /api/v1/email/
"""
from django.urls import path, include
from core.api.routers import SimpleRouter

from . import views

app_name = 'email_service_api'

router = SimpleRouter()
router.register(r'api-keys', views.APIKeyViewSet, basename='apikey')
router.register(r'domains', views.SenderDomainViewSet, basename='domain')
router.register(r'identities', views.SenderIdentityViewSet, basename='identity')
router.register(r'templates', views.EmailTemplateViewSet, basename='template')
router.register(r'messages', views.EmailMessageViewSet, basename='message')
router.register(r'suppressions', views.SuppressionViewSet, basename='suppression')
router.register(r'unsubscribe-groups', views.UnsubscribeGroupViewSet, basename='unsubscribe-group')
router.register(r'webhooks', views.WebhookViewSet, basename='webhook')

urlpatterns = [
    path('', include(router.urls)),
    path('mail/send/', views.MailSendView.as_view(), name='mail_send'),
    path('stats/', views.StatsView.as_view(), name='stats'),
]
